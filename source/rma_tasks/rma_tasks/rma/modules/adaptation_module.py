import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.networks import MLP, EmpiricalNormalization
from rma_tasks.rma.modules import BasePolicy


class AdaptationModule(nn.Module):
    """Encodes history into latent ẑ, then uses frozen teacher policy for action selection."""
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        prev_step_size=48,
        history_len=40,
        z_size=8,
        adaptation_obs_normalization=False,
        hidden_dim=32,
        cnn_hidden_dim=32,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        teacher=None,  # <-- Added: frozen teacher policy
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print("AdaptationModule.__init__ got unexpected args:", list(kwargs.keys()))

        self.z_size = z_size
        self.teacher = teacher
        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.history_len = history_len

        # ----------------------------------------------------------
        # Determine per-step observation dimensions
        num_actor_obs = 0
        for obs_group in obs_groups["adaptation_obs"]:
            num_actor_obs += obs[obs_group].shape[-1]
        assert num_actor_obs % self.history_len == 0, "num_actor_obs must be divisible by history_len"
        self.per_step_dim = num_actor_obs // self.history_len

        # ----------------------------------------------------------
        # Student encoder (MLP + 1D temporal conv)
        self.embed_mlp = nn.Sequential(
            nn.Linear(self.per_step_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, cnn_hidden_dim, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=5, stride=1),
            nn.ReLU(),
        )

        self.fc_out = nn.Linear(cnn_hidden_dim, z_size)
        self.adaptation_obs_normalizer = (
            EmpiricalNormalization(self.per_step_dim)
            if adaptation_obs_normalization else nn.Identity()
        )

    # ----------------------------------------------------------
    # Latent Encoder
    # ----------------------------------------------------------
    def get_latents(self, obs):
        """Compute 8-D latent ẑ given 40×48 flattened history."""
        adaptation_obs = self.get_adaptation_obs(obs)
        B, flat_dim = adaptation_obs.shape
        assert flat_dim == self.history_len * self.per_step_dim, \
            f"Expected {self.history_len * self.per_step_dim}, got {flat_dim}"
        x = adaptation_obs.view(B, self.history_len, self.per_step_dim)
        x = self.adaptation_obs_normalizer(x)
        x = self.embed_mlp(x.reshape(B * self.history_len, self.per_step_dim))
        x = x.view(B, self.history_len, -1).transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.flatten(1)
        z_hat = self.fc_out(x)
        return z_hat

    def forward(self, obs):
        return self.get_latents(obs)

    # ----------------------------------------------------------
    # Acting (via frozen teacher)
    # ----------------------------------------------------------
    def act(self, obs):
        """Encode obs → latent ẑ → pass through frozen teacher to get actions."""
        assert self.teacher is not None, "Teacher policy not attached to AdaptationModule"

        # 1. Student latent
        z_hat = self.get_latents(obs)

        # 2. Teacher actor observations (not encoder obs!)
        policy_obs = self.teacher.get_actor_obs(obs)
        policy_obs = self.teacher.actor_obs_normalizer(policy_obs)

        # 3. Combine [policy_obs, ẑ]
        teacher_input = torch.cat([policy_obs, z_hat], dim=-1)

        # 4. Forward through teacher actor
        actions = self.teacher.actor(teacher_input)
        return actions

    # ----------------------------------------------------------
    def get_adaptation_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["adaptation_obs"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def update_normalization(self, obs):
        if isinstance(self.adaptation_obs_normalizer, EmpiricalNormalization):
            adaptation_obs = self.get_adaptation_obs(obs)
            B, flat_dim = adaptation_obs.shape
            x = adaptation_obs.view(B, self.history_len, self.per_step_dim)
            self.adaptation_obs_normalizer.update(x)

    def reset(self, dones=None):
        pass

    def load_teacher(self, ckpt_path, obs, obs_groups, num_actions, teacher_cfg, device="cpu"):
        """
        Loads a pretrained BasePolicy from checkpoint and attaches it as a frozen teacher.
        """
        print(f"[INFO] Loading teacher checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # -------------------------------
        # 1. Rebuild the BasePolicy (same arch)
        # -------------------------------
        self.teacher = BasePolicy(
            obs,
            obs_groups,
            num_actions,
            **teacher_cfg,
        ).to(device)

        # -------------------------------
        # 2. Load weights (ignore missing keys if extra optimizer keys exist)
        # -------------------------------
        res = self.teacher.load_state_dict(state_dict, strict=False)
        if isinstance(res, (tuple, list)):
            missing, unexpected = res
            if missing:
                print(f"[WARN] Missing keys when loading teacher: {missing}")
            if unexpected:
                print(f"[WARN] Unexpected keys when loading teacher: {unexpected}")
        else:
            print("[INFO] Loaded teacher state_dict (strict=False). No missing/unexpected key info available.")

        # -------------------------------
        # 3. Freeze teacher parameters
        # -------------------------------
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.loaded_teacher = True
        print(f"[INFO] Teacher successfully loaded and frozen from {ckpt_path}")

