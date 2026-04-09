import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.networks import MLP, EmpiricalNormalization

class RMAStudentTeacher(nn.Module):
    """Student-Teacher wrapper for RMA distillation.
    
    Student: AdaptationModule (encoder + actor that uses predicted z)
    Teacher: BasePolicy actor (frozen, uses ground-truth z from privileged obs)
    """
    
    def __init__(
        self,
        student,  # AdaptationModule (encoder only)
        obs,
        obs_groups,
        num_actions,
        student_hidden_dims,
        teacher_hidden_dims,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        teacher_obs_normalization=False,
        **kwargs
    ):
        super().__init__()
        
        if kwargs:
            print(f"[RMAStudentTeacher] Ignoring unexpected kwargs: {list(kwargs.keys())}")
        
        # Store the student encoder
        self.student = student
        self.num_actions = num_actions
        self.obs_groups = obs_groups
        
        # Calculate observation dimensions
        num_actor_obs = 0  # Should be 48
        for obs_group in obs_groups["policy"]:
            num_actor_obs += obs[obs_group].shape[-1]
        
        # Both student and teacher actors take 48 + 8 = 56 dims
        actor_input_dim = num_actor_obs + student.z_size
        
        print(f"[RMAStudentTeacher] Building student actor: input={actor_input_dim}, hidden={student_hidden_dims}, output={num_actions}")
        print(f"[RMAStudentTeacher] Building teacher actor: input={actor_input_dim}, hidden={teacher_hidden_dims}, output={num_actions}")
        
        # Build student actor
        self.student_actor = MLP(
            input_dim=actor_input_dim,
            output_dim=num_actions,
            hidden_dims=student_hidden_dims,
            activation=activation,
        )
        
        # Build teacher actor (frozen)
        self.teacher = MLP(
            input_dim=actor_input_dim,
            output_dim=num_actions,
            hidden_dims=teacher_hidden_dims,
            activation=activation,
        )
        
        # Observation normalizers
        self.teacher_obs_normalizer = nn.Identity()
        self.student_obs_normalizer = nn.Identity()
        
        # Noise std (shared for both)
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=True)
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)), requires_grad=True)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        self.loaded_teacher = False
        
        print(f"[RMAStudentTeacher] Teacher frozen with {sum(p.numel() for p in self.teacher.parameters())} parameters")

    def act(self, observations, privileged_obs=None):
        """Forward pass during rollout collection (only student actions)."""
        policy_obs = []
        for obs_group in self.obs_groups["policy"]:
            policy_obs.append(observations[obs_group])
        policy_obs = torch.cat(policy_obs, dim=-1)

        # Student encoder -> latent z
        # wait do we even need a student actor??? because i thougth z_pred should go through the teacher_actor?
        z_pred = self.student(observations)
        student_input = torch.cat([policy_obs, z_pred], dim=-1)
        student_input = self.student_obs_normalizer(student_input)
        student_mean = self.student_actor(student_input)

        std = (
            self.std.expand_as(student_mean)
            if self.noise_std_type == "scalar"
            else torch.exp(self.log_std).expand_as(student_mean)
        )

        dist = Normal(student_mean, std)
        actions = dist.sample()
        print(f"[INFO] some random actions: {actions}")
        return actions # no need to return teacher actions in this

    # ------------------------------------------------------------------
    # Training-time evaluation (student + teacher)
    # ------------------------------------------------------------------
    def evaluate(self, observations, privileged_obs=None):
        """Used during update() to compute distillation loss."""
        policy_obs = []
        for obs_group in self.obs_groups["policy"]:
            policy_obs.append(observations[obs_group])
        policy_obs = torch.cat(policy_obs, dim=-1)

        # Student forward
        z_pred = self.student(observations)
        student_input = torch.cat([policy_obs, z_pred], dim=-1)
        student_input = self.student_obs_normalizer(student_input)
        student_mean = self.student_actor(student_input)

        # Teacher forward (no grad)
        # not sure if this is right? becuase i thought i need priv obs for teacher (Base)...
        with torch.no_grad():
            teacher_input = torch.cat([policy_obs, z_pred], dim=-1)
            teacher_input = self.teacher_obs_normalizer(teacher_input)
            teacher_mean = self.teacher(teacher_input)

        print(f"[INFO] some random means — student ({type(student_mean)}): {student_mean}, teacher ({type(teacher_mean)}): {teacher_mean}")
        return student_mean, teacher_mean  # should be called by distillation.update()

    # ------------------------------------------------------------------
    # Inference (deployment)
    # ------------------------------------------------------------------
    def act_inference(self, observations):
        """Deterministic inference with student only."""
        policy_obs = []
        for obs_group in self.obs_groups["policy"]:
            policy_obs.append(observations[obs_group])
        policy_obs = torch.cat(policy_obs, dim=-1)

        z_pred = self.student(observations)
        student_input = torch.cat([policy_obs, z_pred], dim=-1)
        student_input = self.student_obs_normalizer(student_input)
        return self.student_actor(student_input)

    # ------------------------------------------------------------------
    # Teacher loading
    # ------------------------------------------------------------------
    def load_teacher(self, checkpoint_path):
        # Load actor MLP into AdaptationModel MLP
        """Load frozen teacher weights from BasePolicy checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=next(self.teacher.parameters()).device)
        full_state = checkpoint.get("model_state_dict", checkpoint)
        print(len(full_state.keys()))
        for k in list(full_state.keys())[:100]:
            print(k)
            
        teacher_state = {k.replace("actor.", ""): v for k, v in full_state.items() if k.startswith("actor.")}
        self.teacher.load_state_dict(teacher_state, strict=False)
        self.loaded_teacher = True
        print(f"[RMAStudentTeacher] Loaded {len(teacher_state)} teacher params from {checkpoint_path}")

    def reset(self, dones=None):
        if hasattr(self.student, "reset"):
            self.student.reset(dones)


# import torch
# import torch.nn as nn
# from torch.distributions import Normal
# from rsl_rl.networks import MLP, EmpiricalNormalization

# class RMAStudentTeacher(nn.Module):
#     """Student–Teacher wrapper for RMA distillation.

#     Student: AdaptationModule encoder + student actor (learned)
#     Teacher: Frozen BasePolicy encoder + actor (loaded from checkpoint)
#     """

#     def __init__(
#         self,
#         student,                   # AdaptationModule (encoder only)
#         obs,
#         obs_groups,
#         num_actions,
#         student_hidden_dims,
#         teacher_hidden_dims,
#         activation="elu",
#         init_noise_std=1.0,
#         noise_std_type="scalar",
#         teacher_obs_normalization=False,
#         encoder_hidden_dims=[256, 256, 256],   # matches BasePolicy encoder
#         z_size=8,
#         **kwargs,
#     ):
#         super().__init__()
#         if kwargs:
#             print(f"[RMAStudentTeacher] Ignoring unexpected kwargs: {list(kwargs.keys())}")

#         self.student = student
#         self.num_actions = num_actions
#         self.obs_groups = obs_groups
#         self.z_size = z_size

#         # ---- Observation dimensions ----
#         num_actor_obs = 0  # ~48
#         for obs_group in obs_groups["policy"]:
#             num_actor_obs += obs[obs_group].shape[-1]
#         num_priv_obs = 0   # ~17
#         for obs_group in obs_groups.get("priv_obs", []):
#             num_priv_obs += obs[obs_group].shape[-1]

#         # ---- Build student + teacher networks ----
#         actor_input_dim = num_actor_obs + z_size
#         print(f"[RMAStudentTeacher] Student actor input={actor_input_dim}, teacher actor input={actor_input_dim}")

#         self.student_actor = MLP(
#             input_dim=actor_input_dim,
#             output_dim=num_actions,
#             hidden_dims=student_hidden_dims,
#             activation=activation,
#         )
#         self.teacher_encoder = MLP(
#             input_dim=num_priv_obs,
#             output_dim=z_size,
#             hidden_dims=encoder_hidden_dims,
#             activation=activation,
#         )
#         self.teacher_actor = MLP(
#             input_dim=actor_input_dim,
#             output_dim=num_actions,
#             hidden_dims=teacher_hidden_dims,
#             activation=activation,
#         )

#         # ---- Normalizers ----
#         self.teacher_obs_normalizer = nn.Identity()
#         self.student_obs_normalizer = nn.Identity()

#         # ---- Action noise std ----
#         self.noise_std_type = noise_std_type
#         if noise_std_type == "scalar":
#             self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=True)
#         elif noise_std_type == "log":
#             self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)), requires_grad=True)

#         # ---- Freeze teacher ----
#         for p in self.teacher_encoder.parameters():
#             p.requires_grad = False
#         for p in self.teacher_actor.parameters():
#             p.requires_grad = False
#         self.teacher_encoder.eval()
#         self.teacher_actor.eval()

#         self.loaded_teacher = False
#         print(f"[RMAStudentTeacher] Teacher frozen with {sum(p.numel() for p in self.teacher_actor.parameters())} actor params")

#     # --------------------------------------------------------------------- #
#     #  Rollout-time action generation (no privileged obs)
#     # --------------------------------------------------------------------- #
#     def act(self, observations, privileged_obs=None):
#         """Called during rollout collection. Student acts; teacher optional."""
#         # -- Student branch --
#         policy_obs = torch.cat([observations[g] for g in self.obs_groups["policy"]], dim=-1)
#         z_pred = self.student(observations)                 # predicted z_hat
#         student_input = torch.cat([policy_obs, z_pred], dim=-1)
#         student_mean = self.student_actor(self.student_obs_normalizer(student_input))

#         if self.noise_std_type == "scalar":
#             std = self.std.expand_as(student_mean)
#         else:
#             std = torch.exp(self.log_std).expand_as(student_mean)

#         student_actions = torch.distributions.Normal(student_mean, std).sample()

#         # -- Optional teacher branch (if privileged_obs provided) --
#         teacher_actions = None
#         if privileged_obs is not None:
#             priv_obs = torch.cat([privileged_obs[g] for g in self.obs_groups["priv_obs"]], dim=-1)
#             z_star = self.teacher_encoder(priv_obs)
#             teacher_input = torch.cat([policy_obs, z_star], dim=-1)
#             teacher_mean = self.teacher_actor(self.teacher_obs_normalizer(teacher_input))
#             teacher_actions = torch.distributions.Normal(teacher_mean, std).sample()

#         return student_actions, teacher_actions

#     # --------------------------------------------------------------------- #
#     #  PPO update-time evaluation (computes student + teacher means)
#     # --------------------------------------------------------------------- #
#     def evaluate(self, observations, privileged_obs=None):
#         """Called during PPO update — returns student_mean, teacher_mean."""
#         # -- Student --
#         policy_obs = torch.cat([observations[g] for g in self.obs_groups["policy"]], dim=-1)
#         z_pred = self.student(observations)
#         student_input = torch.cat([policy_obs, z_pred], dim=-1)
#         student_mean = self.student_actor(self.student_obs_normalizer(student_input))

#         # -- Teacher --
#         teacher_mean = None
#         if privileged_obs is not None:
#             priv_obs = torch.cat([privileged_obs[g] for g in self.obs_groups["priv_obs"]], dim=-1)
#             z_star = self.teacher_encoder(priv_obs)
#             teacher_input = torch.cat([policy_obs, z_star], dim=-1)
#             teacher_mean = self.teacher_actor(self.teacher_obs_normalizer(teacher_input))

#         return (student_mean, teacher_mean) if teacher_mean is not None else student_mean

#     # --------------------------------------------------------------------- #
#     #  Inference-only forward (student only, deterministic)
#     # --------------------------------------------------------------------- #
#     def act_inference(self, observations):
#         policy_obs = torch.cat([observations[g] for g in self.obs_groups["policy"]], dim=-1)
#         z_pred = self.student(observations)
#         student_input = torch.cat([policy_obs, z_pred], dim=-1)
#         student_mean = self.student_actor(self.student_obs_normalizer(student_input))
#         return student_mean

#     # --------------------------------------------------------------------- #
#     #  Teacher checkpoint loader (encoder + actor)
#     # --------------------------------------------------------------------- #
#     def load_teacher(self, checkpoint_path):
#         """Load frozen teacher encoder + actor weights from BasePolicy checkpoint."""
#         checkpoint = torch.load(checkpoint_path, map_location=next(self.teacher_actor.parameters()).device)
#         full_state = checkpoint.get("model_state_dict", checkpoint)

#         actor_sd, encoder_sd = {}, {}
#         for k, v in full_state.items():
#             if k.startswith("actor."):
#                 actor_sd[k.replace("actor.", "")] = v
#             elif k.startswith("encoder."):
#                 encoder_sd[k.replace("encoder.", "")] = v

#         self.teacher_actor.load_state_dict(actor_sd, strict=False)
#         self.teacher_encoder.load_state_dict(encoder_sd, strict=False)
#         self.loaded_teacher = True
#         print(f"[RMAStudentTeacher] Loaded teacher weights: {len(actor_sd)} actor, {len(encoder_sd)} encoder params")

#     # --------------------------------------------------------------------- #
#     def reset(self, dones=None):
#         if hasattr(self.student, "reset"):
#             self.student.reset(dones)