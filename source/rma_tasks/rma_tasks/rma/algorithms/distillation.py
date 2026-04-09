# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from rma_tasks.rma.storage.rollout_rma_storage import RolloutRMAStorage
from rsl_rl.utils import resolve_optimizer


class Distillation:
    """Trains AdaptationModule (student encoder) to match teacher’s latent encoder."""

    def __init__(
        self,
        policy,  # AdaptationModule (contains student + frozen teacher)
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        optimizer="adam",
        device="cpu",
        multi_gpu_cfg=None,
    ):
        self.device = device
        self.policy = policy.to(device)
        self.storage = None
        self.transition = RolloutRMAStorage.Transition()

        # Multi-GPU setup
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Train only student encoder
        self.optimizer = resolve_optimizer(optimizer)(
            self.policy.parameters(), lr=learning_rate
        )

        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.max_grad_norm = max_grad_norm
        self.num_updates = 0

        loss_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        self.loss_fn = loss_dict[loss_type]

    # ----------------------------------------------------------
    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        """Initialize rollout buffer for distillation."""
        self.storage = RolloutRMAStorage(
            training_type=training_type,
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
            latent_dim=self.policy.z_size,   # match AdaptationModule latent dim
            device=self.device,
        )

    # ----------------------------------------------------------
    def act(self, obs):
        """Forward pass for environment interaction (student + teacher)."""
        # ---- Student actions..it uses teacher actor model given input as obs+z_hat ----
        actions = self.policy.act(obs)
        self.transition.actions = actions.detach()

        with torch.no_grad():
            # ---- Student latent ----
            self.transition.latents = self.policy.get_latents(obs).detach()

            # ---- Teacher latent ----
            priv_obs = self.policy.teacher.get_encoder_obs(obs)
            actor_obs = self.policy.teacher.get_actor_obs(obs)

            priv_input = self.policy.teacher.encoder_obs_normalizer(priv_obs)
            teacher_latent = self.policy.teacher.encoder(priv_input).detach()
            self.transition.teacher_latents = teacher_latent

            # ---- Teacher action...basically same as student action but need this since code was erroring out ----
            actor_input = torch.cat([actor_obs, teacher_latent], dim=-1)
            privileged_action = self.policy.teacher.actor(actor_input)
            self.transition.privileged_actions = privileged_action.detach()

        # ---- Save observation ----
        self.transition.observations = obs
        return self.transition.actions

    # ----------------------------------------------------------
    def process_env_step(self, obs, rewards, dones, extras):
        """After every env.step() — store transitions."""
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    # ----------------------------------------------------------
    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0.0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            for obs, _, _, _, _, dones in self.storage.generator():
                # Recompute latents for backprop
                z_student = self.policy.get_latents(obs)
                with torch.no_grad():
                    priv_obs = self.policy.teacher.get_encoder_obs(obs)
                    priv_input = self.policy.teacher.encoder_obs_normalizer(priv_obs)
                    z_teacher = self.policy.teacher.encoder(priv_input)

                loss = self.loss_fn(z_student, z_teacher)

                mean_behavior_loss += loss.item()
                cnt += 1

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        mean_behavior_loss /= max(1, cnt)
        self.storage.clear()
        return {"behavior": mean_behavior_loss}

    # ----------------------------------------------------------
    def broadcast_parameters(self):
        """Sync parameters across GPUs."""
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Average gradients across GPUs."""
        grads = [p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in self.policy.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.data.copy_(all_grads[offset:offset + n].view_as(p.grad))
                offset += n
