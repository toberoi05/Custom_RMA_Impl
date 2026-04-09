# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from collections.abc import Generator
from tensordict import TensorDict
from rsl_rl.utils import split_and_pad_trajectories

# ----------------------------------------------------------------------
# Compatibility alias: HiddenState was removed from rsl_rl.networks.
# ----------------------------------------------------------------------
HiddenState = torch.Tensor


class RolloutRMAStorage:
    """Rollout buffer supporting latent storage for distillation (Option B)."""

    class Transition:
        def __init__(self) -> None:
            # Core
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.privileged_actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None
            self.hidden_states: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)

            # Distillation-specific
            self.latents: torch.Tensor | None = None
            self.teacher_latents: torch.Tensor | None = None

        def clear(self) -> None:
            self.__init__()

    # ------------------------------------------------------------------
    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        latent_dim: int = 8,
        device: str = "cpu",
    ) -> None:
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape
        self.latent_dim = latent_dim

        # Core tensors
        self.observations = TensorDict(
            {k: torch.zeros(num_transitions_per_env, *v.shape, device=device)
             for k, v in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()

        # Distillation tensors
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
            self.latents = torch.zeros(num_transitions_per_env, num_envs, latent_dim, device=device)
            self.teacher_latents = torch.zeros(num_transitions_per_env, num_envs, latent_dim, device=device)

        # Reinforcement-learning tensors
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)

        # RNN hidden states
        self.saved_hidden_state_a = None
        self.saved_hidden_state_c = None
        self.step = 0

    # ------------------------------------------------------------------
    def add_transitions(self, transition: Transition) -> None:
        """Add a single Transition to the buffer."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! Call clear() before adding new transitions.")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)
            if transition.latents is not None:
                self.latents[self.step].copy_(transition.latents)
            if transition.teacher_latents is not None:
                self.teacher_latents[self.step].copy_(transition.teacher_latents)

        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    # ------------------------------------------------------------------
    def _save_hidden_states(self, hidden_states: tuple[torch.Tensor | None, torch.Tensor | None]) -> None:
        if hidden_states == (None, None):
            return
        h_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        h_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        if self.saved_hidden_state_a is None:
            self.saved_hidden_state_a = [
                torch.zeros(self.observations.shape[0], *h_a[i].shape, device=self.device)
                for i in range(len(h_a))
            ]
            self.saved_hidden_state_c = [
                torch.zeros(self.observations.shape[0], *h_c[i].shape, device=self.device)
                for i in range(len(h_c))
            ]
        for i in range(len(h_a)):
            self.saved_hidden_state_a[i][self.step].copy_(h_a[i])
            self.saved_hidden_state_c[i][self.step].copy_(h_c[i])

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset the buffer."""
        self.step = 0

    # ------------------------------------------------------------------
    def generator(self) -> Generator:
        """Iterator used during distillation training."""
        if self.training_type != "distillation":
            raise ValueError("This generator is only for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield (
                self.observations[i],
                self.actions[i],
                self.privileged_actions[i],
                self.latents[i],
                self.teacher_latents[i],
                self.dones[i],
            )
