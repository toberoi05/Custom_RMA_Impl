# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
import inspect
from collections import deque

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state

from rma_tasks.rma.modules import BasePolicy, AdaptationModule
from rma_tasks.rma.modules import RMAStudentTeacher 
from rma_tasks.rma.storage.rollout_rma_storage import RolloutRMAStorage
from rma_tasks.rma.algorithms.distillation import Distillation

class DistillationRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of teacher-student training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", teacher_ckpt=None):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.student_cfg = train_cfg["student"]
        self.teacher_cfg = train_cfg["teacher"]
        self.policy_cfg = self.student_cfg # so that logger doesn't error out
        self.device = device
        self.env = env
        self.teacher_ckpt = teacher_ckpt  # store teacher checkpoint path

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets=["teacher", "critic", "priv_obs", "adaptation_obs"])

        # create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()
        # check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log({"loss_dict": loss_dict, "it": it})
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                if self.logger_type in ["wandb"]:
                    self.writer.callback(it)

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, data):
        """Custom logger for distillation training."""
        if self.disable_logs:
            return

        loss_dict = data.get("loss_dict", {})
        if "behavior" in loss_dict:
            print(f"[LOG] Iter {self.current_learning_iteration} | Behavior Loss: {loss_dict['behavior']:.6f}")
        # write to wandb
        if self.writer:
            self.writer.add_scalar("Loss/behavior", loss_dict["behavior"], self.current_learning_iteration)
        # if self.logger_type in ["wandb"]:
        #     self.writer.callback(data["it"])
    """
    Helper methods.
    """
    
    def _construct_algorithm(self, obs) -> Distillation:
        """Construct the Distillation algorithm using student (AdaptationModule) and teacher (BasePolicy checkpoint)."""

        # ---- Build Student ----
        student = AdaptationModule(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.student_cfg,
        ).to(self.device)
        print(f"[INFO] Built Student: {student.__class__.__name__}")

        # ---- Load Teacher ----
        student.load_teacher(
            ckpt_path=self.teacher_ckpt,
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            num_actions=self.env.num_actions,
            teacher_cfg=self.teacher_cfg,
            device=self.device,
        )
        print("[INFO] Loaded Teacher into Student successfully.")

        # ---- Build Distillation Algorithm ----
        self.alg_cfg.pop("class_name", None)  # this was causing errors so I removed it

        alg = Distillation(
            policy=student,  # student now internally holds teacher
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        print(f"[INFO] Built Distillation: {alg.__class__.__name__}")

        # ---- Initialize Storage ----
        alg.storage = RolloutRMAStorage(
            training_type="distillation",
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            obs=obs,
            actions_shape=[self.env.num_actions],
            latent_dim=student.z_size,
            device=self.device,
        )

        return alg
    
    def _prepare_logging_writer(self):
        """Prepares the logging writers (Tensorboard / W&B / Neptune)."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # determine logger type from config
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)

            elif self.logger_type == "wandb":
                from rma_utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)

            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb', or 'tensorboard'.")


