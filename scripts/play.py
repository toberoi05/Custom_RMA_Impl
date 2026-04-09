# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained RSL-RL checkpoint (supports RMA DistillationRunner student)."""

import argparse
import sys

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# -----------------------------------------------------------------------------#
# CLI arguments
# -----------------------------------------------------------------------------#

parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL (RMA-compatible).")

parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200,
                    help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None,
                    help="Name of the task (e.g., RMA1-Spot-Adaptation-v0).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None,
                    help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the published pre-trained checkpoint from Nucleus (if available).",
)
parser.add_argument(
    "--real-time", action="store_true", default=False,
    help="Run in (approximate) real-time, if possible."
)

# Standard RSL-RL args: includes --checkpoint, --device, etc.
cli_args.add_rsl_rl_args(parser)
# Isaac Sim / AppLauncher args
AppLauncher.add_app_launcher_args(parser)

# Parse CLI + Hydra args
args_cli, hydra_args = parser.parse_known_args()

# If recording video, force cameras on
if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse / Isaac Sim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------#
# Imports that require Isaac Sim to be launched first
# -----------------------------------------------------------------------------#

import os
import time

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner  # For non-RMA tasks

# Use your custom RMA DistillationRunner for RMA checkpoints
from rma_tasks.rma.runners import DistillationRunner as RMADistillationRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401  (default tasks)
import rma_tasks       # noqa: F401  (your RMA tasks / env registration)

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# -----------------------------------------------------------------------------#
# Main: Hydra-configured entry
# -----------------------------------------------------------------------------#


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    """Play a trained RSL-RL agent (RMA-compatible)."""

    # Task name (e.g., "RMA1-Spot-Adaptation-v0" or "...-Play")
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # Override config with non-Hydra CLI args
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Seed + device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Logging root (only used to locate checkpoints / videos)
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # ---------------------------------------------------------------------#
    # Resolve which checkpoint to load
    # ---------------------------------------------------------------------#
    if args_cli.use_pretrained_checkpoint:
        # Published pre-trained (not typically used for your custom RMA runs)
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No published pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        # Directly provided checkpoint (RECOMMENDED for your student)
        # e.g. --checkpoint logs/rsl_rl/spot_rma_adaptation/2025-11-18_12-34-56/model_19500.pt
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # Fall back to "latest" in the experiment directory
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Using checkpoint: {resume_path}")

    # Feed env the log_dir (for consistency)
    env_cfg.log_dir = log_dir

    # ---------------------------------------------------------------------#
    # Create Isaac environment
    # ---------------------------------------------------------------------#
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert multi-agent to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---------------------------------------------------------------------#
    # Construct runner (RMA vs generic)
    # ---------------------------------------------------------------------#
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        # Generic on-policy runner (not your distillation case)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        # Your RMA DistillationRunner (student-only for inference)
        runner = RMADistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # Load weights (this is your STUDENT checkpoint)
    runner.load(resume_path)

    # Get inference policy wrapper (usually handles obs normalization etc.)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Extract the underlying neural network module (student AdaptationModule)
    try:
        # For RSL-RL >= 2.3
        policy_nn = runner.alg.policy
    except AttributeError:
        # For older versions
        policy_nn = runner.alg.actor_critic

    # Extract normalizer (important for RMA student)
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # ---------------------------------------------------------------------#
    # Export policy (useful for real robot deployment)
    # ---------------------------------------------------------------------#
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # ---------------------------------------------------------------------#
    # Play loop
    # ---------------------------------------------------------------------#
    dt = env.unwrapped.step_dt

    # Reset environment
    obs = env.get_observations()
    timestep = 0

    print("[INFO] Starting play loop. Close the Isaac Sim window to exit.")

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # Policy step (this calls your RMA student AdaptationModule under the hood)
            actions = policy(obs)

            # Environment step
            obs, _, dones, _ = env.step(actions)

            # Reset any recurrent / history state on episode termination
            # (Your AdaptationModule should implement .reset(dones) to clear history / z.)
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # Optional real-time pacing
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    # Run main
    main()
    # Then close the sim app
    simulation_app.close()
