from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg
)


from rma_tasks.rma.wrappers import BasePolicyCfg
from rma_tasks.rma.wrappers import AdaptationModuleCfg
from rma_tasks.rma.algorithms.distillation import Distillation
 
# todo: add AdaptativeModuleCfg

# print(RslRlDistillationStudentTeacherCfg.__dataclass_fields__.keys())

# # or to see their types and defaults too:
# print("==== Distillation Config Fields ====")
# for k, v in RslRlDistillationStudentTeacherCfg.__dataclass_fields__.items():
#     print(f"{k}: {v.type} = {v.default}")

# print("==== PPO Config Fields ====")
# for k, v in RslRlPpoAlgorithmCfg.__dataclass_fields__.items():
#     print(f"{k}: {v.type} = {v.default}")

# print("==== OnPolicyRunner Config Fields ====")
# for k, v in RslRlOnPolicyRunnerCfg.__dataclass_fields__.items():
#     print(f"{k}: {v.type} = {v.default}")

@configclass
class Rma1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "spot_rma"
    empirical_normalization = False
    store_code_state = False
    logger = "wandb"
    wandb_project = "rma-training"
    policy = BasePolicyCfg(
        class_name="BasePolicy",
        prev_step_size=48,
        z_size=8, # Size of embedding space for priv obs
        actor_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class Rma1DistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "spot_rma_adaptation"
    empirical_normalization = False
    store_code_state = False
    logger = "wandb"
    wandb_project = "rma-training"
    seed = 42
    class_name = "DistillationRunner"
    
    teacher = BasePolicyCfg(
        class_name="BasePolicy",
        prev_step_size=48,
        z_size=8, # Size of embedding space for priv obs
        actor_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    
    student = AdaptationModuleCfg(
        class_name="AdaptationModule",
        prev_step_size=48,
        history_len=40,
        z_size=8,
        encoder_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
