from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlDistillationStudentTeacherCfg,
)

@configclass
class BasePolicyCfg(RslRlPpoActorCriticCfg):
    encoder_hidden_dims: list = MISSING
    prev_step_size: int = MISSING,
    z_size: int = MISSING
@configclass
class AdaptationModuleCfg(RslRlDistillationStudentTeacherCfg):
    encoder_hidden_dims: list = MISSING
    prev_step_size: int = MISSING
    history_len: int = MISSING
    z_size: int = MISSING
