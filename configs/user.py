from dataclasses import dataclass, field

import obliviator.schemas as oblvsc
from evaluation.probing import ProbConfig


@dataclass(slots=True, kw_only=True)
class UserDeepCls(oblvsc.MLPConfig):
    pass


@dataclass(slots=True, kw_only=True)
class UserClsOptim(oblvsc.OptimConfig):
    pass


@dataclass(slots=True, kw_only=True)
class UserProbUnwanted(ProbConfig):
    mlp_config: oblvsc.MLPConfig = field(default_factory=UserDeepCls)
    optim_config: oblvsc.OptimConfig = field(default_factory=UserClsOptim)
    name: str = "Uwanted"
    color: oblvsc.TermColor = oblvsc.TermColor.BRIGHT_RED


@dataclass(slots=True, kw_only=True)
class UserProbUtility(ProbConfig):
    mlp_config: oblvsc.MLPConfig = field(default_factory=UserDeepCls)
    optim_config: oblvsc.OptimConfig = field(default_factory=UserClsOptim)
    name: str = "Utility"
    color: oblvsc.TermColor = oblvsc.TermColor.BRIGHT_GREEN


@dataclass(slots=True, kw_only=True)
class UserEncoder(oblvsc.MLPConfig):
    pass


@dataclass(slots=True, kw_only=True)
class UserEncoderOptim(oblvsc.OptimConfig):
    pass


@dataclass(slots=True, kw_only=True)
class UserUnsup(oblvsc.UnsupervisedConfig):
    encoder_config: oblvsc.MLPConfig = field(default_factory=UserEncoder)
    optim_config: oblvsc.OptimConfig = field(default_factory=UserEncoderOptim)


@dataclass(slots=True, kw_only=True)
class UserSup(oblvsc.SupervisedConfig):
    encoder_config: oblvsc.MLPConfig = field(default_factory=UserEncoder)
    optim_config: oblvsc.OptimConfig = field(default_factory=UserEncoderOptim)
