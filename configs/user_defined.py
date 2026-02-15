from dataclasses import dataclass

import obliviator.schemas as oblvsc


@dataclass
class UserUnsupConfig(oblvsc.UnsupervisedConfig):
    pass


@dataclass
class UserSupConfig(oblvsc.SupervisedConfig):
    pass
