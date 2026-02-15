from dataclasses import dataclass

import obliviator.schemas as oblvsc


@dataclass
class UserUnsup(oblvsc.UnsupervisedConfig):
    pass


@dataclass
class UserSup(oblvsc.SupervisedConfig):
    pass
