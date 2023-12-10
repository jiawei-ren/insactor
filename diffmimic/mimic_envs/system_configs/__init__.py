from .HUMANOID import _SYSTEM_CONFIG_HUMANOID
from .SWORDSHIELD import _SYSTEM_CONFIG_SWORDSHIELD
from .SMPL import _SYSTEM_CONFIG_SMPL
from .SMPLOLD import _SYSTEM_CONFIG_SMPL_OLD
from .SMPLv2 import _SYSTEM_CONFIG_SMPL as _SYSTEM_CONFIG_SMPL_V2
from .SMPLv2Ext import _SYSTEM_CONFIG_SMPL as _SYSTEM_CONFIG_SMPL_V2EXT
from .SMPLv2Torq import _SYSTEM_CONFIG_SMPL as _SYSTEM_CONFIG_SMPL_V2TORQ
from .SMPLv2Friction import _SYSTEM_CONFIG_SMPL as _SYSTEM_CONFIG_SMPL_V2FRIC

from google.protobuf import text_format
from brax.physics.config_pb2 import Config


def get_system_cfg(system_type):
    return {
      'humanoid': _SYSTEM_CONFIG_HUMANOID,
      'swordshield': _SYSTEM_CONFIG_SWORDSHIELD,
      'smpl': _SYSTEM_CONFIG_SMPL,
      'smpl_old': _SYSTEM_CONFIG_SMPL_OLD,
      'smpl_v2': _SYSTEM_CONFIG_SMPL_V2,
      'smpl_v2ext': _SYSTEM_CONFIG_SMPL_V2EXT,
      'smpl_v2torq': _SYSTEM_CONFIG_SMPL_V2TORQ,
      'smpl_v2fric': _SYSTEM_CONFIG_SMPL_V2FRIC
    }[system_type]


def process_system_cfg(cfg):
    return text_format.Parse(cfg, Config())