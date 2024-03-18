from .HUMANOID import _SYSTEM_CONFIG_HUMANOID
from .SWORDSHIELD import _SYSTEM_CONFIG_SWORDSHIELD
from .SMPL import _SYSTEM_CONFIG_SMPL
from .SMPL_RFC import _SYSTEM_CONFIG_SMPL_RFC
from .SMPL_RFC_1000 import _SYSTEM_CONFIG_SMPL_RFC_1000

from google.protobuf import text_format
from brax.physics.config_pb2 import Config


def get_system_cfg(system_type):
    return {
      'humanoid': _SYSTEM_CONFIG_HUMANOID,
      'swordshield': _SYSTEM_CONFIG_SWORDSHIELD,
      'smpl': _SYSTEM_CONFIG_SMPL,
      'smpl_rfc': _SYSTEM_CONFIG_SMPL_RFC,
      'smpl_rfc_1000': _SYSTEM_CONFIG_SMPL_RFC_1000,
    }[system_type]


def process_system_cfg(cfg):
    return text_format.Parse(cfg, Config())