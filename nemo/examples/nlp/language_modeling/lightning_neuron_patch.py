from typing import Any, Callable, Dict, List, Optional, Union
import functools
import torch
import sys
from lightning_lite.utilities.device_parser import _check_data_type
from lightning_utilities.core.imports import RequirementCache

def auto_device_count_patched() -> int:
    """Get the devices when set to auto."""
    return 2

_XLA_AVAILABLE = RequirementCache("torch_xla")

@functools.lru_cache(maxsize=1)
def is_available() -> bool:
    # check `_XLA_AVAILABLE` again to avoid launching processes
    return bool(_XLA_AVAILABLE)

def _parse_tpu_cores_str_patched(tpu_cores: str) -> Union[int, List[int]]:
    if tpu_cores in ("1", "2", "8", "32"):
        return int(tpu_cores)
    return [int(x.strip()) for x in tpu_cores.split(",") if len(x) > 0]

def _tpu_cores_valid_patched(tpu_cores: Any) -> bool:
    # allow 1 or 8 cores
    ### NEURON: This is the allowed config on Neuron
    if tpu_cores in (1, 2, 8, 32, None):
        return True

    # allow picking 1 of 8 indexes
    if isinstance(tpu_cores, (list, tuple, set)):
        has_1_tpu_idx = len(tpu_cores) == 1
        is_valid_tpu_idx = 1 <= list(tpu_cores)[0] <= 32

        is_valid_tpu_core_choice = has_1_tpu_idx and is_valid_tpu_idx
        return is_valid_tpu_core_choice

    return False

def _parse_tpu_cores_patched(tpu_cores: Optional[Union[int, str, List[int]]]) -> Optional[Union[int, List[int]]]:
    """
    Parses the tpu_cores given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        tpu_cores: An int of 1 or string '1' indicates that 1 core with multi-processing should be used
            An int 8 or string '8' indicates that all 8 cores with multi-processing should be used
            A list of ints or a strings containing a list of comma separated integers
            indicates the specific TPU core to use.

    Returns:
        A list of tpu_cores to be used or ``None`` if no TPU cores were requested

    Raises:
        MisconfigurationException:
            If TPU cores aren't 1, 8 or [<1-8>]
    """
    _check_data_type(tpu_cores)

    if isinstance(tpu_cores, str):
        tpu_cores = _parse_tpu_cores_str_patched(tpu_cores.strip())

    if not _tpu_cores_valid_patched(tpu_cores):
        raise TypeError("`tpu_cores` can only be 1, 2, 8, 32 or [<1-8>]")

    return tpu_cores

import lightning_lite.accelerators.tpu as tpu_module
tpu_module._parse_tpu_cores = _parse_tpu_cores_patched
tpu_module._parse_tpu_cores_str = _parse_tpu_cores_str_patched
tpu_module._tpu_cores_valid = _tpu_cores_valid_patched
tpu_module.TPUAccelerator.auto_device_count = staticmethod(auto_device_count_patched)
tpu_module.TPUAccelerator.is_available = staticmethod(is_available)
