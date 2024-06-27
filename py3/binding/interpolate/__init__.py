from typing import List

import numpy as np
from pykenlm import (
    StringPiece,
    Config as KenLMConfig, 
    SortConfig as KLMSortConfig, 
    InstanceConfig as KLMInstancesConfig
)
import io


class InterpolateConfig:
    in_config_: KenLMConfig 

    def __init__(self, lambdas: np.ndarray, sort_: KLMSortConfig):
        self.in_config_.lambdas = lambdas
        self.in_config_.sort = sort_.sort_cfg

    @property
    def lambdas(self):
        return self.in_config_.lambdas

    @lambdas.setter
    def lambdas(self, v):
        self.in_config_.lambdas = v


class SortConfig:
    
    sort_cfg: KLMSortConfig
    # Filename prefix where temporary files should be placed.
    temp_prefix: KLMSortConfig
    # Size of each input/output buffer.
    buffer_size: np.int64
    # Total memory to use when running alone.
    total_memory: np.int8

    def __init__(self, temp_prefix: str, buffer_size: int, total_memory: int):
        self.sort_cfg.temp_prefix = temp_prefix
        self.sort_cfg.buffer_size = buffer_size
        self.sort_cfg.total_memory = total_memory

    def __del__(self):
        del self.sort_cfg


class InstancesConfig:
    _in_cfg: KLMInstancesConfig 

    def __init__(self, sort_: SortConfig):
        instance_cfg: InstancesConfig = None
        # TODO: FixMe: Cast to Sortconfig
        instance_cfg.sort = sort_.sort_cfg
        self._in_cfg.model_read_chain_mem = instance_cfg.sort.buffer_size
        self._in_cfg.extension_write_chain_mem = instance_cfg.sort.extension_write_chain_mem
        self._in_cfg.lazy_memory = instance_cfg.sort.total_memory
        self._in_cfg = instance_cfg

    def __dealloc__(self):
        del self._in_cfg


def TuneWeights(
    tune_file: str,
    model_names: List,
    config: InstancesConfig,
    weights_out: List[float]
):
    """

    Parameters
    ----------
    tune_file
    model_names
    config
    weights_out

    Returns
    -------

    """
    model_names_: List[StringPiece] = model_names
    # TODO: FixMe
    TuneWeights(tune_file, model_names_, config, weights_out)
