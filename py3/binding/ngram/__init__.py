from pykenlm import Config, WriteMethod, WarningAction, RestFunction
import io
import numpy as np


class NgramConfig:
    """
    Wrapper around lm::ngram::Config.
    Pass this to Model's constructor to set configuration options.
    """
    _c_config: Config

    def __init__(
        self,
        prob_bits: bool,
        backoff_bits: bool,
        pointer_bhiksha_bits: bool,
        unknown_missing_logprob: float,
        probing_multiplier: float,
        temporary_directory_prefix: str,
        building_memory: np.int8,
        write_method: WriteMethod,
        sentence_marker_missing: WarningAction,
        positive_log_probability: WarningAction,
        rest_function: RestFunction,
        include_vocab: bool,
    ):
        self._c_config.rest_function = rest_function
        self._c_config.include_vocab = include_vocab
        self._c_config.write_method = write_method
        self._c_config.sentence_marker_missing = sentence_marker_missing
        self._c_config.positive_log_probability = positive_log_probability
        self._c_config.building_memory = building_memory
        self._c_config.temporary_directory_prefix = temporary_directory_prefix
        self._c_config.prob_bits = prob_bits
        self._c_config.backoff_bits = backoff_bits
        self._c_config.probing_multiplier = probing_multiplier
        self._c_config.unknown_missing_logprob = unknown_missing_logprob
        self._c_config.pointer_bhiksha_bits = pointer_bhiksha_bits

    @property
    def load_method(self):
        return self._c_config.load_method

    @property
    def show_progress(self):
        return self._c_config.show_progress

    @property
    def arpa_complain(self):
        return self._c_config.arpa_complain
