from typing import List, Any
import io
from pykenlm import (
    Discount, 
    KLMChainConfig, 
    WordIndex, 
    FilePiece, 
    ChainPosition, 
    CorpusCount
)


class Py_CorpusCount:
    _counter: CorpusCount

    def __init__(
        self,
        _from: FilePiece,
        vocab_write: int,
        dynamic_vocab: bool,
        token_count: int,
        type_count: WordIndex,
        prune_words: List[bool],
        prune_vocab_filename: str,
        entries_per_block: int
    ):
        self._counter = CorpusCount(
            _from, vocab_write, dynamic_vocab, token_count, type_count,
            prune_vocab_filename, entries_per_block)

    def Run(self, pos: ChainPosition):
        self._counter.Run(pos.c_pos_)

    def VocabUsage(self, vocab_estimate: int):
        return self._counter.VocabUsage(vocab_estimate)

    def DedupeMultiplier(self, order: int):
        return self._counter.DedupeMultiplier(order)


class InitialProbabilitiesConfig:
    
    adder_in: KLMChainConfig
    adder_out: KLMChainConfig
    # SRILM doesn't normally interpolate unigrams.
    _interpolate_unigrams: bool
    _adder_in: bool
    _adder_out: bool

    def __init__(self, interpolate_unigrams: bool = False):
        self._adder_out = None
        self._adder_in = None
        self._interpolate_unigrams = interpolate_unigrams

    @property
    def interpolate_unigrams(self):
        return self._interpolate_unigrams
    
    @interpolate_unigrams.setter
    def interpolate_unigrams(self, v):
        self._interpolate_unigrams = v

    @property
    def adder_in(self):
        return self._adder_in
    
    @adder_in.setter
    def adder_in(self, v):
        self._adder_in = v

    @property
    def adder_out(self):
        return self._adder_out
    
    @adder_out.setter
    def adder_out(self, v):
        self._adder_out = v


class PipelineConfig:
    
    __p_config: PipelineConfig
    order: int
    sort_cfg: SortConfig
    initial_probs: InitialProbabilitiesConfig
    read_backoffs: ChainConfig
    vocab_estimate: WordIndex
    vocab_size_for_unk: int
    minimum_block: int
    block_count: int

    prune_thresholds: List[int]  # mjd
    prune_vocab: bool
    prune_vocab_file: str
    renumber_vocabulary: bool
    discount: DiscountConfig
    output_q: bool

    def __init__(
        self,
        initial_probs: InitialProbabilitiesConfig,
        sort_cfg: SortConfig,
        minimum_block: int,
        block_count: int,
        vocab_estimate: int,
        vocab_size_for_unk: int,
        renumber_vocabulary: bool,
        prune_vocab_file: bool,
        output_q: bool
        # disallowed_symbol_action=catch_error
    ):
        self.initial_probs = initial_probs
        self.sort_cfg = sort_cfg
        self.minimum_block = minimum_block
        self.block_count = block_count
        self.vocab_estimate = vocab_estimate
        self.vocab_size_for_unk = vocab_size_for_unk
        self.renumber_vocabulary = renumber_vocabulary
        self.prune_vocab_file = prune_vocab_file
        self.output_q = output_q

    def __getattr__(self, key: str):
        # assert key in self
        return self.__p_config[key]

    def __setattr__(self, key: str, value: Any):
        # TODO: Check if field present
        self.__p_config[key] = value

    @property
    def config(self):
        return self.__p_config
