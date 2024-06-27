from typing import List, Dict, Any

import sys
import os
from pykenlm import (
    State as KenLMState,
    WordIndex,
    Vocabulary,
    Model as KenLMModel,
    StringPiece,
    Chain,
    Chains,
    QueryPrinter as KenLMQueryPrinter,
    ModelBuffer as KenLMModelBuffer
)
from pykenlm import ngram
import numpy as np
import io


ONE_GB: int = 1 << 30
SIXTY_FOUR_MB: int = 1 << 26
NUMBER_OF_BLOCKS: int = 2


def as_str(data) -> io.BytesIO:
    if isinstance(data, bytes):
        return data
    elif isinstance(data, unicode):
        return data.encode('utf8')
    raise TypeError('Cannot convert %s to string' % type(data))


class FullScoreReturn:
    """
    Wrapper around FullScoreReturn.

    Notes:
        `prob` has been renamed to `log_prob`
        `oov` has been added to flag whether the word is OOV
    """

    log_prob: float
    ngram_length: int
    oov: int

    def __init__(self, log_prob: float, ngram_length: int, oov: int):
        self.log_prob = log_prob
        self.ngram_length = ngram_length
        self.oov = oov

    def __repr__(self):
        return '{0}({1}, {2}, {3})'.format(
            self.__class__.__name__, 
            repr(self.log_prob), repr(self.ngram_length), repr(self.oov)
        )

    @property
    def log_prob(self):
        return self.log_prob
    
    @property
    def ngram_length(self):
        return self.ngram_length
    
    @property
    def oov(self):
        return self.oov


class State:
    """
    Wrapper around lm::ngram::State so that python code can make incremental queries.

    Notes:
        * rich comparisons
        * hashable
    """

    _c_state: KenLMState

    def __richcmp__(self, qa: KenLMState, qb: KenLMState, op: int):
        r = qa._c_state.Compare(qb._c_state)
        if op == 0:  # <
            return r < 0
        elif op == 1:  # <=
            return r <= 0
        elif op == 2:  # ==
            return r == 0
        elif op == 3:  # !=
            return r != 0
        elif op == 4:  # >
            return r > 0
        else:  # >=
            return r >= 0

    def __hash__(self):
        return ngram.hash_value(self._c_state)

    def __copy__(self):
        ret = KenLMState()
        ret._c_state = self._c_state
        return ret

    def __deepcopy__(self):
        return self.__copy__()


class Model:
    """
    Wrapper around lm::ngram::Model.
    """
    
    model: KenLMModel
    path: io.BytesIO
    vocab: Vocabulary

    def __init__(self, path, config: Config = Config()):
        """
        Load the language model.

        :param path: path to an arpa file or a kenlm binary file.
        :param config: configuration options (see lm/config.hh for documentation)
        """
        self.path = os.path.abspath(as_str(path))
        try:
            self.model = ngram.LoadVirtual(self.path, config._c_config)
        except RuntimeError as exception:
            exception_message = str(exception).replace('\n', ' ')
            raise IOError('Cannot read model \'{}\' ({})'.format(path, exception_message)) \
                from exception
        self.vocab = self.model.BaseVocabulary()

    def __dealloc__(self):
        del self.model

    @property
    def order(self):
        return self.model.Order()

    def score(self, sentence, bos: bool=True, eos: bool=True):
        """
        Return the log10 probability of a string.  By default, the string is
        treated as a sentence.
            return log10 p(sentence </s> | <s>)

        If you do not want to condition on the beginning of sentence, pass
            bos = False
        Never includes <s> as part of the string.  That would be predicting the
        beginning of sentence.  Language models are only supposed to condition
        on it as context.

        Similarly, the end of sentence token </s> can be omitted with
            eos = False
        Since language models explicitly predict </s>, it can be part of the
        string.

        Examples:

        # Good: returns log10 p(this is a sentence . </s> | <s>)
        model.score("this is a sentence .")
        # Good: same as the above but more explicit
        model.score("this is a sentence .", bos = True, eos = True)

        # Bad: never includes <s>
        model.score("<s> this is a sentence")
        # Bad: never includes <s>, even if bos = False.
        model.score("<s> this is a sentence", bos = False)

        # Good: returns log10 p(a fragment)
        model.score("a fragment", bos = False, eos = False)

        # Good: returns log10 p(a fragment </s>)
        model.score("a fragment", bos = False, eos = True)

        # Ok, but bad practice: returns log10 p(a fragment </s>)
        # Unlike <s>, the end of sentence token </s> can appear explicitly.
        model.score("a fragment </s>", bos = False, eos = False)
        """
        if bos and eos:
            return ScoreSentence(self.model, str(sentence))
        
        words = str(sentence).split()  # type: List
        state = State()
        
        if bos:
            self.model.BeginSentenceWrite(state)
        else:
            self.model.NullContextWrite(state)
        
        total = 0.0
        for word in words:
            total += self.model.BaseScore(state, self.vocab.Index(word), out_state)
            state = out_state
        if eos:
            total += self.model.BaseScore(state, self.vocab.EndSentence(), out_state)
        return total


    def GetVocabulary(self):
        pass


    def perplexity(self, sentence):
        """
        Compute perplexity of a sentence.
        @param sentence One full sentence to score.  Do not includes <s> or </s>.
        """
        words = len(as_str(sentence).split()) + 1  # For </s>
        return 10.0 ** (-self.score(sentence) / words)


    def full_scores(self, sentence, bos=True, eos=True):
        """
        full_scores(sentence, bos = True, eos = True) -> generate full scores (prob, ngram length, oov)
        @param sentence is a string (do not use boundary symbols)
        @param bos should kenlm add a bos state
        @param eos should kenlm add an eos state
        """
        words = as_str(sentence).split()
        state = State()
        if bos:
            self.model.BeginSentenceWrite(state)
        else:
            self.model.NullContextWrite(state)
        out_state = State()
        ret = FullScoreReturn()
        total: float = 0
        wid = WordIndex()
        
        for word in words:
            wid = self.vocab.Index(word)
            ret = self.model.BaseFullScore(state, wid, out_state)
            yield ret.prob, ret.ngram_length, wid == 0
            state = out_state
        
        if eos:
            ret = self.model.BaseFullScore(state, self.vocab.EndSentence(), out_state)
            yield ret.prob, ret.ngram_length, False


    def BeginSentenceWrite(self, state: State):
        """Change the given state to a BOS state."""
        self.model.BeginSentenceWrite(state._c_state)


    def NullContextWrite(self, state: State):
        """Change the given state to a NULL state."""
        self.model.NullContextWrite(state._c_state)


    def BaseScore(self, in_state: State, word: str, out_state: State) -> float:
        """
        Return p(word|in_state) and update the output state.
        Wrapper around model.BaseScore(in_state, Index(word), out_state)

        :param word: the suffix
        :param state: the context (defaults to NullContext)
        :returns: p(word|state)
        """
        total: float = self.model.BaseScore(in_state._c_state, self.vocab.Index(as_str(word)), out_state._c_state)
        return total

    def BaseFullScore(self, State in_state, str word, State out_state):
        """
        Wrapper around model.BaseFullScore(in_state, Index(word), out_state)

        :param word: the suffix
        :param state: the context (defaults to NullContext)
        :returns: FullScoreReturn(word|state)
        """
        wid: WordIndex = self.vocab.Index(as_str(word))
        ret: FullScoreReturn = self.model.BaseFullScore(in_state._c_state, wid, out_state._c_state)
        return FullScoreReturn(ret.prob, ret.ngram_length, wid == 0)

    def __contains__(self, word):
        w: io.BytesIO = as_str(word)
        return self.vocab.Index(w) != 0

    def __repr__(self):
        return '<Model from {0}>'.format(os.path.basename(self.path))

    def __reduce__(self):
        return Model, (self.path,)

    def Recognize(self):
        pass


class LanguageModel(Model):
    """Backwards compatability stub.  Use Model."""


class QueryPrinter:

    _query_p: KenLMQueryPrinter
    fd: int
    print_word: bool
    print_line: bool
    print_summary: bool

    def __init__(self, options: Dict, flush: bool = False):
        self.print_word = options['word']
        self.print_line = options['line']
        self.print_summary = options['summary']
        self.fd = sys.stdout

        self._query_p = ngram.QueryPrinter(self.fd, self.print_word, self.print_line, self.print_summary, flush)

    def __del__(self):
        del self._query_p

    def Word(self, surface: StringPiece, vocab: WordIndex, ret: FullScoreReturn):
        self._query_p.Word(surface, vocab, ret)

    def Line(self, oov: np.int64, total: np.float64) -> None:
        self._query_p.Line(oov, total)

    def Summary(
        self,
        ppl_including_oov: np.float64,
        ppl_excluding_oov: np.float64,
        corpus_oov: np.uint64,
        corpus_tokens: np.uint64
    ) -> None:
        self._query_p.Summary(ppl_including_oov, ppl_excluding_oov, corpus_oov, corpus_tokens)


class ModelBuffer:
    """
    Construct for writing.  Must call VocabFile() and fill it with null-delimited vocab words.
    """
    _m_buffer: KenLMModelBuffer

    def __init__(self, file_base: StringPiece, keep_buffer: bool, output_q: bool):
        self._m_buffer = ModelBuffer(file_base, keep_buffer, output_q)

    def __dealloc__(self):
        del self._m_buffer

    def Sink(self, chains: Chains, counts: np.ndarray):
        pass

    def Source(self, chains: Chains):
        pass

    def Source(self, order_minus_1: np.int8, chain: Chain):
        pass

    def Order(self) -> int:
        pass

    def Counts(self) -> np.ndarray:
        pass

    def VocabFile(self) -> int:
        pass

    def RawFile(self, order_minus_1: int) -> int:
        pass

    def Keep(self) -> bool:
        pass

    def SlowQuery(self, context: State, word: WordIndex, out: State) -> float:
        pass
