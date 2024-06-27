from typing import List, NewType, Tuple
from argparse import ArgumentParser, Namespace
import inspect
import sys
from coqpit import Coqpit
from dataclasses import dataclass, field, MISSING
from enum import Enum, EnumMeta

# phrase_table_vocab_main.cc

COMMAND_TYPE = NewType('COMMAND_TYPE', str)


class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


cli = ArgumentParser()
subparsers = cli.add_subparsers(dest="subcommand")


def subcommand(args: List[Tuple], parent: ArgumentParser = subparsers):
    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        for arg, kwd in args:
            field(*arg, **kwd)
        parser.set_defaults(func=func)

    return decorator


def argument(*name_or_flags, **kwargs):
    return [*name_or_flags], kwargs


@subcommand([argument("-h", help="Show commands option to be used with KenLM", action="store_true")])
def help(args):
    pass


@dataclass
class QueryParser(Coqpit):
    """"KenLM was compiled with maximum order "
    """
    # _command_: COMMAND_TYPE = "query"

    file_path: str = MISSING
    flush: bool = field(
        default=False,
        metadata=dict(metavar="-b", help="Do not buffer output.")
    )
    model_type: ChoiceEnum([
        "PROBING", "REST_PROBING", "TRIE", "QUANT_TRIE",
        "ARRAY_TRIE", "QUANT_ARRAY_TRIE"]) = field(
        default=None,
        metadata=dict(help="Define model_type to use")
    )
    sentence_context: bool = field(
        default=False,
        metadata=dict(matavar="-n", help="Do not wrap the input in <s> and </s>.")
    )
    # TODO: FixMe
    statistics: ChoiceEnum(["summary", "sentence", "word"]) = field(
        default="summary",
        metadata=dict(
            metavar="-v",
            nargs="?",
            choices=["summary", "sentence", "word"],
            help="summary|sentence|word: Print statistics at this level. "
                 "Can be used multiple times: -v summary -v sentence -v word"
        )
    )
    load_method: ChoiceEnum(["lazy", "populate", "read", "parallel"]) = field(
        default="lazy",
        metadata=dict(
            metavar="-l",
            choices=["lazy", "populate", "read", "parallel"],
            help="Load lazily, with populate, or malloc+read "
                 "The default loading method is populate on Linux and read on others. "
                 "Each word in the output is formatted as: "
                 "word=vocab_id ngram_length log10(p(word|context))"
                 "where ngram_length is the length of n-gram matched. A vocab_id of 0 indicates"
                 "the unknown word. Sentence-level output includes log10 probability of the"
                 "sentence and OOV count."
        )
    )


@dataclass
class BinaryParser(Coqpit):
    """
    Get a memory estimate by passing an ARPA file without an output file name.
    """
    __command__: COMMAND_TYPE = "binary"

    log10_unknown_probability: int = field(
        default=-100,
        metadata=dict(
            metavar="-u",
            help="sets the log10 probability for <unk> if the ARPA file does not have one." +
                 " Default is -100.  The ARPA file will always take precedence."
        )
    )
    silence: bool = field(
        default=False,
        metadata=dict(
            metavar="-s",
            action="store_true",
            help="allows models to be built even if they do not have <s> and </s>."
        )
    )
    positive_log_probability: bool = field(
        default=False,
        metadata=dict(
            action="store_true",
            metavar='-i',
            help="allows buggy models from IRSTLM by mapping positive log probability to 0."
        )
    )
    include_vocab: bool = field(
        default=False,
        metadata=dict(
            metavar='-v',
            action="store_true",
            help="disables inclusion of the vocabulary in the binary file."
        )
    )
    write_method: ChoiceEnum(['mmap', 'after']) = field(
        default='mmap',
        metadata=dict(
            metavar='-w',
            help="determines how writing is done.\n" +
                 "mmap maps the binary file and writes to it. Default for trie.\n" +
                 "after allocates anonymous memory, builds, and writes.  Default for probing."
        )
    )
    model_type: ChoiceEnum(["trie", "probing"]) = field(
        default="trie",
        metadata={"help": "It is either probing or trie.  Default is probing."}
    )
    probe: str = field(
        default=None,
        metadata={"help": "probing uses a probing hash table.  It is the fastest but uses the most memory."}
    )

    # TODO: FixME
    rest_lower_files: ChoiceEnum(["order1.arpa", "order2", "order3", "order4"]) = field(
        default="order1.arpa",
        metadata=dict(
            metavar='-r',
            help="Adds lower-order rest costs from these model files, order1.arpa must be an ARPA file.\n"
                 "All others may be ARPA or the same data structure as being built. All files must have the same "
                 "vocabulary. For probing, the unigrams must be in the same order."
        )
    )
    probing_multiplier: float = field(
        default=1.5,
        metadata=dict(
            metavar='-p',
            help="sets the space multiplier and must be >1.0.  The default is 1.5.\n"
                 "trie is a straightforward trie with bit-level packing.  It uses the least "
                 "memory and is still faster than SRI or IRST.  Building the trie format uses an "
                 "on-disk sort to save memory.\n"
        )
    )
    trie_temporary: str = field(
        default=None,
        metadata=dict(
            metavar='-T',
            help="This is the temporary directory prefix.  Default is the output file name."
        )
    )
    trie_building_mem: str = field(
        default='1G',
        metadata=dict(
            matavar='-S',
            help="determines memory use for sorting.  Default is 1G.  This is compatible "
                 "with GNU sort.  The number is followed by a unit: % for percent of physical "
                 "memory, b for bytes, K for Kilobytes, M for megabytes, then G,T,P,E,Z,Y. "
                 "Default unit is K for Kilobytes."
        )
    )
    qbits: int = field(
        default=8,
        metadata=dict(
            metavar='-q',
            help="turns quantization on and sets the number of bits (e.g. -q 8)."
        )
    )
    bbits: int = field(
        default=-1,
        metadata=dict(
            metavar='-b',
            help="sets backoff quantization bits.  Requires -q and defaults to that value."
        )
    )
    abits: int = field(
        default=-1,
        metadata=dict(
            metavar='-a',
            help="compresses pointers using an array of offsets. The parameter is the "
                 "maximum number of bits encoded by the array.  Memory is minimized subject "
                 "to the maximum, so pick 255 to minimize memory."
        )
    )


class LMPZParser(Coqpit):
    __command__: COMMAND_TYPE = "lmplz"

    order: str = field(default=MISSING, metadata=dict(metavar="-o", required=True))
    intermediate: bool = field(
        default=False,
        metadata=dict(
            help="Write ngrams to intermediate files. Turns off ARPA output (which can be reactivated by --arpa "
                 "file). Forces --renumber on."
        )
    )
    renumber: bool = field(
        default=False,
        metadata=dict(
            help="Renumber the vocabulary identifiers so that they are monotone with the hash of each string. "
                 "This is consistent with the ordering used by the trie data structure."
        )
    )
    collapse_values: bool = field(
        default=False,
        metadata=dict(
            help="Collapse probability and backoff into a single value, q that yields the same "
                 "sentence-level probabilities. See http://kheafield.com/professional/edinburgh/rest_paper.pdf "
                 "for more details, including a proof."
        )
    )
    interpolate_unigrams: bool = field(
        default=True,
        metadata=dict(
            help="Interpolate the unigrams (default) as opposed to giving lots of mass to <unk> like SRI. "
                 "If you want SRI's behavior with a large <unk> and the old lmplz default, "
                 "use --interpolate_unigrams 0."
        )
    )
    skip_symbols: bool = field(
        default=False,
        metadata=dict(help="Treat <s>, </s>, and <unk> as whitespace instead of throwing an exception")
    )
    temp_prefix: str = field(default=None, metadata=dict(metavar='-T', help="Temporary file prefix"))
    memory: str = field(
        default='1G',
        metadata=dict(metavar='-S', help="Sorting memory")
    )
    minimum_block: str = field(
        default='8K',
        metadata=dict(help="Minimum block size to allow")
    )
    sort_block: str = field(
        default='64M',
        metadata=dict(help="Size of IO operations for sort (determines arity)")
    )
    block_count: int = field(
        default=2,
        metadata=dict(help="Block count (per order)")
    )
    vocab_estimate: int = field(
        default=1000000,
        metadata=dict(
            help="Assume this vocabulary size for purposes of calculating memory in step 1 (corpus count) "
                 "and pre-sizing the hash table"
        )
    )
    vocab_pad: int = field(
        default=0,
        metadata=dict(
            help="If the vocabulary is smaller than this value, pad with <unk> to reach this size. "
                 "Requires --interpolate_unigrams"
        )
    )
    verbose_header: bool = field(
        default=False,
        metadata=dict(
            help="Add a verbose header to the ARPA file that includes information such as token count, "
                 "smoothing type, etc.")
    )
    text: str = field(
        default=None, metadata=dict(help="Read text from a file instead of stdin")
    )
    arpa: str = field(
        default=None, metadata=dict(help="Write ARPA to a file instead of stdout")
    )
    discount_fallback: ChoiceEnum(["0.5", "1", "1.5"]) = field(
        default="0.5 1 1.5",
        metadata=dict(
            help="The closed-form estimate for Kneser-Ney discounts does not work without singletons or "
                 "doubletons. It can also fail if these values are out of range.  This option falls back")
    )
    limit_vocab_file: str = field(
        default=None,
        metadata=dict(
            help="Read allowed vocabulary separated by whitespace. N-grams that contain vocabulary "
                 "items not in this list will be pruned. Can be combined with --prune arg")
    )
    prune: str = field(
        default=None,
        metadata=dict(
            help="Prune n-grams with count less than or equal to the given threshold. Specify one value for each "
                 "order i.e. 0 0 1 to prune singleton trigrams and above. The sequence of values must be "
                 "non-decreasing and the last value applies to any remaining orders. "
                 "Default is to not prune, which is equivalent to --prune 0.")
    )


class InterpolateParser(Coqpit):
    __command__: COMMAND_TYPE = 'interpolate'

    model: MISSING = field(
        default=None,
        metadata=dict(
            metavar="-m",
            required=True,
            help="Models to interpolate, which must be in KenLM intermediate format. "
                 "The intermediate format can be generated using the --intermediate argument to lmplz."
        )
    )
    weight: float = field(
        default=0.0,
        metadata=dict(metavar="-w", help="Interpolation weights")
    )
    tuning: str = field(
        default=None,
        metadata=dict(metavar="-t", help="File to tune on: a text file with one sentence per line")
    )
    just_tune: str = field(default=None, metadata=dict(help="Tune and print weights then quit"))
    temp_prefix: str = field(
        default="/tmp/lm",
        metadata=dict(metavar="-T", help="Temporary file prefix")
    )
    memory: str = field(
        default="1G",
        metadata=dict(metavar="-S", help="Sorting memory: this is a very rough guide")
    )
    sort_block: str = field(default="64M", metadata=dict(help="Block size"))


class BenchmarkParser(Coqpit):
    """
    Get the benchmark of LM
    """
    __command__: COMMAND_TYPE = "benchmark"
    model: str = field(
        default=MISSING,
        metadata=dict(metavar="-m", required=True, help="Model to query or convert vocab ids")
    )
    threads: int = field(
        default=4,
        metadata=dict(metavar="-t", help="Threads to use (querying only; TODO vocab conversion)")
    )
    buffer: 4096 = field(
        default=4096,
        metadata=dict(metavar='-b', help="Number of words to buffer per task.")
    )
    vocab: bool = field(
        default=False,
        metadata=dict(
            metavar='-v',
            help="Convert strings to vocab ids"
        )
    )
    query: bool = field(default=False, metadata=dict(metavar='-q', help="Query from vocab ids"))


class FilterParser(Coqpit):
    __command__: COMMAND_TYPE = "filter"

    mode: ChoiceEnum(['copy', 'single', 'multiple', 'union']) = field(
        default="single",
        metadata=dict(
            required=True,
            help="copy mode just copies, but makes the format nicer for e.g. irstlm's broken parser.\n"
                 "single mode treats the entire input as a single sentence.\n"
                 "multiple mode filters to multiple sentences in parallel. Each sentence is on "
                 "a separate line.  A separate file is created for each sentence by appending"
                 "the 0-indexed line number to the output file name.\n"
                 "union mode produces one filtered model that is the union of models created by"
                 "multiple mode."
        )
    )
    context: str = field(
        default=None,
        metadata=dict(
            help="context means only the context (all but last word) has to pass the filter, but "
                 "the entire n-gram is output."
        )
    ),
    phrase: str = field(
        default=None,
        metadata=dict(
            help="phrase means that the vocabulary is actually tab-delimited phrases and that the "
                 "phrases can generate the n-gram when assembled in arbitrary order and clipped. "
                 "Currently works with multiple or union mode."
        )
    )
    format: ChoiceEnum(['raw', 'arpa']) = field(
        default='raw',
        metadata=dict(
            choices=['raw', 'arpa'],
            help="The file format is set by [raw|arpa] with default arpa:"
                 "raw means space-separated tokens, optionally followed by a tab and arbitrary text. "
                 "This is useful for ngram count files. arpa means the ARPA file format for n-gram language models."
        )
    )
    model: str = field(
        default=None,
        metadata=dict(
            help="There are two inputs: vocabulary and model. Either may be given as a file while the other "
                 "is on stdin. Specify the type given as a file using vocab: or model: before the file name. "
                 "For ARPA format, the output must be seekable.  For raw format, it can be "
                 "a\n stream i.e. /dev/stdout\n"
        )
    )
    vocab: str = field(default=None, metadata={"help": ""})
    threads: int = field(
        default=4,
        metadata={"help": "threads:m sets m threads (default: conccurrency detected by boost)"}
    )
    batch_size: int = field(
        default=32,
        metadata=dict(
            help="batch_size:m sets the batch size for threading. Expect memory usage from this of "
                 "2*threads*batch_size n-grams."
        )
    )


class StreamingParser(Coqpit):
    # streaming_example_main.cc
    __command__: COMMAND_TYPE = "streaming"

    ngrams: str = field(
        default=None,
        metadata=dict(metavar="-n", help="ngrams file")
    )
    csortngrams: str = field(
        default=None,
        metadata=dict(metavar="-c", help="context sorted ngrams file")
    )
    backoffs: str = field(
        default=None,
        metadata=dict(metavar="-b", help="backoffs file")
    )
    tmpdir: str = field(
        default=None,
        metadata=dict(metavar="-t", help="Temporary directory")
    )


class NgramCounterParser(Coqpit):
    # count_ngram_main.cc
    """"Get ngram counter"
    """
    __command__: COMMAND_TYPE = "ngram-counter"

    order: int = field(default=None, metadata=dict(metavar='-o', required=True, help="Order"))
    temp_prefix: str = field(default='temp_dir', metadata=dict(metavar='-T', help="Temporary file prefix"))
    memory: str = field(default="80", metadata=dict(metavar='-S', help='RAM'))
    read_vocab_table: str = field(
        default=None,
        metadata=dict(
            help="Vocabulary hash table to read. This should be a probing hash table with size at the beginning."
        )
    )
    write_vocab_list: str = field(
        default=None,
        metadata=dict(help="Vocabulary list to write as null-delimited strings.")
    )


# Add subcommand to parent parser
for cmd in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
    if cmd.startswith('get'):
        cmd()
