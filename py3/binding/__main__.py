from typing import List, Dict, Set

import io
import argparse
import numpy as np
from pykenlm import interpolate, util, builder, benchmark, constant as ct, kenlm, ngram, filter as filter_
import sys
from hashlib import sha256
from .dataclass._dataclass import (
    LMPZParser, 
    QueryParser, 
    BenchmarkParser, 
    NgramCounterParser,
    FilterParser, 
    BinaryParser, 
    InterpolateParser, 
    StreamingParser
)

FILE_NAME = "ngrams"
CONTEXT_SORTED_FILENAME = "csorted-ngrams"
BACKOFF_FILENAME = "backoffs"
TMP_DIR = "/tmp/"
ONE_GB = 1 << 30
SIXTY_FOUR_MB = 1 << 26
NUMBER_OF_BLOCKS = 2


_MODEL_TYPE_MAPPING = \
    {
        ct.Py_ModelType.PROBING: ngram.ProbingModel,
        ct.Py_ModelType.REST_PROBING: ngram.RestProbingModel,
        ct.Py_ModelType.TRIE: ngram.TrieModel,
        ct.Py_ModelType.QUANT_TRIE: ngram.QuantTrieModel,
        ct.Py_ModelType.ARRAY_TRIE: ngram.ArrayTrieModel,
        ct.Py_ModelType.QUANT_ARRAY_TRIE: ngram.QuantArrayTrieModel
    }


def hash_value(m: 'MutablePiece'):
    return sha256(m.behind)


class MutablePiece:
    behind: kenlm.Py_StringPiece = None

    def __init__(self, behind) -> None:
        self.behind = behind

    def __eq__(self, other: 'MutablePiece'):
        return self.behind == other.behind


class InternString:
    strs_: Set[MutablePiece]

    def __init__(self):
        self.strs_ = set([])

    def Add(self, _str: kenlm.StringPiece) -> str:
        mut = MutablePiece(_str)
        mut.behind = util.Py_StringPiece(_str)
        self.strs_.add(mut)
        return mut.behind.data()


class TargetWords:
    # TODO: FixMe - Fix this class
    intern_: InternString
    vocab_: List[Set[str]]
    interns_: List[str]  # Temporary in Add.

    def Introduce(self, source: util.Py_StringPiece):
        temp = []
        self.Add(temp, source)

    def Add(self, sentences: List[int], target: kenlm.Py_StringPiece):
        if len(sentences) == 0:
            return
        interns_ = [ch for ch in target]
        self.vocab_ = self.vocab_ + list(set(interns_))


class Input:
    max_length: int

    # hash of phrase is the key, array of sentences is the value.
    map_: Dict[int, List[int]]

    sentence_id_: int

    # Temporaries in AddSentence.
    canonical_: str = None
    starts_: List[int] = []
    empty_: List[int] = []

    def __init__(self, max_length):
        self.max_length = max_length

    def AddSentence(self, sentence: util.Py_StringPiece, targets: TargetWords):

        canonical_ = [word for word in sentence]
        self.canonical_ = " ".join(canonical_)
        targets.Introduce(self.canonical_)
        self.starts_ = list(range(len(canonical_) + 1))

    def Matches(self, phrase: kenlm.Py_StringPiece) -> List[int]:
        """
        Assumes single space-delimited phrase with no space at the beginning or end.
        Parameters
        ----------
        phrase

        Returns
        -------

        """
        # Map::const_iterator i = map_.find(util::MurmurHash64A(phrase.data(), phrase.size()));
        # return i == map_.end() ? empty_ : i->second
        pass


def Phrase_table_Main():
    input_ = Input(7)
    targets = None  # type: TargetWords

    table = kenlm.FilePiece(0, None)
    line = kenlm.StringPiece()
    pipes = kenlm.StringPiece("|||")

    with open("", 'r') as table:
        line = table.readlines()

    # util::TokenIter<util::MultiCharacter> it(line, pipes);
    # source = kenlm.StringPiece(it)
    # if (!source.empty() && source[source.size() - 1] == ' ')
    # source.remove_suffix(1);
    # targets.Add(input.Matches(source), *++it);


def Query_Main(cfg: QueryParser):
    # TODO: FixMe
    sentence_context = cfg.sentence_context
    _CFG_MAP = \
        {
            'summary': False,
            'word': False,
            'line': False
        }

    _LOAD_METHOD = \
        {
            "populate": util.Py_LoadMethod.POPULATE_OR_READ,
            "read": util.Py_LoadMethod.READ,
            "parallel": util.Py_LoadMethod.PARALLEL_READ,
            "lazy": util.Py_LoadMethod.LAZY
        }

    assert cfg.statistics in _CFG_MAP.keys(), ValueError("Invalid options for stats!")

    print_opts = {k: v for k, v in _CFG_MAP.items() if k in cfg.statistics}
    load_method = util.Py_LoadMethod.POPULATE_OR_READ
    for k, v in _LOAD_METHOD.items():
        if k == cfg.load_method:
            load_method = v
            break

    printer = kenlm.Py_QueryPrinter(print_opts, cfg.flush)
    hparams = open(cfg.file_path, 'r').readlines()
    assert any([cfg.model_type in mtype for mtype in kenlm.Py_ModelType]), ValueError("Invalid model-type")

    model_type = None  # Read from files

    if model_type:

        model = _MODEL_TYPE_MAPPING[model_type](cfg)
        kenlm.Py_Query(model, sentence_context, printer)

    elif kenlm.Model.Recognize(file):
        model = kenlm.Model(file)
        kenlm.Py_Query[kenlm.Model, kenlm.Py_QueryPrinter](model, sentence_context, printer)

    else:
        model = kenlm.Py_ProbingModel(file, cfg)
        kenlm.Py_Query(model, sentence_context, printer)


def Dispatch(_file: str, config: kenlm.Config):
    model_type = kenlm.Py_ModelType  # type: kenlm.Py_ModelType
    model_config = kenlm.Config()  # type: kenlm.Config
    model_config.load_method = kenlm.READ
    model = kenlm.Model(_file, model_config)

    if kenlm.py_RecognizeBinary(_file, model_type):
        assert model_type in _MODEL_TYPE_MAPPING.keys(), RuntimeError("Unrecognized kenlm model type ")
        model = _MODEL_TYPE_MAPPING[model_type](_file, config)

        kenlm.DispatchWidth(model, config)

    raise RuntimeError("Binarize before running benchmarks.")


def benchMark_Main(options: BenchmarkParser):

    config = kenlm.Config(fd_in=0, threads=options.threads, buf_per_thread=options.buffer)  # type kenlm.Config

    assert options.vocab and options.query, \
        ValueError("Specify exactly one of -v (vocab conversion) or -q (query).")
    config.query = options.query
    Dispatch(options.model, config)


def DispatchFunction(model: kenlm.Model, config: benchmark.Config, dtype=kenlm.uint8_t):
    if config.query:
        benchmark.QueryFromBytes(model, config, Width=dtype)
    else:
        benchmark.ConvertToBytes(model, config.fd_in, Width=dtype)


def DispatchWidth(model: kenlm.Model, config: benchmark.Config):
    bound = model.GetVocabulary().Bound()

    if bound <= 256:
        DispatchFunction(model, config, dtype=kenlm.uint8_t)
    elif bound <= 65536:
        DispatchFunction(model, config, dtype=kenlm.uint16_t)
    elif bound <= "(1ULL << 32)":
        DispatchFunction(model, config, dtype=kenlm.uint32_t)
    else:
        DispatchFunction(model, config, dtype=kenlm.uint64_t)

    del model_config


def LMPZ_Main(text: str, options: LMPZParser):
    pipeline: kenlm.PipelineConfig = None

    mem = kenlm.Py_GuessPhysicalMemory()
    if mem:
        print(f"This machine has {mem} bytes of memory.")
    else:
        raise IOError("Unable to determine the amount of memory on this machine.")

    if options.vocab_pad and not options.interpolate_unigrams:
        raise ValueError("--vocab_pad requires --interpolate_unigrams be on")

    initial_probs = builder.Py_InitialProbabilitiesConfig(interpolate_unigrams=options.interpolate_unigrams)
    sort_cfg = interpolate.Py_SortConfig(options.temp_prefix, options.sort_block, options.memory)

    # ParseDiscountFallback parse_discount = ParseDiscountFallback(discount_fallback)
    if options.skip_symbols:
        catch_error = kenlm.WarningAction.COMPLAIN
    else:
        catch_error = kenlm.WarningAction.THROW_UP

    pipeline = builder.Py_PipelineConfig(
        initial_probs=initial_probs,
        sort_cfg=sort_cfg,
        minimum_block=options.minimum_block,
        block_count=options.block_count,
        vocab_estimate=options.vocab_estimate,
        vocab_size_for_unk=options.vocab_pad,
        renumber_vocabulary=options.renumber_vocabulary,
        prune_vocab_file=options.prune_vocab_file,
        output_q=options.output_q,
        # disallowed_symbol_action=catch_error
    )

    if options.discount_fallback:
        pipeline.discount.fallback = kenlm.Py_ParseDiscountFallback(options.discount_fallback)
        pipeline.discount.bad_action = kenlm.WarningAction.COMPLAIN
    else:
        # Unused, just here to prevent the compiler from complaining about uninitialized.
        pipeline.discount.fallback = builder.Py_Discount()
        pipeline.discount.bad_action = kenlm.WarningAction.THROW_UP

    # parse pruning thresholds.  These depend on order, so it is not done as a notifier.
    pipeline.prune_thresholds = kenlm.Py_ParsePruning(pruning, pipeline.order)
    pipeline.prune_vocab = True if options.limit_vocab_file is None else False

    text = options.text
    # intermediate = options.intermediate
    memory = pipeline.sort.total_memory
    arpa = options.arpa

    kenlm.NormalizeTempPrefix_(pipeline.sort.temp_prefix)

    initial = pipeline.initial_probs
    # TODO: evaluate options for these.
    initial.adder_in.total_memory = 32768
    initial.adder_in.block_count = 2
    initial.adder_out.total_memory = 32768
    initial.adder_out.block_count = 2
    pipeline.read_backoffs = initial.adder_out

    # Read from stdin, write to stdout by default
    # cdef scoped_fd in (0), out(1)
    # if self.text:
    #     in.reset(OpenReadOrThrow(self.text))

    # if self.arpa:
    #     out.reset(CreateOrThrow(self.arpa)
    writing_intermediate = None

    try:
        pipeline.renumber_vocabulary = True if options.intermediate else False
        output = kenlm.Output(
            options.intermediate if options.intermediate else pipeline.sort.temp_prefix,
            writing_intermediate,
            pipeline.output_q
        )

        # TODO: FixMe -
        if not writing_intermediate or options.arpa:
            output.Add(kenlm.PrintHook(out.release(), options.verbose_header))

        kenlm.Pipeline(pipeline, in_.release(), output)

    except Exception as e:
        print(f"Try rerunning with a more conservative -S setting than {self.memory}")
        raise e


def Query_Main(model: kenlm.Model, mtype=None):
    """
    fragment_main.cc

    :param _type_ model: _description_
    :param _type_ mtype: _description_, defaults to None
    """
    import sys
    ignored: kenlm.ChartState

    for line in sys.stdin:
        # lm::ngram::RuleScore<Model> scorer(model, ignored);
        # for (util::TokenIter<util::SingleCharacter, true> i(line, ' '); i; ++i) {
        #     scorer.Terminal(model.GetVocabulary().Index(*i));
        # }
        # print(scorer.Finish())
        pass


def fragment_Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, type=int, help="Expected model file name.")
    args = parser.parse_args()

    model_type = kenlm.Py_ModelType()
    kenlm.RecognizeBinary(args.name, model_type)

    if model_type == kenlm.Py_ModelType.PROBING:
        model = kenlm.Py_ProbingModel()
        kenlm.Py_Query(model, mtype=model_type)
        return

    if model_type == kenlm.Py_ModelType.REST_PROBING:
        model = kenlm.Py_RestProbingModel()
        kenlm.Py_Query(model, mtype=model_type)
        return

    raise ValueError("Model type not supported yet.")


def NgramCounter_Main(options: NgramCounterParser):

    # TODO: FixMe
    temp_prefix = None
    vocab_table = None
    vocab_list = None
    ram = options.memory

    if not options.read_vocab_table and options.write_vocab_list:
        raise ValueError("Specify one of --read_vocab_table or --write_vocab_list for vocabulary handling.")

    # kenlm.Py_NormalizeTempPrefix(options.temp_prefix)

    if options.read_vocab_table:
        vocab_file = open(options.write_vocab_list, 'w')  # type: io.TextIO
    else:
        vocab_file = open(options.read_vocab_table, 'r')  # type: io.TextIO

    blocks = 2
    remaining_size = ram - SizeOrThrow(vocab_file.get())

    # This much memory to work with after vocab hash table.
    # Solve for block size including the dedupe multiplier for one block.
    # Chain likes memory expressed in terms of total memory.
    memory_for_chain = (float(remaining_size) /
                        (float(blocks) + builder.Py_CorpusCount.DedupeMultiplier(options.order)) * float(blocks))

    print(f"Using {memory_for_chain} for chains.")

    config = util.Py_ChainConfig(kenlm.NGram.TotalSize(options.order), blocks, memory_for_chain)
    chain = util.Py_Chain(config)
    # TODO: Check this
    fp = kenlm.FilePiece(0, None)
    token_count = 0
    type_count = kenlm.WordIndex(0)
    empty_prune = []
    block_frac = chain.BlockSize() / chain.EntrySize()
    empty_string = ""

    counter = builder.Py_CorpusCount(
        fp,
        vocab_file,
        vocab_table,
        token_count,
        type_count,
        empty_prune,
        empty_string,
        block_frac,
        kenlm.THROW_UP
    )

    buffer_size = (64 * 1024 * 1024)
    # Intended to run in parallel.
    sort_config = kenlm.SortConfig(temp_prefix, buffer_size, remaining_size)

    SortSuffixCount = kenlm.Sort[kenlm.SuffixOrder, kenlm.CombineCounts]
    _sorted = SortSuffixCount(chain, sort_config, kenlm.SuffixOrder(order), kenlm.CombineCounts())

    chain.Wait(True)
    chain2 = kenlm.Py_Chain(
        kenlm.Py_ChainConfig(kenlm.NGram.TotalSize(order, dtype=np.int64),
                             blocks, sort_config.buffer_size)
    )
    _sorted.Output(chain2)

    # Inefficiently copies if there's only one block.
    # WriteAndRecycle(1)
    chain2.Wait(True)


def Filter_Main(options: FilterParser):
    _MODE_CFG = \
        {
            "unset": ct.Py_FilterMode.MODE_UNSET,
            "copy": ct.Py_FilterMode.MODE_COPY,
            "single": ct.Py_FilterMode.MODE_SINGLE,
            "multiple": ct.Py_FilterMode.MODE_MULTIPLE,
            "union": ct.Py_FilterMode.MODE_UNION
        }

    _FMT_CFG = \
        {
            "arpa": ct.Py_Format.FORMAT_ARPA,
            "raw": ct.Py_Format.FORMAT_COUNT
        }
    format_ = ct.Py_Format()
    mode = ct.Py_FilterMode()

    config = kenlm.Py_FilterConfig(format_, mode, options.threads, options.batch_size)

    if (
        config.phrase and
        config.mode is not ct.Py_FilterMode.MODE_UNION and
        config.mode is not ct.Py_FilterMode.MODE_MULTIPLE
    ):
        raise ValueError(
            "Phrase constraint currently only works in multiple or union mode."
            "If you really need it for single, put everything on one line and use union."
        )

    cmd_is_model = True
    if options.vocab is not None:
        cmd_is_model = False
        vocab = options.vocab
    elif options.model:
        cmd_file = open(options.model, 'r')

    # FilePiece model(cmd_is_model ? OpenReadOrThrow(cmd_input) : 0, cmd_is_model ? cmd_input : NULL, &std::cerr)

    if config.format == ct.Py_Format.FORMAT_ARPA:
        filter_.DispatchFilterModes(config, *vocab, kenlm.Py_Format.FORMAT_ARPA)

    elif config.format == ct.Py_Format.FORMAT_COUNT:
        filter_.DispatchFilterModes(config, *vocab, kenlm.Py_Format.FORMAT_COUNT)


def BuildBinary_Main(options: BinaryParser):
    # TODO: FixMe -

    quantize: bool = False
    set_backoff_bits: bool = False
    bhiksha: bool = False
    set_write_method: bool = False
    rest: bool = False
    default_mem: str = "80%" if util.Py_GuessPhysicalMemory() else "1G"

    if not set_backoff_bits:
        set_backoff_bits = bool(options.qbits)
        quantize = True

    if options.bbits:
        set_backoff_bits = bool(options.bbits)

    if options.abits:
        bhiksha = True

    temporary_directory_prefix = options.trie_temporary
    building_memory = min(sys.maxsize, kenlm.Py_ParseSize(options.trie_building_mem))
    write_method = None

    if options.write_method:
        if options.write_method == 'mmap':
            write_method = kenlm.WRITE_MMAP

        if options.write_method == 'after':
            write_method = kenlm.WRITE_AFTER

    sentence_marker_missing = kenlm.SILENT if options.silence else None
    positive_log_probability = kenlm.SILENT if options.positive_log_probability else None

    if options.rest_lower_files:
        rest = True
        kenlm.ParseFileList(options.rest_lower_files, options.rest_lower_files)
        rest_function = kenlm.REST_LOWER

    model_type = options.model_type
    from_file = options.from_file

    if options.model_type == "probing":
        if not options.write_method:
            write_method = kenlm.WRITE_AFTER

        if quantize or set_backoff_bits:
            kenlm.ProbingQuantizationUnsupported()

    config = kenlm.Py_NgramConfig(
        options.qbits,
        set_backoff_bits,
        bhiksha,
        options.log10_unknown_probability,
        options.probing_multiplier,
        temporary_directory_prefix,
        building_memory,
        write_method,
        rest_function=rest_function,
        include_vocab=options.include_vocab
    )

    if not quantize and set_backoff_bits:
        raise ValueError("You specified backoff quantization (-b) but not probability quantization (-q)")

    if rest:
        ngram.RestProbingModel(from_file, config)
    else:
        ngram.ProbingModel(from_file, config)

    assert model_type == "trie" and rest, ValueError("Rest + trie is not supported yet.")

    if not set_write_method:
        config.write_method = kenlm.Config.WRITE_MMAP

    try:
        if quantize:
            if bhiksha:
                ngram.QuantArrayTrieModel(from_file, config)
            else:
                ngram.QuantTrieModel(from_file, config)

        else:
            if bhiksha:
                ngram.ArrayTrieModel(from_file, config)
            else:
                ngram.TrieModel(from_file, config)

    except Exception as e:
        print("Exception: ", str(e))
        raise e


def Interpolate_Main(options: InterpolateParser):
    temp_prefix = options.temp_prefix
    buffer_size = options.sort_block
    total_memory = "50%" if util.Py_GuessPhysicalMemory() else "1G"

    sort_cfg = interpolate.Py_SortConfig(temp_prefix, buffer_size, total_memory)
    pipe_config = interpolate.Py_InterpolateConfig(lamdas=options.weight, sort_=sort_cfg)
    instances_config = interpolate.Py_InstancesConfig(sort_cfg)

    input_models: List[str] = [options.model]
    tuning_file = options.tuning  # type Union[str, Path]

    try:
        interpolate.Py_initParallel()

        if pipe_config.lambdas.empty() and tuning_file is None:
            raise ValueError("Provide a tuning file with -t xor weights with -w.")

        if not pipe_config.lambdas.empty() and not tuning_file:
            raise ValueError("Provide weights xor a tuning file, not both.")

        models = [input_models[i] for i in range(len(input_models))]

        interpolate.Py_TuneWeights(tuning_file, models, instances_config, pipe_config.lambdas)

        if pipe_config.lambdas.size() != len(input_models):
            print(
                f"Number of models ({len(input_models)}) should match "
                "the number of weights ({pipe_config.lambdas.size()})."
            )

        interpolate.Py_Pipeline(models, pipe_config, 1)

    except Exception as e:
        print("Exception: ", str(e))
        raise e


def Streaming_Main(options: StreamingParser):
    """
    The basic strategy here is to have three chains:
     - The first reads the ngram order inputs using ModelBuffer. Those are
       then stripped of their backoff values and fed into the third chain;
       the backoff values *themselves* are written to the second chain.

    - The second chain takes the backoff values and writes them out to a
       file (one for each order).

    - The third chain takes just the probability values and ngrams and
       writes them out, sorted in context-order, to a file (one for each
       order).

    This will be used to read in the binary intermediate files. There is
    one file per order (e.g. ngrams.1, ngrams.2, ...)
    """

    buffer = kenlm.PyModelBuffer(FILE_NAME)  # type: ModelBuffer

    # Create a separate chains for each ngram order for:
    # - Input from the intermediate files
    # - Output to the backoff file
    # - Output to the (context-sorted) probability file

    ngram_inputs = utils.Py_Chains(buffer.Order())
    backoff_chains = utils.Py_Chains(buffer.Order())
    prob_chains = utils.Py_Chains(buffer.Order())

    for i in range(buffer.Order()):
        ngram_cnf = utils.Py_ChainConfig(NGram[ProbBackoff].TotalSize(i + 1), NUMBER_OF_BLOCKS, ONE_GB)
        ngram_inputs.push_back(ngram_cnf)

        bckoff_cnf = utils.Py_ChainConfig(sizeof(float), NUMBER_OF_BLOCKS, ONE_GB)
        backoff_chains.push_back(bckoff_cnf)

        prob_chain_cnf = utils.Py_ChainConfig(
            sizeof(WordIndex) * (i + 1) + sizeof(float), NUMBER_OF_BLOCKS, ONE_GB
        )
        prob_chains.push_back(prob_chain_cnf)

    # This sets the input for each of the ngram order chains to the appropriate file
    buffer.Source(ngram_inputs)

    workers = utils.FixedArrayWorker(buffer.Order())

    for i in range(buffer.Order()):
        # Attach a SplitWorker to each of the ngram input chains, writing to the
        # corresponding order's backoff and probability chains
        _split_worker = interpolate.Py_SplitWorker(i + 1, backoff_chains[i], prob_chains[i])
        workers.push_back(_split_worker)
    # ngram_inputs[i] >> boost::ref(*workers.back())

    sort_cfg = interpolate.Py_SortConfig(TMP_DIR, SIXTY_FOUR_MB, ONE_GB)

    # This will parallel merge sort the individual order files, putting
    # them in context-order instead of suffix-order.
    # Two new threads will be running, each owned by the chains[i] object.
    #   - The first executes BlockSorter.Run() to sort the n-gram entries
    #   - The second executes WriteAndRecycle.Run() to write each sorted
    # block to disk as a temporary file

    sorts = utils.PySorts(buffer.Order())

    for i in range(prob_chains.size()):
        sorts.push_back(prob_chains[i], sort_cfg, ContextOrder(i + 1))

    # Set the sort output to be on the same chain
    for i in range(prob_chains.size() + 1):
        # The following call to Chain::Wait() joins the threads owned by chains[i].
        # As such the following call won't return until all threads owned by chains[i] have completed.

        # The following call also resets chain[i] so that it can be reused
        # (including free'ing the memory previously used by the chain)
        prob_chains[i].Wait()

        # In an ideal world (without memory restrictions)
        # we could merge all of the previously sorted blocks
        # by reading them all completely into memory
        # and then running merge sort over them.

        # In the real world, we have memory restrictions;
        # depending on how many blocks we have,
        # and how much memory we can use to read from each block
        # (sort_config.buffer_size)
        # it may be the case that we have insufficient memory
        # to read sort_config.buffer_size of data from each block from disk.

        # If this occurs, then it will be necessary to perform one or more rounds
        # of merge sort on disk;
        # doing so will reduce the number of blocks that we will eventually
        # need to read from
        # when performing the final round of merge sort in memory.

        # So, the following call determines whether it is necessary
        # to perform one or more rounds of merge sort on disk;
        # if such on-disk merge sorting is required, such sorting is performed.

        # Finally, the following method launches a thread that calls
        # `OwningMergingReader.Run()` to perform the final round of merge sort in memory.

        # Merge sort could have be invoked directly so that merge sort memory doesn't coexist with Chain memory.

        sorts[i].Output(prob_chains[i])

    # Create another model buffer for our output on e.g. csorted-ngrams.1, csorted-ngrams.2, ...

    output_buf = interpolate.ModelBuffer(CONTEXT_SORTED_FILENAME, True, False)
    boff_buf = interpolate.ModelBuffer(BACKOFF_FILENAME, True, False)

    output_buf.Sink(prob_chains, buffer.Counts())

    # Create a third model buffer for our backoff output on e.g. backoff.1, backoff.2, ...
    boff_buf.Sink(backoff_chains, buffer.Counts())

    # Joins all threads that chains owns,
    # and does a for loop over each chain object in chains,
    # calling chain.Wait() on each such chain object
    ngram_inputs.Wait(True)
    backoff_chains.Wait(True)
    prob_chains.Wait(True)
