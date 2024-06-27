use crate::constant::{ARPALoadComplain, RestFunction, WarningAction, WriteMethod};

use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::fs::{File, FileType};
use std::io::prelude::*;
use std::io::{BufRead, BufReader};

type Node = NodeRange;

#[derive(Debug, Clone, Copy)]
pub struct State;

#[derive(Debug, Clone, Copy)]
pub struct LoadMethod;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    probing_multiplier: f64,

    load_method: LoadMethod,

    show_progress: bool,

    // Level of complaining to do when loading from ARPA instead of binary format.
    arpa_complain: ARPALoadComplain,

    // Where to log messages including the progress bar.  Set to NULL for silence.
    // messages: ostream,

    // This will be called with every string in the vocabulary by the constructor; it need
    // only exist for the lifetime of the constructor. See enumerate_vocab.hh for more detail.
    // Config does not take ownership; just delete/let it go out of scope after the constructor exits.
    enumerate_vocab: EnumerateVocab,

    // ONLY EFFECTIVE WHEN READING ARPA
    // What to do when <unk> isn't in the provided model.
    unknown_missing: WarningAction,
    // What to do when <s> or </s> is missing from the model.
    // If THROW_UP, the exception will be of type util::SpecialWordMissingException.
    sentence_marker_missing: WarningAction,

    // What to do with a positive log probability. For COMPLAIN and SILENT, map to 0.
    positive_log_probability: WarningAction,

    // The probability to substitute for <unk> if it's missing from the model.
    // No effect if the model has <unk> or unknown_missing == THROW_UP.
    unknown_missing_logprob: f64,

    // Size multiplier for probing hash table. Must be > 1. Space is linear in
    // this. Time is probing_multiplier / (probing_multiplier - 1). No effect for sorted variant.
    // If you find yourself setting this to a low number, consider using the TrieModel which has lower memory consumption.
    probing_multiplier: f64,

    //  Amount of memory to use for building.  The actual memory usage will be higher since this
    // just sets sort buffer size.  Only applies to trie models.
    building_memory: i64,

    // Template for temporary directory appropriate for passing to mkdtemp.
    // The characters XXXXXX are appended before passing to mkdtemp. Only applies to trie.
    // If empty, defaults to write_mmap. If that's NULL, defaults to input file name.
    temporary_directory_prefix: String,

    //  While loading an ARPA file, also write out this binary format file. Set to NULL to disable.
    write_mmap: &str,

    write_method: WriteMethod,

    // Include the vocab in the binary file?  Only effective if write_mmap != NULL.
    include_vocab: bool,

    rest_function: RestFunction, // Only used for REST_LOWER.

    rest_lower_files: Vec<String>,

    // Quantization options.  Only effective for QuantTrieModel.  One value is
    // reserved for each of prob and backoff, so 2^bits - 1 buckets will be used
    // to quantize (and one of the remaining backoffs will be 0).
    prob_bits: u8,
    backoff_bits: u8,

    // Bhiksha compression (simple form).  Only works with trie.
    pointer_bhiksha_bits: u8,
}

#[derive(Debug)]
pub(crate) struct NodeRange {
    begin: u64,
    end: u64,
}

#[derive(Debug)]
pub struct UnigramPointer;

#[derive(Debug)]
pub struct Unigram;

#[derive(Debug)]
pub struct BitPacked;

#[derive(Debug)]
pub struct BitPackedMiddle<Bhiksha>;

#[derive(Debug)]
pub(crate) struct BitPackedLongest;

#[derive(Debug)]
pub(crate) struct BhikshaBase;

#[derive(Debug)]
pub(crate) struct SortedVocabulary;

#[derive(Debug)]
pub struct ProbingVocabulary;

trait Bhiksha {
    pub fn new(self, max_offset: u64, max_value: u64, config: &Config) -> Self {}
}

#[derive(Debug)]
pub struct ArrayBhiksha {
    kModelTypeAdd: ModelType,
}

#[derive(Debug)]
pub struct DontBhiksha {
    kModelTypeAdd: ModelType,
}

#[derive(Debug, Clone, Copy)]
pub struct GenericModel<Search, VocabularyT> {
    // This is the model type returned by RecognizeBinary.
    kModelType: ModelType,
    kVersion: i64,
}


#[derive(Debug, Default)]
pub struct ProbingModel;

#[derive(Debug, Default)]
pub struct RestProbingModel;

#[derive(Debug, Default)]
pub struct TrieModel;

#[derive(Debug, Default)]
pub struct ArrayTrieModel;

#[derive(Debug, Default)]
pub struct QuantTrieModel;

#[derive(Debug, Default)]
pub struct QuantArrayTrieModel;

pub trait ModelTrait {
    fn new() -> Self {

    }
}

impl ModelTrait for ProbingModel {
    fn new() -> Self {

    }
}

impl ModelTrait for RestProbingModel {
    fn new() -> Self {

    }
}

impl ModelTrait for TrieModel {
    fn new() -> Self {

    }
}

impl ModelTrait for ArrayTrieModel {
    fn new() -> Self {

    }
}

impl ModelTrait for QuantTrieModel {
    fn new() -> Self {

    }
}

impl ModelTrait for QuantArrayTrieModel {
    fn new() -> Self {

    }
}

#[derive(Debug)]
pub struct LongestPointer;

#[derive(Debug)]
pub struct BinaryFormat;

#[derive(Debug)]
pub(crate) struct DontQuantize;

#[derive(Debug)]
pub(crate) struct Bins;

#[derive(Debug)]
pub struct MiddlePointer;

#[derive(Debug)]
pub(crate) struct SeparatelyQuantize;

#[derive(Debug)]
pub(crate) struct QueryPrinter;

trait Value {
    fn new() -> Self;
}

#[derive(Debug)]
pub struct RestValue;

#[derive(Debug)]
pub struct BackoffValue;

#[derive(Debug)]
pub struct HashedSearch<Value>;

#[derive(Debug, Clone, Copy)]
pub struct TrieSearch<Quant, Bhiksha>;

trait Search {
    pub fn new() -> Self {}
}

impl BhikshaBase {
    fn UpdateConfigFromBinary(self, file: &BinaryFormat, offset: u64, config: &Config);

    fn Size(&self, max_offset: u64, max_next: u64, config: &Config) -> u64;

    fn InlineBits(&self, max_offset: u64, max_next: u64, config: &Config) -> u8;

    fn ReadNext(&self, base: RefCell, bit_offset: u64, index: u64, total_bits: u8, out: &NodeRange);
}

impl Bhiksha for ArrayBhiksha {}

impl Bhiksha for DontBhiksha {}

impl Value for RestValue {
    fn new() -> Self {
        todo!()
    }
}

impl Value for BackoffValue {
    fn new() -> Self {
        todo!()
    }
}

impl Search for TrieSearch<Quant, Bhiksha> {
    fn new() -> Self {}

    fn UpdateConfigFromBinary(
        &self,
        file: &BinaryFormat,
        counts: &Vec<u64>,
        offset: u64,
        config: &Config,
    );

    fn Size(&self, counts: &Vec<u64>, config: &Config) -> u64;

    fn SetupMemory(&self, start: u8, counts: &Vec<u64>, config: &Config) -> &u8;

    fn InitializeFromARPA(
        &self,
        file: &str,
        f: &FilePiece,
        counts: &Vec<u64>,
        config: &Config,
        vocab: &SortedVocabulary,
        backing: &BinaryFormat,
    );

    fn LookupUnigram(
        &self,
        word: WordIndex,
        node: &Node,
        independent_left: &bool,
        extend_left: &u64,
    ) -> UnigramPointer;

    fn UnknownUnigram(&self) -> ProbBackoff;

    fn Unpack(&self, extend_pointer: u64, extend_length: &str, node: &Node) -> MiddlePointer;

    fn LookupMiddle(
        &self,
        order_minus_2: &str,
        word: WordIndex,
        node: &Node,
        independent_left: &bool,
        extend_left: &u64,
    ) -> MiddlePointer;
}

impl Config {
    // pub fn ProgressMessages() -> ostream;
}

impl State {
    pub fn Compare(&self, other: &State);
}

impl GenericModel<Search, VocabularyT> {
    pub fn new(file: &str, init_config: &Config) -> Self;

    // Get the size of memory that will be mapped given ngram counts. This
    // does not includes small non-mapped control structures, such as this class itself.
    fn Size(&self, counts: &Vec<u64>, config: &Config) -> i64;

    // Get the state for a context. Don't use this if you can avoid it. Use
    // BeginSentenceState or NullContextState and extend from those. If
    // you're only going to use this state to call FullScore once, use FullScoreForgotState.
    // To use this function, make an array of WordIndex containing the context
    // vocabulary ids in reverse order.  Then, pass the bounds of the array:
    // [context_rbegin, context_rend).
    pub fn GetState(&self, context_rbegin: &WordIndex, context_rend: &WordIndex, out_state: &State);

    // Score p(new_word | in_state) and incorporate new_word into out_state.
    // Note that in_state and out_state must be different references:
    // &in_state != &out_state.
    pub fn FullScore(
        &self,
        in_state: &State,
        new_word: WordIndex,
        out_state: &State,
    ) -> FullScoreReturn;

    // Slower call without in_state. Try to remember state, but sometimes it
    // would cost too much memory or your decoder isn't setup properly.
    // To use this function, make an array of WordIndex containing the context
    // vocabulary ids in reverse order.  Then, pass the bounds of the array:
    // [context_rbegin, context_rend).  The new_word is not part of the context
    // array unless you intend to repeat words.
    pub fn FullScoreForgotState(
        &self,
        context_rbegin: &WordIndex,
        context_rend: &WordIndex,
        new_word: WordIndex,
        out_state: &State,
    ) -> FullScoreReturn;
}

impl MiddlePointer {
    pub fn Found(&self) -> bool;
    pub fn Prob(&self) -> f64;
    pub fn Backoff(&self) -> f64;
    pub fn Rest(&self) -> f64;
    pub fn Write(&self, prob: f64, backoff: f64);
}

trait Quant {
    pub fn new() -> Self {}

    pub fn UpdateConfigFromBinary(&self, bin_format: &BinaryFormat, inp: u64, config: &Config);

    pub fn Size(&self, inp: u64, config: &Config) -> u64;

    pub fn MiddleBits(&self, config: &Config) -> u8;

    pub fn LongestBits(&self, config: &Config) -> u8;

    pub fn SetupMemory(&self, void: RefCell<i8>, inp_char: &str, config: &Config);

    pub fn Train(&self, inp: u8, &vector1: Vec<f64>, vector2: Vec<f64>);

    pub fn TrainProb(&self, inp: u8, &vector: Vec<f64>);

    pub fn FinishedLoading(&self, config: &Config);

    pub fn GetTables(&self, inp_char: &str, order_minus_2: i8) -> &Bins;

    pub fn LongestTable(&self) -> &Bins;
}

impl Quant for DontQuantize {
    fn new() -> Self {
        todo!()
    }
}

impl Quant for SeparatelyQuantize {
    fn new() -> Self {
        todo!()
    }
}

impl QueryPrinter {
    pub fn new(
        fd: i64,
        print_word: bool,
        print_line: bool,
        print_summary: bool,
        flush: bool,
    ) -> Self;

    pub fn Word(&self, surface: StringPiece, vocab: WordIndex, ret: &FullScoreReturn);

    pub fn Line(&self, oov: u64, total: f64);

    pub fn Summary(
        &self,
        ppl_including_oov: f64,
        ppl_excluding_oov: f64,
        corpus_oov: u64,
        corpus_tokens: u64,
    );
}

impl Search for HashedSearch<Value> {
    fn FastMakeNode(&self, begin: WordIndex, end: WordIndex, node: &Node) -> bool;

    fn InitializeFromARPA(
        &self,
        file: &str,
        f: &FilePiece,
        counts: &Vec<u64>,
        config: &Config,
        vocab: &SortedVocabulary,
        backing: &BinaryFormat,
    );

    fn LookupLongest(&self, word: WordIndex, node: &Node) -> LongestPointer;

    fn LookupMiddle(
        &self,
        order_minus_2: &str,
        word: WordIndex,
        node: &Node,
        independent_left: &bool,
        extend_left: &str,
    ) -> MiddlePointer;

    fn LookupUnigram(
        &self,
        word: WordIndex,
        node: &Node,
        independent_left: &bool,
        extend_left: &u64,
    ) -> UnigramPointer;

    fn Order(&self) -> &str;

    fn SetupMemory(&self, start: u8, counts: &Vec<u64>, config: &Config) -> u8;

    fn Size(&self, counts: &Vec<u64>, config: &Config) -> u64;

    fn UnknownUnigram(&self) -> &ProbBackoff;

    fn Unpack(&self, extend_pointer: u64, extend_length: &str, node: &Node) -> MiddlePointer;

    fn UpdateConfigFromBinary(
        &self,
        binary_format: &BinaryFormat,
        vector: &Vec<u64>,
        myint: uint64_t,
        config: &Config,
    );
}

trait VocabularyT {
    fn new() -> Self {}

    pub fn Index(&self, inp_str: &StringPiece) -> WordIndex {}
}

impl VocabularyT for SortedVocabulary {
    fn new() -> Self {
        return SortedVocabulary();
    }

    fn Index(&self, inp_str: &StringPiece) -> WordIndex {}
}

impl VocabularyT for ProbingVocabulary {
    fn new() -> Self {
        return ProbingVocabulary();
    }

    fn Index(&self, inp_str: &StringPiece) -> WordIndex {}
}

pub fn Query<Model, Printer>(
    file: &str,
    config: &Config,
    sentence_context: bool,
    printer: &QueryPrinter,
) {
}

pub fn RecognizeBinary(file: &str, recognized: &ModelType) -> bool {}

pub fn LookupLongest(word: WordIndex, node: &Node) -> LongestPointer {}

pub fn FastMakeNode(begin: WordIndex, end: WordIndex, node: &Node) -> bool {}

pub fn ShowSizes(file: &str, config: &Config) {}
