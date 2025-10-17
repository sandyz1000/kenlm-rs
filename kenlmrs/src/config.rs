use crate::constant::{ARPALoadComplain, LoadMethod, RestFunction, WarningAction, WriteMethod};
use crate::types::EnumerateVocab;

pub struct Config {
    load_method: LoadMethod,

    show_progress: bool,

    // Level of complaining to do when loading from ARPA instead of binary format.
    arpa_complain: ARPALoadComplain,

    // Where to log messages including the progress bar.  Set to NULL for silence.
    // messages: ostream,

    // This will be called with every string in the vocabulary by the constructor; it need
    // only exist for the lifetime of the constructor. See enumerate_vocab.hh for more detail.
    // Config does not take ownership; just delete/let it go out of scope after the constructor exits.
    enumerate_vocab: Option<Box<dyn EnumerateVocab>>,

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
    write_mmap: String,

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

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("load_method", &self.load_method)
            .field("show_progress", &self.show_progress)
            .field("arpa_complain", &self.arpa_complain)
            .field("enumerate_vocab", &self.enumerate_vocab.is_some())
            .field("unknown_missing", &self.unknown_missing)
            .field("sentence_marker_missing", &self.sentence_marker_missing)
            .field("positive_log_probability", &self.positive_log_probability)
            .field("unknown_missing_logprob", &self.unknown_missing_logprob)
            .field("probing_multiplier", &self.probing_multiplier)
            .field("building_memory", &self.building_memory)
            .field("write_mmap", &self.write_mmap)
            .field("write_method", &self.write_method)
            .field("include_vocab", &self.include_vocab)
            .field("rest_function", &self.rest_function)
            .field("prob_bits", &self.prob_bits)
            .field("backoff_bits", &self.backoff_bits)
            .field("pointer_bhiksha_bits", &self.pointer_bhiksha_bits)
            .finish()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            load_method: LoadMethod::PopulateOrRead,
            show_progress: true,
            arpa_complain: ARPALoadComplain::All,
            enumerate_vocab: None,
            unknown_missing: WarningAction::Complain,
            sentence_marker_missing: WarningAction::ThrowUp,
            positive_log_probability: WarningAction::ThrowUp,
            unknown_missing_logprob: -100.0,
            probing_multiplier: 1.5,
            building_memory: 1073741824, // 1GB
            temporary_directory_prefix: String::new(),
            write_mmap: String::new(),
            write_method: WriteMethod::WriteAfter,
            include_vocab: true,
            rest_function: RestFunction::RestMax,
            rest_lower_files: Vec::new(),
            prob_bits: 8,
            backoff_bits: 8,
            pointer_bhiksha_bits: 0,
        }
    }
}
