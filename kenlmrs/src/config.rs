use crate::constant::{ARPALoadComplain, LoadMethod, RestFunction, WarningAction, WriteMethod};



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
