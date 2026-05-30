/// Maximum order supported by KenLM
pub const KENLM_MAX_ORDER: usize = 6;

/// Special word indices
pub const UNK_WORD: u32 = 0;
pub const BOS_WORD: u32 = 1;
pub const EOS_WORD: u32 = 2;

/// Default values
pub const DEFAULT_UNKNOWN_PROB: f32 = -100.0;
pub const DEFAULT_PROBING_MULTIPLIER: f32 = 1.5;

/// Binary format magic string
pub const BINARY_MAGIC: &str = "mmap lm http://kheafield.com/code";

/// Model version
pub const MODEL_VERSION: u32 = 5;

/// Extension constants for quantization
pub const K_EXTENSION_QUANT: u64 = 1;
pub const K_NO_EXTENSION_QUANT: u64 = 0;
pub const K_EXTENSION_BACKOFF: f32 = std::f32::NEG_INFINITY;
// Negative zero: sign bit set means "no higher-order n-gram extends from here".
// set_extension() converts this to +0.0 when a child n-gram is discovered.
pub const K_NO_EXTENSION_BACKOFF: f32 = f32::from_bits(0x80000000u32); // -0.0

#[derive(Debug, Clone)]
pub enum ARPALoadComplain {
    All,
    Expensive,
    None,
}

#[derive(Debug, Clone)]
pub enum WriteMethod {
    WriteMmap, // Map the file directly.
    WriteAfter, // Write after we're done.
}

// Left rest options. Only used when the model includes rest costs.
#[derive(Debug, Clone)]
pub enum RestFunction {
    RestMax, // Maximum of any score to the left
    RestLower, // Use lower-order files given below.
}

#[derive(Debug)]
pub enum FilterMode {
    ModeCopy,
    ModeSingle,
    ModeMultiple,
    ModeUnion,
    ModeUnset,
}

#[derive(Debug, Clone, Default)]
pub enum WarningAction {
    ThrowUp,
    Complain,
    #[default]
    Silent,
}

#[derive(Debug)]
pub enum Format {
    FormatArpa,
    FormatCount,
}

#[derive(Debug)]
pub enum LoadMethod {
    Lazy,
    PopulateOrLazy,
    PopulateOrRead,
    Read,
    ParallelRead,
}

#[derive(Debug)]
pub enum FormatEnum {
    FormatArpa,
    FormatCount,
}

#[derive(Debug)]
pub enum HookType {
    // Probability and backoff (or just q). Output must process the orders in
    // parallel or there will be a deadlock.
    ProbParallelHook,
    // Probability and backoff (or just q). Output can process orders any way it likes.
    // This requires writing the data to disk then reading.  Useful for ARPA files, which put unigrams first etc.
    ProbSequentialHook,
    // Keep this last so we know how many values there are.
    NumberOfHooks,
}
