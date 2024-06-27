#[derive(Debug, Clone)]
pub(crate) enum ARPALoadComplain {
    All,
    Expensive,
    None,
}

#[derive(Debug, Clone)]
pub enum WriteMethod {
    WriteMmap,  // Map the file directly.
    WriteAfter, // Write after we're done.
}

// Left rest options. Only used when the model includes rest costs.
#[derive(Debug, Clone)]
pub enum RestFunction {
    RestMax,   // Maximum of any score to the left
    RestLower, // Use lower-order files given below.
}

#[derive(Debug)]
pub(crate) enum FilterMode {
    ModeCopy,
    ModeSingle,
    ModeMultiple,
    ModeUnion,
    ModeUnset,
}

#[derive(Debug, Clone)]
pub(crate) enum WarningAction {
    ThrowUp,
    Complain,
    Silent,
}

// ------------------
// Constant PYX below
// ------------------
#[derive(Debug)]
pub(crate) enum ModelType {
    Probing,
    RestProbing,
    Trie,
    QuantTrie,
    ArrayTrie,
    QuantArrayTrie,
}

#[derive(Debug)]
pub(crate) enum Format {
    FormatArpa,
    FormatCount,
}

#[derive(Debug)]
pub(crate) enum LoadMethod {
    Lazy,
    PopulateOrLazy,
    PopulateOrRead,
    Read,
    ParallelRead,
}

#[derive(Debug)]
pub(crate) enum FormatEnum {
    FormatArpa,
    FormatCount,
}

#[derive(Debug)]
pub(crate) enum HookType {
    // Probability and backoff (or just q). Output must process the orders in
    // parallel or there will be a deadlock.
    ProbParallelHook,
    // Probability and backoff (or just q). Output can process orders any way it likes.
    // This requires writing the data to disk then reading.  Useful for ARPA files, which put unigrams first etc.
    ProbSequentialHook,
    // Keep this last so we know how many values there are.
    NumberOfHooks,
}
