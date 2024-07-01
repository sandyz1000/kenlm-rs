#[derive(Debug, Clone)]
pub enum ARPALoadComplain {
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
pub enum FilterMode {
    ModeCopy,
    ModeSingle,
    ModeMultiple,
    ModeUnion,
    ModeUnset,
}

#[derive(Debug, Clone)]
pub enum WarningAction {
    ThrowUp,
    Complain,
    Silent,
}

// ------------------
// Constant PYX below
// ------------------
#[derive(Debug)]
pub enum ModelType {
    Probing,
    RestProbing,
    Trie,
    QuantTrie,
    ArrayTrie,
    QuantArrayTrie,
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
