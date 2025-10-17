use crate::constant::*;

/// Word index type used throughout the library
pub type WordIndex = u32;

/// Model type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelType {
    Probing = 0,
    RestProbing = 1,
    Trie = 2,
    QuantTrie = 3,
    ArrayTrie = 4,
    QuantArrayTrie = 5,
}

impl ModelType {
    /// Historical names mapping
    pub const HASH_PROBING: ModelType = ModelType::Probing;
    pub const TRIE_SORTED: ModelType = ModelType::Trie;
    pub const QUANT_TRIE_SORTED: ModelType = ModelType::QuantTrie;
    pub const ARRAY_TRIE_SORTED: ModelType = ModelType::ArrayTrie;
    pub const QUANT_ARRAY_TRIE_SORTED: ModelType = ModelType::QuantArrayTrie;

    pub const K_QUANT_ADD: u8 = ModelType::QuantTrie as u8 - ModelType::Trie as u8;
    pub const K_ARRAY_ADD: u8 = ModelType::ArrayTrie as u8 - ModelType::Trie as u8;
}

/// State used for n-gram context in language model queries
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// Word indices for the context
    pub words: [WordIndex; KENLM_MAX_ORDER - 1],
    /// Backoff weights for each position
    pub backoff: [f32; KENLM_MAX_ORDER - 1],
    /// Length of the context
    pub length: u8,
}

impl State {
    pub fn new() -> Self {
        Self {
            words: [0; KENLM_MAX_ORDER - 1],
            backoff: [0.0; KENLM_MAX_ORDER - 1],
            length: 0,
        }
    }

    /// Zero out remaining entries for consistent memory comparison
    pub fn zero_remaining(&mut self) {
        for i in self.length as usize..(KENLM_MAX_ORDER - 1) {
            self.words[i] = 0;
            self.backoff[i] = 0.0;
        }
    }

    /// Get the length of the state
    pub fn len(&self) -> u8 {
        self.length
    }

    /// Three-way comparison function
    pub fn compare(&self, other: &State) -> std::cmp::Ordering {
        match self.length.cmp(&other.length) {
            std::cmp::Ordering::Equal => {
                self.words[..self.length as usize].cmp(&other.words[..other.length as usize])
            }
            other => other,
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.compare(other))
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.compare(other)
    }
}

impl Eq for State {}

/// Return type for full scoring operations
#[derive(Debug, Clone, PartialEq)]
pub struct FullScoreReturn {
    /// log10 probability
    pub prob: f32,
    /// Length of n-gram matched
    pub ngram_length: u8,
    /// Whether probability is independent of words to the left
    pub independent_left: bool,
    /// Extension information for left context
    pub extend_left: u64,
    /// Rest cost for extension to the left
    pub rest: f32,
}

impl FullScoreReturn {
    pub fn new() -> Self {
        Self {
            prob: 0.0,
            ngram_length: 0,
            independent_left: false,
            extend_left: 0,
            rest: 0.0,
        }
    }
}

impl Default for FullScoreReturn {
    fn default() -> Self {
        Self::new()
    }
}

/// Right context state (alias for State)
pub type Right = State;

/// Probability and backoff pair
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbBackoff {
    pub prob: f32,
    pub backoff: f32,
}

impl ProbBackoff {
    pub fn new(prob: f32, backoff: f32) -> Self {
        Self { prob, backoff }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Default for ProbBackoff {
    fn default() -> Self {
        Self::zero()
    }
}

/// Configuration for language model loading and operation
pub struct Config {
    /// How to handle unknown words
    pub unknown_missing: UnknownMissing,
    /// Probability for unknown words when they're missing
    pub unknown_missing_logprob: f32,
    /// Probing hash table multiplier
    pub probing_multiplier: f32,
    /// Memory used for building
    pub building_memory: usize,
    /// Temporary directory prefix
    pub temporary_directory_prefix: Option<String>,
    /// ARPA complaint level
    pub arpa_complain: ARPAComplain,
    /// Write method for binary files
    pub write_method: WriteMethod,
    /// Write memory map file
    pub write_mmap: Option<String>,
    /// Load method
    pub load_method: LoadMethod,
    /// Show progress messages
    pub messages: Option<Box<dyn std::io::Write>>,
    /// Enumerate vocabulary during loading
    pub enumerate_vocab: Option<Box<dyn EnumerateVocab>>,
    /// Warn about positive log probabilities
    pub positive_log_probability: PositiveLogProbability,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("unknown_missing", &self.unknown_missing)
            .field("unknown_missing_logprob", &self.unknown_missing_logprob)
            .field("probing_multiplier", &self.probing_multiplier)
            .field("building_memory", &self.building_memory)
            .field(
                "temporary_directory_prefix",
                &self.temporary_directory_prefix,
            )
            .field("arpa_complain", &self.arpa_complain)
            .field("write_method", &self.write_method)
            .field("write_mmap", &self.write_mmap)
            .field("load_method", &self.load_method)
            .field("messages", &self.messages.is_some())
            .field("enumerate_vocab", &self.enumerate_vocab.is_some())
            .field("positive_log_probability", &self.positive_log_probability)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownMissing {
    ThrowUp,
    ComplainEndlessly,
    Silent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ARPAComplain {
    All,
    Expensive,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMethod {
    WriteMethod,
    WriteAfter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMethod {
    ReadMethod,
    MMapMethod,
    LazyMethod,
    PopulateOrRead,
    PopulateOrLazy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositiveLogProbability {
    Throw,
    Complain,
    Silent,
}

/// Trait for vocabulary enumeration during loading
pub trait EnumerateVocab {
    fn add(&mut self, word_index: WordIndex, word: &str);
}

impl Default for Config {
    fn default() -> Self {
        Self {
            unknown_missing: UnknownMissing::ThrowUp,
            unknown_missing_logprob: -100.0,
            probing_multiplier: 1.5,
            building_memory: 1 << 30, // 1GB
            temporary_directory_prefix: None,
            arpa_complain: ARPAComplain::All,
            write_method: WriteMethod::WriteMethod,
            write_mmap: None,
            load_method: LoadMethod::ReadMethod,
            messages: None,
            enumerate_vocab: None,
            positive_log_probability: PositiveLogProbability::Throw,
        }
    }
}

impl Config {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if progress messages should be shown
    pub fn progress_messages(&self) -> bool {
        self.messages.is_some()
    }
}
