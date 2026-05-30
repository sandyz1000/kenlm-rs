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
        let start = self.length as usize;
        self.words[start..].fill(0);
        self.backoff[start..].fill(0.0);
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

/// Left-context state used in chart-based decoding (e.g., SMT phrase-table scoring).
///
/// Stores pointers into the trie for each left-context word and a `full` flag
/// indicating whether the context is saturated up to the model order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Left {
    pub pointers: [u64; KENLM_MAX_ORDER - 1],
    pub length: u8,
    /// True when the left context cannot extend further (reached model order).
    pub full: bool,
}

impl Left {
    pub fn new() -> Self {
        Self {
            pointers: [0; KENLM_MAX_ORDER - 1],
            length: 0,
            full: false,
        }
    }

    /// Zeros out pointer slots beyond `length` so raw memory comparisons are stable.
    pub fn zero_remaining(&mut self) {
        for p in &mut self.pointers[self.length as usize..] {
            *p = 0;
        }
    }

    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        match self.length.cmp(&other.length) {
            Equal if self.length == 0 => Equal,
            Equal => {
                let last = (self.length - 1) as usize;
                match self.pointers[last].cmp(&other.pointers[last]) {
                    Equal => self.full.cmp(&other.full),
                    ord => ord,
                }
            }
            ord => ord,
        }
    }

    /// MurmurHash seed for use in `ChartState::hash_value`.
    pub fn hash_value(&self) -> u64 {
        use crate::utils::hash::murmur_hash_64a;
        let seed = if self.length > 0 {
            self.pointers[(self.length - 1) as usize]
        } else {
            0
        };
        let add = [self.length, self.full as u8];
        murmur_hash_64a(&add, seed)
    }
}

impl Default for Left {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialOrd for Left {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.compare(other))
    }
}

impl Ord for Left {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.compare(other)
    }
}

/// Combined left+right state for chart-based decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChartState {
    pub left: Left,
    pub right: State,
}

impl ChartState {
    pub fn new() -> Self {
        Self {
            left: Left::new(),
            right: State::new(),
        }
    }

    /// Zeros out unused slots in both `left` and `right` for stable comparisons.
    pub fn zero_remaining(&mut self) {
        self.left.zero_remaining();
        self.right.zero_remaining();
    }

    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        match self.left.compare(&other.left) {
            std::cmp::Ordering::Equal => self.right.compare(&other.right),
            ord => ord,
        }
    }

    /// Hash combining both left and right context hashes.
    pub fn hash_value(&self) -> u64 {
        use crate::utils::hash::murmur_hash_64a;
        let left_hash = self.left.hash_value();
        // Collect context word bytes in native byte order — safe, no raw pointers needed.
        let bytes: Vec<u8> = self.right.words[..self.right.length as usize]
            .iter()
            .flat_map(|w| w.to_ne_bytes())
            .collect();
        murmur_hash_64a(&bytes, left_hash)
    }
}

impl Default for ChartState {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialOrd for ChartState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.compare(other))
    }
}

impl Ord for ChartState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.compare(other)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    // ── State ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_state_default_zero_length() {
        let s = State::default();
        assert_eq!(s.length, 0);
        assert_eq!(s.words, [0; KENLM_MAX_ORDER - 1]);
    }

    #[test]
    fn test_state_len() {
        let mut s = State::new();
        s.length = 3;
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn test_state_zero_remaining() {
        let mut s = State::new();
        s.length = 2;
        s.words[0] = 10;
        s.words[1] = 20;
        s.words[2] = 99; // should be zeroed
        s.backoff[2] = 5.0;
        s.zero_remaining();
        assert_eq!(s.words[2], 0);
        assert_eq!(s.backoff[2], 0.0);
        // Filled portion is untouched
        assert_eq!(s.words[0], 10);
    }

    #[test]
    fn test_state_compare_equal() {
        let s1 = State::default();
        let s2 = State::default();
        assert_eq!(s1.compare(&s2), Ordering::Equal);
    }

    #[test]
    fn test_state_compare_shorter_less() {
        let mut s1 = State::new();
        s1.length = 1;
        s1.words[0] = 5;
        let mut s2 = State::new();
        s2.length = 2;
        s2.words[0] = 5;
        s2.words[1] = 6;
        assert_eq!(s1.compare(&s2), Ordering::Less);
        assert_eq!(s2.compare(&s1), Ordering::Greater);
    }

    #[test]
    fn test_state_compare_same_length_word_order() {
        let mut s1 = State::new();
        s1.length = 2;
        s1.words[0] = 1;
        s1.words[1] = 2;
        let mut s2 = State::new();
        s2.length = 2;
        s2.words[0] = 1;
        s2.words[1] = 3;
        assert_eq!(s1.compare(&s2), Ordering::Less);
    }

    #[test]
    fn test_state_ord_trait() {
        let s1 = State::default();
        let mut s2 = State::new();
        s2.length = 1;
        s2.words[0] = 1;
        assert!(s1 < s2);
    }

    // ── FullScoreReturn ───────────────────────────────────────────────────────

    #[test]
    fn test_full_score_return_default() {
        let r = FullScoreReturn::default();
        assert_eq!(r.prob, 0.0);
        assert_eq!(r.ngram_length, 0);
        assert!(!r.independent_left);
        assert_eq!(r.extend_left, 0);
        assert_eq!(r.rest, 0.0);
    }

    #[test]
    fn test_full_score_return_new() {
        let r = FullScoreReturn::new();
        assert_eq!(r, FullScoreReturn::default());
    }

    // ── ProbBackoff ───────────────────────────────────────────────────────────

    #[test]
    fn test_prob_backoff_new() {
        let pb = ProbBackoff::new(-1.5, -0.3);
        assert_eq!(pb.prob, -1.5);
        assert_eq!(pb.backoff, -0.3);
    }

    #[test]
    fn test_prob_backoff_zero() {
        let pb = ProbBackoff::zero();
        assert_eq!(pb.prob, 0.0);
        assert_eq!(pb.backoff, 0.0);
    }

    #[test]
    fn test_prob_backoff_default_is_zero() {
        assert_eq!(ProbBackoff::default(), ProbBackoff::zero());
    }

    // ── Config ────────────────────────────────────────────────────────────────

    #[test]
    fn test_config_new_defaults() {
        let c = Config::new();
        assert_eq!(c.probing_multiplier, 1.5);
        assert_eq!(c.unknown_missing_logprob, -100.0);
        assert!(!c.progress_messages());
        assert!(c.write_mmap.is_none());
        assert!(c.enumerate_vocab.is_none());
    }

    // ── ModelType ─────────────────────────────────────────────────────────────

    #[test]
    fn test_model_type_values() {
        assert_eq!(ModelType::Probing as u8, 0);
        assert_eq!(ModelType::Trie as u8, 2);
        assert_eq!(ModelType::QuantArrayTrie as u8, 5);
    }

    // ── Left ─────────────────────────────────────────────────────────────────

    #[test]
    fn left_should_default_to_zero_length() {
        let l = Left::default();
        assert_eq!(l.length, 0);
        assert!(!l.full);
        assert_eq!(l.pointers, [0u64; KENLM_MAX_ORDER - 1]);
    }

    #[test]
    fn left_compare_should_order_by_length_first() {
        let mut shorter = Left::new();
        shorter.length = 1;
        shorter.pointers[0] = 999;

        let mut longer = Left::new();
        longer.length = 2;
        longer.pointers[0] = 1;
        longer.pointers[1] = 2;

        assert_eq!(shorter.compare(&longer), Ordering::Less);
        assert_eq!(longer.compare(&shorter), Ordering::Greater);
    }

    #[test]
    fn left_compare_should_use_last_pointer_when_same_length() {
        let mut a = Left::new();
        a.length = 1;
        a.pointers[0] = 10;

        let mut b = Left::new();
        b.length = 1;
        b.pointers[0] = 20;

        assert_eq!(a.compare(&b), Ordering::Less);
    }

    #[test]
    fn left_compare_should_use_full_flag_as_tiebreaker() {
        let mut a = Left::new();
        a.length = 1;
        a.pointers[0] = 5;
        a.full = false;

        let mut b = Left::new();
        b.length = 1;
        b.pointers[0] = 5;
        b.full = true;

        assert_eq!(a.compare(&b), Ordering::Less); // false < true
    }

    #[test]
    fn left_zero_remaining_should_clear_slots_beyond_length() {
        let mut l = Left::new();
        l.length = 1;
        l.pointers[0] = 42;
        l.pointers[1] = 99; // should be cleared
        l.zero_remaining();
        assert_eq!(l.pointers[0], 42); // kept
        assert_eq!(l.pointers[1], 0);  // cleared
    }

    #[test]
    fn left_hash_value_should_be_deterministic() {
        let mut l = Left::new();
        l.length = 1;
        l.pointers[0] = 12345;
        assert_eq!(l.hash_value(), l.hash_value());
    }

    // ── ChartState ────────────────────────────────────────────────────────────

    #[test]
    fn chart_state_should_default_to_zero() {
        let cs = ChartState::default();
        assert_eq!(cs.left.length, 0);
        assert_eq!(cs.right.length, 0);
    }

    #[test]
    fn chart_state_compare_should_order_by_left_first() {
        let mut a = ChartState::new();
        a.left.length = 1;
        a.left.pointers[0] = 1;

        let mut b = ChartState::new();
        b.left.length = 2;
        b.left.pointers[0] = 1;
        b.left.pointers[1] = 2;

        assert_eq!(a.compare(&b), Ordering::Less);
    }

    #[test]
    fn chart_state_hash_value_should_differ_from_right_only_hash() {
        let mut cs = ChartState::new();
        cs.right.length = 1;
        cs.right.words[0] = 7;

        // Left has non-zero data → hash should differ from a state with zero left
        cs.left.length = 1;
        cs.left.pointers[0] = 999;

        let cs_no_left = ChartState::new(); // left length=0
        assert_ne!(cs.hash_value(), cs_no_left.hash_value());
    }
}
