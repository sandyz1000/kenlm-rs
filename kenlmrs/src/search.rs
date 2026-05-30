use crate::model::{is_independent_left, mark_extends_left, mark_independent, scoring_prob, set_extension};
use crate::types::{Config, ProbBackoff, WordIndex};
use std::marker::PhantomData;

/// Trait for search implementations in language models
pub trait Search: Default + Sized {
    type Node;
    type UnigramPointer: Pointer;
    type MiddlePointer: Pointer;
    type LongestPointer: Pointer;

    /// Model type for this search implementation
    const K_MODEL_TYPE: u8;

    /// Whether this search uses different rest costs
    const K_DIFFERENT_REST: bool;

    /// Version of this search implementation
    const K_VERSION: u32;

    /// Calculate memory size needed for this search structure
    fn size(counts: &[u64], config: &Config) -> u64;

    /// Setup memory layout for search structure
    fn setup_memory(&mut self, start: &mut [u8], counts: &[u64], config: &Config) -> &mut [u8];

    /// Initialize from ARPA file
    fn initialize_from_arpa(
        &mut self,
        file: &str,
        counts: &[u64],
        config: &Config,
        vocab: &mut dyn crate::vocabulary::Vocabulary
    ) -> Result<(), crate::error::LMError>;

    /// Get the order of this n-gram model
    fn order(&self) -> u8;

    /// Lookup unigram probability
    fn lookup_unigram(
        &self,
        word: WordIndex,
        node: &mut Self::Node,
        independent_left: &mut bool,
        extend_left: &mut u64
    ) -> Self::UnigramPointer;

    /// Lookup middle n-gram probability
    fn lookup_middle(
        &self,
        order_minus_2: u8,
        word: WordIndex,
        node: &mut Self::Node,
        independent_left: &mut bool,
        extend_left: &mut u64
    ) -> Self::MiddlePointer;

    /// Lookup longest n-gram probability
    fn lookup_longest(&self, word: WordIndex, node: &Self::Node) -> Self::LongestPointer;

    /// Fast node creation for efficiency
    fn fast_make_node(&self, begin: &[WordIndex], node: &mut Self::Node) -> bool;

    /// Unpack pointer information
    fn unpack(
        &self,
        extend_pointer: u64,
        extend_length: u8,
        node: &mut Self::Node
    ) -> Self::MiddlePointer;

    /// Get unknown unigram weights
    fn unknown_unigram(&mut self) -> &mut ProbBackoff;
}

/// Trait for pointer types returned by search operations
pub trait Pointer {
    /// Check if the pointer points to a valid entry
    fn found(&self) -> bool;

    /// Get the probability value
    fn prob(&self) -> f32;

    /// Get the backoff value (if applicable)
    fn backoff(&self) -> f32 {
        0.0
    }

    /// Get the rest cost (if applicable)
    fn rest(&self) -> f32 {
        0.0
    }

    /// Check if this entry is independent of left context
    fn independent_left(&self) -> bool {
        false
    }
}

/// Hash-based search implementation for probing models
#[derive(Debug)]
pub struct HashedSearch<V> {
    unigram: UnigramTable,
    middle: Vec<MiddleTable>,
    longest: LongestTable,
    _phantom: PhantomData<V>,
}

impl<V: Value> HashedSearch<V> {
    /// Reads an optional backoff value from `file_piece` after a probability or word token.
    /// Expects either `\t<float>\n` (backoff present) or `\n`/`\r` (no backoff → K_NO_EXTENSION_BACKOFF).
    fn read_backoff(
        file_piece: &mut crate::utils::pieces::file::FilePiece,
    ) -> Result<f32, crate::error::LMError> {
        use crate::constant::K_NO_EXTENSION_BACKOFF;
        match file_piece.get()? {
            '\t' => {
                let bo = file_piece.read_float()?;
                let nl = file_piece.get()?;
                if nl != '\n' && nl != '\r' {
                    return Err(crate::error::LMError::InvalidArpa(
                        "Expected newline after backoff".to_string(),
                    ));
                }
                Ok(bo)
            }
            '\n' | '\r' => Ok(K_NO_EXTENSION_BACKOFF),
            other => Err(crate::error::LMError::InvalidArpa(format!(
                "Expected tab or newline for backoff, got '{other}'"
            ))),
        }
    }

    /// Builds incremental hash keys for an n-gram whose words are in reversed ARPA order
    /// (vocab_ids[0] = new word, vocab_ids[1] = most-recent context, etc.).
    ///
    /// Returns a `Vec` of length `n-1` where `keys[k]` is the hash of the (k+2)-gram
    /// formed by vocab_ids[0..=k+1]. The final key `keys[n-2]` is the hash for the whole
    /// n-gram and matches C++ `ReadNGrams::keys[n-2]`.
    fn build_ngram_keys(vocab_ids: &[WordIndex]) -> Vec<u64> {
        assert!(vocab_ids.len() >= 2, "build_ngram_keys requires at least a bigram");
        vocab_ids[1..]
            .iter()
            .scan(vocab_ids[0] as u64, |acc, &w| {
                *acc = combine_word_hash(*acc, w);
                Some(*acc)
            })
            .collect()
    }

    /// C++ FindLower + AdjustLower: clears the `independent_left` sign bit on the new word's
    /// unigram AND on any missing intermediate scoring-chain n-grams (inserting blank entries).
    ///
    /// Blank entries are inserted at any scoring-chain level that doesn't yet have an entry,
    /// so that `lookup_middle` can chain through them to reach the actual higher-order n-gram.
    fn find_lower_and_mark_extends(
        unigrams: &mut Vec<ProbBackoff>,
        middle: &mut Vec<MiddleTable>,
        vocab_ids: &[WordIndex],
        keys: &[u64],
    ) {
        use crate::constant::K_NO_EXTENSION_BACKOFF;

        let n = vocab_ids.len();
        let new_word = vocab_ids[0] as usize;

        if n == 2 {
            // Bigram: mark the new word's unigram as extending left.
            if new_word < unigrams.len() {
                unigrams[new_word].prob = mark_extends_left(unigrams[new_word].prob);
            }
            return;
        }

        // n ≥ 3: Walk scoring-chain levels from (n-3) down to 0.
        // Insert blank entries for any missing intermediate levels and record where we stopped.
        let mut found_level: Option<usize> = None;
        for level in (0..=(n - 3)).rev() {
            let key = keys[level];
            if middle[level].data.contains_key(&key) {
                found_level = Some(level);
                break;
            }
            // Insert blank: backoff = -0.0 (no extension until a child is found).
            middle[level].data.insert(key, ProbBackoff {
                prob: mark_independent(0.0_f32),
                backoff: K_NO_EXTENSION_BACKOFF,
            });
        }

        // Mark all levels from the bottom (found or 0) up to n-3 as extending left.
        let bottom = found_level.unwrap_or(0);
        for level in bottom..=(n - 3) {
            let key = keys[level];
            if let Some(entry) = middle[level].data.get_mut(&key) {
                entry.prob = mark_extends_left(entry.prob);
            }
        }

        // If we fell all the way to the unigram level, clear its sign bit too.
        if found_level.is_none() && new_word < unigrams.len() {
            unigrams[new_word].prob = mark_extends_left(unigrams[new_word].prob);
        }
    }

    /// C++ ActivateLowerMiddle / ActivateUnigram: sets the extension bit on the backoff of
    /// the ARPA-order context n-gram (so `state.length` reflects that a child exists).
    ///
    /// The ARPA-order context uses key chain starting from `vocab_ids[1]`, which is the
    /// scoring key for the (n-1)-gram whose new_word is vocab_ids[1].
    fn activate_context_backoff(
        unigrams: &mut Vec<ProbBackoff>,
        middle: &mut Vec<MiddleTable>,
        vocab_ids: &[WordIndex],
    ) {
        let n = vocab_ids.len();
        if n == 2 {
            // Context is the unigram vocab_ids[1]; set extension on its backoff.
            let ctx = vocab_ids[1] as usize;
            if ctx < unigrams.len() {
                set_extension(&mut unigrams[ctx].backoff);
            }
        } else {
            // Context is the (n-1)-gram stored in middle[n-3].
            // Its scoring key is comb_chain(vocab_ids[1], vocab_ids[2], ..., vocab_ids[n-1]).
            let mut hash = vocab_ids[1] as u64;
            for &w in &vocab_ids[2..] {
                hash = combine_word_hash(hash, w);
            }
            let ctx_level = n - 3;
            if let Some(entry) = middle[ctx_level].data.get_mut(&hash) {
                set_extension(&mut entry.backoff);
            }
        }
    }

    /// Reads and stores unigrams from an already-positioned `FilePiece`.
    /// Each probability is stored with the sign bit set (`mark_independent`) to indicate that
    /// no higher-order match has been discovered yet.
    fn load_unigrams(
        &mut self,
        file_piece: &mut crate::utils::pieces::file::FilePiece,
        count: u64,
        vocab: &mut dyn crate::vocabulary::Vocabulary,
        warn: &crate::arpa_reader::PositiveProbWarn,
    ) -> Result<(), crate::error::LMError> {
        use crate::arpa_reader::read_ngram_header;
        use crate::constant::K_NO_EXTENSION_BACKOFF;

        read_ngram_header(file_piece, 1)?;
        // +1 so slot 0 (UNK) is always valid even if the ARPA omits <unk>
        self.unigram = UnigramTable::with_capacity(count as usize + 1);

        // Pre-register the three special words so they always land at the
        // canonical indices (0=<unk>, 1=<s>, 2=</s>) matching C++ KenLM.
        vocab.add_word("<unk>");
        vocab.add_word("<s>");
        vocab.add_word("</s>");

        for _ in 0..count {
            let raw_prob = file_piece.read_float()?;
            let raw_prob = if raw_prob > 0.0 {
                warn.warn(raw_prob);
                0.0_f32
            } else {
                raw_prob
            };

            let c = file_piece.get()?;
            if c != '\t' {
                return Err(crate::error::LMError::InvalidArpa(format!(
                    "Expected tab after probability, got '{c}'"
                )));
            }

            let word_str = file_piece.read_delimited(&crate::arpa_reader::ARPA_SPACES)?;
            // add_word inserts if new, returns existing index if already present.
            let word = vocab.add_word(&word_str);

            if (word as usize) < self.unigram.data.len() {
                // Mark as independent: no higher-order n-gram has claimed this as context yet.
                self.unigram.data[word as usize].prob = mark_independent(raw_prob);

                self.unigram.data[word as usize].backoff = Self::read_backoff(file_piece)?;
            }
        }
        Ok(())
    }

    /// Reads and inserts all n-grams at order `n` from an already-positioned `FilePiece`,
    /// then marks each n-gram's context as having a left extension.
    fn load_order(
        &mut self,
        file_piece: &mut crate::utils::pieces::file::FilePiece,
        n: usize,
        count: u64,
        total_orders: usize,
        vocab: &mut dyn crate::vocabulary::Vocabulary,
        warn: &crate::arpa_reader::PositiveProbWarn,
    ) -> Result<(), crate::error::LMError> {
        use crate::arpa_reader::read_ngram_header;

        read_ngram_header(file_piece, n as u32)?;

        for _ in 0..count {
            let raw_prob = file_piece.read_float()?;
            let raw_prob = if raw_prob > 0.0 {
                warn.warn(raw_prob);
                0.0_f32
            } else {
                raw_prob
            };

            // vocab_ids stored in REVERSED ARPA order: vocab_ids[0]=new_word, vocab_ids[1..]=context
            let mut vocab_ids = vec![0u32; n];
            for slot in vocab_ids.iter_mut().rev() {
                let word_str = file_piece.read_delimited(&crate::arpa_reader::ARPA_SPACES)?;
                *slot = vocab.index(&word_str);
            }

            let backoff = Self::read_backoff(file_piece)?;

            let keys = Self::build_ngram_keys(&vocab_ids);
            let ngram_key = keys[n - 2];

            // New n-grams start as independent (sign bit set); context marking below may
            // clear the sign bit of a lower-order entry.
            let stored_prob = mark_independent(raw_prob);

            if n == total_orders {
                // Longest order: no backoff stored (nothing extends further).
                self.longest.insert(ngram_key, stored_prob);
            } else {
                // Middle order
                let table_idx = n - 2;
                self.middle[table_idx].insert(ngram_key, ProbBackoff { prob: stored_prob, backoff });
            }

            // Insert blank scoring-chain intermediates and mark new_word as extending left.
            let (unigrams, middle) = (&mut self.unigram.data, &mut self.middle);
            Self::find_lower_and_mark_extends(unigrams, middle, &vocab_ids, &keys);
            // Set the extension bit on the ARPA-order context's backoff (for state.length).
            Self::activate_context_backoff(unigrams, middle, &vocab_ids);
        }
        Ok(())
    }

    pub(crate) fn compute_hash_key(&self, vocab_ids: &[WordIndex]) -> u64 {
        if vocab_ids.is_empty() {
            return 0;
        }
        let mut key = vocab_ids[0] as u64;
        for &word in &vocab_ids[1..] {
            key = combine_word_hash(key, word);
        }
        key
    }
}

impl<V: Value> Default for HashedSearch<V> {
    fn default() -> Self {
        Self {
            unigram: UnigramTable::new(),
            middle: Vec::new(),
            longest: LongestTable::new(),
            _phantom: PhantomData,
        }
    }
}

impl<V: Value> Search for HashedSearch<V> {
    type Node = u64;
    type UnigramPointer = HashedUnigramPointer<V>;
    type MiddlePointer = HashedMiddlePointer<V>;
    type LongestPointer = HashedLongestPointer;

    const K_MODEL_TYPE: u8 = V::K_PROBING_MODEL_TYPE;
    const K_DIFFERENT_REST: bool = V::K_DIFFERENT_REST;
    const K_VERSION: u32 = 0;

    fn size(counts: &[u64], config: &Config) -> u64 {
        let mut ret = UnigramTable::size(counts[0]);
        for n in 1..counts.len() - 1 {
            ret += MiddleTable::size(counts[n], config.probing_multiplier);
        }
        ret + LongestTable::size(counts[counts.len() - 1], config.probing_multiplier)
    }

    fn setup_memory(&mut self, _start: &mut [u8], counts: &[u64], _config: &Config) -> &mut [u8] {
        // For now, just initialize with defaults
        self.unigram = UnigramTable::with_capacity(counts[0] as usize);
        self.middle.clear();
        for _n in 1..counts.len() - 1 {
            self.middle.push(MiddleTable::with_capacity(1000));
        }
        self.longest = LongestTable::with_capacity(counts[counts.len() - 1] as usize);

        // Return empty slice for now
        &mut []
    }

    fn initialize_from_arpa(
        &mut self,
        file: &str,
        counts: &[u64],
        config: &Config,
        vocab: &mut dyn crate::vocabulary::Vocabulary,
    ) -> Result<(), crate::error::LMError> {
        use crate::arpa_reader::{read_arpa_counts, read_end, PositiveProbWarn};
        use crate::constant::WarningAction;
        use crate::utils::pieces::file::FilePiece;

        let mut fp = FilePiece::open(file)?;

        // Consume the \data\ header so the file is positioned at the first \N-grams: section.
        read_arpa_counts(&mut fp)?;

        let warn = PositiveProbWarn::new(WarningAction::Complain);
        let total_orders = counts.len();

        self.load_unigrams(&mut fp, counts[0], vocab, &warn)?;

        // Allocate middle tables for orders 2 .. total_orders-1
        self.middle.clear();
        for n in 2..total_orders {
            let cap = ((counts[n - 1] as f32) * config.probing_multiplier) as usize;
            self.middle.push(MiddleTable::with_capacity(cap));
        }
        let longest_cap = ((counts[total_orders - 1] as f32) * config.probing_multiplier) as usize;
        self.longest = LongestTable::with_capacity(longest_cap);

        for n in 2..=total_orders {
            self.load_order(&mut fp, n, counts[n - 1], total_orders, vocab, &warn)?;
        }

        read_end(&mut fp)
    }

    fn order(&self) -> u8 {
        (self.middle.len() + 2) as u8
    }

    fn lookup_unigram(
        &self,
        word: WordIndex,
        node: &mut Self::Node,
        independent_left: &mut bool,
        extend_left: &mut u64
    ) -> Self::UnigramPointer {
        *extend_left = word as u64;
        *node = *extend_left;
        let pointer = self.unigram.lookup(word);
        *independent_left = pointer.independent_left();
        pointer
    }

    fn lookup_middle(
        &self,
        order_minus_2: u8,
        word: WordIndex,
        node: &mut Self::Node,
        independent_left: &mut bool,
        extend_left: &mut u64
    ) -> Self::MiddlePointer {
        let key = combine_word_hash(*node, word);
        *extend_left = key;
        *node = key;
        let pointer = self.middle[order_minus_2 as usize].lookup(key);
        *independent_left = pointer.independent_left();
        pointer
    }

    fn lookup_longest(&self, word: WordIndex, node: &Self::Node) -> Self::LongestPointer {
        let key = combine_word_hash(*node, word);
        self.longest.lookup(key)
    }

    fn fast_make_node(&self, begin: &[WordIndex], node: &mut Self::Node) -> bool {
        if begin.is_empty() {
            return false;
        }

        *node = begin[0] as u64;
        for &word in &begin[1..] {
            *node = combine_word_hash(*node, word);
        }
        true
    }

    fn unpack(
        &self,
        extend_pointer: u64,
        extend_length: u8,
        node: &mut Self::Node
    ) -> Self::MiddlePointer {
        *node = extend_pointer;
        if extend_length == 1 {
            // Convert unigram pointer to middle pointer
            let unigram_ptr = self.unigram.lookup(extend_pointer as WordIndex);
            HashedMiddlePointer::from_unigram(unigram_ptr)
        } else {
            self.middle[(extend_length - 2) as usize].lookup(extend_pointer)
        }
    }

    fn unknown_unigram(&mut self) -> &mut ProbBackoff {
        self.unigram.unknown_mut()
    }
}

/// Trie-based search (structure exists; ARPA loading not yet implemented).
#[derive(Debug)]
pub struct TrieSearch<Q, B> {
    unknown_weights: ProbBackoff,
    _phantom: PhantomData<(Q, B)>,
}

impl<Q: Quantization, B: Bhiksha> Default for TrieSearch<Q, B> {
    fn default() -> Self {
        Self {
            unknown_weights: ProbBackoff::default(),
            _phantom: PhantomData,
        }
    }
}

impl<Q: Quantization, B: Bhiksha> Search for TrieSearch<Q, B> {
    type Node = TrieNode;
    type UnigramPointer = TrieUnigramPointer<Q>;
    type MiddlePointer = TrieMiddlePointer<Q>;
    type LongestPointer = TrieLongestPointer<Q>;

    const K_MODEL_TYPE: u8 = 2;
    const K_DIFFERENT_REST: bool = false;
    const K_VERSION: u32 = 1;

    fn size(_counts: &[u64], _config: &Config) -> u64 {
        0 // Not yet implemented
    }

    fn setup_memory(&mut self, _start: &mut [u8], _counts: &[u64], _config: &Config) -> &mut [u8] {
        &mut [] // Not yet implemented
    }

    fn initialize_from_arpa(
        &mut self,
        _file: &str,
        _counts: &[u64],
        _config: &Config,
        _vocab: &mut dyn crate::vocabulary::Vocabulary,
    ) -> Result<(), crate::error::LMError> {
        Err(crate::error::LMError::LoadError(
            "TrieSearch: ARPA loading is not yet implemented".into(),
        ))
    }

    fn order(&self) -> u8 {
        0
    }

    fn lookup_unigram(
        &self,
        _word: WordIndex,
        _node: &mut Self::Node,
        independent_left: &mut bool,
        _extend_left: &mut u64,
    ) -> Self::UnigramPointer {
        *independent_left = true;
        TrieUnigramPointer(PhantomData)
    }

    fn lookup_middle(
        &self,
        _order_minus_2: u8,
        _word: WordIndex,
        _node: &mut Self::Node,
        independent_left: &mut bool,
        _extend_left: &mut u64,
    ) -> Self::MiddlePointer {
        *independent_left = true;
        TrieMiddlePointer(PhantomData)
    }

    fn lookup_longest(&self, _word: WordIndex, _node: &Self::Node) -> Self::LongestPointer {
        TrieLongestPointer(PhantomData)
    }

    fn fast_make_node(&self, _begin: &[WordIndex], _node: &mut Self::Node) -> bool {
        false
    }

    fn unpack(
        &self,
        _extend_pointer: u64,
        _extend_length: u8,
        _node: &mut Self::Node,
    ) -> Self::MiddlePointer {
        TrieMiddlePointer(PhantomData)
    }

    fn unknown_unigram(&mut self) -> &mut ProbBackoff {
        &mut self.unknown_weights
    }
}

/// Trait for value types used in language models
pub trait Value {
    const K_PROBING_MODEL_TYPE: u8;
    const K_DIFFERENT_REST: bool;
}

/// Standard backoff value type
#[derive(Debug, Clone, Copy)]
pub struct BackoffValue;

impl Value for BackoffValue {
    const K_PROBING_MODEL_TYPE: u8 = 0; // PROBING
    const K_DIFFERENT_REST: bool = false;
}

/// Rest value type for models with rest costs
#[derive(Debug, Clone, Copy)]
pub struct RestValue;

impl Value for RestValue {
    const K_PROBING_MODEL_TYPE: u8 = 1; // REST_PROBING
    const K_DIFFERENT_REST: bool = true;
}

pub use crate::bhiksha::{ ArrayBhiksha, DontBhiksha };
/// Re-export quantization and Bhiksha traits and types
pub use crate::quantize::{ DontQuantize, SeparatelyQuantize };

// Traits for quantization and Bhiksha
pub trait Quantization {}
pub trait Bhiksha {}

impl Quantization for DontQuantize {}
impl Quantization for SeparatelyQuantize {}
impl Bhiksha for DontBhiksha {}
impl Bhiksha for ArrayBhiksha {}

// Placeholder implementations for hash table structures
use std::collections::HashMap;

#[derive(Debug)]
struct UnigramTable {
    data: Vec<ProbBackoff>,
    unknown: ProbBackoff,
}

impl UnigramTable {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            unknown: ProbBackoff::default(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: vec![ProbBackoff::default(); capacity + 1], // +1 for <unk>
            unknown: ProbBackoff::default(),
        }
    }

    fn size(_count: u64) -> u64 {
        // Calculate actual size needed
        1024 // Placeholder
    }

    fn from_memory(_memory: &mut [u8]) -> Self {
        Self::new()
    }

    fn lookup<V: Value>(&self, word: WordIndex) -> HashedUnigramPointer<V> {
        if (word as usize) < self.data.len() {
            let weights = &self.data[word as usize];
            HashedUnigramPointer::new(weights.prob, weights.backoff, true)
        } else {
            HashedUnigramPointer::new(self.unknown.prob, self.unknown.backoff, false)
        }
    }

    fn unknown_mut(&mut self) -> &mut ProbBackoff {
        &mut self.unknown
    }
}

#[derive(Debug)]
struct MiddleTable {
    data: HashMap<u64, ProbBackoff>,
}

impl MiddleTable {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
        }
    }

    fn size(_count: u64, _multiplier: f32) -> u64 {
        1024 // Placeholder
    }

    fn from_memory(_memory: &mut [u8]) -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn insert(&mut self, key: u64, weights: ProbBackoff) {
        self.data.insert(key, weights);
    }

    fn lookup<V: Value>(&self, key: u64) -> HashedMiddlePointer<V> {
        if let Some(weights) = self.data.get(&key) {
            HashedMiddlePointer::new(weights.prob, weights.backoff, true)
        } else {
            HashedMiddlePointer::new(0.0, 0.0, false)
        }
    }
}

#[derive(Debug)]
struct LongestTable {
    data: HashMap<u64, f32>,
}

impl LongestTable {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
        }
    }

    fn size(_count: u64, _multiplier: f32) -> u64 {
        1024 // Placeholder
    }

    fn from_memory(_memory: &mut [u8]) -> Self {
        Self::new()
    }

    fn insert(&mut self, key: u64, prob: f32) {
        self.data.insert(key, prob);
    }

    fn lookup(&self, key: u64) -> HashedLongestPointer {
        if let Some(&prob) = self.data.get(&key) {
            HashedLongestPointer::new_found(prob)
        } else {
            HashedLongestPointer::new(0.0)
        }
    }
}

// Pointer implementations
#[derive(Debug)]
pub struct HashedUnigramPointer<V> {
    prob: f32,
    backoff: f32,
    rest: f32,
    found: bool,
    _phantom: PhantomData<V>,
}

impl<V: Value> HashedUnigramPointer<V> {
    fn new(prob: f32, backoff: f32, found: bool) -> Self {
        // rest is same as prob if no different rest, otherwise separate
        let rest = prob; // rest == prob for now; RestValue will compute differently once trie is wired
        Self {
            prob,
            backoff,
            rest,
            found,
            _phantom: PhantomData,
        }
    }
}

impl<V> Pointer for HashedUnigramPointer<V> {
    fn found(&self) -> bool {
        self.found
    }
    fn prob(&self) -> f32 {
        // Force sign bit ON so the returned value is always a valid negative log-prob.
        scoring_prob(self.prob)
    }
    fn backoff(&self) -> f32 {
        self.backoff
    }
    fn rest(&self) -> f32 {
        self.rest
    }
    fn independent_left(&self) -> bool {
        // Sign bit SET on stored prob = no higher-order n-gram uses this as context.
        is_independent_left(self.prob)
    }
}

#[derive(Debug)]
pub struct HashedMiddlePointer<V> {
    prob: f32,
    backoff: f32,
    rest: f32,
    found: bool,
    _phantom: PhantomData<V>,
}

impl<V: Value> HashedMiddlePointer<V> {
    fn new(prob: f32, backoff: f32, found: bool) -> Self {
        let rest = prob; // rest == prob for now; RestValue will compute differently once trie is wired
        Self {
            prob,
            backoff,
            rest,
            found,
            _phantom: PhantomData,
        }
    }

    fn from_unigram(unigram: HashedUnigramPointer<V>) -> Self {
        Self {
            prob: unigram.prob,
            backoff: unigram.backoff,
            rest: unigram.rest,
            found: unigram.found,
            _phantom: PhantomData,
        }
    }
}

impl<V> Pointer for HashedMiddlePointer<V> {
    fn found(&self) -> bool {
        self.found
    }
    fn prob(&self) -> f32 {
        scoring_prob(self.prob)
    }
    fn backoff(&self) -> f32 {
        self.backoff
    }
    fn rest(&self) -> f32 {
        self.rest
    }
    fn independent_left(&self) -> bool {
        is_independent_left(self.prob)
    }
}

#[derive(Debug)]
pub struct HashedLongestPointer {
    prob: f32,
    found: bool,
}

impl HashedLongestPointer {
    fn new(prob: f32) -> Self {
        Self { prob, found: false }
    }

    fn new_found(prob: f32) -> Self {
        Self { prob, found: true }
    }
}

impl Pointer for HashedLongestPointer {
    fn found(&self) -> bool {
        self.found
    }
    fn prob(&self) -> f32 {
        self.prob
    }
}

// Trie pointer types (placeholders)
#[derive(Debug)]
pub struct TrieNode;
#[derive(Debug)]
pub struct TrieUnigramPointer<Q>(PhantomData<Q>);
#[derive(Debug)]
pub struct TrieMiddlePointer<Q>(PhantomData<Q>);
#[derive(Debug)]
pub struct TrieLongestPointer<Q>(PhantomData<Q>);

impl<Q> Pointer for TrieUnigramPointer<Q> {
    fn found(&self) -> bool { false }
    fn prob(&self) -> f32 { 0.0 }
}

impl<Q> Pointer for TrieMiddlePointer<Q> {
    fn found(&self) -> bool { false }
    fn prob(&self) -> f32 { 0.0 }
}

impl<Q> Pointer for TrieLongestPointer<Q> {
    fn found(&self) -> bool { false }
    fn prob(&self) -> f32 { 0.0 }
}

/// Hash function for combining word indices
fn combine_word_hash(current: u64, next: WordIndex) -> u64 {
    current.wrapping_mul(8978948897894561157) ^
        (1 + (next as u64)).wrapping_mul(17894857484156487943)
}

// Default implementation for TrieNode
impl Default for TrieNode {
    fn default() -> Self {
        TrieNode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Config;

    fn default_config() -> Config { Config::new() }

    // ── combine_word_hash ─────────────────────────────────────────────────────

    #[test]
    fn test_combine_word_hash_deterministic() {
        let h1 = combine_word_hash(123, 456);
        let h2 = combine_word_hash(123, 456);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_combine_word_hash_different_inputs() {
        let h1 = combine_word_hash(1, 2);
        let h2 = combine_word_hash(1, 3);
        let h3 = combine_word_hash(2, 2);
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
    }

    // ── HashedSearch ─────────────────────────────────────────────────────────

    #[test]
    fn test_hashed_search_default_order() {
        let s: HashedSearch<BackoffValue> = HashedSearch::default();
        // Default has empty middle → order = 0 + 2 = 2
        assert_eq!(s.order(), 2);
    }

    #[test]
    fn test_hashed_search_size_calculation() {
        let counts = vec![100u64, 500u64, 50u64];
        let config = default_config();
        let size = HashedSearch::<BackoffValue>::size(&counts, &config);
        assert!(size > 0);
    }

    #[test]
    fn test_hashed_search_setup_memory_updates_order() {
        let mut s: HashedSearch<BackoffValue> = HashedSearch::default();
        let counts = vec![5u64, 10u64, 3u64]; // unigrams=5, bigrams=10, trigrams=3
        let config = default_config();
        let mut mem: &mut [u8] = &mut [];
        s.setup_memory(&mut mem, &counts, &config);
        // middle has 1 entry (order 2 = bigrams), so order = 1 + 2 = 3
        assert_eq!(s.order(), 3);
    }

    #[test]
    fn test_hashed_search_unknown_unigram() {
        let mut s: HashedSearch<BackoffValue> = HashedSearch::default();
        let unk = s.unknown_unigram();
        assert_eq!(unk.prob, 0.0);
        assert_eq!(unk.backoff, 0.0);
    }

    #[test]
    fn test_hashed_search_lookup_unigram_not_found_when_empty() {
        let s: HashedSearch<BackoffValue> = HashedSearch::default();
        let mut node = 0u64;
        let mut independent_left = false;
        let mut extend_left = 0u64;
        let ptr = s.lookup_unigram(999, &mut node, &mut independent_left, &mut extend_left);
        // unigram table is empty → found=false
        assert!(!ptr.found());
    }

    #[test]
    fn test_hashed_search_fast_make_node_empty_returns_false() {
        let s: HashedSearch<BackoffValue> = HashedSearch::default();
        let mut node = 0u64;
        assert!(!s.fast_make_node(&[], &mut node));
    }

    #[test]
    fn test_hashed_search_fast_make_node_single_word() {
        let s: HashedSearch<BackoffValue> = HashedSearch::default();
        let mut node = 0u64;
        assert!(s.fast_make_node(&[42], &mut node));
        assert_eq!(node, 42);
    }

    #[test]
    fn test_hashed_search_compute_hash_key_empty() {
        let s: HashedSearch<BackoffValue> = HashedSearch::default();
        let key = s.compute_hash_key(&[]);
        assert_eq!(key, 0);
    }

    #[test]
    fn test_hashed_search_compute_hash_key_deterministic() {
        let s: HashedSearch<BackoffValue> = HashedSearch::default();
        let k1 = s.compute_hash_key(&[1, 2, 3]);
        let k2 = s.compute_hash_key(&[1, 2, 3]);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_hashed_search_unpack_length1() {
        let s: HashedSearch<BackoffValue> = HashedSearch::default();
        let mut node = 0u64;
        // extend_length=1 means unigram; with empty table, ptr.found() is false
        let ptr = s.unpack(0, 1, &mut node);
        assert!(!ptr.found()); // empty table
    }

    // ── Value marker types ────────────────────────────────────────────────────

    #[test]
    fn test_backoff_value_constants() {
        assert_eq!(BackoffValue::K_PROBING_MODEL_TYPE, 0);
        assert!(!BackoffValue::K_DIFFERENT_REST);
    }

    #[test]
    fn test_rest_value_constants() {
        assert_eq!(RestValue::K_PROBING_MODEL_TYPE, 1);
        assert!(RestValue::K_DIFFERENT_REST);
    }

    // ── Pointer trait defaults ────────────────────────────────────────────────

    #[test]
    fn test_hashed_unigram_pointer_fields() {
        let ptr: HashedUnigramPointer<BackoffValue> = HashedUnigramPointer::new(-1.5, -0.3, true);
        assert!(ptr.found());
        assert!((ptr.prob() - (-1.5)).abs() < 1e-6);
        assert!((ptr.backoff() - (-0.3)).abs() < 1e-6);
        assert!(ptr.independent_left());
    }

    #[test]
    fn test_hashed_longest_pointer_found_not_found() {
        let found = HashedLongestPointer::new_found(-2.0);
        let not_found = HashedLongestPointer::new(-2.0);
        assert!(found.found());
        assert!(!not_found.found());
        assert!((found.prob() - (-2.0)).abs() < 1e-6);
    }

    // ── Sign-bit encoding ─────────────────────────────────────────────────────

    #[test]
    fn independent_left_should_be_true_for_negative_stored_prob() {
        // All log probs start negative → sign bit set → independent by default
        let ptr: HashedUnigramPointer<BackoffValue> =
            HashedUnigramPointer::new(-1.5_f32, -0.0_f32, true);
        assert!(ptr.independent_left());
    }

    #[test]
    fn independent_left_should_be_false_after_sign_bit_cleared() {
        // Simulate a unigram whose prob had its sign bit cleared by mark_extends_left
        let cleared = mark_extends_left(-1.5_f32);
        let ptr: HashedUnigramPointer<BackoffValue> =
            HashedUnigramPointer::new(cleared, -0.0_f32, true);
        assert!(!ptr.independent_left());
    }

    #[test]
    fn prob_should_always_return_negative_for_scoring() {
        // Even with sign bit cleared, prob() forces sign bit ON
        let cleared = mark_extends_left(-1.5_f32);
        let ptr: HashedUnigramPointer<BackoffValue> =
            HashedUnigramPointer::new(cleared, 0.0_f32, true);
        assert!(ptr.prob() < 0.0, "scoring prob must be negative");
        assert!((ptr.prob() - (-1.5_f32)).abs() < 1e-6);
    }

    // ── ARPA integration smoke test ───────────────────────────────────────────

    #[test]
    fn scoring_should_use_bigram_when_context_is_available() {
        use std::io::Write;
        use tempfile::NamedTempFile;
        use crate::types::{Config, State};
        use crate::vocabulary::{ProbingVocabulary, Vocabulary};
        use crate::model::GenericModel;

        // Trigram ARPA: bigrams go to middle[0], trigrams go to longest.
        // This lets us verify middle-table lookup works correctly.
        let arpa = "\
\\data\\
ngram 1=3
ngram 2=1
ngram 3=1

\\1-grams:
-99\t<unk>
-1.5\thello\t-0.3
-2.0\tworld\t0

\\2-grams:
-0.8\thello\tworld\t-0.1

\\3-grams:
-0.5\t<unk>\thello\tworld

\\end\\
";
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{}", arpa).unwrap();
        f.flush().unwrap();

        let counts = vec![3u64, 1u64, 1u64];
        let config = Config::new();
        let mut vocab = ProbingVocabulary::new();
        vocab.add_word("<unk>");
        vocab.add_word("hello");
        vocab.add_word("world");

        let mut search: HashedSearch<BackoffValue> = HashedSearch::default();
        search
            .initialize_from_arpa(
                f.path().to_str().unwrap(),
                &counts,
                &config,
                &mut vocab,
            )
            .unwrap();

        let hello = vocab.index("hello");
        let world = vocab.index("world");

        let mut node = 0u64;
        let mut independent_left = false;
        let mut extend_left = 0u64;

        // Scoring follows the same path as full_score: look up the NEW WORD first (world),
        // then extend with the CONTEXT word (hello). Keys are built outward from new_word.

        // C++ KenLM clears the sign bit of the NEW WORD (world), not the context (hello).
        // hello is the CONTEXT of the bigram → only its backoff gets set_extension, not its prob.
        {
            let mut dummy_node = 0u64;
            let mut il = false;
            let mut el = 0u64;
            let uni_world_check = search.lookup_unigram(world, &mut dummy_node, &mut il, &mut el);
            assert!(!uni_world_check.independent_left(), "world is new_word of bigram → must NOT be independent");
            let uni_hello = search.lookup_unigram(hello, &mut dummy_node, &mut il, &mut el);
            assert!(uni_hello.independent_left(), "hello is only context, not new_word → must be independent");
        }

        // Scoring P(world|hello): new_word=world, context=[hello]
        // 1. lookup_unigram(world) → node = world_id, independent_left=false (world appears as new_word in bigram)
        let uni_world = search.lookup_unigram(world, &mut node, &mut independent_left, &mut extend_left);
        assert!(uni_world.found());
        assert!((uni_world.prob() - (-2.0_f32)).abs() < 0.01, "unigram world prob should be -2.0");

        // 2. lookup_middle(0, hello, node=world) → key = combine(world, hello)
        let bigram = search.lookup_middle(0, hello, &mut node, &mut independent_left, &mut extend_left);
        assert!(bigram.found(), "bigram hello→world must be found in middle[0]");
        assert!((bigram.prob() - (-0.8_f32)).abs() < 0.01, "bigram prob should be -0.8");
    }
}
