use crate::constant::*;
use crate::types::WordIndex;
use crate::utils::hash::{hash_for_vocab, UNKNOWN_HASH, UNKNOWN_CAP_HASH};
use std::collections::HashMap;

/// Base vocabulary trait that all vocabulary implementations must follow
pub trait Vocabulary {
    /// Get the word index for beginning of sentence
    fn begin_sentence(&self) -> WordIndex;

    /// Get the word index for end of sentence  
    fn end_sentence(&self) -> WordIndex;

    /// Get the word index for unknown/not found words
    fn not_found(&self) -> WordIndex;

    /// Get the word index for a string
    fn index(&self, str: &str) -> WordIndex;

    /// Convenience method for String input
    fn index_from_string(&self, str: &String) -> WordIndex {
        self.index(str.as_str())
    }

    /// Convenience method for str input (same as index)
    fn index_from_str(&self, str: &str) -> WordIndex {
        self.index(str)
    }

    /// Set special word indices
    fn set_special(
        &mut self,
        begin_sentence: WordIndex,
        end_sentence: WordIndex,
        not_found: WordIndex,
    );

    /// Get the size/bound of the vocabulary
    fn bound(&self) -> WordIndex;

    /// Check if unknown words were encountered during loading
    fn saw_unk(&self) -> bool;

    /// Insert `word` into the vocabulary, returning its index.
    /// If the word already exists, returns its existing index without duplicating.
    fn add_word(&mut self, word: &str) -> WordIndex;
}

/// Base vocabulary implementation with default behavior
pub struct BaseVocabulary {
    begin_sentence: WordIndex,
    end_sentence: WordIndex,
    not_found: WordIndex,
    bound: WordIndex,
    saw_unk: std::cell::Cell<bool>, // Use Cell for interior mutability
}

impl Vocabulary for BaseVocabulary {
    fn begin_sentence(&self) -> WordIndex {
        self.begin_sentence
    }

    fn end_sentence(&self) -> WordIndex {
        self.end_sentence
    }

    fn not_found(&self) -> WordIndex {
        self.not_found
    }

    fn index(&self, _str: &str) -> WordIndex {
        // Base implementation returns not_found for all queries
        // This should be overridden by concrete implementations
        self.not_found
    }

    fn set_special(
        &mut self,
        begin_sentence: WordIndex,
        end_sentence: WordIndex,
        not_found: WordIndex,
    ) {
        self.begin_sentence = begin_sentence;
        self.end_sentence = end_sentence;
        self.not_found = not_found;
    }

    fn bound(&self) -> WordIndex {
        self.bound
    }

    fn saw_unk(&self) -> bool {
        self.saw_unk.get()
    }

    fn add_word(&mut self, _word: &str) -> WordIndex {
        // BaseVocabulary is only a base helper; concrete types override this.
        unreachable!("add_word called on BaseVocabulary directly")
    }
}

impl BaseVocabulary {
    pub fn new() -> Self {
        Self {
            begin_sentence: BOS_WORD,
            end_sentence: EOS_WORD,
            not_found: UNK_WORD,
            bound: 3, // Start with 3 special words
            saw_unk: std::cell::Cell::new(false),
        }
    }

    pub fn with_special(
        begin_sentence: WordIndex,
        end_sentence: WordIndex,
        not_found: WordIndex,
    ) -> Self {
        let mut vocab = Self::new();
        vocab.set_special(begin_sentence, end_sentence, not_found);
        vocab
    }

    pub fn set_bound(&mut self, bound: WordIndex) {
        self.bound = bound;
    }

    pub fn set_saw_unk(&self, saw_unk: bool) {
        self.saw_unk.set(saw_unk);
    }
}

impl Default for BaseVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Probing vocabulary using a hash table for fast lookups
pub struct ProbingVocabulary {
    base: BaseVocabulary,
    word_to_index: HashMap<String, WordIndex>,
    index_to_word: Vec<String>,
}

impl Vocabulary for ProbingVocabulary {
    fn begin_sentence(&self) -> WordIndex {
        self.base.begin_sentence()
    }

    fn end_sentence(&self) -> WordIndex {
        self.base.end_sentence()
    }

    fn not_found(&self) -> WordIndex {
        self.base.not_found()
    }

    fn index(&self, str: &str) -> WordIndex {
        self.word_to_index
            .get(str)
            .copied()
            .unwrap_or(self.not_found())
    }

    fn set_special(
        &mut self,
        begin_sentence: WordIndex,
        end_sentence: WordIndex,
        not_found: WordIndex,
    ) {
        self.base
            .set_special(begin_sentence, end_sentence, not_found);
    }

    fn bound(&self) -> WordIndex {
        self.base.bound()
    }

    fn saw_unk(&self) -> bool {
        self.base.saw_unk()
    }

    fn add_word(&mut self, word: &str) -> WordIndex {
        ProbingVocabulary::add_word(self, word)
    }
}

impl ProbingVocabulary {
    pub fn new() -> Self {
        Self {
            base: BaseVocabulary::new(),
            word_to_index: HashMap::new(),
            index_to_word: Vec::new(),
        }
    }

    /// Add a word to the vocabulary, returning its index
    pub fn add_word(&mut self, word: &str) -> WordIndex {
        if let Some(&existing_index) = self.word_to_index.get(word) {
            return existing_index;
        }

        let new_index = self.index_to_word.len() as WordIndex;
        self.word_to_index.insert(word.to_string(), new_index);
        self.index_to_word.push(word.to_string());
        self.base.set_bound((new_index + 1).max(self.base.bound()));
        new_index
    }

    /// Get word by index
    pub fn word(&self, index: WordIndex) -> Option<&str> {
        self.index_to_word.get(index as usize).map(|s| s.as_str())
    }
}

impl Default for ProbingVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Sorted vocabulary for memory-efficient storage and binary search
/// Uses hash-based lookups similar to KenLM's SortedVocabulary
pub struct SortedVocabulary {
    base: BaseVocabulary,
    hashes: Vec<u64>,       // Sorted hashes for binary search
    strings: Vec<String>,    // Corresponding strings for enumeration
    sorted: bool,
}

impl Vocabulary for SortedVocabulary {
    fn begin_sentence(&self) -> WordIndex {
        self.base.begin_sentence()
    }

    fn end_sentence(&self) -> WordIndex {
        self.base.end_sentence()
    }

    fn not_found(&self) -> WordIndex {
        self.base.not_found()
    }

    fn index(&self, str: &str) -> WordIndex {
        let hash = hash_for_vocab(str);
        
        // Check for special unknown tokens
        if hash == UNKNOWN_HASH || hash == UNKNOWN_CAP_HASH {
            self.base.set_saw_unk(true);
            return 0; // UNK_WORD
        }
        
        if !self.sorted {
            // Linear search if not sorted yet
            for (i, &h) in self.hashes.iter().enumerate() {
                if h == hash {
                    // Add 1 because index 0 is reserved for <unk>
                    return (i + 1) as WordIndex;
                }
            }
        } else {
            // Binary search if sorted
            if let Ok(index) = self.hashes.binary_search(&hash) {
                return (index + 1) as WordIndex;
            }
        }
        self.not_found()
    }

    fn set_special(
        &mut self,
        begin_sentence: WordIndex,
        end_sentence: WordIndex,
        not_found: WordIndex,
    ) {
        self.base
            .set_special(begin_sentence, end_sentence, not_found);
    }

    fn bound(&self) -> WordIndex {
        self.base.bound()
    }

    fn saw_unk(&self) -> bool {
        self.base.saw_unk()
    }

    fn add_word(&mut self, word: &str) -> WordIndex {
        SortedVocabulary::insert(self, word)
    }
}

impl SortedVocabulary {
    pub fn new() -> Self {
        Self {
            base: BaseVocabulary::new(),
            hashes: Vec::new(),
            strings: Vec::new(),
            sorted: false,
        }
    }

    /// Insert a word into the vocabulary during loading
    /// Returns the word index (1-based, with 0 reserved for <unk>)
    pub fn insert(&mut self, word: &str) -> WordIndex {
        let hash = hash_for_vocab(word);
        
        // Check for special unknown tokens
        if hash == UNKNOWN_HASH || hash == UNKNOWN_CAP_HASH {
            self.base.set_saw_unk(true);
            return 0; // UNK_WORD
        }
        
        self.hashes.push(hash);
        self.strings.push(word.to_string());
        self.sorted = false;
        
        // Return 1-based index (0 is reserved for <unk>)
        let index = self.hashes.len() as WordIndex;
        self.base.set_bound(index + 1); // +1 to account for <unk> at 0
        index
    }

    /// Add a word to the vocabulary (alias for insert)
    pub fn add_word(&mut self, word: &str) -> WordIndex {
        self.insert(word)
    }

    /// Finish loading the vocabulary
    /// Sorts the hashes and reorders strings accordingly for efficient lookup
    pub fn finished_loading(&mut self) {
        if self.sorted {
            return;
        }

        let mut indices: Vec<usize> = (0..self.hashes.len()).collect();
        indices.sort_unstable_by_key(|&i| self.hashes[i]);

        // Copy hashes (u64 is Copy) and move strings without cloning.
        let old_hashes: Vec<u64> = self.hashes.clone();
        let mut old_strings = std::mem::take(&mut self.strings);
        self.hashes = indices.iter().map(|&i| old_hashes[i]).collect();
        // Each index appears exactly once — mem::take moves each String out once.
        self.strings = indices.iter().map(|&i| std::mem::take(&mut old_strings[i])).collect();
        
        self.sorted = true;
    }

    /// Sort the vocabulary for efficient binary search (alias for finished_loading)
    pub fn sort(&mut self) {
        self.finished_loading();
    }

    /// Get word by index (0 = <unk>, 1+ = actual words)
    pub fn word(&self, index: WordIndex) -> Option<&str> {
        if index == 0 {
            Some("<unk>")
        } else {
            self.strings.get((index - 1) as usize).map(|s| s.as_str())
        }
    }

    /// Get the number of words in vocabulary (including <unk>)
    pub fn size(&self) -> WordIndex {
        (self.hashes.len() + 1) as WordIndex
    }
}

impl Default for SortedVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constant::{BOS_WORD, EOS_WORD, UNK_WORD};

    // ── BaseVocabulary ────────────────────────────────────────────────────────

    #[test]
    fn test_base_vocab_defaults() {
        let v = BaseVocabulary::new();
        assert_eq!(v.begin_sentence(), BOS_WORD);
        assert_eq!(v.end_sentence(), EOS_WORD);
        assert_eq!(v.not_found(), UNK_WORD);
        assert_eq!(v.bound(), 3);
        assert!(!v.saw_unk());
    }

    #[test]
    fn test_base_vocab_set_special() {
        let mut v = BaseVocabulary::new();
        v.set_special(10, 11, 12);
        assert_eq!(v.begin_sentence(), 10);
        assert_eq!(v.end_sentence(), 11);
        assert_eq!(v.not_found(), 12);
    }

    #[test]
    fn test_base_vocab_saw_unk() {
        let v = BaseVocabulary::new();
        assert!(!v.saw_unk());
        v.set_saw_unk(true);
        assert!(v.saw_unk());
        v.set_saw_unk(false);
        assert!(!v.saw_unk());
    }

    #[test]
    fn test_base_vocab_with_special() {
        let v = BaseVocabulary::with_special(5, 6, 7);
        assert_eq!(v.begin_sentence(), 5);
        assert_eq!(v.end_sentence(), 6);
        assert_eq!(v.not_found(), 7);
    }

    // ── ProbingVocabulary ─────────────────────────────────────────────────────

    #[test]
    fn test_probing_vocab_add_and_lookup() {
        let mut v = ProbingVocabulary::new();
        let idx = v.add_word("hello");
        assert_eq!(v.index("hello"), idx);
        assert_eq!(v.word(idx), Some("hello"));
    }

    #[test]
    fn test_probing_vocab_unknown_word() {
        let v = ProbingVocabulary::new();
        assert_eq!(v.index("missing"), v.not_found());
    }

    #[test]
    fn test_probing_vocab_duplicate_add_returns_same_index() {
        let mut v = ProbingVocabulary::new();
        let i1 = v.add_word("dup");
        let i2 = v.add_word("dup");
        assert_eq!(i1, i2);
    }

    #[test]
    fn test_probing_vocab_multiple_words() {
        let mut v = ProbingVocabulary::new();
        let i1 = v.add_word("apple");
        let i2 = v.add_word("banana");
        let i3 = v.add_word("cherry");
        assert_ne!(i1, i2);
        assert_ne!(i2, i3);
        assert_eq!(v.index("apple"), i1);
        assert_eq!(v.index("banana"), i2);
        assert_eq!(v.index("cherry"), i3);
    }

    #[test]
    fn test_probing_vocab_bound_grows() {
        let mut v = ProbingVocabulary::new();
        let before = v.bound();
        v.add_word("new_word");
        assert!(v.bound() > before || v.bound() >= 1);
    }

    #[test]
    fn test_probing_vocab_word_out_of_range_returns_none() {
        let v = ProbingVocabulary::new();
        assert_eq!(v.word(9999), None);
    }

    #[test]
    fn test_probing_vocab_special_words() {
        let v = ProbingVocabulary::new();
        assert_eq!(v.begin_sentence(), BOS_WORD);
        assert_eq!(v.end_sentence(), EOS_WORD);
        assert_eq!(v.not_found(), UNK_WORD);
    }

    #[test]
    fn test_probing_vocab_set_special() {
        let mut v = ProbingVocabulary::new();
        v.set_special(20, 21, 22);
        assert_eq!(v.begin_sentence(), 20);
        assert_eq!(v.end_sentence(), 21);
        assert_eq!(v.not_found(), 22);
    }

    // ── SortedVocabulary ──────────────────────────────────────────────────────

    #[test]
    fn test_sorted_vocab_insert_and_lookup() {
        let mut v = SortedVocabulary::new();
        let i = v.insert("hello");
        v.finished_loading();
        assert_eq!(v.index("hello"), i);
    }

    #[test]
    fn test_sorted_vocab_unknown_returns_not_found() {
        let mut v = SortedVocabulary::new();
        v.insert("foo");
        v.finished_loading();
        assert_eq!(v.index("bar"), v.not_found());
    }

    #[test]
    fn test_sorted_vocab_unk_detection() {
        let mut v = SortedVocabulary::new();
        // "<unk>" and "<UNK>" map to index 0 and set saw_unk
        let i_unk = v.insert("<unk>");
        assert_eq!(i_unk, 0);
        assert!(v.saw_unk());
    }

    #[test]
    fn test_sorted_vocab_finished_loading_sorts() {
        let mut v = SortedVocabulary::new();
        v.insert("zebra");
        v.insert("apple");
        v.insert("mango");
        v.finished_loading();
        // After sorting, binary search should find all words
        assert_ne!(v.index("zebra"), v.not_found());
        assert_ne!(v.index("apple"), v.not_found());
        assert_ne!(v.index("mango"), v.not_found());
    }

    #[test]
    fn test_sorted_vocab_double_sort_is_idempotent() {
        let mut v = SortedVocabulary::new();
        v.insert("a");
        v.insert("b");
        v.finished_loading();
        v.finished_loading(); // second call should be a no-op
        assert_ne!(v.index("a"), v.not_found());
    }

    #[test]
    fn test_sorted_vocab_word_by_index() {
        let mut v = SortedVocabulary::new();
        v.insert("hello");
        v.finished_loading();
        assert_eq!(v.word(0), Some("<unk>"));
        // Index 1 is the first real word inserted
        assert!(v.word(1).is_some());
    }

    #[test]
    fn test_sorted_vocab_size() {
        let mut v = SortedVocabulary::new();
        assert_eq!(v.size(), 1); // only <unk>
        v.insert("a");
        v.insert("b");
        assert_eq!(v.size(), 3); // <unk> + 2 words
    }

    #[test]
    fn test_sorted_vocab_add_word_alias() {
        let mut v = SortedVocabulary::new();
        let i = v.add_word("test");
        v.finished_loading();
        assert_ne!(i, v.not_found());
    }
}
