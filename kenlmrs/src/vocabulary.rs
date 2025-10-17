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
            match self.hashes.binary_search(&hash) {
                Ok(index) => {
                    // Add 1 because index 0 is reserved for <unk>
                    return (index + 1) as WordIndex;
                }
                Err(_) => {}
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

        // Create indices for sorting
        let mut indices: Vec<usize> = (0..self.hashes.len()).collect();
        
        // Sort indices by hash values
        indices.sort_by_key(|&i| self.hashes[i]);
        
        // Reorder both hashes and strings
        let old_hashes = self.hashes.clone();
        let old_strings = self.strings.clone();
        
        for (new_pos, &old_pos) in indices.iter().enumerate() {
            self.hashes[new_pos] = old_hashes[old_pos];
            self.strings[new_pos] = old_strings[old_pos].clone();
        }
        
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
