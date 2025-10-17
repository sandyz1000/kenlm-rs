use crate::constant::*;
use crate::error::LMError;
use crate::types::{Config, WordIndex};
use std::collections::HashMap;
use std::hash::BuildHasher;

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
    saw_unk: bool,
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
        self.saw_unk
    }
}

impl BaseVocabulary {
    pub fn new() -> Self {
        Self {
            begin_sentence: BOS_WORD,
            end_sentence: EOS_WORD,
            not_found: UNK_WORD,
            bound: 3, // Start with 3 special words
            saw_unk: false,
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

    pub fn set_saw_unk(&mut self, saw_unk: bool) {
        self.saw_unk = saw_unk;
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
pub struct SortedVocabulary {
    base: BaseVocabulary,
    words: Vec<String>,
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
        if !self.sorted {
            // Linear search if not sorted yet
            for (i, word) in self.words.iter().enumerate() {
                if word == str {
                    return i as WordIndex;
                }
            }
        } else {
            // Binary search if sorted
            match self.words.binary_search(&str.to_string()) {
                Ok(index) => return index as WordIndex,
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
            words: Vec::new(),
            sorted: false,
        }
    }

    /// Add a word to the vocabulary
    pub fn add_word(&mut self, word: &str) {
        self.words.push(word.to_string());
        self.sorted = false;
        self.base.set_bound(self.words.len() as WordIndex);
    }

    /// Sort the vocabulary for efficient binary search
    pub fn sort(&mut self) {
        self.words.sort();
        self.sorted = true;
    }

    /// Get word by index
    pub fn word(&self, index: WordIndex) -> Option<&str> {
        self.words.get(index as usize).map(|s| s.as_str())
    }
}

impl Default for SortedVocabulary {
    fn default() -> Self {
        Self::new()
    }
}
