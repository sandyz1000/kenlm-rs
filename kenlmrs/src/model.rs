use crate::bhiksha::{ArrayBhiksha, DontBhiksha};
use crate::error::LMError;
use crate::quantize::{DontQuantize, SeparatelyQuantize};
use crate::search::{BackoffValue, HashedSearch, Pointer, RestValue, Search, TrieSearch};
use crate::types::{Config, FullScoreReturn, ModelType, State, WordIndex};
use crate::vocabulary::{ProbingVocabulary, SortedVocabulary, Vocabulary};
use std::marker::PhantomData;

/// Binary format handler for reading/writing model files
#[derive(Debug)]
pub struct BinaryFormat {
    // Placeholder implementation
}

impl BinaryFormat {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for BinaryFormat {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for vocabularies that can report their size
pub trait VocabularySize {
    fn vocab_size(&self) -> usize;
}

impl VocabularySize for ProbingVocabulary {
    fn vocab_size(&self) -> usize {
        // TODO: Implement actual size
        0
    }
}

impl VocabularySize for SortedVocabulary {
    fn vocab_size(&self) -> usize {
        // TODO: Implement actual size
        0
    }
}

/// Trait for vocabularies that can be created
pub trait VocabularyNew {
    fn new() -> Self;
}

impl VocabularyNew for ProbingVocabulary {
    fn new() -> Self {
        ProbingVocabulary::new()
    }
}

impl VocabularyNew for SortedVocabulary {
    fn new() -> Self {
        SortedVocabulary::new()
    }
}

/// Generic model implementation that works with different search and vocabulary types
pub struct GenericModel<S, V>
where
    S: Search,
    V: Vocabulary + VocabularyNew + VocabularySize,
{
    backing: BinaryFormat,
    vocab: V,
    search: S,
    _phantom: PhantomData<(S, V)>,
}

impl<S, V> GenericModel<S, V>
where
    S: Search,
    S::Node: Default,
    S::UnigramPointer: Pointer,
    S::MiddlePointer: Pointer,
    S::LongestPointer: Pointer,
    V: Vocabulary + VocabularyNew + VocabularySize,
{
    /// Create a new generic model
    pub fn new(_file_name: &str, _config: &Config) -> Result<Self, LMError> {
        // TODO: Implement actual loading
        Ok(Self {
            backing: BinaryFormat::new(),
            vocab: V::new(),
            search: S::default(), // Use Default trait instead of new
            _phantom: PhantomData,
        })
    }

    /// Get the order of this model
    pub fn order(&self) -> u8 {
        self.search.order()
    }

    /// Get reference to vocabulary
    pub fn vocab(&self) -> &V {
        &self.vocab
    }

    /// Score a full sequence and return the result
    /// This is the main API for scoring p(word | context)
    pub fn full_score(
        &self,
        in_state: &State,
        new_word: WordIndex,
        out_state: &mut State,
    ) -> FullScoreReturn {
        // Score except for backoff weights
        let mut ret = self.score_except_backoff(
            &in_state.words[..in_state.length as usize],
            new_word,
            out_state,
        );

        // Add backoff weights from the input state
        // Start from where the n-gram matched
        for i in (ret.ngram_length - 1) as usize..in_state.length as usize {
            ret.prob += in_state.backoff[i];
        }

        ret
    }

    /// Score without considering backoff weights from context
    /// This is the core scoring function
    fn score_except_backoff(
        &self,
        context_rbegin: &[WordIndex],
        new_word: WordIndex,
        out_state: &mut State,
    ) -> FullScoreReturn {
        let mut ret = FullScoreReturn {
            prob: 0.0,
            ngram_length: 1,
            independent_left: false,
            extend_left: 0,
            rest: 0.0,
        };

        // Lookup the unigram for the new word
        let mut node = S::Node::default();
        let uni = self.search.lookup_unigram(
            new_word,
            &mut node,
            &mut ret.independent_left,
            &mut ret.extend_left,
        );

        out_state.backoff[0] = uni.backoff();
        ret.prob = uni.prob();
        ret.rest = uni.rest();

        // This is the length of context that should be used for continuation to the right
        out_state.length = if has_extension(out_state.backoff[0]) {
            1
        } else {
            0
        };

        // Write the word anyway since it will probably be used
        out_state.words[0] = new_word;

        if context_rbegin.is_empty() {
            return ret;
        }

        // Continue scoring with context
        self.resume_score(
            context_rbegin,
            0,
            &mut node,
            &mut out_state.backoff[1..],
            &mut out_state.length,
            &mut ret,
        );

        // Copy remaining history
        self.copy_remaining_history(context_rbegin, out_state);

        ret
    }

    /// Resume scoring with additional context
    fn resume_score(
        &self,
        context: &[WordIndex],
        mut order_minus_2: u8,
        node: &mut S::Node,
        backoff_out: &mut [f32],
        next_use: &mut u8,
        ret: &mut FullScoreReturn,
    ) {
        let max_order = self.order();

        for (i, &word) in context.iter().enumerate() {
            if ret.independent_left {
                return;
            }

            if order_minus_2 == max_order - 2 {
                // We've reached the longest n-gram
                break;
            }

            let pointer = self.search.lookup_middle(
                order_minus_2,
                word,
                node,
                &mut ret.independent_left,
                &mut ret.extend_left,
            );

            if !pointer.found() {
                return;
            }

            if i < backoff_out.len() {
                backoff_out[i] = pointer.backoff();
            }

            ret.prob = pointer.prob();
            ret.rest = pointer.rest();
            ret.ngram_length = order_minus_2 + 2;

            if has_extension(backoff_out[i]) {
                *next_use = ret.ngram_length;
            }

            order_minus_2 += 1;
        }

        // Check longest n-gram if we have more context
        if order_minus_2 == max_order - 2 && order_minus_2 < context.len() as u8 {
            ret.independent_left = true;
            let longest = self
                .search
                .lookup_longest(context[order_minus_2 as usize], node);

            if longest.found() {
                ret.prob = longest.prob();
                ret.rest = ret.prob;
                ret.ngram_length = max_order;
            }
        }
    }

    /// Copy remaining history words to output state
    fn copy_remaining_history(&self, context: &[WordIndex], out_state: &mut State) {
        let copy_len = (out_state.length as usize).saturating_sub(1);
        if copy_len > 0 && copy_len <= context.len() {
            out_state.words[1..=copy_len].copy_from_slice(&context[..copy_len]);
        }
    }
}

/// Check if a backoff value indicates there's an extension to higher order
fn has_extension(backoff: f32) -> bool {
    // In KenLM, a backoff of 0.0 or very close to 0.0 means no extension
    // Non-zero backoff means there are higher-order n-grams
    backoff.abs() > 1e-8
}

/// Common model trait
pub trait Model: Sized {
    fn full_score(&self, context: &State, word: WordIndex, state: &mut State) -> FullScoreReturn;
    fn base_score(&self, context: &State, word: WordIndex, state: &mut State) -> f32;
    fn short_score(&self, context: &State, word: WordIndex, state: &mut State) -> f32;
    fn new(file_name: &str, config: &Config) -> Result<Self, LMError>;
}

/// Macro for defining concrete model types
macro_rules! define_model {
    ($name:ident, $search_type:ty, $vocab_type:ty) => {
        pub struct $name {
            inner: GenericModel<$search_type, $vocab_type>,
        }

        impl $name {
            pub fn new(file_name: &str, config: &Config) -> Result<Self, LMError> {
                Ok(Self {
                    inner: GenericModel::new(file_name, config)?,
                })
            }
        }

        impl Model for $name {
            fn full_score(
                &self,
                context: &State,
                word: WordIndex,
                state: &mut State,
            ) -> FullScoreReturn {
                self.inner.full_score(context, word, state)
            }

            fn base_score(&self, _context: &State, _word: WordIndex, _state: &mut State) -> f32 {
                // TODO: Implement base scoring
                0.0
            }

            fn short_score(&self, _context: &State, _word: WordIndex, _state: &mut State) -> f32 {
                // TODO: Implement short scoring
                0.0
            }

            fn new(file_name: &str, config: &Config) -> Result<Self, LMError> {
                Self::new(file_name, config)
            }
        }
    };
}

// Define concrete model types
define_model!(ProbingModel, HashedSearch<BackoffValue>, ProbingVocabulary);
define_model!(RestProbingModel, HashedSearch<RestValue>, ProbingVocabulary);
define_model!(TrieModel, TrieSearch<DontQuantize, DontBhiksha>, SortedVocabulary);
define_model!(ArrayTrieModel, TrieSearch<DontQuantize, ArrayBhiksha>, SortedVocabulary);
define_model!(QuantTrieModel, TrieSearch<SeparatelyQuantize, DontBhiksha>, SortedVocabulary);
define_model!(QuantArrayTrieModel, TrieSearch<SeparatelyQuantize, ArrayBhiksha>, SortedVocabulary);

// Type aliases for convenience
pub type DefaultVocabulary = ProbingVocabulary;
pub type DefaultModel = ProbingModel;

/// Load a model with automatic type detection
/// Note: Model trait is not dyn compatible due to Sized requirement, so we return concrete types
pub fn load_virtual(
    file_name: &str,
    config: &Config,
    if_arpa: ModelType,
) -> Result<DefaultModel, LMError> {
    // TODO: Implement binary format recognition
    // For now, just return the default model type
    match if_arpa {
        ModelType::Probing => ProbingModel::new(file_name, config),
        ModelType::RestProbing => {
            // Convert to default for now since we can't return different types
            ProbingModel::new(file_name, config)
        }
        ModelType::Trie => {
            // Convert to default for now since we can't return different types
            ProbingModel::new(file_name, config)
        }
        ModelType::ArrayTrie => {
            // Convert to default for now since we can't return different types
            ProbingModel::new(file_name, config)
        }
        ModelType::QuantTrie => {
            // Convert to default for now since we can't return different types
            ProbingModel::new(file_name, config)
        }
        ModelType::QuantArrayTrie => {
            // Convert to default for now since we can't return different types
            ProbingModel::new(file_name, config)
        }
    }
}
