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
        self.bound() as usize
    }
}

impl VocabularySize for SortedVocabulary {
    fn vocab_size(&self) -> usize {
        self.size() as usize
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
    /// Load a model from an ARPA file.
    pub fn new(file_name: &str, config: &Config) -> Result<Self, LMError> {
        use crate::arpa_reader::read_arpa_counts;
        use crate::utils::pieces::file::FilePiece;

        // First pass: read the n-gram counts from the \data\ header.
        let counts = {
            let mut fp = FilePiece::open(file_name)?;
            read_arpa_counts(&mut fp)?
        };

        let mut vocab = V::new();
        let mut search = S::default();

        // Second pass: populate vocabulary and hash tables.
        // initialize_from_arpa re-opens the file, skips the counts section,
        // then loads unigrams (inserting words into vocab) and all n-gram orders.
        search.initialize_from_arpa(file_name, &counts, config, &mut vocab)?;

        Ok(Self {
            backing: BinaryFormat::new(),
            vocab,
            search,
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

    /// Returns a zero-length state representing no context (start of a sentence or null context).
    pub fn empty_context_state() -> State {
        State::new()
    }

    /// Returns a one-word state seeded with the begin-of-sentence marker `<s>`.
    pub fn begin_sentence_state(&self) -> State {
        let bos = self.vocab.begin_sentence();
        let mut state = State::new();
        state.words[0] = bos;
        state.length = 1;

        let mut node = S::Node::default();
        let mut independent_left = false;
        let mut extend_left = 0u64;
        let uni = self.search.lookup_unigram(bos, &mut node, &mut independent_left, &mut extend_left);
        state.backoff[0] = uni.backoff();
        state
    }

    /// Reconstructs a [`State`] from an explicit context word slice.
    ///
    /// `context` must be in **reverse order** (most-recent word first), matching
    /// the layout of [`State::words`]. Words beyond the model order are ignored.
    pub fn get_state(&self, context: &[WordIndex]) -> State {
        let max_ctx = (self.order() as usize).saturating_sub(1);
        let context = &context[..context.len().min(max_ctx)];

        let mut state = State::new();
        let Some(&first) = context.first() else {
            return state;
        };

        let mut node = S::Node::default();
        let mut independent_left = false;
        let mut extend_left = 0u64;
        let uni = self.search.lookup_unigram(first, &mut node, &mut independent_left, &mut extend_left);
        state.backoff[0] = uni.backoff();
        if has_extension(state.backoff[0]) {
            state.length = 1;
        }

        for (i, &word) in context[1..].iter().enumerate() {
            let ptr = self.search.lookup_middle(
                i as u8, word, &mut node, &mut independent_left, &mut extend_left,
            );
            if !ptr.found() {
                break;
            }
            state.backoff[i + 1] = ptr.backoff();
            if has_extension(state.backoff[i + 1]) {
                state.length = (i + 2) as u8;
            }
        }

        state.words[..state.length as usize]
            .copy_from_slice(&context[..state.length as usize]);
        state
    }

    /// Scores `new_word` given an explicit context slice, without a pre-built [`State`].
    ///
    /// This is the stateless API: it calls [`score_except_backoff`] for the base probability
    /// and then adds the applicable backoff weights from any further context n-grams found.
    ///
    /// `context` is in **reverse order** (most-recent word first).
    pub fn full_score_forgot_state(
        &self,
        context: &[WordIndex],
        new_word: WordIndex,
        out_state: &mut State,
    ) -> FullScoreReturn {
        let max_ctx = (self.order() as usize).saturating_sub(1);
        let context = &context[..context.len().min(max_ctx)];

        let mut ret = self.score_except_backoff(context, new_word, out_state);

        let start = ret.ngram_length as usize;
        if context.len() < start {
            return ret;
        }

        let mut node = S::Node::default();
        let mut independent_left = false;
        let mut extend_left = 0u64;

        if start <= 1 {
            let uni = self.search.lookup_unigram(
                context[0], &mut node, &mut independent_left, &mut extend_left,
            );
            ret.prob += uni.backoff();
        } else if !self.search.fast_make_node(&context[..start - 1], &mut node) {
            return ret;
        }

        for (i, &word) in context[start - 1..].iter().enumerate() {
            let order = (start - 1 + i) as u8;
            if order >= self.order() - 2 {
                break;
            }
            let ptr = self.search.lookup_middle(
                order, word, &mut node, &mut independent_left, &mut extend_left,
            );
            if !ptr.found() {
                break;
            }
            ret.prob += ptr.backoff();
        }
        ret
    }
}

/// Returns true when the backoff value indicates a higher-order n-gram exists for this context.
///
/// C++ KenLM encodes "no extension" as -0.0 (bit pattern 0x80000000). Everything else —
/// including +0.0 and any nonzero float — means "has extension".
pub(crate) fn has_extension(backoff: f32) -> bool {
    backoff.to_bits() != 0x80000000
}

/// Marks a backoff slot as having a higher-order child by converting -0.0 → +0.0.
pub(crate) fn set_extension(backoff: &mut f32) {
    if backoff.to_bits() == 0x80000000 {
        *backoff = 0.0_f32;
    }
}

/// Sets the IEEE 754 sign bit, encoding "no higher-order match extends from here".
pub(crate) fn mark_independent(prob: f32) -> f32 {
    f32::from_bits(prob.to_bits() | 0x80000000)
}

/// Clears the sign bit, encoding "at least one higher-order n-gram uses this as context".
pub(crate) fn mark_extends_left(prob: f32) -> f32 {
    f32::from_bits(prob.to_bits() & !0x80000000u32)
}

/// Checks the stored sign bit to determine whether context search should stop here.
pub(crate) fn is_independent_left(prob: f32) -> bool {
    prob.to_bits() & 0x80000000 != 0
}

/// Returns the probability for scoring: forces sign bit ON so the value stays negative.
pub(crate) fn scoring_prob(stored_prob: f32) -> f32 {
    f32::from_bits(stored_prob.to_bits() | 0x80000000)
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

            pub fn empty_context_state() -> State {
                GenericModel::<$search_type, $vocab_type>::empty_context_state()
            }

            pub fn begin_sentence_state(&self) -> State {
                self.inner.begin_sentence_state()
            }

            pub fn get_state(&self, context: &[WordIndex]) -> State {
                self.inner.get_state(context)
            }

            pub fn full_score_forgot_state(
                &self,
                context: &[WordIndex],
                new_word: WordIndex,
                out_state: &mut State,
            ) -> FullScoreReturn {
                self.inner.full_score_forgot_state(context, new_word, out_state)
            }

            /// Score `word` given `in_state` context, writing the new context to `out_state`.
            pub fn full_score(
                &self,
                in_state: &State,
                word: WordIndex,
                out_state: &mut State,
            ) -> FullScoreReturn {
                self.inner.full_score(in_state, word, out_state)
            }

            /// N-gram order of this model.
            pub fn order(&self) -> u8 { self.inner.order() }

            /// Access the vocabulary.
            pub fn vocab(&self) -> &$vocab_type { self.inner.vocab() }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::State;

    #[test]
    fn test_has_extension_zero() {
        // +0.0 means "has extension" (context appears with higher-order match)
        assert!(has_extension(0.0_f32));
        // -0.0 means "no extension" (KenLM kNoExtensionBackoff)
        assert!(!has_extension(-0.0_f32));
    }

    #[test]
    fn test_has_extension_near_zero() {
        // Small nonzero values also have extension (only -0.0 is the sentinel)
        assert!(has_extension(1e-9_f32));
        assert!(has_extension(-1e-9_f32));
    }

    #[test]
    fn test_has_extension_nonzero() {
        assert!(has_extension(1.0_f32));
        assert!(has_extension(-0.5_f32));
        assert!(has_extension(0.1_f32));
    }

    #[test]
    fn test_binary_format_default() {
        let bf = BinaryFormat::default();
        let bf2 = BinaryFormat::new();
        // Both are empty structs — test they can be constructed
        drop(bf);
        drop(bf2);
    }

    fn make_tiny_arpa() -> (tempfile::NamedTempFile, ProbingModel) {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "\
\\data\\
ngram 1=3
ngram 2=1

\\1-grams:
-99\t<unk>
-0.5\t<s>\t-0.3
-0.8\t</s>

\\2-grams:
-0.4\t<s> </s>

\\end\\
").unwrap();
        f.flush().unwrap();
        let path = f.path().to_str().unwrap().to_string();
        let model = ProbingModel::new(&path, &Config::new()).unwrap();
        (f, model)
    }

    #[test]
    fn test_probing_model_new_fails_for_nonexistent_file() {
        let config = Config::new();
        let model = ProbingModel::new("definitely_does_not_exist.arpa", &config);
        assert!(model.is_err(), "nonexistent file must return Err");
    }

    #[test]
    fn test_generic_model_full_score_empty_context() {
        let (_f, model) = make_tiny_arpa();
        let in_state = State::default();
        let mut out_state = State::default();
        let _result = model.full_score(&in_state, 0, &mut out_state);
    }

    #[test]
    fn test_generic_model_order() {
        let (_f, model) = make_tiny_arpa();
        // 2-gram model → order 2
        assert_eq!(model.order(), 2);
    }

    // ── has_extension / set_extension bit semantics ───────────────────────────

    #[test]
    fn has_extension_should_distinguish_positive_zero_from_negative_zero() {
        assert!(has_extension(0.0_f32));           // +0.0 → has extension
        assert!(!has_extension(-0.0_f32));          // -0.0 → no extension
    }

    #[test]
    fn has_extension_should_return_true_for_nonzero_backoff() {
        assert!(has_extension(-0.3_f32));
        assert!(has_extension(0.5_f32));
    }

    #[test]
    fn set_extension_should_convert_negative_zero_to_positive_zero() {
        let mut b = -0.0_f32;
        set_extension(&mut b);
        assert_eq!(b.to_bits(), 0x00000000); // +0.0
    }

    #[test]
    fn set_extension_should_leave_nonzero_backoff_unchanged() {
        let mut b = -0.3_f32;
        let original = b.to_bits();
        set_extension(&mut b);
        assert_eq!(b.to_bits(), original);
    }

    // ── sign-bit encoding helpers ─────────────────────────────────────────────

    #[test]
    fn mark_independent_should_set_sign_bit() {
        let p = mark_independent(-1.5_f32);
        assert!(p.to_bits() & 0x80000000 != 0);
        assert!(is_independent_left(p));
    }

    #[test]
    fn mark_extends_left_should_clear_sign_bit() {
        let p = mark_extends_left(-1.5_f32);
        assert!(p.to_bits() & 0x80000000 == 0);
        assert!(!is_independent_left(p));
    }

    #[test]
    fn scoring_prob_should_always_return_negative_float() {
        // Even if stored as positive (sign bit cleared), scoring forces sign bit back ON
        let stored = mark_extends_left(-1.5_f32); // bit-flipped → positive
        assert!(stored > 0.0);
        let scored = scoring_prob(stored);
        assert!(scored < 0.0);
        assert!((scored - (-1.5_f32)).abs() < 1e-6);
    }

    // ── empty_context_state / begin_sentence_state ─────────────────────────────

    #[test]
    fn empty_context_state_should_have_zero_length() {
        let s = ProbingModel::empty_context_state();
        assert_eq!(s.length, 0);
    }

    #[test]
    fn begin_sentence_state_should_have_length_one() {
        let (_f, model) = make_tiny_arpa();
        let s = model.begin_sentence_state();
        assert_eq!(s.length, 1);
    }

    // ── get_state ─────────────────────────────────────────────────────────────

    #[test]
    fn get_state_should_return_empty_state_for_empty_context() {
        let (_f, model) = make_tiny_arpa();
        let s = model.get_state(&[]);
        assert_eq!(s.length, 0);
    }

    #[test]
    fn get_state_should_clamp_context_to_model_order() {
        let (_f, model) = make_tiny_arpa();
        let bos = model.vocab().begin_sentence();
        // order=2 → max context = 1; passing 5 words is clamped to 1
        let s = model.get_state(&[bos, bos, bos, bos, bos]);
        assert!(s.length <= 1);
    }

    // ── full_score_forgot_state ────────────────────────────────────────────────

    #[test]
    fn full_score_forgot_state_should_not_panic_with_empty_context() {
        let (_f, model) = make_tiny_arpa();
        let mut out = State::default();
        let _ret = model.full_score_forgot_state(&[], 0, &mut out);
    }
}

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
