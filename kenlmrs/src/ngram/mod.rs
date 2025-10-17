pub mod binary_format;
pub mod query;
pub mod search;
use crate::ngram::query::QueryPrinter;
use crate::ngram::search::LongestPointer;
use crate::types::{Config, ModelType, State, WordIndex};
use crate::utils::pieces::string::StringPiece;
use crate::vocabulary::{ProbingVocabulary, SortedVocabulary};
use std::marker::PhantomData;

type Node = NodeRange;

#[derive(Debug, Clone, Copy)]
pub struct LoadMethod;

#[derive(Debug)]
pub(crate) struct NodeRange {
    begin: u64,
    end: u64,
}

#[derive(Debug)]
pub struct UnigramPointer;

#[derive(Debug)]
pub struct Unigram;

#[derive(Debug)]
pub struct BitPacked;

#[derive(Debug)]
pub struct BitPackedMiddle<Bhiksha> {
    _phantom: PhantomData<Bhiksha>,
}

#[derive(Debug)]
pub struct BitPackedLongest;

#[derive(Debug, Clone, Copy)]
pub struct GenericModel<Search, VocabularyT> {
    // This is the model type returned by RecognizeBinary.
    k_model_type: ModelType,
    k_version: i64,
    _phantom: PhantomData<(Search, VocabularyT)>,
}

#[derive(Debug)]
pub struct RestValue;

#[derive(Debug)]
pub struct BackoffValue;

#[derive(Debug)]
pub struct HashedSearch<Value> {
    _phantom: PhantomData<Value>,
}

#[derive(Debug, Clone, Copy)]
pub struct TrieSearch<Quant, Bhiksha> {
    _phantom: PhantomData<(Quant, Bhiksha)>,
}

/// Trait for value types in search implementations
pub trait Value {
    fn new() -> Self;
}

impl Value for RestValue {
    fn new() -> Self {
        RestValue
    }
}

impl Value for BackoffValue {
    fn new() -> Self {
        BackoffValue
    }
}

impl State {
    pub fn Compare(&self, _other: &State) {
        // TODO: Implement state comparison
        todo!()
    }
}

pub trait VocabularyT {
    fn new() -> Self;
    fn index(&self, inp_str: &StringPiece) -> WordIndex;
}

impl VocabularyT for SortedVocabulary {
    fn new() -> Self {
        SortedVocabulary::new()
    }

    fn index(&self, inp_str: &StringPiece) -> WordIndex {
        use crate::vocabulary::Vocabulary;
        Vocabulary::index(self, inp_str.as_str())
    }
}

impl VocabularyT for ProbingVocabulary {
    fn new() -> Self {
        ProbingVocabulary::new()
    }

    fn index(&self, inp_str: &StringPiece) -> WordIndex {
        use crate::vocabulary::Vocabulary;
        Vocabulary::index(self, inp_str.as_str())
    }
}

pub fn Query<Model, Printer>(
    _file: &str,
    _config: &Config,
    _sentence_context: bool,
    _printer: &QueryPrinter,
) {
    // TODO: Implement query
}

pub fn RecognizeBinary(_file: &str, _recognized: &ModelType) -> bool {
    // TODO: Implement binary format recognition
    false
}

pub fn LookupLongest(_word: WordIndex, _node: &Node) -> LongestPointer {
    // TODO: Implement longest lookup
    LongestPointer
}

pub fn FastMakeNode(_begin: WordIndex, _end: WordIndex, _node: &Node) -> bool {
    // TODO: Implement fast node creation
    false
}

pub fn ShowSizes(_file: &str, _config: &Config) {
    // TODO: Implement size display
}
