pub mod binary_format;
pub mod query;
pub mod search;

use crate::constant::{ARPALoadComplain, RestFunction, WarningAction, WriteMethod};
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::fs::{File, FileType};
use std::io::prelude::*;
use std::io::{BufRead, BufReader};

type Node = NodeRange;

#[derive(Debug, Clone, Copy)]
pub struct State;

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
pub struct BitPackedMiddle<Bhiksha>;

#[derive(Debug)]
pub struct BitPackedLongest;


#[derive(Debug, Clone, Copy)]
pub struct GenericModel<Search, VocabularyT> {
    // This is the model type returned by RecognizeBinary.
    k_model_type: ModelType,
    k_version: i64,
}

#[derive(Debug)]
pub struct RestValue;

#[derive(Debug)]
pub struct BackoffValue;

#[derive(Debug)]
pub struct HashedSearch<Value>;

#[derive(Debug, Clone, Copy)]
pub struct TrieSearch<Quant, Bhiksha>;


impl Value for RestValue {
    fn new() -> Self {
        todo!()
    }
}

impl Value for BackoffValue {
    fn new() -> Self {
        todo!()
    }
}


impl Config {
    // pub fn ProgressMessages() -> ostream;
}

impl State {
    pub fn Compare(&self, other: &State);
}

trait VocabularyT {
    fn new() -> Self {}

    pub fn Index(&self, inp_str: &StringPiece) -> WordIndex {}
}

impl VocabularyT for SortedVocabulary {
    fn new() -> Self {
        return SortedVocabulary();
    }

    fn Index(&self, inp_str: &StringPiece) -> WordIndex {}
}

impl VocabularyT for ProbingVocabulary {
    fn new() -> Self {
        return ProbingVocabulary();
    }

    fn Index(&self, inp_str: &StringPiece) -> WordIndex {}
}

pub fn Query<Model, Printer>(
    file: &str,
    config: &Config,
    sentence_context: bool,
    printer: &QueryPrinter,
) {
}

pub fn RecognizeBinary(file: &str, recognized: &ModelType) -> bool {}

pub fn LookupLongest(word: WordIndex, node: &Node) -> LongestPointer {}

pub fn FastMakeNode(begin: WordIndex, end: WordIndex, node: &Node) -> bool {}

pub fn ShowSizes(file: &str, config: &Config) {}
