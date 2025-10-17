use crate::constant::{FilterMode, FormatEnum};
use crate::utils::pieces::file::FilePiece;
use crate::utils::pieces::string::StringPiece;
use std::collections::HashSet;
use std::process::Output;

// Placeholder types
pub struct Words;
pub struct IStream;

#[derive(Debug)]
pub enum FilterType {
    Threaded,
    Context,
    Binary,
}

#[derive(Debug)]
pub(crate) struct Config {
    batch_size: i32, // 25000
    threads: i8,
    mode: FilterMode,   // MODE_COPY
    phrase: bool,       // False
    context: bool,      // False
    format: FormatEnum, // FORMAT_ARPA
}

impl Default for Config {
    fn default() -> Self {
        Config {
            batch_size: 25_000,
            threads: 4,
            mode: FilterMode::ModeCopy,
            phrase: false,
            context: false,
            format: FormatEnum::FormatArpa,
        }
    }
}

pub(crate) fn RunThreadedFilter<FilterType, Format, OutputBuffer, F: Filter>(
    config: &Config,
    in_lm: &FilePiece,
    filter_: &F,
    output: Output,
) {
}

#[derive(Debug, Clone)]
pub struct PhraseFilter;

#[derive(Debug, Clone)]
pub struct VocabFilter;

trait Filter {
    fn new() -> Self;
}

impl Filter for PhraseFilter {
    fn new() -> Self {
        PhraseFilter
    }
}

impl Filter for VocabFilter {
    fn new() -> Self {
        VocabFilter
    }
}

#[derive(Debug, Clone)]
pub struct Substrings;

#[derive(Debug, Clone)]
pub struct Multiple;

#[derive(Debug, Clone)]
pub struct Union;

impl Union {
    fn new(self, vocabs: &Words) -> Self {
        // TODO: Implement Union constructor
        todo!()
    }
}

impl Multiple {
    fn AddNGram(
        &self,
        begin: &Iterator,
        end: &Iterator,
        ngram: &StringPiece,
        line: &StringPiece,
        output: &Output,
    ) {
        // TODO: Implement AddNGram
        todo!()
    }
}

pub fn ReadSingle(in_: IStream, out: &HashSet<String>) {
    // TODO: Implement ReadSingle
    todo!()
}

pub fn ReadMultiple(in_: IStream, out_: Substrings) -> i64 {
    // TODO: Implement ReadMultiple
    todo!()
}
