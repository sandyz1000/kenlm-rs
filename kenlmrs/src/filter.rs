use crate::constant::{FilterMode, FormatEnum};
use std::collections::HashSet;
use std::process::Output;

#[derive(Debug)]
pub enum FilterType {
    Threaded,
    Context,
    Binary,
}

#[derive(Debug, Default)]
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

pub(crate) fn RunThreadedFilter<FilterType, Format, OutputBuffer>(
    config: &Config,
    in_lm: &FilePiece,
    filter_: &Filter,
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

impl Filter for PhraseFilter {}

impl Filter for VocabFilter {}

#[derive(Debug, Clone)]
pub struct Substrings;

#[derive(Debug, Clone)]
pub struct Multiple;

#[derive(Debug, Clone)]
pub struct Union;

impl Union {
    fn new(self, vocabs: &Words) -> Self;
}

impl Multiple {
    fn AddNGram(
        &self,
        begin: &Iterator,
        end: &Iterator,
        ngram: &StringPiece,
        line: &StringPiece,
        output: &Output,
    );
}

pub fn ReadSingle(in_: IStream, out: &HashSet<String>);
pub fn ReadMultiple(in_: IStream, out_: Substrings) -> i64;
