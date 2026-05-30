use crate::constant::{FilterMode, FormatEnum};
use crate::utils::pieces::file::FilePiece;
use std::process::Output;

#[derive(Debug)]
pub(crate) struct Config {
    batch_size: i32,
    threads: i8,
    mode: FilterMode,
    phrase: bool,
    context: bool,
    format: FormatEnum,
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

pub(crate) trait Filter {
    fn filter_ngram(&self, ngram: &str) -> bool;
}

pub(crate) fn run_threaded_filter<F: Filter>(
    _config: &Config,
    _in_lm: &FilePiece,
    _filter: &F,
    _output: Output,
) {
    // TODO: Implement threaded filtering
}
