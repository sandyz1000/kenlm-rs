/// ARPA file format handling
///
/// This module provides a compatibility layer and re-exports the main
/// ARPA reading functionality from the arpa_reader module.
// Re-export everything from arpa_reader for backward compatibility
pub use crate::arpa_reader::{
    read_1gram, read_1grams, read_arpa_counts, read_end, read_ngram, read_ngram_header,
    PositiveProbWarn, ARPA_SPACES,
};
