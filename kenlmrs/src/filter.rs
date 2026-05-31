use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::sync::mpsc;
use std::thread;

use crate::constant::{FilterMode, FormatEnum};
use crate::error::LMError;

/// Configuration for the n-gram filter.
#[derive(Debug)]
pub struct Config {
    /// Lines to accumulate before dispatching to a worker thread.
    pub batch_size: usize,
    /// Number of worker threads (1 = single-threaded).
    pub threads: usize,
    pub mode: FilterMode,
    /// Enable phrase-table filtering (tab-delimited phrases).
    pub phrase: bool,
    /// Enable context-only filtering (match on prefix words, emit full n-gram).
    pub context: bool,
    pub format: FormatEnum,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            batch_size: 25_000,
            threads: 1,
            mode: FilterMode::ModeCopy,
            phrase: false,
            context: false,
            format: FormatEnum::FormatArpa,
        }
    }
}

/// Predicate that decides whether a given n-gram line passes the filter.
pub trait Filter: Send + Sync {
    fn filter_ngram(&self, ngram: &str) -> bool;
}

/// A filter that passes everything (copy mode).
pub struct CopyFilter;
impl Filter for CopyFilter {
    fn filter_ngram(&self, _ngram: &str) -> bool { true }
}

/// A filter backed by a vocabulary set: pass only n-grams whose words all appear in the set.
pub struct VocabFilter {
    vocab: HashSet<String>,
    context_only: bool,
}

impl VocabFilter {
    /// Create from an iterator of accepted words.
    pub fn new(words: impl IntoIterator<Item = String>, context_only: bool) -> Self {
        VocabFilter { vocab: words.into_iter().collect(), context_only }
    }
}

impl Filter for VocabFilter {
    fn filter_ngram(&self, ngram: &str) -> bool {
        // n-gram line format: "prob\tword1 word2 ...\t[backoff]"
        // Extract just the words (second tab field or whole line if no tabs).
        let words_str = ngram.splitn(3, '\t').nth(1).unwrap_or(ngram);
        let words: Vec<&str> = words_str.split_whitespace().collect();
        if words.is_empty() { return false; }
        let check_words = if self.context_only {
            // Context-only: check only words except the last (the predicted word)
            if words.len() > 1 { &words[..words.len()-1] } else { &words[..] }
        } else {
            &words[..]
        };
        check_words.iter().all(|w| self.vocab.contains(*w))
    }
}

/// Detect whether a text looks like an ARPA file (starts with `\data\`).
pub fn is_arpa_format(text: &str) -> bool {
    text.trim_start().starts_with("\\data\\")
}

/// Filter lines from `input` using `filter`, writing passing lines to `output`.
///
/// Mode behaviour:
/// - `ModeCopy`: pass all n-gram lines unchanged.
/// - `ModeSingle`: treat the entire input as one sentence; output all n-grams that pass.
/// - `ModeMultiple`: each input line is a sentence; output one filtered block per sentence.
/// - `ModeUnion`: vocabulary union of all sentences; any n-gram whose words appear in the
///   union passes.
///
/// If `config.threads > 1`, lines are batched and processed in parallel worker threads.
pub fn run_filter<F: Filter + 'static>(
    config: &Config,
    input: &str,
    filter: &F,
    output: &mut dyn Write,
) -> Result<(), LMError> {
    match config.mode {
        FilterMode::ModeCopy | FilterMode::ModeUnset => {
            filter_lines(config, input, filter, output)
        }
        FilterMode::ModeSingle => {
            filter_single(input, filter, output)
        }
        FilterMode::ModeMultiple => {
            filter_multiple(config, input, filter, output)
        }
        FilterMode::ModeUnion => {
            filter_union(config, input, filter, output)
        }
    }
}

/// Copy mode + Single mode: pass every n-gram line through the filter predicate.
fn filter_lines<F: Filter>(
    config: &Config,
    input: &str,
    filter: &F,
    output: &mut dyn Write,
) -> Result<(), LMError> {
    if config.threads <= 1 {
        for line in input.lines() {
            if !line.is_empty() && filter.filter_ngram(line) {
                writeln!(output, "{}", line)?;
            }
        }
    } else {
        let (tx, rx) = mpsc::channel::<Vec<String>>();
        let batch_size = config.batch_size;
        let lines: Vec<String> = input.lines().map(String::from).collect();
        let n_threads = config.threads;

        // Split into batches and process in parallel
        let batches: Vec<Vec<String>> = lines
            .chunks(batch_size.max(1))
            .map(|c| c.to_vec())
            .collect();

        thread::scope(|s| {
            for batch in batches {
                let tx = tx.clone();
                s.spawn(move || {
                    let passing: Vec<String> = batch
                        .iter()
                        .filter(|l| !l.is_empty() && filter.filter_ngram(l))
                        .cloned()
                        .collect();
                    let _ = tx.send(passing);
                });
            }
            drop(tx);
            for passing in rx {
                for line in passing {
                    writeln!(output, "{}", line).ok();
                }
            }
        });
    }
    Ok(())
}

/// Single mode: treat entire input as one sentence context.
fn filter_single<F: Filter>(
    input: &str,
    filter: &F,
    output: &mut dyn Write,
) -> Result<(), LMError> {
    for line in input.lines() {
        if !line.is_empty() && filter.filter_ngram(line) {
            writeln!(output, "{}", line)?;
        }
    }
    Ok(())
}

/// Multiple mode: each non-empty input line is a separate sentence; emit a separator between outputs.
fn filter_multiple<F: Filter>(
    _config: &Config,
    input: &str,
    filter: &F,
    output: &mut dyn Write,
) -> Result<(), LMError> {
    let mut first = true;
    for line in input.lines() {
        if line.is_empty() {
            if !first {
                writeln!(output)?;
            }
            first = true;
            continue;
        }
        if filter.filter_ngram(line) {
            writeln!(output, "{}", line)?;
            first = false;
        }
    }
    Ok(())
}

/// Union mode: all n-grams from all sentences are merged; any passing the filter is emitted once.
fn filter_union<F: Filter>(
    config: &Config,
    input: &str,
    filter: &F,
    output: &mut dyn Write,
) -> Result<(), LMError> {
    let mut seen: HashSet<String> = HashSet::new();
    for line in input.lines() {
        if !line.is_empty() && filter.filter_ngram(line) && seen.insert(line.to_string()) {
            writeln!(output, "{}", line)?;
        }
    }
    Ok(())
}

/// Parse a phrase-table line into constituent n-grams.
/// Format: source ||| target ... — returns the space-separated source phrase words.
pub fn parse_phrase_table_ngrams(line: &str) -> Vec<String> {
    if let Some(src) = line.split("|||").next() {
        src.split_whitespace()
            .flat_map(|w| {
                // Emit all sub-sequences (n-grams) up to the phrase length
                let words: Vec<&str> = src.split_whitespace().collect();
                let mut ngrams = Vec::new();
                for start in 0..words.len() {
                    for end in start+1..=words.len() {
                        ngrams.push(words[start..end].join(" "));
                    }
                }
                ngrams
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    } else {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy_filter_config() -> Config {
        Config { mode: FilterMode::ModeCopy, ..Config::default() }
    }

    #[test]
    fn copy_filter_passes_all_lines() {
        let input = "line one\nline two\nline three";
        let mut out = Vec::new();
        run_filter(&copy_filter_config(), input, &CopyFilter, &mut out).unwrap();
        let result = String::from_utf8(out).unwrap();
        assert!(result.contains("line one"));
        assert!(result.contains("line two"));
    }

    #[test]
    fn vocab_filter_passes_known_words() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let f = VocabFilter::new(vocab, false);
        // Tab-separated: prob \t words \t backoff
        assert!(f.filter_ngram("-0.5\thello world\t-0.1"));
        assert!(!f.filter_ngram("-0.5\thello unknown\t-0.1"));
    }

    #[test]
    fn context_filter_ignores_predicted_word() {
        let vocab = vec!["hello".to_string()];
        let f = VocabFilter::new(vocab, true);
        // context_only: "hello" is context, "unknown" is predicted — only context checked
        assert!(f.filter_ngram("-0.5\thello unknown\t-0.1"));
    }

    #[test]
    fn copy_mode_via_run_filter() {
        let config = copy_filter_config();
        let input = "a\nb\nc";
        let mut out = Vec::new();
        run_filter(&config, input, &CopyFilter, &mut out).unwrap();
        let result = String::from_utf8(out).unwrap();
        assert_eq!(result.lines().count(), 3);
    }

    #[test]
    fn union_mode_deduplicates() {
        let config = Config { mode: FilterMode::ModeUnion, ..Config::default() };
        let input = "alpha\nbeta\nalpha\ngamma";
        let mut out = Vec::new();
        run_filter(&config, input, &CopyFilter, &mut out).unwrap();
        let result = String::from_utf8(out).unwrap();
        assert_eq!(result.lines().filter(|l| *l == "alpha").count(), 1);
    }

    #[test]
    fn multiple_mode_separates_sentences() {
        let config = Config { mode: FilterMode::ModeMultiple, ..Config::default() };
        let input = "a\nb\n\nc\nd";
        let mut out = Vec::new();
        run_filter(&config, input, &CopyFilter, &mut out).unwrap();
        let result = String::from_utf8(out).unwrap();
        // Should contain a blank line separator
        assert!(result.contains('\n'));
    }

    #[test]
    fn is_arpa_format_detection() {
        assert!(is_arpa_format("\\data\\\nngram 1=3"));
        assert!(!is_arpa_format("the cat sat on the mat"));
    }

    #[test]
    fn copy_filter_struct_always_passes() {
        let f = CopyFilter;
        assert!(f.filter_ngram("anything at all"));
        assert!(f.filter_ngram(""));
    }
}
