/// ARPA file reading functionality
///
/// This module provides functions for reading ARPA format language model files,
/// matching the C++ KenLM implementation.
use crate::constant::{WarningAction, K_NO_EXTENSION_BACKOFF};
use crate::error::LMError;
use crate::types::{ProbBackoff, WordIndex};
use crate::utils::pieces::file::FilePiece;
use crate::vocabulary::Vocabulary;

/// ARPA delimiter table: true for \t, \n, \r, and space
/// Matches kARPASpaces from C++ KenLM
pub const ARPA_SPACES: [bool; 256] = {
    let mut arr = [false; 256];
    arr[b'\t' as usize] = true;
    arr[b'\n' as usize] = true;
    arr[b'\r' as usize] = true;
    arr[b' ' as usize] = true;
    arr
};

/// Read a single 1-gram (unigram) from the ARPA file
/// Format: prob \t word [\t backoff] \n
pub fn read_1gram<V: Vocabulary>(
    f: &mut FilePiece,
    vocab: &mut V,
    unigrams: &mut [ProbBackoff],
    warn: &PositiveProbWarn,
) -> Result<(), LMError> {
    // Read probability
    let mut prob = f.read_float()?;
    if prob > 0.0 {
        warn.warn(prob);
        prob = 0.0;
    }

    // Expect tab after probability
    let c = f.get()?;
    if c != '\t' {
        return Err(LMError::InvalidArpa(format!(
            "Expected tab after probability, got '{}'",
            c
        )));
    }

    // Read word
    let word_str = f.read_delimited(&ARPA_SPACES)?;
    let word = vocab.index(&word_str);

    // Store probability
    if (word as usize) < unigrams.len() {
        unigrams[word as usize].prob = prob;

        // Read backoff if present
        read_backoff_probbackoff(f, &mut unigrams[word as usize])?;
    } else {
        return Err(LMError::InvalidArpa(format!(
            "Word index {} out of bounds for unigram array of size {}",
            word,
            unigrams.len()
        )));
    }

    Ok(())
}

/// Read all 1-grams from the ARPA file
pub fn read_1grams<V: Vocabulary>(
    f: &mut FilePiece,
    count: usize,
    vocab: &mut V,
    unigrams: &mut [ProbBackoff],
    warn: &PositiveProbWarn,
) -> Result<(), LMError> {
    // Read header like "\1-grams:"
    read_ngram_header(f, 1)?;

    // Read each unigram
    for _ in 0..count {
        read_1gram(f, vocab, unigrams, warn)?;
    }

    Ok(())
}

/// Read an n-gram (n > 1) from the ARPA file
/// Format: prob \t word1 word2 ... wordn [\t backoff] \n
pub fn read_ngram<V: Vocabulary>(
    f: &mut FilePiece,
    n: u8,
    vocab: &V,
    indices_out: &mut [WordIndex],
    weights: &mut ProbBackoff,
    warn: &PositiveProbWarn,
) -> Result<(), LMError> {
    // Read probability
    weights.prob = f.read_float()?;
    if weights.prob > 0.0 {
        warn.warn(weights.prob);
        weights.prob = 0.0;
    }

    // Read n words
    for i in 0..n as usize {
        let word_str = f.read_delimited(&ARPA_SPACES)?;
        let index = vocab.index(&word_str);
        indices_out[i] = index;

        // Check for words mapped to <unk> that are not the string <unk>
        if index == 0 && word_str != "<unk>" && word_str != "<UNK>" {
            return Err(LMError::InvalidArpa(format!(
                "Word '{}' was not seen in the unigrams but appears in {}-gram",
                word_str, n
            )));
        }
    }

    // Read backoff if present
    read_backoff_probbackoff(f, weights)?;

    Ok(())
}

/// Helper to check if a line is entirely whitespace
fn is_entirely_whitespace(line: &str) -> bool {
    line.chars().all(|c| c.is_whitespace())
}

/// Read the n-gram section header (e.g., "\1-grams:")
pub fn read_ngram_header(f: &mut FilePiece, length: u32) -> Result<(), LMError> {
    // Skip empty lines
    let mut line;
    loop {
        line = f.read_line('\n', true)?;
        if !is_entirely_whitespace(&line) {
            break;
        }
    }

    let expected = format!("\\{}-grams:", length);
    let actual = line.trim();

    if actual != expected {
        return Err(LMError::InvalidArpa(format!(
            "Expected n-gram header '{}' but got '{}'",
            expected, actual
        )));
    }

    Ok(())
}

/// Read backoff weight into a ProbBackoff structure
fn read_backoff_probbackoff(f: &mut FilePiece, weights: &mut ProbBackoff) -> Result<(), LMError> {
    match f.get()? {
        '\t' => {
            weights.backoff = f.read_float()?;

            // Check for NaN or Inf
            if weights.backoff.is_nan() || weights.backoff.is_infinite() {
                return Err(LMError::InvalidArpa(format!(
                    "Bad backoff value: {}",
                    weights.backoff
                )));
            }

            // Convert extension backoff marker
            if weights.backoff == 0.0 {
                weights.backoff = K_NO_EXTENSION_BACKOFF;
            }

            // Expect newline after backoff
            match f.get()? {
                '\r' => {
                    if f.get()? != '\n' {
                        return Err(LMError::InvalidArpa(
                            "Expected newline after backoff".to_string(),
                        ));
                    }
                }
                '\n' => {}
                got => {
                    return Err(LMError::InvalidArpa(format!(
                        "Expected newline after backoff, got '{}'",
                        got
                    )));
                }
            }
        }
        '\r' => {
            if f.get()? != '\n' {
                return Err(LMError::InvalidArpa(
                    "Expected newline after backoff".to_string(),
                ));
            }
            weights.backoff = K_NO_EXTENSION_BACKOFF;
        }
        '\n' => {
            weights.backoff = K_NO_EXTENSION_BACKOFF;
        }
        _ => {
            return Err(LMError::InvalidArpa(
                "Expected tab or newline for backoff".to_string(),
            ));
        }
    }

    Ok(())
}

/// Read ARPA counts from the \data\ section
pub fn read_arpa_counts(f: &mut FilePiece) -> Result<Vec<u64>, LMError> {
    let mut number = Vec::new();

    // Read until we find \data\
    loop {
        let line = f.read_line('\n', true)?;
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Check if this is the \data\ section
        if trimmed == "\\data\\" {
            break;
        }

        // Check for binary format
        if line.len() >= 2 && line.as_bytes()[0] == 0x1f && line.as_bytes()[1] == 0x8b {
            return Err(LMError::InvalidArpa(
                "Looks like a gzip file. Pipe through zcat first.".to_string(),
            ));
        }

        // Check for other formats
        if line.starts_with("mmap lm http://kheafield.com/code") {
            return Err(LMError::InvalidArpa(
                "This looks like a binary file sent to ARPA parser.".to_string(),
            ));
        }

        return Err(LMError::InvalidArpa(format!(
            "First non-empty line was '{}' not \\data\\",
            trimmed
        )));
    }

    // Now read the counts
    loop {
        let line = f.read_line('\n', true)?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            break;
        }

        // Parse lines like "ngram 1=1000"
        if !trimmed.starts_with("ngram ") {
            return Err(LMError::InvalidArpa(format!(
                "Count line '{}' doesn't begin with 'ngram '",
                trimmed
            )));
        }

        // Extract the order and count
        let remaining = &trimmed[6..]; // Skip "ngram "
        let parts: Vec<&str> = remaining.split('=').collect();

        if parts.len() != 2 {
            return Err(LMError::InvalidArpa(format!(
                "Invalid count line format: {}",
                trimmed
            )));
        }

        let order: usize = parts[0]
            .trim()
            .parse()
            .map_err(|_| LMError::InvalidArpa(format!("Invalid order number: {}", parts[0])))?;

        let count: u64 = parts[1]
            .trim()
            .parse()
            .map_err(|_| LMError::InvalidArpa(format!("Invalid count: {}", parts[1])))?;

        // Orders should be consecutive starting with 1
        if order != number.len() + 1 {
            return Err(LMError::InvalidArpa(format!(
                "ngram orders should be consecutive starting with 1, got {}",
                order
            )));
        }

        number.push(count);
    }

    if number.is_empty() {
        return Err(LMError::InvalidArpa(
            "No ngram counts found in \\data\\ section".to_string(),
        ));
    }

    Ok(number)
}

/// Read the \end\ marker at the end of an ARPA file
pub fn read_end(f: &mut FilePiece) -> Result<(), LMError> {
    // Skip empty lines
    let mut line;
    loop {
        match f.read_line('\n', true) {
            Ok(l) => {
                line = l;
                if !is_entirely_whitespace(&line) {
                    break;
                }
            }
            Err(LMError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(LMError::InvalidArpa(
                    "Expected \\end\\ but got EOF".to_string(),
                ));
            }
            Err(e) => return Err(e),
        }
    }

    if line.trim() != "\\end\\" {
        return Err(LMError::InvalidArpa(format!(
            "Expected \\end\\ but got '{}'",
            line.trim()
        )));
    }

    // Check for trailing non-empty lines (these would be warnings but we'll be permissive)
    loop {
        match f.read_line('\n', true) {
            Ok(line) => {
                if !is_entirely_whitespace(&line) {
                    // Just warn, don't error
                    eprintln!("Warning: Trailing line after \\end\\: {}", line);
                }
            }
            Err(LMError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(e),
        }
    }

    Ok(())
}

/// Positive log probability warning handler
pub struct PositiveProbWarn {
    action: WarningAction,
    warned: std::cell::Cell<bool>,
}

impl PositiveProbWarn {
    pub fn new(action: WarningAction) -> Self {
        Self {
            action,
            warned: std::cell::Cell::new(false),
        }
    }

    pub fn warn(&self, prob: f32) {
        match self.action {
            WarningAction::ThrowUp => {
                panic!(
                    "Positive log probability {} in the model. This is a bug in IRSTLM.",
                    prob
                );
            }
            WarningAction::Complain => {
                if !self.warned.get() {
                    eprintln!(
                        "Warning: Positive log probability {} in ARPA file. \
                         This and subsequent entries will be mapped to 0.",
                        prob
                    );
                    self.warned.set(true);
                }
            }
            WarningAction::Silent => {}
        }
    }
}

impl Default for PositiveProbWarn {
    fn default() -> Self {
        Self::new(WarningAction::ThrowUp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_entirely_whitespace() {
        assert!(is_entirely_whitespace("   "));
        assert!(is_entirely_whitespace("\t\n"));
        assert!(is_entirely_whitespace(""));
        assert!(!is_entirely_whitespace("  a  "));
    }

    #[test]
    fn test_positive_prob_warn() {
        let warn = PositiveProbWarn::new(WarningAction::Silent);
        warn.warn(0.5); // Should not panic with Silent
    }
}
