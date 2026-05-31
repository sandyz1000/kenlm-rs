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
    use crate::constant::WarningAction;
    use crate::vocabulary::ProbingVocabulary;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_file(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{}", content).unwrap();
        f.flush().unwrap();
        f
    }

    fn open_piece(content: &str) -> FilePiece {
        let f = make_file(content);
        FilePiece::open(f.path()).unwrap()
    }

    // ── is_entirely_whitespace ────────────────────────────────────────────────

    #[test]
    fn test_is_entirely_whitespace() {
        assert!(is_entirely_whitespace("   "));
        assert!(is_entirely_whitespace("\t\n"));
        assert!(is_entirely_whitespace(""));
        assert!(!is_entirely_whitespace("  a  "));
    }

    // ── PositiveProbWarn ──────────────────────────────────────────────────────

    #[test]
    fn test_positive_prob_warn_silent() {
        let warn = PositiveProbWarn::new(WarningAction::Silent);
        warn.warn(0.5); // must not panic
    }

    #[test]
    fn test_positive_prob_warn_complain_once() {
        let warn = PositiveProbWarn::new(WarningAction::Complain);
        warn.warn(0.1); // prints to stderr but does not panic
        warn.warn(0.2); // second call is suppressed by warned flag
    }

    #[test]
    #[should_panic(expected = "Positive log probability")]
    fn test_positive_prob_warn_throw_up() {
        let warn = PositiveProbWarn::new(WarningAction::ThrowUp);
        warn.warn(0.5);
    }

    #[test]
    fn test_positive_prob_warn_default_throws() {
        // Default action is ThrowUp
        let result = std::panic::catch_unwind(|| {
            let warn = PositiveProbWarn::default();
            warn.warn(0.1);
        });
        assert!(
            result.is_err(),
            "Default warn action should panic on positive prob"
        );
    }

    // ── read_ngram_header ─────────────────────────────────────────────────────

    #[test]
    fn test_read_ngram_header_valid() {
        let mut fp = open_piece("\\1-grams:\n");
        read_ngram_header(&mut fp, 1).unwrap();
    }

    #[test]
    fn test_read_ngram_header_skips_blank_lines() {
        let mut fp = open_piece("\n  \n\\2-grams:\n");
        read_ngram_header(&mut fp, 2).unwrap();
    }

    #[test]
    fn test_read_ngram_header_wrong_order() {
        let mut fp = open_piece("\\3-grams:\n");
        let err = read_ngram_header(&mut fp, 2).unwrap_err();
        matches!(err, LMError::InvalidArpa(_));
    }

    // ── read_arpa_counts ──────────────────────────────────────────────────────

    #[test]
    fn test_read_arpa_counts_basic() {
        let content = "\
\\data\\
ngram 1=5
ngram 2=12
ngram 3=3

";
        let mut fp = open_piece(content);
        let counts = read_arpa_counts(&mut fp).unwrap();
        assert_eq!(counts, vec![5, 12, 3]);
    }

    #[test]
    fn test_read_arpa_counts_with_comments_and_blank() {
        let content = "\
# this is a comment

\\data\\
ngram 1=100

";
        let mut fp = open_piece(content);
        let counts = read_arpa_counts(&mut fp).unwrap();
        assert_eq!(counts, vec![100]);
    }

    #[test]
    fn test_read_arpa_counts_empty_data_section() {
        let content = "\\data\\\n\n";
        let mut fp = open_piece(content);
        let err = read_arpa_counts(&mut fp).unwrap_err();
        matches!(err, LMError::InvalidArpa(_));
    }

    #[test]
    fn test_read_arpa_counts_gzip_magic_bytes() {
        // gzip magic: 0x1f 0x8b
        let mut content = vec![0x1fu8, 0x8b];
        content.extend_from_slice(b" rest of binary garbage\n");
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&content).unwrap();
        f.flush().unwrap();
        let mut fp = FilePiece::open(f.path()).unwrap();
        let err = read_arpa_counts(&mut fp).unwrap_err();
        matches!(err, LMError::InvalidArpa(_));
    }

    #[test]
    fn test_read_arpa_counts_non_consecutive_orders() {
        let content = "\\data\\\nngram 1=5\nngram 3=2\n\n";
        let mut fp = open_piece(content);
        let err = read_arpa_counts(&mut fp).unwrap_err();
        matches!(err, LMError::InvalidArpa(_));
    }

    #[test]
    fn test_read_arpa_counts_wrong_prefix() {
        let content = "\\data\\\nbad 1=5\n\n";
        let mut fp = open_piece(content);
        let err = read_arpa_counts(&mut fp).unwrap_err();
        matches!(err, LMError::InvalidArpa(_));
    }

    // ── read_end ──────────────────────────────────────────────────────────────

    #[test]
    fn test_read_end_valid() {
        let mut fp = open_piece("\\end\\\n");
        read_end(&mut fp).unwrap();
    }

    #[test]
    fn test_read_end_with_blank_lines_before() {
        let mut fp = open_piece("\n\n\\end\\\n");
        read_end(&mut fp).unwrap();
    }

    #[test]
    fn test_read_end_wrong_marker() {
        let mut fp = open_piece("\\done\\\n");
        let err = read_end(&mut fp).unwrap_err();
        matches!(err, LMError::InvalidArpa(_));
    }

    // ── read_1gram / read_1grams ──────────────────────────────────────────────

    #[test]
    fn test_read_1gram_with_backoff() {
        // Format: prob TAB word TAB backoff NEWLINE
        let mut vocab = ProbingVocabulary::new();
        vocab.add_word("hello");
        let mut unigrams = vec![ProbBackoff::default(); 2];
        let warn = PositiveProbWarn::new(WarningAction::Silent);

        let mut fp = open_piece("-1.5\thello\t-0.3\n");
        read_1gram(&mut fp, &mut vocab, &mut unigrams, &warn).unwrap();
        assert!((unigrams[0].prob - (-1.5)).abs() < 1e-5);
    }

    #[test]
    fn test_read_1gram_no_backoff() {
        let mut vocab = ProbingVocabulary::new();
        vocab.add_word("world");
        let mut unigrams = vec![ProbBackoff::default(); 2];
        let warn = PositiveProbWarn::new(WarningAction::Silent);

        let mut fp = open_piece("-2.0\tworld\n");
        read_1gram(&mut fp, &mut vocab, &mut unigrams, &warn).unwrap();
        assert!((unigrams[0].prob - (-2.0)).abs() < 1e-5);
        assert_eq!(unigrams[0].backoff, crate::constant::K_NO_EXTENSION_BACKOFF);
    }

    #[test]
    fn test_read_1gram_positive_prob_clamped() {
        let mut vocab = ProbingVocabulary::new();
        vocab.add_word("word");
        let mut unigrams = vec![ProbBackoff::default(); 2];
        let warn = PositiveProbWarn::new(WarningAction::Silent);

        let mut fp = open_piece("0.5\tword\n");
        read_1gram(&mut fp, &mut vocab, &mut unigrams, &warn).unwrap();
        // Positive prob gets clamped to 0.0
        assert_eq!(unigrams[0].prob, 0.0);
    }

    // ── full round-trip for a minimal bigram ARPA ─────────────────────────────

    #[test]
    fn test_full_round_trip_unigram_arpa() {
        let content = "\
\\data\\
ngram 1=3

\\1-grams:
-99\t<unk>\t0
-1.5\t<s>\t-0.3
-1.0\t</s>

\\end\\
";
        let mut fp = open_piece(content);
        let counts = read_arpa_counts(&mut fp).unwrap();
        assert_eq!(counts, vec![3]);

        let mut vocab = ProbingVocabulary::new();
        vocab.add_word("<unk>");
        vocab.add_word("<s>");
        vocab.add_word("</s>");
        let mut unigrams = vec![ProbBackoff::default(); 3];
        let warn = PositiveProbWarn::new(WarningAction::Silent);

        read_1grams(
            &mut fp,
            counts[0] as usize,
            &mut vocab,
            &mut unigrams,
            &warn,
        )
        .unwrap();
        read_end(&mut fp).unwrap();
    }
}
