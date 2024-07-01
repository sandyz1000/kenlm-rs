use std::collections::HashMap;
use std::io::{self, Read};
use std::str::FromStr;

fn read_arpa_counts(in_file: &mut FilePiece, number: &mut Vec<u64>) {
    // Implementation for reading ARPA counts
}

fn read_ngram_header(in_file: &mut FilePiece, length: u32) {
    // Implementation for reading NGram header
}

fn read_backoff(in_file: &mut FilePiece, weights: &mut f32) {
    *weights = in_file.read_float();
}

fn read_end(in_file: &mut FilePiece) {
    // Implementation for reading end
}

const K_ARPA_SPACES: [bool; 256] = [false; 256];

pub enum WarningAction {
    ThrowUp,
}

pub struct PositiveProbWarn {
    action: WarningAction,
}

impl PositiveProbWarn {
    pub fn new() -> Self {
        PositiveProbWarn {
            action: WarningAction::ThrowUp,
        }
    }

    pub fn with_action(action: WarningAction) -> Self {
        PositiveProbWarn { action }
    }

    pub fn warn(&self, prob: f32) {
        // Implementation for warning
    }
}

#[derive(Debug)]
pub struct Vocab {
    map: HashMap<String, u32>,
}

impl Vocab {
    pub fn new() -> Self {
        Vocab {
            map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, word: String) -> u32 {
        let index = self.map.len() as u32;
        self.map.insert(word, index);
        index
    }

    pub fn index(&self, word: &str) -> u32 {
        *self.map.get(word).unwrap_or(&0)
    }

    pub fn finished_loading(&self) {
        // Implementation for finished loading
    }
}

#[derive(Debug)]
pub struct Weights {
    prob: f32,
    backoff: f32,
}

pub fn read_1gram(
    f: &mut FilePiece,
    vocab: &mut Vocab,
    unigrams: &mut [Weights],
    warn: &PositiveProbWarn,
) {
    let mut prob = f.read_float();
    if prob > 0.0 {
        warn.warn(prob);
        prob = 0.0;
    }
    assert_eq!(f.get(), '\t');
    f.offset += 1;
    let word = f.read_delimited(&K_ARPA_SPACES);
    let word_index = vocab.insert(word);
    unigrams[word_index as usize].prob = prob;
    read_backoff(f, &mut unigrams[word_index as usize].backoff);
}

pub fn read_1grams(
    f: &mut FilePiece,
    count: usize,
    vocab: &mut Vocab,
    unigrams: &mut [Weights],
    warn: &PositiveProbWarn,
) {
    read_ngram_header(f, 1);
    for _ in 0..count {
        read_1gram(f, vocab, unigrams, warn);
    }
    vocab.finished_loading();
}

pub fn read_ngram<Voc, Weights>(
    f: &mut FilePiece,
    n: u8,
    vocab: &Voc,
    indices_out: &mut [u32],
    weights: &mut Weights,
    warn: &PositiveProbWarn,
) where
    Voc: Vocab,
    Weights: ProbBackoff,
{
    weights.prob = f.read_float();
    if weights.prob > 0.0 {
        warn.warn(weights.prob);
        weights.prob = 0.0;
    }
    for i in 0..n {
        let word = f.read_delimited(&K_ARPA_SPACES);
        let index = vocab.index(&word);
        indices_out[i as usize] = index;
        if index == 0 && word != "<unk>" && word != "<UNK>" {
            panic!(
                "Word {} was not seen in the unigrams but appears in the n-grams",
                word
            );
        }
    }
    read_backoff(f, weights);
}

const K_ARPA_SPACES: [bool; 256] = {
    let mut spaces = [false; 256];
    spaces[b'\t' as usize] = true;
    spaces[b'\n' as usize] = true;
    spaces[b'\r' as usize] = true;
    spaces[b' ' as usize] = true;
    spaces
};

fn is_entirely_whitespace(line: &str) -> bool {
    line.chars().all(|c| c.is_whitespace())
}

const BINARY_MAGIC: &str = "mmap lm http://kheafield.com/code";

fn read_count(from: &str) -> u64 {
    from.parse::<u64>().expect(&format!("Bad count {}", from))
}

pub fn read_arpa_counts(in_file: &mut FilePiece, number: &mut Vec<u64>) {
    number.clear();
    let mut line = in_file.read_line();
    while is_entirely_whitespace(&line) || line.starts_with('#') {
        line = in_file.read_line();
    }

    if line != "\\data\\" {
        if line.as_bytes().len() >= 2 && line.as_bytes()[0] == 0x1f && line.as_bytes()[1] == 0x8b {
            panic!("Looks like a gzip file. If this is an ARPA file, pipe {} through zcat. If this already in binary format, you need to decompress it because mmap doesn't work on top of gzip.", in_file.file_name());
        }
        if line.as_bytes().len() >= BINARY_MAGIC.len()
            && &line[..BINARY_MAGIC.len()] == BINARY_MAGIC
        {
            panic!("This looks like a binary file but got sent to the ARPA parser. Did you compress the binary file or pass a binary file where only ARPA files are accepted?");
        }
        if line.as_bytes().len() >= 4 && &line[..4] == "blmt" {
            panic!("This looks like an IRSTLM binary file. Did you forget to pass --text yes to compile-lm?");
        }
        if line == "iARPA" {
            panic!("This looks like an IRSTLM iARPA file. You need an ARPA file. Run\n  compile-lm --text yes {} {}.arpa\nfirst.", in_file.file_name(), in_file.file_name());
        }
        panic!("first non-empty line was \"{}\" not \\data\\.", line);
    }
    while !is_entirely_whitespace(&line) {
        if line.len() < 6 || &line[..6] != "ngram " {
            panic!("count line \"{}\" doesn't begin with \"ngram \"", line);
        }
        let remaining = &line[6..];
        let length = remaining
            .split_whitespace()
            .next()
            .unwrap()
            .parse::<u32>()
            .unwrap();
        if length as usize - 1 != number.len() {
            panic!(
                "ngram count lengths should be consecutive starting with 1: {}",
                line
            );
        }
        let count_str = remaining.split('=').nth(1).unwrap();
        number.push(read_count(count_str));
        line = in_file.read_line();
    }
}

fn read_ngram_header(in_file: &mut FilePiece, length: u32) {
    let mut line = in_file.read_line();
    while is_entirely_whitespace(&line) {
        line = in_file.read_line();
    }
    let expected = format!("\\{}-grams:", length);
    if line != expected {
        panic!(
            "Was expecting n-gram header {} but got {} instead",
            expected, line
        );
    }
}

fn consume_newline(in_file: &mut FilePiece) {
    let follow = in_file.get();
    if follow != '\n' {
        panic!("Expected newline got '{}'", follow);
    }
}

fn read_backoff(in_file: &mut FilePiece, weights: &mut f32) {
    match in_file.get() {
        '\t' => {
            let got = in_file.read_float();
            if got != 0.0 {
                panic!(
                    "Non-zero backoff {} provided for an n-gram that should have no backoff",
                    got
                );
            }
        }
        '\r' => consume_newline(in_file),
        '\n' => (),
        _ => panic!("Expected tab or newline for backoff"),
    }
}

fn read_backoff_float(in_file: &mut FilePiece, backoff: &mut f32) {
    match in_file.get() {
        '\t' => {
            *backoff = in_file.read_float();
            if backoff.is_nan() || backoff.is_infinite() {
                panic!("Bad backoff {}", backoff);
            }
            match in_file.get() {
                '\r' => consume_newline(in_file),
                '\n' => (),
                got => panic!("Expected newline after backoffs, got '{}'", got),
            }
        }
        '\r' => consume_newline(in_file),
        '\n' => *backoff = 0.0,
        _ => panic!("Expected tab or newline for backoff"),
    }
}

fn read_end(in_file: &mut FilePiece) {
    let mut line = in_file.read_line();
    while is_entirely_whitespace(&line) {
        line = in_file.read_line();
    }
    if line != "\\end\\" {
        panic!("Expected \\end\\ but the ARPA file has {}", line);
    }
    while let Ok(line) = in_file.read_line() {
        if !is_entirely_whitespace(&line) {
            panic!("Trailing line {}", line);
        }
    }
}

impl PositiveProbWarn {
    pub fn warn(&self, prob: f32) {
        match self.action {
            WarningAction::ThrowUp => {
                let err_msg = "Positive log probability {} in the model. 
                    This is a bug in IRSTLM; you can set config.positive_log_probability = SILENT or 
                    pass -i to build_binary to substitute 0.0 for the log probability."; 
                panic!(err_msg, prob)
            },
            WarningAction::Complain => {
                let err_msg = "There's a positive log probability {} in the ARPA file, probably because of a bug in IRSTLM. This and subsequent entries will be mapped to 0 log probability.";
                eprintln!(err_msg, prob)
            },
            WarningAction::Silent => (),
        }
    }
}
