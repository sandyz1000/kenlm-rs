mod config;
mod count;
mod error;
mod interpolate;
pub mod pipeline;
mod proba;

use crate::constant::WarningAction;
use crate::types::{ ProbBackoff, WordIndex };
use crate::vocabulary::Vocabulary as _;
use std::cmp;

#[derive(Debug, Default, Clone)]
pub struct Discount {
    amount: [u64; 4],
}

impl Discount {
    fn get(&self, count: u64) -> u64 {
        self.amount[cmp::min(count as usize, 3)]
    }

    fn apply(&self, count: u64) -> u64 {
        count - self.get(count)
    }
}

#[derive(Debug)]
pub struct HeaderInfo {
    input_file: String,
    token_count: u64,
    counts_pruned: Vec<u64>,
}

impl HeaderInfo {
    fn new(input_file_in: &str, token_count_in: u64, counts_pruned_in: &Vec<u64>) -> Self {
        HeaderInfo {
            input_file: input_file_in.to_string(),
            token_count: token_count_in,
            counts_pruned: counts_pruned_in.clone(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Uninterpolated {
    prob: f32, // Uninterpolated probability
    gamma: f32, // Interpolation weight for lower order
}

#[derive(Copy, Clone, Debug)]
pub enum PayloadType {
    Count(u64),
    Uninterp(Uninterpolated),
    Complete(ProbBackoff),
}

#[derive(Copy, Clone, Debug)]
pub struct BuildingPayload {
    payload: PayloadType,
}

impl BuildingPayload {
    pub fn is_marked(&self) -> bool {
        match self.payload {
            PayloadType::Count(count) => (count >> (std::mem::size_of::<u64>() * 8 - 1)) != 0,
            _ => false,
        }
    }

    pub fn mark(&mut self) {
        if let PayloadType::Count(ref mut count) = self.payload {
            *count |= 1 << (std::mem::size_of::<u64>() * 8 - 1);
        }
    }

    pub fn unmark(&mut self) {
        if let PayloadType::Count(ref mut count) = self.payload {
            *count &= !(1 << (std::mem::size_of::<u64>() * 8 - 1));
        }
    }

    pub fn unmarked_count(&self) -> u64 {
        match self.payload {
            PayloadType::Count(count) => count & !(1 << (std::mem::size_of::<u64>() * 8 - 1)),
            _ => 0,
        }
    }

    pub fn cutoff_count(&self) -> u64 {
        if self.is_marked() { 0 } else { self.unmarked_count() }
    }
}

// Assuming ProbBackoff and WordIndex are defined somewhere else.
const BOS: WordIndex = 1;
const EOS: WordIndex = 2;

#[derive(Debug, Clone)]
pub struct CorpusCount {
    pub token_count: u64,
    pub type_count: WordIndex,
}

impl CorpusCount {
    /// Memory multiplier per byte of block data for deduplication at a given order.
    /// Derived from C++ KenLM: sizeof(NGram) * order + (WordIndex + sizeof(uint64_t)) = 4*order + 12.
    /// Divide by sizeof(uint64_t) = 8 for bytes-per-byte ratio.
    pub fn dedupe_multiplier(order: usize) -> f64 {
        (4 * order + 12) as f64 / 8.0
    }

    /// Estimated bytes for a probing vocabulary hash table of `vocab_estimate` entries.
    pub fn vocab_usage(vocab_estimate: WordIndex) -> usize {
        (vocab_estimate as usize) * 8 + 1024
    }

    /// Initialize a CorpusCount state. The actual counting happens in `run`.
    pub fn new() -> Self {
        Self { token_count: 0, type_count: 0 }
    }

    /// Count n-grams from a plain-text input file.
    ///
    /// Reads the file line by line, tokenises on whitespace, wraps each sentence
    /// in `<s>` / `</s>` markers, and counts all n-grams up to `order`.
    /// Returns a Vec of (ngram_words, count) pairs.
    pub fn run(
        &mut self,
        input_path: &str,
        order: usize,
        vocab: &mut crate::vocabulary::ProbingVocabulary,
    ) -> Result<Vec<(Vec<crate::types::WordIndex>, u64)>, crate::error::LMError> {
        use std::io::{BufRead, BufReader};
        use std::fs::File;
        use std::collections::HashMap;

        let file = File::open(input_path)?;
        let reader = BufReader::new(file);

        // Register special tokens
        vocab.add_word("<unk>");
        vocab.add_word("<s>");
        vocab.add_word("</s>");
        let bos = vocab.index("<s>");
        let eos = vocab.index("</s>");

        let mut ngram_counts: HashMap<Vec<crate::types::WordIndex>, u64> = HashMap::new();
        let mut token_count = 0u64;

        for line in reader.lines() {
            let line = line.map_err(crate::error::LMError::IoError)?;
            let words: Vec<&str> = line.split_whitespace().collect();
            if words.is_empty() { continue; }

            // Build sentence token sequence: <s> w1 w2 ... wn </s>
            let mut sent = vec![bos];
            for w in &words {
                let id = vocab.add_word(w);
                sent.push(id);
                token_count += 1;
            }
            sent.push(eos);

            // Count all n-grams up to `order`
            for start in 0..sent.len() {
                let end = (start + order).min(sent.len());
                for len in 1..=(end - start) {
                    let ngram = sent[start..start + len].to_vec();
                    *ngram_counts.entry(ngram).or_insert(0) += 1;
                }
            }
        }

        self.token_count = token_count;
        self.type_count = vocab.bound();

        Ok(ngram_counts.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_count_dedupe_multiplier_grows_with_order() {
        let d1 = CorpusCount::dedupe_multiplier(1);
        let d3 = CorpusCount::dedupe_multiplier(3);
        assert!(d3 > d1, "dedupe multiplier should grow with order");
    }

    #[test]
    fn corpus_count_vocab_usage_is_positive() {
        let usage = CorpusCount::vocab_usage(1000);
        assert!(usage > 0);
    }

    #[test]
    fn parse_discount_fallback_parses_csv() {
        use crate::builder::pipeline::ParseDiscountFallback;
        let params = vec!["0.5,0.75,1.0".to_string()];
        let d = ParseDiscountFallback(&params);
        assert!((d.amount[0] - 0.5).abs() < 1e-5);
        assert!((d.amount[1] - 0.75).abs() < 1e-5);
    }

    #[test]
    fn parse_pruning_pads_to_order() {
        use crate::builder::pipeline::ParsePruning;
        let params = vec!["0 0 1".to_string()];
        let thresh = ParsePruning(&params, 5);
        assert_eq!(thresh.len(), 5);
        assert_eq!(thresh[2], 1);
        assert_eq!(thresh[4], 0);
    }
}
