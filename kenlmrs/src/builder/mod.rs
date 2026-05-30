mod config;
mod count;
mod error;
mod interpolate;
mod pipeline;
mod proba;

use crate::constant::WarningAction;
use crate::types::{ ProbBackoff, WordIndex };
// TODO: Add proper imports for util types when they exist
// use crate::util::{Chains, FilePiece, SortConfig, ChainConfig, ChainPosition, StringPiece};
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
    token_count: i64,
}

impl CorpusCount {
    // Memory usage will be DedupeMultipler(order) * block_size + total_chain_size + unknown vocab_hash_size
    fn DedupeMultiplier(order: i8) -> f64 {
        todo!()
    }

    // How much memory vocabulary will use based on estimated size of the vocab.
    fn VocabUsage(&self, vocab_estimate: i8) -> i8 {
        todo!()
    }

    // type_count aka vocabulary size.  Initialize to an estimate.  It is set to the exact value.
    fn new(
        from_: &crate::utils::pieces::file::FilePiece,
        vocab_write: i64,
        dynamic_vocab: bool,
        token_count: &u64,
        type_count: &WordIndex,
        prune_words: &Vec<bool>,
        prune_vocab_filename: &str,
        entries_per_block: i8,
        disallowed_symbol: WarningAction
    ) -> Self {
        Self { token_count: 0 }
    }

    fn Run(&self, position: &crate::builder::proba::ChainPosition) {
        todo!()
    }
}
