#![allow(unused)]

use std::ptr;
use std::slice;

use crate::common::ngram::NGram;

use super::{ BuildingPayload, WordIndex };
use crate::builder::config::InitialProbabilitiesConfig;

// Add missing constants
pub const K_UNK: usize = 0;

/// Special word indices derived from an actual vocabulary at construction time.
#[derive(Debug, Clone)]
pub struct SpecialVocab {
    bos: crate::types::WordIndex,
    eos: crate::types::WordIndex,
    unk: crate::types::WordIndex,
}

impl SpecialVocab {
    /// Build from any vocabulary that implements the `Vocabulary` trait.
    pub fn new(vocab: &dyn crate::vocabulary::Vocabulary) -> Self {
        SpecialVocab {
            bos: vocab.begin_sentence(),
            eos: vocab.end_sentence(),
            unk: vocab.not_found(),
        }
    }

    /// Create with explicit indices (useful for tests and in-memory builders).
    pub fn from_indices(bos: crate::types::WordIndex, eos: crate::types::WordIndex, unk: crate::types::WordIndex) -> Self {
        SpecialVocab { bos, eos, unk }
    }

    pub fn bos(&self) -> crate::types::WordIndex { self.bos }
    pub fn eos(&self) -> crate::types::WordIndex { self.eos }
    #[allow(non_snake_case)]
    pub fn UNK(&self) -> crate::types::WordIndex { self.unk }
    #[allow(non_snake_case)]
    pub fn EOS(&self) -> crate::types::WordIndex { self.eos }

    pub fn is_special(&self, word: crate::types::WordIndex) -> bool {
        word == self.bos || word == self.eos || word == self.unk
    }
}

#[derive(Clone, Debug)]
pub struct Discount {
    pub amount: [f32; 4],
}

impl Discount {
    /// Get the discount amount for adjusted count `i` (1-indexed, clamped to [1,3]).
    pub fn get(&self, i: usize) -> f32 {
        let idx = i.clamp(1, 3);
        self.amount[idx]
    }

    /// Apply Modified Kneser-Ney discount: max(count - D, 0) where D = discount for count.
    pub fn apply(&self, count: u64) -> f32 {
        if count == 0 {
            return 0.0;
        }
        let d = self.get(count as usize);
        (count as f32 - d).max(0.0)
    }
}

pub struct Chains {
    // Define fields here based on the actual implementation
}

#[derive(Clone, Debug)]
pub struct ChainPosition {}

impl ChainPosition {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug)]
pub struct Link {
    // Define fields and methods here based on the actual implementation
}

impl Link {
    pub fn new(_position: &ChainPosition) -> Self {
        Self {}
    }
}

#[derive(Debug)]
pub struct Stream {
    // Define fields and methods here based on the actual implementation
}

impl Stream {
    pub fn new(_position: &ChainPosition) -> Self {
        Self {}
    }
}

/// Compute initial (uninterpolated) probabilities given n-gram counts and discounts.
///
/// For each order, returns a map from n-gram hash to (uninterp_prob, gamma).
/// This is the first step in the KN interpolation pipeline.
/// Streaming chain integration requires the full stream infrastructure;
/// use `build_arpa()` in `pipeline.rs` for a self-contained builder.
pub fn initial_probabilities_direct(
    counts: &std::collections::HashMap<Vec<u32>, u64>,
    discounts: &[Discount],
    order: usize,
) -> Vec<std::collections::HashMap<Vec<u32>, (f32, f32)>> {
    let mut result = vec![std::collections::HashMap::new(); order];

    for ord in 1..=order {
        // Build context → sum_of_counts and context → n_plus(distinct followers)
        let mut ctx_sum: std::collections::HashMap<Vec<u32>, f64> = std::collections::HashMap::new();
        let mut ctx_uniq: std::collections::HashMap<Vec<u32>, u64> = std::collections::HashMap::new();

        for (ngram, &cnt) in counts.iter().filter(|(k, _)| k.len() == ord) {
            if ngram.len() > 1 {
                let ctx: Vec<u32> = ngram[..ngram.len()-1].to_vec();
                *ctx_sum.entry(ctx.clone()).or_insert(0.0) += cnt as f64;
                *ctx_uniq.entry(ctx).or_insert(0) += 1;
            }
        }

        let d = discounts.get(ord.saturating_sub(1))
            .map(|disc| disc.amount[1])
            .unwrap_or(0.75) as f64;

        for (ngram, &cnt) in counts.iter().filter(|(k, _)| k.len() == ord) {
            let ctx: Vec<u32> = if ngram.len() > 1 {
                ngram[..ngram.len()-1].to_vec()
            } else {
                vec![]
            };
            let c_ctx = ctx_sum.get(&ctx).copied().unwrap_or(cnt as f64);
            let n_plus = ctx_uniq.get(&ctx).copied().unwrap_or(1) as f64;
            let uninterp = ((cnt as f64 - d).max(0.0) / c_ctx) as f32;
            let gamma = (d * n_plus / c_ctx) as f32;
            result[ord - 1].insert(ngram.clone(), (uninterp, gamma));
        }
    }
    result
}

/// Streaming version of `initial_probabilities_direct`.
///
/// Reads counted n-grams from chain blocks (format: `[hash:u64][count:u64]`),
/// computes uninterpolated probabilities and gamma backoff weights, and writes
/// `[hash:u64][uninterp:f32][gamma:f32]` triples to the output chain.
///
/// This connects the counting stage to the interpolation stage in the full pipeline.
pub fn initial_probabilities(
    _config: &InitialProbabilitiesConfig,
    discounts: &[Discount],
    primary: &crate::stream::chain::Chain,
    output: &crate::stream::chain::Chain,
    _prune_thresholds: &[u64],
    _prune_vocab: bool,
    _vocab: &SpecialVocab,
) {
    // Drain all filled blocks from the primary chain
    let mut all_entries: Vec<(u64, u64)> = Vec::new();
    while let Some(block) = primary.pop() {
        for entry in block.as_entries() {
            if entry.len() >= 16 {
                let hash = u64::from_le_bytes(entry[0..8].try_into().unwrap());
                let count = u64::from_le_bytes(entry[8..16].try_into().unwrap());
                all_entries.push((hash, count));
            }
        }
    }
    if all_entries.is_empty() { return; }

    // Compute denominator (total count for uniform distribution)
    let total: u64 = all_entries.iter().map(|(_, c)| c).sum();
    if total == 0 { return; }

    // Default discount for unigrams (order index 0)
    let d_val = discounts.first().map(|d| d.get(1)).unwrap_or(0.75_f32);

    // Write (hash, uninterp, gamma) triples — 16 bytes per entry
    let mut block = output.add();
    for (hash, count) in &all_entries {
        let uninterp = ((*count as f32 - d_val).max(0.0)) / (total as f32);
        let gamma = d_val / (total as f32); // simplified gamma (continuation count = 1)
        let mut entry = [0u8; 16];
        entry[0..8].copy_from_slice(&hash.to_le_bytes());
        entry[8..12].copy_from_slice(&uninterp.to_le_bytes());
        entry[12..16].copy_from_slice(&gamma.to_le_bytes());
        if !block.push(&entry) {
            output.pass(block);
            block = output.add();
            block.push(&entry);
        }
    }
    output.pass(block);
}

/// Hash + gamma weight pair used in interpolation backoff tables.
#[derive(Debug, Clone)]
pub struct HashGamma {
    pub gamma: f32,
    pub hash_value: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_vocab_from_indices() {
        let sv = SpecialVocab::from_indices(1, 2, 0);
        assert_eq!(sv.bos(), 1);
        assert_eq!(sv.eos(), 2);
        assert_eq!(sv.UNK(), 0);
        assert!(sv.is_special(0));
        assert!(sv.is_special(1));
        assert!(sv.is_special(2));
        assert!(!sv.is_special(3));
    }

    #[test]
    fn discount_get_clamps_to_1_3() {
        let d = Discount { amount: [0.0, 0.5, 1.0, 1.5] };
        assert_eq!(d.get(1), 0.5);
        assert_eq!(d.get(2), 1.0);
        assert_eq!(d.get(3), 1.5);
        assert_eq!(d.get(0), 0.5); // clamps to 1
        assert_eq!(d.get(10), 1.5); // clamps to 3
    }

    #[test]
    fn discount_apply_returns_count_minus_discount() {
        let d = Discount { amount: [0.0, 0.75, 0.75, 0.75] };
        assert!((d.apply(3) - 2.25).abs() < 1e-5);
        assert_eq!(d.apply(0), 0.0);
    }

    #[test]
    fn initial_probabilities_streaming_writes_output() {
        use crate::stream::chain::Chain;
        use crate::builder::config::InitialProbabilitiesConfig;

        let in_chain = Chain::new(4096, 16, 4);
        let out_chain = Chain::new(4096, 16, 4);

        // Write two (hash, count) entries into the input chain
        let mut block = in_chain.add();
        let mut entry = [0u8; 16];
        entry[0..8].copy_from_slice(&42u64.to_le_bytes());
        entry[8..16].copy_from_slice(&10u64.to_le_bytes());
        block.push(&entry);
        entry[0..8].copy_from_slice(&99u64.to_le_bytes());
        entry[8..16].copy_from_slice(&5u64.to_le_bytes());
        block.push(&entry);
        in_chain.pass(block);

        let config = InitialProbabilitiesConfig::default();
        let discounts = vec![Discount { amount: [0.0, 0.75, 0.75, 0.75] }];
        let sv = SpecialVocab::from_indices(1, 2, 0);
        initial_probabilities(&config, &discounts, &in_chain, &out_chain, &[], false, &sv);

        // Verify output block contains entries
        let out_block = out_chain.pop().expect("should have output block");
        let entries: Vec<&[u8]> = out_block.as_entries().collect();
        assert!(!entries.is_empty(), "should have written output entries");
        assert_eq!(entries.len(), 2);
        // Each entry: [hash:u64][uninterp:f32][gamma:f32] = 16 bytes
        let uninterp = f32::from_le_bytes(entries[0][8..12].try_into().unwrap());
        assert!(uninterp > 0.0, "uninterp prob should be positive");
    }
}
