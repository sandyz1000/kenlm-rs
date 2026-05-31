use std::collections::HashMap;
use std::vec::Vec;

use crate::builder::proba::{HashGamma, SpecialVocab};
use crate::common::ordering::ChainPositions;

struct ProbBackoff {
    prob: f32,
    backoff: f32,
}

trait GramOutput {
    fn gram(&mut self, order_minus_1: usize, full_backoff: f32, out: &mut ProbBackoff);
}

impl GramOutput for OutputQ {
    fn gram(&mut self, order_minus_1: usize, full_backoff: f32, out: &mut ProbBackoff) {
        OutputQ::gram(self, order_minus_1, full_backoff, out);
    }
}

impl GramOutput for OutputProbBackoff {
    fn gram(&mut self, order_minus_1: usize, full_backoff: f32, out: &mut ProbBackoff) {
        OutputProbBackoff::gram(self, order_minus_1, full_backoff, out);
    }
}

struct OutputQ {
    q_delta: Vec<f32>,
}

impl OutputQ {
    pub fn new(order: usize) -> Self {
        Self {
            q_delta: vec![0.0; order],
        }
    }

    pub fn gram(&mut self, order_minus_1: usize, full_backoff: f32, out: &mut ProbBackoff) {
        if order_minus_1 > 0 {
            let prev_q = self.q_delta[order_minus_1 - 1];
            let q_del = &mut self.q_delta[order_minus_1];
            *q_del = (prev_q / out.backoff) * full_backoff;
        } else {
            self.q_delta[order_minus_1] = full_backoff;
        }
        let q_val = self.q_delta[order_minus_1];
        out.prob = (out.prob * q_val).log10();
        // TODO: stop wastefully outputting this!
        out.backoff = 0.0;
    }
}

struct OutputProbBackoff;

impl OutputProbBackoff {
    pub fn new(_order: usize) -> Self {
        Self {}
    }

    pub fn gram(&mut self, _order_minus_1: usize, full_backoff: f32, out: &mut ProbBackoff) {
        out.prob = out.prob.log10().min(0.0);
        out.backoff = full_backoff.log10();
    }
}

pub struct Callback<Output> {
    backoffs: Vec<HashMap<u64, HashGamma>>,
    probs: Vec<f32>,
    prune_thresholds: Vec<u64>,
    prune_vocab: bool,
    output: Output,
    specials: SpecialVocab,
}

impl<Output: GramOutput> Callback<Output> {
    pub fn new(
        uniform_prob: f32,
        backoffs: Vec<HashMap<u64, HashGamma>>,
        prune_thresholds: Vec<u64>,
        prune_vocab: bool,
        specials: SpecialVocab,
        output: Output,
    ) -> Self {
        let probs = vec![uniform_prob; backoffs.len() + 2];
        Self {
            backoffs,
            probs,
            prune_thresholds,
            prune_vocab,
            output,
            specials,
        }
    }

    pub fn enter(
        &mut self,
        order_minus_1: usize,
        ngram_ids: &[u32],
        uninterp_prob: f32,
        gamma: f32,
    ) {
        // Compute interpolated probability: p(w|ctx) = uninterp + gamma * p_lower
        let p_lower = if order_minus_1 > 0 { self.probs[order_minus_1] } else { 0.0 };
        let interp_prob = (uninterp_prob + gamma * p_lower).min(0.0);
        self.probs[order_minus_1 + 1] = interp_prob;

        // Compute backoff weight for this n-gram context (used by higher-order n-grams).
        // Skip BOS/EOS/UNK as context words — they don't have meaningful backoff weights.
        let out_backoff = if order_minus_1 < self.backoffs.len()
            && !ngram_ids.is_empty()
            && !self.specials.is_special(*ngram_ids.last().unwrap())
            && !self.backoffs[order_minus_1].is_empty()
        {
            if self.prune_vocab
                || self.prune_thresholds.get(order_minus_1 + 1).copied().unwrap_or(0) > 0
            {
                let hash = util_murmur_hash_native(ngram_ids);
                self.backoffs[order_minus_1]
                    .get(&hash)
                    .map(|hb| hb.gamma)
                    .unwrap_or(1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };

        let mut out = ProbBackoff { prob: interp_prob, backoff: out_backoff };
        self.output.gram(order_minus_1, out_backoff, &mut out);
    }
}

struct Interpolate {
    uniform_prob: f32,
    backoffs: Vec<HashMap<u64, HashGamma>>,
    // backoffs: ChainPositions,
    prune_thresholds: Vec<u64>,
    prune_vocab: bool,
    output_q: bool,
    specials: SpecialVocab,
}

impl Interpolate {
    pub fn new(
        vocab_size: u64,
        backoffs: Vec<HashMap<u64, HashGamma>>,
        prune_thresholds: Vec<u64>,
        prune_vocab: bool,
        output_q: bool,
        specials: SpecialVocab,
    ) -> Self {
        let uniform_prob = 1.0 / (vocab_size as f32);
        Self {
            uniform_prob,
            backoffs,
            prune_thresholds,
            prune_vocab,
            output_q,
            specials,
        }
    }

    pub fn run_with_chain(&self, _positions: &ChainPositions) {
        // Streaming chain-based interpolation — requires the full stream infrastructure.
        // Use build_arpa() in pipeline.rs for a self-contained alternative.
    }

    pub fn run(&self, positions: Vec<&[u32]>) {
        assert_eq!(positions.len(), self.backoffs.len() + 1);
        if self.output_q {
            let mut callback = Callback::new(
                self.uniform_prob,
                self.backoffs.clone(),
                self.prune_thresholds.clone(),
                self.prune_vocab,
                self.specials.clone(),
                OutputQ::new(positions.len()),
            );
            for (order_minus_1, position) in positions.iter().enumerate() {
                callback.enter(order_minus_1, position, 0.0, 0.0);
            }
        } else {
            let mut callback = Callback::new(
                self.uniform_prob,
                self.backoffs.clone(),
                self.prune_thresholds.clone(),
                self.prune_vocab,
                self.specials.clone(),
                OutputProbBackoff::new(positions.len()),
            );
            for (order_minus_1, position) in positions.iter().enumerate() {
                callback.enter(order_minus_1, position, 0.0, 0.0);
            }
        }
    }
}

/// Hash a slice of word-ids for backoff table lookup, matching C++ `util::MurmurHashNative`.
fn util_murmur_hash_native(data: &[u32]) -> u64 {
    let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
    crate::utils::hash::murmur_hash_64a(&bytes, 0)
}
