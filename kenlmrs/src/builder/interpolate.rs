
use std::collections::HashMap;
use std::f32::consts::LOG10_E;
use std::vec::Vec;

use crate::common::NGram;

struct ProbBackoff {
    prob: f32,
    backoff: f32,
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
        let q_del = &mut self.q_delta[order_minus_1];
        if order_minus_1 > 0 {
            *q_del = self.q_delta[order_minus_1 - 1] / out.backoff * full_backoff;
        } else {
            *q_del = full_backoff;
        }
        out.prob = (out.prob * *q_del).log10();
        // TODO: stop wastefully outputting this!
        out.backoff = 0.0;
    }
}

struct OutputProbBackoff;

impl OutputProbBackoff {
    pub fn new(_order: usize) -> Self {
        Self {}
    }

    pub fn gram(&self, _order_minus_1: usize, full_backoff: f32, out: &mut ProbBackoff) {
        // Correcting for numerical precision issues. Take that IRST.
        out.prob = (out.prob).log10().min(0.0);
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

impl<Output> Callback<Output> {
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

    pub fn enter(&mut self, order_minus_1: usize, data: &[u32]) {
        let gram = NGram::new(data, order_minus_1 + 1);
        let mut pay = gram.value().clone();
        pay.complete.prob = pay.uninterp.prob + pay.uninterp.gamma * self.probs[order_minus_1];
        self.probs[order_minus_1 + 1] = pay.complete.prob;

        let out_backoff = if order_minus_1 < self.backoffs.len()
            && gram.end().last() != Some(&self.specials.UNK())
            && gram.end().last() != Some(&self.specials.EOS())
            && !self.backoffs[order_minus_1].is_empty()
        {
            if self.prune_vocab || self.prune_thresholds[order_minus_1 + 1] > 0 {
                let current_hash = util_murmur_hash_native(gram.begin());
                if let Some(hashed_backoff) = self.backoffs[order_minus_1].get(&current_hash) {
                    hashed_backoff.gamma
                } else {
                    1.0
                }
            } else {
                1.0
            }
        } else {
            1.0
        };

        self.output.gram(order_minus_1, out_backoff, &mut pay.complete);
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
        let uniform_prob = 1.0 / vocab_size as f32;
        Self {
            uniform_prob,
            backoffs,
            prune_thresholds,
            prune_vocab,
            output_q,
            specials,
        }
    }

    pub fn run_with_chain(&self, positions: &ChainPositions) {
        todo!()
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
                callback.enter(order_minus_1, position);
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
                callback.enter(order_minus_1, position);
            }
        }
    }
}

// This is a placeholder for the actual MurmurHash implementation
fn util_murmur_hash_native(data: &[u32]) -> u64 {
    // Implement the MurmurHash function here
    unimplemented!()
}
