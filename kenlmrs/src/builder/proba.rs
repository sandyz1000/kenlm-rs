#![allow(unused)]

use std::ptr;
use std::slice;

use crate::common::ngram::NGram;

use super::{ BuildingPayload, WordIndex };
use crate::builder::config::InitialProbabilitiesConfig;

// Add missing constants
pub const K_UNK: usize = 0;

// Add missing types
#[derive(Debug, Clone)]
pub struct SpecialVocab;

impl SpecialVocab {
    pub fn is_special(&self, _word: usize) -> bool {
        false
    }
    pub fn bos(&self) -> usize {
        1
    }
    pub fn eos(&self) -> usize {
        2
    }
    pub fn UNK(&self) -> usize {
        0
    }
    pub fn EOS(&self) -> usize {
        2
    }
}

#[derive(Clone, Debug)]
pub struct Discount {
    pub amount: [f32; 4],
}

impl Discount {
    pub fn get(&self, _i: usize) -> f32 {
        0.0
    }

    pub fn apply(&self, _count: u64) -> f32 {
        0.0
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

// Function to compute initial (uninterpolated) probabilities
fn initial_probabilities(
    _config: &InitialProbabilitiesConfig,
    _discounts: &[Discount],
    _primary: &mut Chains,
    _second_in: &mut Chains,
    _gamma_out: &mut Chains,
    _prune_thresholds: &[u64],
    _prune_vocab: bool,
    _vocab: &SpecialVocab
) {
    todo!()
}

/// Hash + gamma weight pair used in interpolation backoff tables.
#[derive(Debug, Clone)]
pub struct HashGamma {
    pub gamma: f32,
    pub hash_value: u64,
}
