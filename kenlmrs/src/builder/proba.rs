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

// NOTE: The following structs are currently unimplemented stubs
// They will need proper implementation based on the C++ KenLM code

/* Commented out until NGram supports required methods

#[derive(Debug, Clone)]
pub struct PruneNGramStream<'a> {
    current: NGram<'a, BuildingPayload>,
    dest: NGram<'a, BuildingPayload>,
    current_count: u64,
    block: ChainPosition,
    specials: &'a SpecialVocab,
}

impl<'a> PruneNGramStream<'a> {
    pub fn new(position: &'a ChainPosition, specials: &'a SpecialVocab) -> Self {
        todo!("PruneNGramStream needs NGram pointer-based methods")
    }
}
*/

#[derive(Debug)]
pub struct HashBufferEntry {
    pub gamma: f32,
    pub hash_value: u64,
}

#[derive(Debug, Clone)]
pub struct HashGamma {
    pub gamma: f32,
    pub hash_value: u64,
}

#[derive(Debug)]
pub struct BufferEntry {
    pub denominator: f32,
    pub gamma: f32,
}

#[derive(Debug)]
pub struct Uninterpolated {
    pub prob: f32,
    pub gamma: f32,
}

/* Commented out until stream infrastructure is implemented

#[derive(Debug)]
pub struct OnlyGamma {
    pruning: bool,
}

impl OnlyGamma {
    pub fn new(pruning: bool) -> Self {
        Self { pruning }
    }

    pub fn run(&self, position: &ChainPosition) {
        todo!("OnlyGamma needs stream infrastructure")
    }
}

#[derive(Debug)]
pub struct AddRight<'a> {
    discount: &'a Discount,
    input: ChainPosition,
    pruning: bool,
}

impl<'a> AddRight<'a> {
    pub fn new(discount: &'a Discount, input: ChainPosition, pruning: bool) -> Self {
        Self {
            discount,
            input,
            pruning,
        }
    }

    pub fn run(&self, output: &ChainPosition) {
        todo!("AddRight needs stream infrastructure")
    }
}

#[derive(Debug)]
pub struct MergeRight<'a> {
    interpolate_unigrams: bool,
    from_adder: ChainPosition,
    discount: &'a Discount,
    specials: &'a SpecialVocab,
}

impl<'a> MergeRight<'a> {
    pub fn new(
        interpolate_unigrams: bool,
        from_adder: ChainPosition,
        discount: &'a Discount,
        specials: &'a SpecialVocab,
    ) -> Self {
        Self {
            interpolate_unigrams,
            from_adder,
            discount,
            specials,
        }
    }

    pub fn run(&self, primary: &ChainPosition) {
        todo!("MergeRight needs stream infrastructure")
    }
}

*/
