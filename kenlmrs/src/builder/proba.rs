#![allow(unused)]

use std::ptr;
use std::slice;

use crate::common::ngram::NGram;

use super::{BuildingPayload, WordIndex};
use crate::builder::config::InitialProbabilitiesConfig;

// Add missing constants
pub const K_UNK: usize = 0;

// Add missing types
#[derive(Debug)]
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
}

#[derive(Clone, Debug)]
pub struct Discount {
    pub amount: [f32; 4],
}

#[derive(Debug)]
pub struct PruneNGramStream;

impl PruneNGramStream {
    pub fn new(_primary: (), _specials: &SpecialVocab) -> Self {
        Self
    }
    pub fn begin(&self) -> usize {
        0
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
    pub fn new(position: &ChainPosition) -> Self {
        Self {}
    }
}

#[derive(Debug)]
pub struct Stream {
    // Define fields and methods here based on the actual implementation
}

impl Stream {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug)]
pub struct Discount {
    // Define fields here based on the actual implementation
}

#[derive(Debug)]
struct SpecialVocab {
    // Define fields here based on the actual implementation
}

// Function to compute initial (uninterpolated) probabilities
fn initial_probabilities(
    config: &InitialProbabilitiesConfig,
    discounts: &[Discount],
    primary: &mut Chains,
    second_in: &mut Chains,
    gamma_out: &mut Chains,
    prune_thresholds: &[u64],
    prune_vocab: bool,
    vocab: &SpecialVocab,
) {
    todo!()
}

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
        let entry_size = position.get_chain().entry_size();
        let current = NGram::new(ptr::null_mut(), NGram::order_from_size(entry_size));
        let dest = NGram::new(ptr::null_mut(), NGram::order_from_size(entry_size));
        let mut stream = PruneNGramStream {
            current,
            dest,
            current_count: 0,
            block: position.clone(),
            specials,
        };
        stream.start_block();
        stream
    }

    pub fn current(&self) -> &NGram<BuildingPayload> {
        &self.current
    }

    pub fn current_mut(&mut self) -> &mut NGram<BuildingPayload> {
        &mut self.current
    }
}

impl<'a> std::ops::Deref for PruneNGramStream<'a> {
    type Target = NGram<'a, BuildingPayload>;

    fn deref(&self) -> &Self::Target {
        &self.current
    }
}

impl<'a> std::ops::DerefMut for PruneNGramStream<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.current
    }
}

impl<'a> std::iter::Iterator for PruneNGramStream<'a> {
    type Item = NGram<'a, BuildingPayload>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.block.is_valid() {
            return None;
        }

        if self.current.order() == 1 && self.specials.is_special(self.current.begin()) {
            self.dest.next_in_memory();
        } else if self.current_count > 0 {
            if self.dest.base() < self.current.base() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        self.current.base(),
                        self.dest.base(),
                        self.current.total_size(),
                    );
                }
            }
            self.dest.next_in_memory();
        }

        self.current.next_in_memory();

        let block_base = self.block.get() as *const u8;
        if self.current.base() == unsafe { block_base.add(self.block.valid_size()) } {
            self.block
                .set_valid_size(self.dest.base() as usize - block_base as usize);
            self.block.advance();
            self.start_block();
            if self.block.is_valid() {
                self.current_count = self.current.value().cutoff_count();
            }
        } else {
            self.current_count = self.current.value().cutoff_count();
        }

        Some(self.current.clone())
    }
}

impl<'a> PruneNGramStream<'a> {
    fn start_block(&mut self) {
        while self.block.is_valid() {
            if self.block.valid_size() > 0 {
                break;
            }
            self.block.advance();
        }

        if self.block.is_valid() {
            self.current.rebase(self.block.get());
            self.current_count = self.current.value().cutoff_count();

            self.dest.rebase(self.block.get());
        }
    }
}

#[derive(Debug)]
pub struct HashBufferEntry {
    pub gamma: f32,
    pub hash_value: u64,
    // Additional fields as necessary
}

#[derive(Debug)]
pub struct HashGamma {
    pub gamma: f32,
    pub hash_value: u64,
}

#[derive(Debug)]
pub struct BufferEntry {
    pub denominator: f32,
    pub gamma: f32,
    // Additional fields as necessary
}

#[derive(Debug)]
pub struct Uninterpolated {
    prob: f32,
    gamma: f32,
}

#[derive(Debug)]
pub struct OnlyGamma {
    pruning: bool,
}

impl OnlyGamma {
    pub fn new(pruning: bool) -> Self {
        Self { pruning }
    }

    pub fn run(&self, position: &ChainPosition) {
        let mut block_it = Link::new(position);
        while block_it.get() != std::ptr::null_mut() {
            if self.pruning {
                let in_ptr = block_it.get() as *const HashBufferEntry;
                let end_ptr = block_it.valid_end() as *const HashBufferEntry;

                let mut out_ptr = block_it.get() as *mut HashGamma;

                let in_slice = unsafe {
                    std::slice::from_raw_parts(in_ptr, end_ptr.offset_from(in_ptr) as usize)
                };
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr, end_ptr.offset_from(in_ptr) as usize)
                };

                for (in_entry, out_entry) in in_slice.iter().zip(out_slice.iter_mut()) {
                    out_entry.gamma = in_entry.gamma;
                    out_entry.hash_value = in_entry.hash_value;
                }

                block_it.set_valid_size(
                    (block_it.valid_size() * std::mem::size_of::<HashGamma>())
                        / std::mem::size_of::<HashBufferEntry>(),
                );
            } else {
                let out_ptr = block_it.get() as *mut f32;
                let in_ptr = out_ptr;
                let end_ptr = block_it.valid_end() as *const f32;

                let in_slice = unsafe {
                    std::slice::from_raw_parts(in_ptr, end_ptr.offset_from(in_ptr) as usize / 2)
                };
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_ptr,
                        end_ptr.offset_from(in_ptr) as usize / 2,
                    )
                };

                for (i, out_entry) in out_slice.iter_mut().enumerate() {
                    *out_entry = in_slice[i * 2 + 1];
                }

                block_it.set_valid_size(block_it.valid_size() / 2);
            }
        }
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
        let mut in_stream = NGramStream::<BuildingPayload>::new(self.input);
        let mut out_stream = Stream::new(output);

        let mut previous = vec![0; in_stream.order() - 1];
        let previous_raw = if previous.is_empty() {
            std::ptr::null()
        } else {
            previous.as_ptr() as *const std::ffi::c_void
        };

        let size = std::mem::size_of::<WordIndex>() * previous.len();

        while in_stream.has_next() {
            out_stream.next();
            previous.copy_from_slice(in_stream.begin());

            let mut denominator = 0;
            let mut normalizer = 0;
            let mut counts = [0u64; 4];

            loop {
                let value = in_stream.value();
                denominator += value.unmarked_count();
                normalizer += value.unmarked_count() - value.cutoff_count();

                if value.cutoff_count() > 0 {
                    counts[std::cmp::min(value.cutoff_count() as usize, 3)] += 1;
                }

                if !in_stream.next() || previous != in_stream.begin() {
                    break;
                }
            }

            let entry = out_stream.get() as *mut BufferEntry;
            entry.denominator = denominator as f32;
            entry.gamma = 0.0;

            for i in 1..=3 {
                entry.gamma += self.discount.get(i) * counts[i] as f32;
            }

            entry.gamma += normalizer as f32;
            entry.gamma /= entry.denominator;

            if self.pruning {
                let hash_entry = entry as *mut HashBufferEntry;
                hash_entry.hash_value = util::murmur_hash_native(previous_raw, size);
            }
        }

        out_stream.poison();
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
        let mut summed = Stream::new(&self.from_adder);
        let mut grams = PruneNGramStream::new(primary, self.specials);

        if grams.order() == 1 {
            let sums = summed.get() as *const BufferEntry;

            assert_eq!(grams.begin(), K_UNK);

            let gamma_assign = if self.interpolate_unigrams {
                sums.gamma
            } else {
                0.0
            };

            grams.value().uninterp.prob = 0.0;

            while grams.next() {
                if grams.begin() == specials.bos() {
                    break;
                }

                grams.value().uninterp.prob =
                    self.discount.apply(grams.value().count) / sums.denominator;
                grams.value().uninterp.gamma = gamma_assign;
            }

            assert_eq!(grams.begin(), specials.bos());
            grams.value().uninterp.prob = 1.0;
            grams.value().uninterp.gamma = 0.0;

            while grams.next() {
                grams.value().uninterp.prob =
                    self.discount.apply(grams.value().count) / sums.denominator;
                grams.value().uninterp.gamma = gamma_assign;
            }

            summed.next();
            return;
        }

        let mut previous = vec![0; grams.order() - 1];
        let size = std::mem::size_of::<WordIndex>() * previous.len();

        while grams.has_next() {
            summed.next();
            previous.copy_from_slice(grams.begin());

            let sums = summed.get() as *const BufferEntry;

            loop {
                let pay = grams.value();
                pay.uninterp.prob = self.discount.apply(pay.unmarked_count()) / sums.denominator;
                pay.uninterp.gamma = sums.gamma;

                if !grams.next() || previous != grams.begin() {
                    break;
                }
            }
        }
    }
}
