/// Bhiksha compression implementation
///
/// Simple implementation of Bhiksha Raj and Ed Whittaker's compression scheme
/// from "Lossless Compression of Language Model Structure and Word Identifiers"
/// IEEE ICASSP 2003, pages 388-391.
///
/// Currently only used for next pointers in the trie.
use crate::utils::bit_packing::{read_int57, required_bits, write_int57, BitsMask};
use std::cmp::min;

/// Placeholder for BinaryFormat (actual implementation is in ngram::binary_format)
pub struct BinaryFormat;

impl BinaryFormat {
    pub fn read_for_config(&self, _offset: u64) -> [u8; 2] {
        [0; 2]
    }
}

/// Configuration for Bhiksha compression
#[derive(Debug, Clone)]
pub struct BhikshaConfig {
    pub pointer_bhiksha_bits: u8,
}

impl Default for BhikshaConfig {
    fn default() -> Self {
        Self {
            pointer_bhiksha_bits: 0,
        }
    }
}

/// Node range for trie navigation
#[derive(Debug)]
pub struct NodeRange {
    pub begin: u64,
    pub end: u64,
}

/// No compression - stores pointers directly
#[derive(Debug)]
pub struct DontBhiksha {
    next: BitsMask,
}

impl DontBhiksha {
    pub const K_MODEL_TYPE_ADD: u8 = 0;

    pub fn new(max_next: u64) -> Self {
        DontBhiksha {
            next: BitsMask::by_max(max_next),
        }
    }

    pub fn update_config_from_binary(
        _binary: &BinaryFormat,
        _offset: u64,
        _config: &mut BhikshaConfig,
    ) {
        // No configuration needed for uncompressed
    }

    pub fn size(_max_offset: u64, _max_next: u64, _config: &BhikshaConfig) -> u64 {
        0
    }

    pub fn inline_bits(_max_offset: u64, max_next: u64, _config: &BhikshaConfig) -> u8 {
        required_bits(max_next)
    }

    pub fn read_next(&self, base: &[u8], bit_offset: u64, total_bits: u8, out: &mut NodeRange) {
        out.begin = read_int57(base, bit_offset, self.next.bits, self.next.mask);
        out.end = read_int57(
            base,
            bit_offset + total_bits as u64,
            self.next.bits,
            self.next.mask,
        );
    }

    pub fn write_next(&self, base: &mut [u8], bit_offset: u64, value: u64) {
        write_int57(base, bit_offset, self.next.bits, value);
    }

    pub fn finished_loading(&self, _config: &BhikshaConfig) {}

    pub fn get_inline_bits(&self) -> u8 {
        self.next.bits
    }
}

/// Array-based Bhiksha compression
#[derive(Debug)]
pub struct ArrayBhiksha {
    next_inline: BitsMask,
    offset_begin: Vec<u64>,
    offset_end: Vec<u64>,
    write_to: Vec<u64>,
    original_base: Vec<u8>,
}

impl ArrayBhiksha {
    pub const K_MODEL_TYPE_ADD: u8 = 1;

    pub fn new(base: &[u8], max_offset: u64, max_value: u64, config: &BhikshaConfig) -> Self {
        let next_inline_bits = ArrayBhiksha::inline_bits(max_offset, max_value, config);

        // Parse the aligned bytes as u64 values
        let aligned = align_to_8(base);
        let count = array_count(max_offset, max_value, config);
        let mut offset_begin = Vec::with_capacity(count);
        let mut offset_end = Vec::with_capacity(count);

        // Read u64 values from the aligned bytes
        for i in 0..count {
            let offset = i * 8;
            if offset + 8 <= aligned.len() {
                let bytes: [u8; 8] = aligned[offset..offset + 8].try_into().unwrap();
                let value = u64::from_le_bytes(bytes);
                offset_begin.push(value);
                offset_end.push(value);
            }
        }

        ArrayBhiksha {
            next_inline: BitsMask::by_bits(next_inline_bits),
            offset_begin,
            offset_end,
            write_to: vec![0; 1], // first entry is 0
            original_base: base.to_vec(),
        }
    }

    pub fn update_config_from_binary(file: &BinaryFormat, offset: u64, config: &mut BhikshaConfig) {
        let buffer = file.read_for_config(offset);
        let version = buffer[0];
        let configured_bits = buffer[1];
        if version != Self::K_MODEL_TYPE_ADD {
            panic!(
                "This file has sorted array compression version {} but the code expects version {}",
                version,
                Self::K_MODEL_TYPE_ADD
            );
        }
        config.pointer_bhiksha_bits = configured_bits;
    }

    pub fn size(max_offset: u64, max_next: u64, config: &BhikshaConfig) -> u64 {
        let array_count = array_count(max_offset, max_next, config);
        std::mem::size_of::<u64>() as u64 * (1 + array_count as u64) + 7 // 8-byte alignment
    }

    pub fn inline_bits(max_offset: u64, max_next: u64, config: &BhikshaConfig) -> u8 {
        required_bits(max_next) - chop_bits(max_offset, max_next, config)
    }

    pub fn read_next(
        &self,
        base: &[u8],
        bit_offset: u64,
        index: u64,
        total_bits: u8,
        out: &mut NodeRange,
    ) {
        // std::upper_bound returns the first element that is greater.
        // Want the last element that is <= to the index.
        let begin_it = self
            .offset_begin
            .iter()
            .position(|&x| x > index)
            .unwrap_or(self.offset_begin.len())
            .saturating_sub(1);

        let mut end_it = begin_it + 1;
        while end_it < self.offset_end.len() && self.offset_end[end_it] <= index + 1 {
            end_it += 1;
        }
        end_it = end_it.saturating_sub(1);

        out.begin = ((begin_it as u64) << self.next_inline.bits)
            | read_int57(
                base,
                bit_offset,
                self.next_inline.bits,
                self.next_inline.mask,
            );
        out.end = ((end_it as u64) << self.next_inline.bits)
            | read_int57(
                base,
                bit_offset + total_bits as u64,
                self.next_inline.bits,
                self.next_inline.mask,
            );

        assert!(out.end >= out.begin);
    }

    pub fn write_next(&mut self, base: &mut [u8], bit_offset: u64, index: u64, value: u64) {
        let encode = value >> self.next_inline.bits;
        while (self.write_to.len() as u64) <= encode {
            self.write_to.push(index);
        }
        write_int57(
            base,
            bit_offset,
            self.next_inline.bits,
            value & self.next_inline.mask,
        );
    }

    pub fn finished_loading(&mut self, config: &BhikshaConfig) {
        self.write_to[0] = 0;

        if self.write_to != self.offset_end {
            panic!("Did not get all the array entries that were expected.");
        }

        let mut head_write = self.original_base.clone();
        if head_write.len() >= 2 {
            head_write[0] = Self::K_MODEL_TYPE_ADD;
            head_write[1] = config.pointer_bhiksha_bits;
        }
    }
}

fn chop_bits(max_offset: u64, max_next: u64, config: &BhikshaConfig) -> u8 {
    let required = required_bits(max_next);
    let mut best_chop = 0;
    let mut lowest_change = i64::MAX;

    for chop in 0..=min(required, config.pointer_bhiksha_bits) {
        let change = (max_next >> (required - chop)) as i64 * 64 - max_offset as i64 * chop as i64;
        if change < lowest_change {
            lowest_change = change;
            best_chop = chop;
        }
    }
    best_chop
}

fn array_count(max_offset: u64, max_next: u64, config: &BhikshaConfig) -> usize {
    let required = required_bits(max_next);
    let chopping = chop_bits(max_offset, max_next, config);
    (max_next >> (required - chopping)) as usize + 1
}

fn align_to_8(from: &[u8]) -> &[u8] {
    let remainder = from.as_ptr() as usize % 8;
    if remainder == 0 {
        from
    } else {
        &from[(8 - remainder)..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dont_bhiksha() {
        let bhiksha = DontBhiksha::new(255);
        assert_eq!(bhiksha.get_inline_bits(), 8);
    }

    #[test]
    fn test_chop_bits() {
        let config = BhikshaConfig {
            pointer_bhiksha_bits: 8,
        };
        let chop = chop_bits(1000, 255, &config);
        assert!(chop <= 8);
    }
}
