use std::cmp::min;
use std::fmt;

#[derive(Debug)]
pub struct Config {
    pub pointer_bhiksha_bits: u8,
}

#[derive(Debug)]
pub struct NodeRange {
    pub begin: u64,
    pub end: u64,
}

#[derive(Debug, Default)]
pub struct BitsMask {
    bits: u8,
    mask: u64,
}

mod util {
    pub fn required_bits(value: u64) -> u8 {
        64 - value.leading_zeros() as u8
    }

    pub fn read_int57(base: &[u8], bit_offset: u64, bits: u8, mask: u64) -> u64 {
        // Implement your logic here
        0
    }

    pub fn write_int57(base: &mut [u8], bit_offset: u64, bits: u8, value: u64) {
        // Implement your logic here
    }
}

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

    pub fn update_config_from_binary(_binary: &BinaryFormat, _offset: u64, _config: &mut Config) {}

    pub fn size(_max_offset: u64, _max_next: u64, _config: &Config) -> u64 {
        0
    }

    pub fn inline_bits(_max_offset: u64, max_next: u64, _config: &Config) -> u8 {
        util::required_bits(max_next)
    }

    pub fn read_next(&self, base: &[u8], bit_offset: u64, total_bits: u8, out: &mut NodeRange) {
        out.begin = util::read_int57(base, bit_offset, self.next.bits, self.next.mask);
        out.end = util::read_int57(base, bit_offset + total_bits as u64, self.next.bits, self.next.mask);
    }

    pub fn write_next(&self, base: &mut [u8], bit_offset: u64, value: u64) {
        util::write_int57(base, bit_offset, self.next.bits, value);
    }

    pub fn finished_loading(&self, _config: &Config) {}

    pub fn inline_bits(&self) -> u8 {
        self.next.bits
    }
}

impl BitsMask {
    pub fn by_max(max: u64) -> Self {
        let bits = util::required_bits(max);
        BitsMask {
            bits,
            mask: (1 << bits) - 1,
        }
    }

    pub fn by_bits(bits: u8) -> Self {
        BitsMask {
            bits,
            mask: (1 << bits) - 1,
        }
    }
}

pub struct ArrayBhiksha {
    next_inline: BitsMask,
    offset_begin: Vec<u64>,
    offset_end: Vec<u64>,
    write_to: Vec<u64>,
    original_base: Vec<u8>,
}

impl ArrayBhiksha {
    pub const K_MODEL_TYPE_ADD: u8 = 1;

    pub fn new(base: &[u8], max_offset: u64, max_value: u64, config: &Config) -> Self {
        let next_inline_bits = ArrayBhiksha::inline_bits(max_offset, max_value, config);
        let offset_begin = align_to_8(base).to_vec();
        let offset_end = offset_begin.clone();

        ArrayBhiksha {
            next_inline: BitsMask::by_bits(next_inline_bits),
            offset_begin,
            offset_end,
            write_to: vec![0; 1], // first entry is 0
            original_base: base.to_vec(),
        }
    }

    pub fn update_config_from_binary(file: &BinaryFormat, offset: u64, config: &mut Config) {
        let buffer = file.read_for_config(offset);
        let version = buffer[0];
        let configured_bits = buffer[1];
        if version != Self::K_MODEL_TYPE_ADD {
            panic!("This file has sorted array compression version {} but the code expects version {}",
                   version, Self::K_MODEL_TYPE_ADD);
        }
        config.pointer_bhiksha_bits = configured_bits;
    }

    pub fn size(max_offset: u64, max_next: u64, config: &Config) -> u64 {
        let array_count = array_count(max_offset, max_next, config);
        std::mem::size_of::<u64>() as u64 * (1 + array_count as u64) + 7 // 8-byte alignment
    }

    pub fn inline_bits(max_offset: u64, max_next: u64, config: &Config) -> u8 {
        util::required_bits(max_next) - chop_bits(max_offset, max_next, config)
    }

    pub fn read_next(&self, base: &[u8], bit_offset: u64, index: u64, total_bits: u8, out: &mut NodeRange) {
        let begin_it = self.offset_begin.iter().position(|&x| x > index).unwrap_or(0) - 1;
        let end_it = self.offset_end.iter().position(|&x| x > index + 1).unwrap_or(0) - 1;

        out.begin = ((begin_it as u64) << self.next_inline.bits) |
            util::read_int57(base, bit_offset, self.next_inline.bits, self.next_inline.mask);
        out.end = ((end_it as u64) << self.next_inline.bits) |
            util::read_int57(base, bit_offset + total_bits as u64, self.next_inline.bits, self.next_inline.mask);

        assert!(out.end >= out.begin);
    }

    pub fn write_next(&mut self, base: &mut [u8], bit_offset: u64, index: u64, value: u64) {
        let encode = value >> self.next_inline.bits;
        while self.write_to.len() <= self.offset_begin.len() + encode as usize {
            self.write_to.push(index);
        }
        util::write_int57(base, bit_offset, self.next_inline.bits, value & self.next_inline.mask);
    }

    pub fn finished_loading(&mut self, config: &Config) {
        self.write_to[0] = 0;

        if self.write_to != self.offset_end {
            panic!("Did not get all the array entries that were expected.");
        }

        let mut head_write = self.original_base.clone();
        head_write[0] = Self::K_MODEL_TYPE_ADD;
        head_write[1] = config.pointer_bhiksha_bits;
    }
}

fn chop_bits(max_offset: u64, max_next: u64, config: &Config) -> u8 {
    let required = util::required_bits(max_next);
    let mut best_chop = 0;
    let mut lowest_change = i64::MAX;

    for chop in 0..=min(required, config.pointer_bhiksha_bits) {
        let change = (max_next >> (required - chop)) as i64 * 64
            - max_offset as i64 * chop as i64;
        if change < lowest_change {
            lowest_change = change;
            best_chop = chop;
        }
    }
    best_chop
}

fn array_count(max_offset: u64, max_next: u64, config: &Config) -> usize {
    let required = util::required_bits(max_next);
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

impl BinaryFormat {
    pub fn read_for_config(&self, offset: u64) -> [u8; 2] {
        // Implement the logic to read the binary format
        [0; 2]
    }
}
