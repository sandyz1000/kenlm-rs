/// Quantization support for KenLM models
///
/// Quantization reduces memory footprint by representing probabilities and backoffs
/// with fewer bits.

use crate::ngram::binary_format::BinaryFormat;
use crate::utils::bit_packing::{
    read_float32, read_int25, read_non_positive_float31, write_float32, write_int57,
    write_non_positive_float31, BitAddress,
};
use std::cmp::Ordering;

const KENLM_MAX_ORDER: usize = 6;
const K_EXTENSION_QUANT: u64 = 1;
const K_NO_EXTENSION_QUANT: u64 = 0;

/// Configuration for quantization
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub prob_bits: u8,
    pub backoff_bits: u8,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            prob_bits: 8,
            backoff_bits: 8,
        }
    }
}

/// No quantization - stores full float values
#[derive(Debug, Default, Clone)]
pub struct DontQuantize;

impl DontQuantize {
    pub const MODEL_TYPE_ADD: u8 = 0;

    pub fn update_config_from_binary(
        _file: &BinaryFormat,
        _offset: u64,
        _config: &mut QuantConfig,
    ) {
        // No configuration needed for unquantized models
    }

    pub fn size(_order: u8, _config: &QuantConfig) -> u64 {
        0 // No quantization tables needed
    }

    pub fn middle_bits(_config: &QuantConfig) -> u8 {
        63 // Full 63-bit probability + 32-bit backoff
    }

    pub fn longest_bits(_config: &QuantConfig) -> u8 {
        31 // Full 31-bit probability
    }

    pub fn setup_memory(&mut self, _start: &mut [u8], _order: u8, _config: &QuantConfig) {
        // No setup needed
    }

    pub fn train(&mut self, _order: u8, _prob: &mut Vec<f32>, _backoff: &mut Vec<f32>) {
        // No training needed for unquantized
    }

    pub fn train_prob(&mut self, _order: u8, _prob: &mut Vec<f32>) {
        // No training needed
    }

    pub fn finished_loading(&mut self, _config: &QuantConfig) {
        // Nothing to finalize
    }
}

/// Pointer for middle n-grams (unquantized)
#[derive(Debug, Clone)]
pub struct DontQuantizeMiddlePointer {
    address: BitAddress,
}

impl DontQuantizeMiddlePointer {
    pub fn new(_quant: &DontQuantize, _order_minus_2: u8, address: BitAddress) -> Self {
        Self { address }
    }

    pub fn found(&self) -> bool {
        !self.address.base.is_empty()
    }

    pub fn prob(&self) -> f32 {
        read_non_positive_float31(&self.address.base, self.address.offset)
    }

    pub fn backoff(&self) -> f32 {
        read_float32(&self.address.base, self.address.offset + 31)
    }

    pub fn rest(&self) -> f32 {
        self.prob()
    }

    pub fn write(&mut self, prob: f32, backoff: f32) {
        write_non_positive_float31(&mut self.address.base, self.address.offset, prob);
        write_float32(&mut self.address.base, self.address.offset + 31, backoff);
    }
}

/// Pointer for longest n-grams (unquantized)
#[derive(Debug, Clone)]
pub struct DontQuantizeLongestPointer {
    address: BitAddress,
}

impl DontQuantizeLongestPointer {
    pub fn new(_quant: &DontQuantize, address: BitAddress) -> Self {
        Self { address }
    }

    pub fn found(&self) -> bool {
        !self.address.base.is_empty()
    }

    pub fn prob(&self) -> f32 {
        read_non_positive_float31(&self.address.base, self.address.offset)
    }

    pub fn write(&mut self, prob: f32) {
        write_non_positive_float31(&mut self.address.base, self.address.offset, prob);
    }
}

/// Quantization bins for probability and backoff values
#[derive(Debug, Clone)]
pub struct Bins {
    begin: Vec<f32>,
    bits: u8,
    mask: u64,
}

impl Bins {
    fn new(bits: u8, begin: Vec<f32>) -> Self {
        let mask = if bits > 0 { (1u64 << bits) - 1 } else { 0 };
        Self { begin, bits, mask }
    }

    pub fn populate(&mut self) -> &mut Vec<f32> {
        &mut self.begin
    }

    pub fn encode_prob(&self, value: f32) -> u64 {
        self.encode(value, 0)
    }

    pub fn encode_backoff(&self, value: f32) -> u64 {
        if value == 0.0 {
            if self.has_extension(value) {
                K_EXTENSION_QUANT
            } else {
                K_NO_EXTENSION_QUANT
            }
        } else {
            self.encode(value, 2)
        }
    }

    pub fn decode(&self, off: usize) -> f32 {
        if off < self.begin.len() {
            self.begin[off]
        } else {
            0.0
        }
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn mask(&self) -> u64 {
        self.mask
    }

    fn encode(&self, value: f32, reserved: usize) -> u64 {
        if self.begin.is_empty() {
            return 0;
        }

        // lower_bound: first element >= value
        let above = self.begin[reserved..]
            .binary_search_by(|probe| probe.partial_cmp(&value).unwrap_or(Ordering::Equal));

        match above {
            Ok(index) | Err(index) if index == 0 => reserved as u64,
            Ok(index) | Err(index) if index == self.begin.len() - reserved => {
                (self.begin.len() - 1) as u64
            }
            Ok(index) | Err(index) => {
                let index = index + reserved;
                // Return closer quantization bin
                if value - self.begin[index - 1] < self.begin[index] - value {
                    index as u64 - 1
                } else {
                    index as u64
                }
            }
        }
    }

    fn has_extension(&self, _value: f32) -> bool {
        // TODO: Implement extension detection
        false
    }
}

/// Separately quantize probability and backoff values
#[derive(Debug, Clone)]
pub struct SeparatelyQuantize {
    tables: Vec<[Bins; 2]>,
    longest: Bins,
    actual_base: Vec<u8>,
    prob_bits: u8,
    backoff_bits: u8,
}

impl Default for SeparatelyQuantize {
    fn default() -> Self {
        Self::new()
    }
}

impl SeparatelyQuantize {
    pub const MODEL_TYPE_ADD: u8 = 1;
    const VERSION: u8 = 2;

    pub fn new() -> Self {
        let mut tables = Vec::new();
        for _ in 0..KENLM_MAX_ORDER - 1 {
            tables.push([Bins::new(0, Vec::new()), Bins::new(0, Vec::new())]);
        }

        Self {
            tables,
            longest: Bins::new(0, Vec::new()),
            actual_base: Vec::new(),
            prob_bits: 0,
            backoff_bits: 0,
        }
    }

    pub fn update_config_from_binary(
        file: &BinaryFormat,
        offset: u64,
        config: &mut QuantConfig,
    ) {
        // Read 3 bytes: version, prob_bits, backoff_bits
        let mut buffer = [0u8; 3];

        if file.read_for_config(&mut buffer, offset).is_ok() {
            let version = buffer[0];
            if version != Self::VERSION {
                panic!(
                    "Quantization version mismatch: file has {}, code expects {}",
                    version,
                    Self::VERSION
                );
            }
            config.prob_bits = buffer[1];
            config.backoff_bits = buffer[2];
        }
    }

    pub fn setup_memory(&mut self, start: &mut [u8], order: u8, config: &QuantConfig) {
        self.prob_bits = config.prob_bits;
        self.backoff_bits = config.backoff_bits;

        if config.prob_bits == 0 {
            panic!("Cannot quantize probability to zero bits");
        }
        if config.backoff_bits == 0 {
            panic!("Cannot quantize backoff to zero bits");
        }
        if config.prob_bits > 25 {
            panic!(
                "Quantizing probability supports at most 25 bits. Currently {} bits requested.",
                config.prob_bits
            );
        }
        if config.backoff_bits > 25 {
            panic!(
                "Quantizing backoff supports at most 25 bits. Currently {} bits requested.",
                config.backoff_bits
            );
        }

        self.actual_base = start.to_vec();

        // Setup tables for each order
        for i in 0..order as usize - 2 {
            // Allocate vectors for bins
            self.tables[i][0] = Bins::new(config.prob_bits, vec![0.0; 1 << config.prob_bits]);
            self.tables[i][1] = Bins::new(config.backoff_bits, vec![0.0; 1 << config.backoff_bits]);
        }

        self.longest = Bins::new(config.prob_bits, vec![0.0; 1 << config.prob_bits]);
    }

    pub fn size(order: u8, config: &QuantConfig) -> u64 {
        let longest_table = (1u64 << config.prob_bits as u64) * std::mem::size_of::<f32>() as u64;
        let middle_table =
            (1u64 << config.backoff_bits as u64) * std::mem::size_of::<f32>() as u64
                + longest_table;
        (order as u64 - 2) * middle_table + longest_table + 8
    }

    pub fn middle_bits(config: &QuantConfig) -> u8 {
        config.prob_bits + config.backoff_bits
    }

    pub fn longest_bits(config: &QuantConfig) -> u8 {
        config.prob_bits
    }

    pub fn get_tables(&self, order_minus_2: usize) -> &[Bins; 2] {
        &self.tables[order_minus_2]
    }

    pub fn longest_table(&self) -> &Bins {
        &self.longest
    }

    pub fn train(&mut self, order: u8, mut prob: Vec<f32>, mut backoff: Vec<f32>) {
        self.train_prob(order, &mut prob);

        let centers = self.tables[order as usize - 2][1].populate();
        // Reserve first 2 slots for special backoff values
        if centers.len() >= 2 {
            centers[0] = crate::constant::K_NO_EXTENSION_BACKOFF;
            centers[1] = crate::constant::K_EXTENSION_BACKOFF;
        }
        make_bins(&mut backoff, &mut centers[2..], (1 << self.backoff_bits) - 2);
    }

    pub fn train_prob(&mut self, order: u8, prob: &mut Vec<f32>) {
        let centers = self.tables[order as usize - 2][0].populate();
        make_bins(prob, centers, 1 << self.prob_bits);
    }

    pub fn finished_loading(&mut self, config: &QuantConfig) {
        if self.actual_base.len() >= 3 {
            self.actual_base[0] = Self::VERSION;
            self.actual_base[1] = config.prob_bits;
            self.actual_base[2] = config.backoff_bits;
        }
    }
}

/// Pointer for middle n-grams (quantized)
#[derive(Debug, Clone)]
pub struct QuantizedMiddlePointer<'a> {
    bins: &'a [Bins; 2],
    address: BitAddress,
}

impl<'a> QuantizedMiddlePointer<'a> {
    pub fn new(quant: &'a SeparatelyQuantize, order_minus_2: u8, address: BitAddress) -> Self {
        Self {
            bins: quant.get_tables(order_minus_2 as usize),
            address,
        }
    }

    pub fn found(&self) -> bool {
        !self.address.base.is_empty()
    }

    pub fn prob(&self) -> f32 {
        let offset = self.address.offset + self.backoff_bins().bits() as u64;
        let encoded = read_int25(
            &self.address.base,
            offset,
            self.prob_bins().bits(),
            self.prob_bins().mask() as u32,
        );
        self.prob_bins().decode(encoded as usize)
    }

    pub fn backoff(&self) -> f32 {
        let encoded = read_int25(
            &self.address.base,
            self.address.offset,
            self.backoff_bins().bits(),
            self.backoff_bins().mask() as u32,
        );
        self.backoff_bins().decode(encoded as usize)
    }

    pub fn rest(&self) -> f32 {
        self.prob()
    }

    pub fn write(&mut self, prob: f32, backoff: f32) {
        let prob_encoded = self.prob_bins().encode_prob(prob);
        let backoff_encoded = self.backoff_bins().encode_backoff(backoff);
        let backoff_bits = self.backoff_bins().bits();
        let prob_bits = self.prob_bins().bits();
        let total_bits = prob_bits + backoff_bits;

        #[cfg(target_endian = "little")]
        let encoded = (prob_encoded << backoff_bits) | backoff_encoded;

        #[cfg(target_endian = "big")]
        let encoded = (backoff_encoded << prob_bits) | prob_encoded;

        write_int57(
            &mut self.address.base,
            self.address.offset,
            total_bits,
            encoded,
        );
    }

    fn prob_bins(&self) -> &Bins {
        &self.bins[0]
    }

    fn backoff_bins(&self) -> &Bins {
        &self.bins[1]
    }
}

/// Pointer for longest n-grams (quantized)
#[derive(Debug, Clone)]
pub struct QuantizedLongestPointer<'a> {
    table: &'a Bins,
    address: BitAddress,
}

impl<'a> QuantizedLongestPointer<'a> {
    pub fn new(quant: &'a SeparatelyQuantize, address: BitAddress) -> Self {
        Self {
            table: quant.longest_table(),
            address,
        }
    }

    pub fn found(&self) -> bool {
        !self.address.base.is_empty()
    }

    pub fn prob(&self) -> f32 {
        let encoded = read_int25(
            &self.address.base,
            self.address.offset,
            self.table.bits(),
            self.table.mask() as u32,
        );
        self.table.decode(encoded as usize)
    }

    pub fn write(&mut self, prob: f32) {
        let encoded = self.table.encode_prob(prob);
        write_int57(
            &mut self.address.base,
            self.address.offset,
            self.table.bits(),
            encoded,
        );
    }
}

/// Create quantization bins from values
fn make_bins(values: &mut Vec<f32>, centers: &mut [f32], bins: u32) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut start = 0;

    for i in 0..bins as usize {
        if i >= centers.len() {
            break;
        }

        let finish = ((values.len() * (i + 1)) as u64 / bins as u64) as usize;
        if finish == start {
            centers[i] = if i > 0 {
                centers[i - 1]
            } else {
                f32::NEG_INFINITY
            };
        } else {
            centers[i] = values[start..finish].iter().sum::<f32>() / (finish - start) as f32;
        }
        start = finish;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dont_quantize() {
        let _dq = DontQuantize;
        assert_eq!(DontQuantize::MODEL_TYPE_ADD, 0);
        assert_eq!(DontQuantize::middle_bits(&QuantConfig::default()), 63);
    }

    #[test]
    fn test_bins_creation() {
        let bins = Bins::new(8, vec![0.0, 0.5, 1.0]);
        assert_eq!(bins.bits(), 8);
        assert_eq!(bins.mask(), 255);
    }

    #[test]
    fn test_separately_quantize() {
        let sq = SeparatelyQuantize::new();
        assert_eq!(sq.prob_bits, 0);
        assert_eq!(sq.backoff_bits, 0);
    }

    #[test]
    fn test_make_bins() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut centers = vec![0.0; 3];
        make_bins(&mut values, &mut centers, 3);
        assert!(centers[0] < centers[1]);
        assert!(centers[1] < centers[2]);
    }
}
