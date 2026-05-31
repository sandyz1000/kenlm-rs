use std::marker::PhantomData;

use crate::bhiksha::{ArrayBhiksha, DontBhiksha};
use crate::ngram::binary_format::BinaryFormat;
use crate::quantize::{DontQuantize, QuantConfig, SeparatelyQuantize};
use crate::types::{ModelType, ProbBackoff, WordIndex};
use crate::utils::bit_packing::BitAddress;

type Config = QuantConfig;

#[derive(Default, Clone, Copy)]
pub struct MiddlePointer {
    pub prob: f32,
    pub backoff: f32,
    pub found: bool,
}

impl MiddlePointer {
    pub fn new<Q: Quantization>(quant: &Q, order_minus_2: u8, address: BitAddress) -> Self {
        if address.base.is_empty() {
            return MiddlePointer::default();
        }
        let (prob, backoff) = quant.read_middle_entry(&address.base, address.offset, order_minus_2);
        MiddlePointer {
            prob,
            backoff,
            found: true,
        }
    }

    pub fn found(&self) -> bool {
        self.found
    }
    pub fn prob(&self) -> f32 {
        self.prob
    }
    pub fn backoff(&self) -> f32 {
        self.backoff
    }
}

#[derive(Default, Clone, Copy)]
pub struct LongestPointer {
    pub prob: f32,
    pub found: bool,
}

impl LongestPointer {
    pub fn new<Q: Quantization>(quant: &Q, address: BitAddress) -> Self {
        if address.base.is_empty() {
            return LongestPointer::default();
        }
        let prob = quant.read_longest_entry(&address.base, address.offset);
        LongestPointer { prob, found: true }
    }

    pub fn found(&self) -> bool {
        self.found
    }
    pub fn prob(&self) -> f32 {
        self.prob
    }
}

const TRIE_SORTED: u8 = 0;
const MODEL_TYPE: u8 = 1;

pub trait Quantization: Default {
    fn update_config_from_binary(file: &BinaryFormat, offset: u64, config: &mut Config);
    fn size(order: usize, config: &Config) -> u64;
    fn middle_bits(config: &Config) -> u8;
    fn longest_bits(config: &Config) -> u8;

    /// Prepare quantization tables before ARPA loading begins.
    fn setup_tables(&mut self, order: u8, config: &Config);

    /// Train middle-layer tables from collected prob/backoff values (called once per layer).
    fn train_middle(&mut self, order: u8, probs: Vec<f32>, backoffs: Vec<f32>);

    /// Train the longest-layer table from collected prob values.
    fn train_longest(&mut self, probs: Vec<f32>);

    /// Serialize quantization tables (no-op for DontQuantize).
    fn write_quant_binary<W: std::io::Write>(
        &self,
        w: &mut W,
        n_middles: usize,
    ) -> Result<(), crate::error::LMError>;

    /// Deserialize quantization tables into self (no-op for DontQuantize).
    fn load_quant_binary<R: std::io::Read>(
        &mut self,
        r: &mut R,
        n_middles: usize,
    ) -> Result<(), crate::error::LMError>;

    /// Write a (prob, backoff) pair into `base` at bit offset `bit_offset`.
    fn write_middle_entry(
        &self,
        base: &mut Vec<u8>,
        bit_offset: u64,
        order_minus_2: u8,
        prob: f32,
        backoff: f32,
    );

    /// Write a prob into `base` at bit offset `bit_offset` (longest layer, no backoff).
    fn write_longest_entry(&self, base: &mut Vec<u8>, bit_offset: u64, prob: f32);

    /// Read (prob, backoff) from `base` at bit offset `bit_offset`.
    fn read_middle_entry(&self, base: &[u8], bit_offset: u64, order_minus_2: u8) -> (f32, f32);

    /// Read prob from `base` at bit offset `bit_offset` (longest layer).
    fn read_longest_entry(&self, base: &[u8], bit_offset: u64) -> f32;
}

pub trait BhikshaImpl {
    fn update_config_from_binary(file: &BinaryFormat, offset: u64, config: &mut Config);
}

impl Quantization for DontQuantize {
    fn update_config_from_binary(_file: &BinaryFormat, _offset: u64, _config: &mut Config) {}
    fn size(_order: usize, _config: &Config) -> u64 {
        0
    }
    fn middle_bits(config: &Config) -> u8 {
        DontQuantize::middle_bits(config)
    }
    fn longest_bits(config: &Config) -> u8 {
        DontQuantize::longest_bits(config)
    }
    fn setup_tables(&mut self, _order: u8, _config: &Config) {}
    fn train_middle(&mut self, _order: u8, _probs: Vec<f32>, _backoffs: Vec<f32>) {}
    fn train_longest(&mut self, _probs: Vec<f32>) {}
    fn write_quant_binary<W: std::io::Write>(
        &self,
        _w: &mut W,
        _n_middles: usize,
    ) -> Result<(), crate::error::LMError> {
        Ok(())
    }
    fn load_quant_binary<R: std::io::Read>(
        &mut self,
        _r: &mut R,
        _n_middles: usize,
    ) -> Result<(), crate::error::LMError> {
        Ok(())
    }
    fn write_middle_entry(
        &self,
        base: &mut Vec<u8>,
        bit_offset: u64,
        _order_minus_2: u8,
        prob: f32,
        backoff: f32,
    ) {
        use crate::model::mark_independent;
        use crate::utils::bit_packing::{write_float32, write_non_positive_float31};
        write_non_positive_float31(base, bit_offset, mark_independent(prob));
        write_float32(base, bit_offset + 31, backoff);
    }
    fn write_longest_entry(&self, base: &mut Vec<u8>, bit_offset: u64, prob: f32) {
        use crate::model::mark_independent;
        use crate::utils::bit_packing::write_non_positive_float31;
        write_non_positive_float31(base, bit_offset, mark_independent(prob));
    }
    fn read_middle_entry(&self, base: &[u8], bit_offset: u64, _order_minus_2: u8) -> (f32, f32) {
        use crate::utils::bit_packing::{read_float32, read_non_positive_float31};
        (
            read_non_positive_float31(base, bit_offset),
            read_float32(base, bit_offset + 31),
        )
    }
    fn read_longest_entry(&self, base: &[u8], bit_offset: u64) -> f32 {
        use crate::utils::bit_packing::read_non_positive_float31;
        read_non_positive_float31(base, bit_offset)
    }
}

impl Quantization for SeparatelyQuantize {
    fn update_config_from_binary(_file: &BinaryFormat, _offset: u64, _config: &mut Config) {}
    fn size(order: usize, config: &Config) -> u64 {
        SeparatelyQuantize::size(order as u8, config)
    }
    fn middle_bits(config: &Config) -> u8 {
        SeparatelyQuantize::middle_bits(config)
    }
    fn longest_bits(config: &Config) -> u8 {
        SeparatelyQuantize::longest_bits(config)
    }
    fn setup_tables(&mut self, order: u8, config: &Config) {
        self.setup_tables_for_arpa(order, config);
    }
    fn train_middle(&mut self, order: u8, probs: Vec<f32>, backoffs: Vec<f32>) {
        self.train(order, probs, backoffs);
    }
    fn train_longest(&mut self, mut probs: Vec<f32>) {
        self.train_longest_probs(&mut probs);
    }
    fn write_middle_entry(
        &self,
        base: &mut Vec<u8>,
        bit_offset: u64,
        order_minus_2: u8,
        prob: f32,
        backoff: f32,
    ) {
        use crate::utils::bit_packing::write_int57;
        let tables = self.get_tables(order_minus_2 as usize);
        let backoff_bits = tables[1].bits();
        let prob_bits = tables[0].bits();
        let total_bits = prob_bits + backoff_bits;
        let encoded = self.encode_middle(order_minus_2 as usize, prob, backoff);
        write_int57(base, bit_offset, total_bits, encoded);
    }
    fn write_longest_entry(&self, base: &mut Vec<u8>, bit_offset: u64, prob: f32) {
        use crate::utils::bit_packing::write_int57;
        let prob_bits = self.prob_bits();
        let encoded = self.encode_longest_prob(prob);
        write_int57(base, bit_offset, prob_bits, encoded);
    }
    fn read_middle_entry(&self, base: &[u8], bit_offset: u64, order_minus_2: u8) -> (f32, f32) {
        use crate::utils::bit_packing::read_int57;
        let tables = self.get_tables(order_minus_2 as usize);
        let backoff_bits = tables[1].bits();
        let prob_bits = tables[0].bits();
        let total_bits = prob_bits + backoff_bits;
        let mask = if total_bits < 64 {
            (1u64 << total_bits) - 1
        } else {
            u64::MAX
        };
        let encoded = read_int57(base, bit_offset, total_bits, mask);
        self.decode_middle(order_minus_2 as usize, encoded)
    }
    fn read_longest_entry(&self, base: &[u8], bit_offset: u64) -> f32 {
        use crate::utils::bit_packing::read_int57;
        let prob_bits = self.prob_bits();
        let mask = if prob_bits < 64 {
            (1u64 << prob_bits) - 1
        } else {
            u64::MAX
        };
        let encoded = read_int57(base, bit_offset, prob_bits, mask);
        self.decode_longest_prob(encoded)
    }
    fn write_quant_binary<W: std::io::Write>(
        &self,
        w: &mut W,
        n_middles: usize,
    ) -> Result<(), crate::error::LMError> {
        use std::io::Write;
        w.write_all(&[self.prob_bits(), self.backoff_bits()])?;
        for i in 0..n_middles {
            let tables = self.get_tables(i);
            let prob_bins = tables[0].populate_ref();
            w.write_all(&(prob_bins.len() as u32).to_le_bytes())?;
            for &v in prob_bins {
                w.write_all(&v.to_le_bytes())?;
            }
            let backoff_bins = tables[1].populate_ref();
            w.write_all(&(backoff_bins.len() as u32).to_le_bytes())?;
            for &v in backoff_bins {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        let longest_bins = self.longest_table().populate_ref();
        w.write_all(&(longest_bins.len() as u32).to_le_bytes())?;
        for &v in longest_bins {
            w.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }
    fn load_quant_binary<R: std::io::Read>(
        &mut self,
        r: &mut R,
        n_middles: usize,
    ) -> Result<(), crate::error::LMError> {
        use std::io::Read;
        let mut buf2 = [0u8; 2];
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf2)?;
        let config = crate::quantize::QuantConfig {
            prob_bits: buf2[0],
            backoff_bits: buf2[1],
        };
        self.setup_tables_for_arpa((n_middles + 2) as u8, &config);
        for i in 0..n_middles {
            let tables = self.get_tables_mut(i);
            r.read_exact(&mut buf4)?;
            let n = u32::from_le_bytes(buf4) as usize;
            let mut bins = vec![0.0f32; n];
            for b in bins.iter_mut() {
                r.read_exact(&mut buf4)?;
                *b = f32::from_le_bytes(buf4);
            }
            *tables[0].populate() = bins;
            r.read_exact(&mut buf4)?;
            let n = u32::from_le_bytes(buf4) as usize;
            let mut bins = vec![0.0f32; n];
            for b in bins.iter_mut() {
                r.read_exact(&mut buf4)?;
                *b = f32::from_le_bytes(buf4);
            }
            *tables[1].populate() = bins;
        }
        r.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;
        let mut bins = vec![0.0f32; n];
        for b in bins.iter_mut() {
            r.read_exact(&mut buf4)?;
            *b = f32::from_le_bytes(buf4);
        }
        *self.longest_table_mut().populate() = bins;
        Ok(())
    }
}

impl BhikshaImpl for DontBhiksha {
    fn update_config_from_binary(_file: &BinaryFormat, _offset: u64, _config: &mut Config) {}
}

impl BhikshaImpl for ArrayBhiksha {
    fn update_config_from_binary(_file: &BinaryFormat, _offset: u64, _config: &mut Config) {}
}

/// Trie-based search structure.
///
/// The middle layers are stored as a `Vec<Middle>` (one entry per order between
/// unigrams and the longest order). `Vec` acts as the owning smart pointer here:
/// it manages the heap allocation and drops all elements automatically.
pub struct TrieSearch<Quant: Quantization, Bhiksha: BhikshaImpl> {
    /// One `Middle` per intermediate order (order 2 … max_order-1).
    middles: Vec<Middle>,
    quant: Quant,
    longest: Longest,
    unigram: Unigram,
    _phantom: PhantomData<Bhiksha>,
}

impl<Quant: Quantization + Default, Bhiksha: BhikshaImpl> Default for TrieSearch<Quant, Bhiksha> {
    fn default() -> Self {
        TrieSearch::new()
    }
}

impl<Quant: Quantization, Bhiksha: BhikshaImpl> TrieSearch<Quant, Bhiksha> {
    pub const K_DIFFERENT_REST: bool = false;
    pub const K_MODEL_TYPE: ModelType = ModelType::Trie;
    pub const K_VERSION: u8 = 1;

    pub fn new() -> Self {
        TrieSearch {
            middles: Vec::new(),
            quant: Default::default(),
            longest: Longest::new(),
            unigram: Unigram::new(),
            _phantom: PhantomData,
        }
    }

    pub fn update_config_from_binary(
        file: &BinaryFormat,
        counts: &[u64],
        offset: u64,
        config: &mut Config,
    ) {
        Quant::update_config_from_binary(file, offset, config);
        if counts.len() > 2 {
            Bhiksha::update_config_from_binary(
                file,
                offset + Quant::size(counts.len(), config) + Unigram::size(counts[0]),
                config,
            );
        }
    }

    pub fn size(counts: &[u64], config: &Config) -> u64 {
        let mut ret = Quant::size(counts.len(), config) + Unigram::size(counts[0]);
        for i in 1..counts.len() - 1 {
            ret += Middle::size(
                Quant::middle_bits(config),
                counts[i],
                counts[0],
                counts[i + 1],
                config,
            );
        }
        if !counts.is_empty() {
            ret + Longest::size(
                Quant::longest_bits(config),
                counts[counts.len() - 1],
                counts[0],
            )
        } else {
            ret
        }
    }

    /// Allocates owned storage for each trie layer based on the n-gram counts.
    pub fn setup_layers(&mut self, counts: &[u64], config: &Config) {
        let unigram_count = (counts[0] + 2) as usize;
        self.unigram
            .init(vec![UnigramValue::default(); unigram_count]);

        let quant_bits = Quant::middle_bits(config);
        let middle_count = counts.len().saturating_sub(2);
        self.middles = (0..middle_count)
            .map(|i| Middle::new(quant_bits, counts[i + 1], counts[0], counts[i + 2]))
            .collect();

        let longest_bits = Quant::longest_bits(config);
        self.longest = Longest::init_storage(longest_bits, *counts.last().unwrap(), counts[0]);
    }

    /// Read optional backoff value: `\t float \n` or `\n` → `K_NO_EXTENSION_BACKOFF`.
    fn read_backoff_fp(
        fp: &mut crate::utils::pieces::file::FilePiece,
    ) -> Result<f32, crate::error::LMError> {
        use crate::constant::K_NO_EXTENSION_BACKOFF;
        match fp.peek()? {
            '\t' => {
                fp.get()?;
                let bo = fp.read_float()?;
                let nl = fp.get()?;
                if nl != '\n' && nl != '\r' {
                    return Err(crate::error::LMError::InvalidArpa(
                        "Expected newline after backoff".into(),
                    ));
                }
                Ok(bo)
            }
            '\n' | '\r' => {
                fp.get()?;
                Ok(K_NO_EXTENSION_BACKOFF)
            }
            _ => Ok(K_NO_EXTENSION_BACKOFF),
        }
    }

    /// Load from an ARPA file. Signature matches the `Search` trait.
    pub fn initialize_from_arpa(
        &mut self,
        file: &str,
        counts: &[u64],
        _model_config: &crate::types::Config,
        vocab: &mut dyn crate::vocabulary::Vocabulary,
    ) -> Result<(), crate::error::LMError> {
        use crate::arpa_reader::{
            read_arpa_counts, read_end, read_ngram_header, PositiveProbWarn, ARPA_SPACES,
        };
        use crate::constant::{WarningAction, K_NO_EXTENSION_BACKOFF};
        use crate::model::{mark_independent, set_extension};
        use crate::utils::pieces::file::FilePiece;

        let quant_config = QuantConfig::default();
        let total_orders = counts.len();
        let warn = PositiveProbWarn::new(WarningAction::Complain);

        // === Phase 1: Register vocabulary and collect unigram (word_id, prob, backoff) ===
        let mut unigram_data: Vec<(u32, f32, f32)> = Vec::with_capacity(counts[0] as usize);
        {
            let mut fp = FilePiece::open(file)?;
            read_arpa_counts(&mut fp)?;
            read_ngram_header(&mut fp, 1)?;
            vocab.add_word("<unk>");
            vocab.add_word("<s>");
            vocab.add_word("</s>");
            for _ in 0..counts[0] {
                let raw_prob = fp.read_float()?;
                let raw_prob = if raw_prob > 0.0 {
                    warn.warn(raw_prob);
                    0.0f32
                } else {
                    raw_prob
                };
                fp.get()?; // consume \t
                let word_str = fp.read_delimited(&ARPA_SPACES)?;
                let word_id = vocab.add_word(&word_str);
                let backoff = Self::read_backoff_fp(&mut fp)?;
                unigram_data.push((word_id, raw_prob, backoff));
            }
        }

        // === Phase 2: Collect n-gram data for orders 2..N ===
        // ngram_data[k] = n-grams of order k+2; each entry: (ids, prob, backoff)
        let mut ngram_data: Vec<Vec<(Vec<u32>, f32, f32)>> = (0..total_orders.saturating_sub(1))
            .map(|_| Vec::new())
            .collect();

        if total_orders >= 2 {
            let mut fp = FilePiece::open(file)?;
            read_arpa_counts(&mut fp)?;
            // Skip unigram section
            read_ngram_header(&mut fp, 1)?;
            for _ in 0..counts[0] {
                fp.read_float()?;
                fp.get()?;
                fp.read_delimited(&ARPA_SPACES)?;
                Self::read_backoff_fp(&mut fp)?;
            }
            for order in 2..=total_orders {
                let n = counts[order - 1] as usize;
                read_ngram_header(&mut fp, order as u32)?;
                for _ in 0..n {
                    let raw_prob = fp.read_float()?;
                    let raw_prob = if raw_prob > 0.0 {
                        warn.warn(raw_prob);
                        0.0f32
                    } else {
                        raw_prob
                    };
                    fp.get()?; // consume \t
                    let mut ids = vec![0u32; order];
                    for id in ids.iter_mut() {
                        let w = fp.read_delimited(&ARPA_SPACES)?;
                        *id = vocab.index(&w);
                    }
                    let backoff = Self::read_backoff_fp(&mut fp)?;
                    ngram_data[order - 2].push((ids, raw_prob, backoff));
                }
            }
        }

        // === Phase 3: Sort in SUFFIX ORDER (C++ KenLM trie layout) ===
        // The trie is organized by new_word (last) as parent, context as child.
        // Unigram[w].next = range of bigrams where w is the NEW word (last word).
        // Within that range, bigrams are sorted by their context word (first word).
        // So sort key is REVERSED ids: (ids[n-1], ids[n-2], ..., ids[0]).
        for data in ngram_data.iter_mut() {
            data.sort_unstable_by(|(a, ..), (b, ..)| a.iter().rev().cmp(b.iter().rev()));
        }

        // === Phase 4: Setup trie layer storage ===
        self.setup_layers(counts, &quant_config);

        // === Phase 4b: Train quantization tables (no-op for DontQuantize) ===
        self.quant.setup_tables(total_orders as u8, &quant_config);
        for order in 2..total_orders {
            let layer = order - 2;
            let probs: Vec<f32> = ngram_data[layer].iter().map(|(_, p, _)| *p).collect();
            let backoffs: Vec<f32> = ngram_data[layer].iter().map(|(_, _, b)| *b).collect();
            self.quant.train_middle(order as u8, probs, backoffs);
        }
        if total_orders >= 2 {
            let probs: Vec<f32> = ngram_data[total_orders - 2]
                .iter()
                .map(|(_, p, _)| *p)
                .collect();
            self.quant.train_longest(probs);
        }

        // === Phase 5: Populate unigram weights ===
        for (word_id, prob, backoff) in &unigram_data {
            let idx = *word_id as usize;
            if idx < self.unigram.unigram.len() {
                self.unigram.unigram[idx].weights.prob = mark_independent(*prob);
                self.unigram.unigram[idx].weights.backoff = *backoff;
            }
        }

        // === Phase 6: Build trie layers with correct next pointers ===
        // For each layer, after sorting n-grams by path, we can walk unigrams/middle-entries
        // in index order and set next = start-of-children, then insert children in order.

        if total_orders == 1 {
            return Ok(()); // unigram-only model
        }

        // Helper: compute per-entry next-layer start indices (suffix-order).
        // Uses binary search so orphan entries (missing parent) don't corrupt the scan.
        // Invariant: next entries belonging to cur[ci] satisfy next[j].ids[1..] == cur[ci].ids,
        // which in reversed order means next[j]'s reversed ids start with reversed(cur[ci].ids).
        let compute_next_starts =
            |cur: &[(Vec<u32>, f32, f32)], next: &[(Vec<u32>, f32, f32)]| -> Vec<u64> {
                let mut starts = vec![next.len() as u64; cur.len() + 1];
                for (ci, (cids, ..)) in cur.iter().enumerate() {
                    // Find the first next-entry whose reversed ids >= reversed(cids).
                    // next is sorted by reversed ids, so partition_point works correctly.
                    let pos = next.partition_point(|(nids, ..)| {
                        nids[1..].iter().rev().cmp(cids.iter().rev()) == std::cmp::Ordering::Less
                    });
                    starts[ci] = pos as u64;
                }
                // starts[cur.len()] stays at next.len() (default sentinel).
                starts
            };

        // -- Bigrams (layer 0) --
        // Suffix-order layout: sorted by (ids[1], ids[0]) = (new_word, context_word).
        // unigram[w].next = start of bigrams where w is the NEW word (ids[1] == w).
        // Within each group, entries are sorted by context_word (ids[0]) for binary search.
        // BitPackedMiddle stores the CONTEXT WORD (ids[0]) as the key in each entry.
        {
            let bigrams = &ngram_data[0];
            // Set unigram.next: iterate by new word (ids[1])
            let mut bi = 0usize;
            for word in 0..self.unigram.unigram.len() {
                self.unigram.unigram[word].next = bi as u64;
                while bi < bigrams.len() && bigrams[bi].0[1] as usize == word {
                    bi += 1;
                }
            }
            // Compute bigram → next layer starts
            let next_layer_starts = if total_orders >= 3 {
                compute_next_starts(bigrams, &ngram_data[1])
            } else {
                vec![0u64; bigrams.len() + 1]
            };
            // Insert bigrams: key = context word (ids[0])
            for (bi, (ids, prob, backoff)) in bigrams.iter().enumerate() {
                let context_word = ids[0];
                let addr = self.middles[0].inner.insert(context_word);
                self.quant.write_middle_entry(
                    &mut self.middles[0].inner.bit_packed.base,
                    addr.offset,
                    0,
                    *prob,
                    *backoff,
                );
                self.middles[0]
                    .inner
                    .write_next(bi as u64, next_layer_starts[bi]);
                // Mark the NEW WORD's unigram backoff as extending (it has a bigram child)
                let new_word = ids[1] as usize;
                if new_word < self.unigram.unigram.len() {
                    set_extension(&mut self.unigram.unigram[new_word].weights.backoff);
                }
            }
            let sentinel = *next_layer_starts.last().unwrap_or(&0);
            self.middles[0].inner.finished_loading(sentinel);
        }

        // -- Middle orders 3..N-1 (layers 1..) --
        // Entry sorted by reversed ids: (ids[order-1], ids[order-2], ..., ids[0]).
        // Key inserted = ids[0] (oldest context word).
        for order in 3..total_orders {
            let layer = order - 2;
            let cur = &ngram_data[layer];
            let next_layer_starts = if order + 1 <= total_orders {
                compute_next_starts(cur, &ngram_data[layer + 1])
            } else {
                vec![0u64; cur.len() + 1]
            };
            for (ci, (ids, prob, backoff)) in cur.iter().enumerate() {
                let oldest_context = ids[0]; // key = oldest context word
                let addr = self.middles[layer].inner.insert(oldest_context);
                self.quant.write_middle_entry(
                    &mut self.middles[layer].inner.bit_packed.base,
                    addr.offset,
                    layer as u8,
                    *prob,
                    *backoff,
                );
                self.middles[layer]
                    .inner
                    .write_next(ci as u64, next_layer_starts[ci]);
            }
            let sentinel = *next_layer_starts.last().unwrap_or(&0);
            self.middles[layer].inner.finished_loading(sentinel);
        }

        // -- Longest layer --
        // Sorted by reversed ids: (ids[order-1], .., ids[0]).
        // Key inserted = ids[0] (oldest context word).
        {
            let longest_data = &ngram_data[total_orders - 2];
            for (ids, prob, _backoff) in longest_data.iter() {
                let oldest_context = ids[0]; // suffix-order: key = oldest context word
                let addr = self.longest.inner.insert(oldest_context);
                self.quant.write_longest_entry(
                    &mut self.longest.inner.bit_packed.base,
                    addr.offset,
                    *prob,
                );
            }
        }

        Ok(())
    }

    /// Number of n-gram orders this search covers.
    /// Middle layers = orders 2..(max_order-1); add 2 for unigrams + longest.
    pub fn order(&self) -> u8 {
        self.middles.len() as u8 + 2
    }

    pub fn unknown_unigram(&mut self) -> &mut ProbBackoff {
        self.unigram.unknown()
    }

    /// Serialize the trie to a writer.
    ///
    /// Layout (little-endian):
    ///   quant tables (if SeparatelyQuantize)
    ///   [unigram_count: u32][UnigramValue × n: prob:f32, backoff:f32, next:u64]
    ///   [n_middles: u8]
    ///   for each middle: [quant_bits:u8][next_bits:u8][max_vocab:u64][insert_index:u64]
    ///                    [buf_len:u64][bytes...]
    ///   [longest_quant_bits:u8][longest_max_vocab:u64][longest_insert_index:u64]
    ///   [buf_len:u64][bytes...]
    pub fn write_binary<W: std::io::Write>(&self, w: &mut W) -> Result<(), crate::error::LMError> {
        use std::io::Write;
        let n_middles = self.middles.len();
        // n_middles is written first so read_binary can pass it to load_quant_binary
        w.write_all(&[n_middles as u8])?;
        self.quant.write_quant_binary(w, n_middles)?;
        // Unigrams
        w.write_all(&(self.unigram.unigram.len() as u32).to_le_bytes())?;
        for v in &self.unigram.unigram {
            w.write_all(&v.weights.prob.to_le_bytes())?;
            w.write_all(&v.weights.backoff.to_le_bytes())?;
            w.write_all(&v.next.to_le_bytes())?;
        }
        // Middles
        for m in &self.middles {
            let bp = &m.inner.bit_packed;
            w.write_all(&[m.inner.quant_bits, m.inner.next_bits])?;
            w.write_all(&bp.max_vocab.to_le_bytes())?;
            w.write_all(&bp.insert_index.to_le_bytes())?;
            w.write_all(&(bp.base.len() as u64).to_le_bytes())?;
            w.write_all(&bp.base)?;
        }
        // Longest
        {
            let bp = &self.longest.inner.bit_packed;
            let quant_bits = if bp.total_bits >= bp.word_bits {
                bp.total_bits - bp.word_bits
            } else {
                0
            };
            w.write_all(&[quant_bits])?;
            w.write_all(&bp.max_vocab.to_le_bytes())?;
            w.write_all(&bp.insert_index.to_le_bytes())?;
            w.write_all(&(bp.base.len() as u64).to_le_bytes())?;
            w.write_all(&bp.base)?;
        }
        Ok(())
    }

    /// Deserialize the trie from a reader.
    pub fn read_binary<R: std::io::Read>(r: &mut R) -> Result<Self, crate::error::LMError> {
        use std::io::Read;
        let mut buf1 = [0u8; 1];
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Peek at n_middles first to pass to load_quant_binary — read after quant
        // We must read quant first. For DontQuantize, load_quant_binary is a no-op.
        // For SeparatelyQuantize, it reads prob_bits/backoff_bits + tables.
        // We read n_middles from the stream *after* quant headers; use a two-pass approach:
        // Actually quant binary is written first, so we read it first.
        // But we need n_middles to know how many middle tables to read in quant.
        // Solution: for SeparatelyQuantize, read n_middles temporarily from ahead.
        // Simpler: store n_middles BEFORE quant tables in write_binary.
        // But that would break the format. Instead: for DontQuantize no-op, for
        // SeparatelyQuantize we need n_middles. We store a leading u8 n_middles for quant.

        // Actually let's just re-order: write n_middles before quant tables.
        // Re-implementing: unigrams first (quant-independent), then n_middles, then quant, then layers.
        // But write_binary already wrote quant first. Let me fix write_binary to put n_middles before quant.
        // For now, since quant leading byte for SeparatelyQuantize starts with prob_bits+backoff_bits (2 bytes),
        // and DontQuantize writes nothing, we need n_middles before we can call load_quant_binary for SeparatelyQuantize.

        // REVISED FORMAT (consistent with write_binary which is also updated):
        // [n_middles: u8] FIRST, then quant, then unigrams, then layers.
        r.read_exact(&mut buf1)?;
        let n_middles = buf1[0] as usize;

        let mut search = Self::new();
        search.quant.load_quant_binary(r, n_middles)?;

        // Unigrams
        r.read_exact(&mut buf4)?;
        let n_uni = u32::from_le_bytes(buf4) as usize;
        let mut unigram_data = Vec::with_capacity(n_uni);
        for _ in 0..n_uni {
            r.read_exact(&mut buf4)?;
            let prob = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?;
            let backoff = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf8)?;
            let next = u64::from_le_bytes(buf8);
            unigram_data.push(UnigramValue {
                weights: ProbBackoff { prob, backoff },
                next,
            });
        }
        search.unigram.init(unigram_data);

        // Middles (n_middles was read at the top)
        for _ in 0..n_middles {
            r.read_exact(&mut buf1)?;
            let quant_bits = buf1[0];
            r.read_exact(&mut buf1)?;
            let next_bits = buf1[0];
            r.read_exact(&mut buf8)?;
            let max_vocab = u64::from_le_bytes(buf8);
            r.read_exact(&mut buf8)?;
            let insert_index = u64::from_le_bytes(buf8);
            r.read_exact(&mut buf8)?;
            let buf_len = u64::from_le_bytes(buf8) as usize;
            let mut base = vec![0u8; buf_len];
            r.read_exact(&mut base)?;
            // Reconstruct BitPackedMiddle from raw fields
            use crate::utils::bit_packing::required_bits;
            let next_mask = if next_bits > 0 {
                (1u64 << next_bits) - 1
            } else {
                0
            };
            let remaining_bits = quant_bits + next_bits;
            let mut bit_packed = BitPacked::new();
            bit_packed.base_init(base, max_vocab, remaining_bits);
            bit_packed.insert_index = insert_index;
            let inner = BitPackedMiddle {
                bit_packed,
                quant_bits,
                next_bits,
                next_mask,
            };
            search.middles.push(Middle { inner });
        }

        // Longest
        r.read_exact(&mut buf1)?;
        let quant_bits = buf1[0];
        r.read_exact(&mut buf8)?;
        let max_vocab = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let insert_index = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let buf_len = u64::from_le_bytes(buf8) as usize;
        let mut base = vec![0u8; buf_len];
        r.read_exact(&mut base)?;
        let mut bp = BitPacked::new();
        bp.base_init(base, max_vocab, quant_bits);
        bp.insert_index = insert_index;
        search.longest.inner.bit_packed = bp;

        Ok(search)
    }

    pub fn lookup_unigram(
        &self,
        word: WordIndex,
        next: &mut NodeRange,
        independent_left: &mut bool,
        extend_left: &mut u64,
    ) -> UnigramPointer {
        *extend_left = word as u64;
        let ret = self.unigram.find(word as usize, next);
        *independent_left = next.begin == next.end;
        ret
    }

    pub fn unpack(
        &self,
        extend_pointer: u64,
        extend_length: u8,
        node: &mut NodeRange,
    ) -> MiddlePointer {
        let idx = (extend_length - 2) as usize;
        let address = self.middles[idx].read_entry(extend_pointer, node);
        MiddlePointer::new(&self.quant, extend_length - 2, address)
    }

    pub fn lookup_middle(
        &self,
        order_minus_2: u8,
        word: WordIndex,
        node: &mut NodeRange,
        independent_left: &mut bool,
        extend_left: &mut u64,
    ) -> MiddlePointer {
        let address = self.middles[order_minus_2 as usize].find(word, node, extend_left);
        *independent_left = address.base.is_empty() || node.begin == node.end;
        MiddlePointer::new(&self.quant, order_minus_2, address)
    }

    pub fn lookup_longest(&self, word: WordIndex, node: &NodeRange) -> LongestPointer {
        LongestPointer::new(&self.quant, self.longest.find(word, node))
    }

    pub fn fast_make_node(
        &self,
        begin: &[WordIndex],
        end: &[WordIndex],
        node: &mut NodeRange,
    ) -> bool {
        assert!(!begin.is_empty());
        let mut independent_left = false;
        let mut ignored = 0;
        self.lookup_unigram(begin[0], node, &mut independent_left, &mut ignored);
        for (idx, word) in begin.iter().enumerate().skip(1) {
            if idx >= end.len() {
                break;
            }
            if independent_left
                || !self
                    .lookup_middle(idx as u8, *word, node, &mut independent_left, &mut ignored)
                    .found()
            {
                return false;
            }
        }
        true
    }
}

// No manual Drop needed: `middles: Vec<Middle>` drops all elements automatically.

pub struct Middle {
    pub(crate) inner: BitPackedMiddle,
}

impl Middle {
    pub fn new(quant_bits: u8, count: u64, max_vocab: u64, next_count: u64) -> Self {
        let size = BitPackedMiddle::size(quant_bits, count, max_vocab, next_count) as usize;
        let buffer = vec![0u8; size];
        Middle {
            inner: BitPackedMiddle::new(buffer, quant_bits, count, max_vocab, next_count),
        }
    }

    pub fn size(
        middle_bits: u8,
        count: u64,
        base_count: u64,
        next_count: u64,
        _config: &Config,
    ) -> u64 {
        BitPackedMiddle::size(middle_bits, count, base_count, next_count)
    }

    pub fn insert(&mut self, word: WordIndex) -> BitAddress {
        self.inner.insert(word)
    }

    pub fn write_next(&mut self, entry_index: u64, next_value: u64) {
        self.inner.write_next(entry_index, next_value)
    }

    pub fn finished_loading(&mut self, next_end: u64) {
        self.inner.finished_loading(next_end)
    }

    pub fn insert_index(&self) -> u64 {
        self.inner.insert_index()
    }

    pub fn read_entry(&self, extend_pointer: u64, node: &mut NodeRange) -> BitAddress {
        self.inner.read_entry(extend_pointer, node)
    }

    pub fn find(&self, word: WordIndex, node: &mut NodeRange, extend_left: &mut u64) -> BitAddress {
        self.inner.find(word, node, extend_left)
    }
}

pub struct Longest {
    pub(crate) inner: BitPackedLongest,
}

impl Longest {
    pub fn new() -> Self {
        Longest {
            inner: BitPackedLongest::new(),
        }
    }

    pub fn init_storage(quant_bits: u8, count: u64, max_vocab: u64) -> Self {
        let size = BitPackedLongest::size(quant_bits, count, max_vocab) as usize;
        let buffer = vec![0u8; size];
        let mut inner = BitPackedLongest::new();
        inner.init(buffer, quant_bits, max_vocab);
        Longest { inner }
    }

    pub fn size(longest_bits: u8, count: u64, base_count: u64) -> u64 {
        BitPackedLongest::size(longest_bits, count, base_count)
    }

    pub fn insert(&mut self, word: WordIndex) -> BitAddress {
        self.inner.insert(word)
    }

    pub fn insert_index(&self) -> u64 {
        self.inner.bit_packed.insert_index
    }

    pub fn find(&self, word: WordIndex, node: &NodeRange) -> BitAddress {
        self.inner.find(word, node)
    }
}

impl<Quant: Quantization, Bhiksha: BhikshaImpl> TrieSearch<Quant, Bhiksha> {
    pub fn middle_bits(config: &Config) -> u8 {
        Quant::middle_bits(config)
    }

    pub fn longest_bits(config: &Config) -> u8 {
        Quant::longest_bits(config)
    }
}

#[derive(Default, Clone, Copy)]
pub struct NodeRange {
    pub begin: u64,
    pub end: u64,
}

// Definition of UnigramValue struct
#[derive(Default, Clone, Copy)]
struct UnigramValue {
    weights: ProbBackoff,
    next: u64,
}

impl UnigramValue {
    fn next(&self) -> u64 {
        self.next
    }
}

// Definition of UnigramPointer struct (owns its data — no lifetime)
#[derive(Default, Clone, Copy)]
pub struct UnigramPointer {
    pub prob: f32,
    pub backoff: f32,
    pub found: bool,
}

impl UnigramPointer {
    pub fn new(to: &ProbBackoff) -> Self {
        UnigramPointer {
            prob: to.prob,
            backoff: to.backoff,
            found: true,
        }
    }

    pub fn found(&self) -> bool {
        self.found
    }
    pub fn prob(&self) -> f32 {
        self.prob
    }
    pub fn backoff(&self) -> f32 {
        self.backoff
    }
    pub fn rest(&self) -> f32 {
        self.prob
    }
    pub fn independent_left(&self) -> bool {
        crate::model::is_independent_left(self.prob)
    }
}

// Definition of Unigram struct
struct Unigram {
    unigram: Vec<UnigramValue>,
}

impl Unigram {
    pub fn new() -> Self {
        Unigram {
            unigram: Vec::new(),
        }
    }

    pub fn init(&mut self, start: Vec<UnigramValue>) {
        self.unigram = start;
    }

    pub fn size(count: u64) -> u64 {
        (count + 2) * (std::mem::size_of::<UnigramValue>() as u64)
    }

    pub fn lookup(&self, index: usize) -> &ProbBackoff {
        &self.unigram[index].weights
    }

    pub fn unknown(&mut self) -> &mut ProbBackoff {
        &mut self.unigram[0].weights
    }

    pub fn raw(&mut self) -> &mut [UnigramValue] {
        &mut self.unigram
    }

    pub fn find(&self, word: usize, next: &mut NodeRange) -> UnigramPointer {
        if word >= self.unigram.len() {
            return UnigramPointer::default();
        }
        let val = &self.unigram[word];
        next.begin = val.next;
        next.end = self.unigram.get(word + 1).map_or(val.next, |v| v.next);
        UnigramPointer::new(&val.weights)
    }
}

// Definition of BitPacked struct
#[derive(Debug, Default)]
pub struct BitPacked {
    word_bits: u8,
    total_bits: u8,
    word_mask: u64,
    base: Vec<u8>,
    insert_index: u64,
    max_vocab: u64,
}

impl BitPacked {
    pub fn new() -> Self {
        BitPacked {
            word_bits: 0,
            total_bits: 0,
            word_mask: 0,
            base: Vec::new(),
            insert_index: 0,
            max_vocab: 0,
        }
    }

    pub fn insert_index(&self) -> u64 {
        self.insert_index
    }

    /// Calculate memory size needed for bit-packed storage
    ///
    /// # Arguments
    /// * `entries` - Number of n-gram entries to store
    /// * `max_vocab` - Maximum vocabulary size (determines bits needed per word)
    /// * `remaining_bits` - Additional bits per entry (for probabilities, next pointers, etc.)
    ///
    /// # Returns
    /// Size in bytes needed for the bit-packed array
    pub fn base_size(entries: u64, max_vocab: u64, remaining_bits: u8) -> u64 {
        use crate::utils::bit_packing::required_bits;

        let word_bits = required_bits(max_vocab);
        let total_bits = word_bits + remaining_bits;

        // +1 for extra entry at the end (for end pointer)
        // +7 to round up when converting bits to bytes
        // +8 for safety margin to prevent ReadInt57 from going out of bounds
        ((entries + 1) * (total_bits as u64)).div_ceil(8) + 8
    }

    /// Initialize bit-packed array
    ///
    /// # Arguments
    /// * `base` - Pre-allocated byte buffer (must be zeroed)
    /// * `max_vocab` - Maximum vocabulary size
    /// * `remaining_bits` - Additional bits per entry beyond word index
    pub fn base_init(&mut self, base: Vec<u8>, max_vocab: u64, remaining_bits: u8) {
        use crate::utils::bit_packing::required_bits;

        // Calculate bits needed for word indices
        self.word_bits = required_bits(max_vocab);

        // Create mask for extracting word bits
        self.word_mask = if self.word_bits > 0 {
            (1u64 << self.word_bits) - 1
        } else {
            0
        };

        // Total bits = word bits + probability/next pointer bits
        self.total_bits = self.word_bits + remaining_bits;

        // Store the buffer
        self.base = base;

        // Start inserting at index 0
        self.insert_index = 0;

        // Store max vocab for validation
        self.max_vocab = max_vocab;
    }
}

// Definition of BitPackedMiddle struct
#[derive(Debug)]
struct BitPackedMiddle {
    bit_packed: BitPacked,
    quant_bits: u8,
    next_bits: u8,
    next_mask: u64,
}

impl BitPackedMiddle {
    pub fn insert_index(&self) -> u64 {
        self.bit_packed.insert_index
    }

    /// Create a new middle layer
    ///
    /// # Arguments
    /// * `base` - Pre-allocated zero-filled byte buffer
    /// * `quant_bits` - Bits for quantized probability/backoff
    /// * `entries` - Number of n-grams in this layer
    /// * `max_vocab` - Maximum vocabulary size (determines word bits)
    /// * `max_next` - Maximum next pointer value (determines next bits)
    pub fn new(base: Vec<u8>, quant_bits: u8, entries: u64, max_vocab: u64, max_next: u64) -> Self {
        use crate::utils::bit_packing::required_bits;

        // Calculate bits needed for next pointers
        let next_bits = required_bits(max_next);
        let next_mask = if next_bits > 0 {
            (1u64 << next_bits) - 1
        } else {
            0
        };

        // Total bits per entry = word_bits + quant_bits + next_bits
        let remaining_bits = quant_bits + next_bits;

        let mut bit_packed = BitPacked::new();
        bit_packed.base_init(base, max_vocab, remaining_bits);

        BitPackedMiddle {
            bit_packed,
            quant_bits,
            next_bits,
            next_mask,
        }
    }

    /// Calculate memory size for middle layer
    pub fn size(quant_bits: u8, entries: u64, max_vocab: u64, max_next: u64) -> u64 {
        use crate::utils::bit_packing::required_bits;
        let next_bits = required_bits(max_next);
        let remaining_bits = quant_bits + next_bits;
        BitPacked::base_size(entries, max_vocab, remaining_bits)
    }

    /// Insert a word during trie construction
    ///
    /// Returns the BitAddress where probability should be written
    /// The next pointer will be written by the caller after determining the next index
    pub fn insert(&mut self, word: WordIndex) -> BitAddress {
        use crate::utils::bit_packing::write_int57;

        assert!(
            (word as u64) <= self.bit_packed.word_mask,
            "Word index {} exceeds max vocab",
            word
        );

        // Calculate bit offset for this entry
        let at_pointer = self.bit_packed.insert_index * (self.bit_packed.total_bits as u64);

        // Write the word index
        write_int57(
            &mut self.bit_packed.base,
            at_pointer,
            self.bit_packed.word_bits,
            word as u64,
        );

        // Probability will be written at: at_pointer + word_bits
        let prob_offset = at_pointer + (self.bit_packed.word_bits as u64);

        // Next pointer will be at: prob_offset + quant_bits
        // (written later by write_next)

        // Increment insert index
        self.bit_packed.insert_index += 1;

        BitAddress::new(self.bit_packed.base.clone(), prob_offset)
    }

    /// Write next pointer for an entry
    ///
    /// Call this after insert() to set the next pointer value
    pub fn write_next(&mut self, entry_index: u64, next_value: u64) {
        use crate::utils::bit_packing::write_int57;

        let at_pointer = entry_index * (self.bit_packed.total_bits as u64);
        let next_offset =
            at_pointer + (self.bit_packed.word_bits as u64) + (self.quant_bits as u64);

        write_int57(
            &mut self.bit_packed.base,
            next_offset,
            self.next_bits,
            next_value,
        );
    }

    /// Finalize after all entries loaded
    pub fn finished_loading(&mut self, next_end: u64) {
        // Write the final next pointer (points past the end)
        if self.bit_packed.insert_index > 0 {
            self.write_next(self.bit_packed.insert_index, next_end);
        }
    }

    /// Binary search to find a word in the given range
    ///
    /// Returns BitAddress of probability and updates pointer/range for next layer
    pub fn find(&self, word: WordIndex, range: &mut NodeRange, pointer: &mut u64) -> BitAddress {
        use crate::utils::bit_packing::read_int57;

        // Binary search in range [range.begin, range.end)
        let mut left = range.begin;
        let mut right = range.end;

        while left < right {
            let mid = (left + right) / 2;

            // Read word at mid index
            let mid_offset = mid * (self.bit_packed.total_bits as u64);
            let mid_word = read_int57(
                &self.bit_packed.base,
                mid_offset,
                self.bit_packed.word_bits,
                self.bit_packed.word_mask,
            ) as WordIndex;

            if mid_word < word {
                left = mid + 1;
            } else if mid_word > word {
                right = mid;
            } else {
                // Found! Read the next pointer to determine range for next layer
                *pointer = mid;

                let prob_offset = mid_offset + (self.bit_packed.word_bits as u64);
                let next_offset = prob_offset + (self.quant_bits as u64);

                // Read this entry's next pointer
                let next_begin = read_int57(
                    &self.bit_packed.base,
                    next_offset,
                    self.next_bits,
                    self.next_mask,
                );

                // Read next entry's next pointer (for end of range)
                let next_entry_offset = (mid + 1) * (self.bit_packed.total_bits as u64);
                let next_entry_next_offset = next_entry_offset
                    + (self.bit_packed.word_bits as u64)
                    + (self.quant_bits as u64);
                let next_end = read_int57(
                    &self.bit_packed.base,
                    next_entry_next_offset,
                    self.next_bits,
                    self.next_mask,
                );

                // Update range for next layer lookup
                range.begin = next_begin;
                range.end = next_end;

                return BitAddress::new(self.bit_packed.base.clone(), prob_offset);
            }
        }

        // Not found - return null address
        BitAddress::new(Vec::new(), 0)
    }

    /// Read an entry by direct pointer access
    ///
    /// Used when we already know the index (from unpacking a pointer)
    pub fn read_entry(&self, pointer: u64, range: &mut NodeRange) -> BitAddress {
        use crate::utils::bit_packing::read_int57;

        let at_pointer = pointer * (self.bit_packed.total_bits as u64);
        let prob_offset = at_pointer + (self.bit_packed.word_bits as u64);
        let next_offset = prob_offset + (self.quant_bits as u64);

        // Read this entry's next pointer
        let next_begin = read_int57(
            &self.bit_packed.base,
            next_offset,
            self.next_bits,
            self.next_mask,
        );

        // Read next entry's next pointer (for end of range)
        let next_entry_offset = (pointer + 1) * (self.bit_packed.total_bits as u64);
        let next_entry_next_offset =
            next_entry_offset + (self.bit_packed.word_bits as u64) + (self.quant_bits as u64);
        let next_end = read_int57(
            &self.bit_packed.base,
            next_entry_next_offset,
            self.next_bits,
            self.next_mask,
        );

        // Update range
        range.begin = next_begin;
        range.end = next_end;

        BitAddress::new(self.bit_packed.base.clone(), prob_offset)
    }
}

// Definition of BitPackedLongest struct
pub struct BitPackedLongest {
    bit_packed: BitPacked,
}

impl BitPackedLongest {
    pub fn new() -> Self {
        BitPackedLongest {
            bit_packed: BitPacked::new(),
        }
    }

    /// Calculate memory size for longest n-gram layer
    pub fn size(quant_bits: u8, entries: u64, max_vocab: u64) -> u64 {
        BitPacked::base_size(entries, max_vocab, quant_bits)
    }

    /// Initialize the longest layer
    ///
    /// # Arguments
    /// * `base` - Pre-allocated zero-filled byte buffer
    /// * `quant_bits` - Bits needed for probability/quantization value
    /// * `max_vocab` - Maximum vocabulary size
    pub fn init(&mut self, base: Vec<u8>, quant_bits: u8, max_vocab: u64) {
        self.bit_packed.base_init(base, max_vocab, quant_bits);
    }

    /// Insert a word during trie construction
    /// Returns the BitAddress where the probability should be written
    pub fn insert(&mut self, word: WordIndex) -> BitAddress {
        use crate::utils::bit_packing::write_int57;

        assert!(
            (word as u64) <= self.bit_packed.word_mask,
            "Word index {} exceeds max vocab",
            word
        );

        // Calculate bit offset for this entry
        let at_pointer = self.bit_packed.insert_index * (self.bit_packed.total_bits as u64);

        // Write the word index
        write_int57(
            &mut self.bit_packed.base,
            at_pointer,
            self.bit_packed.word_bits,
            word as u64,
        );

        // Probability will be written at: at_pointer + word_bits
        let prob_offset = at_pointer + (self.bit_packed.word_bits as u64);

        // Increment insert index for next entry
        self.bit_packed.insert_index += 1;

        BitAddress::new(self.bit_packed.base.clone(), prob_offset)
    }

    /// Binary search to find a word in the given node range
    /// Returns BitAddress of the probability if found
    pub fn find(&self, word: WordIndex, node: &NodeRange) -> BitAddress {
        use crate::utils::bit_packing::read_int57;

        // Binary search in range [node.begin, node.end)
        let mut left = node.begin;
        let mut right = node.end;

        while left < right {
            let mid = (left + right) / 2;

            // Read word at mid index
            let mid_offset = mid * (self.bit_packed.total_bits as u64);
            let mid_word = read_int57(
                &self.bit_packed.base,
                mid_offset,
                self.bit_packed.word_bits,
                self.bit_packed.word_mask,
            ) as WordIndex;

            if mid_word < word {
                left = mid + 1;
            } else if mid_word > word {
                right = mid;
            } else {
                // Found! Return address of probability
                let prob_offset = mid_offset + (self.bit_packed.word_bits as u64);
                return BitAddress::new(self.bit_packed.base.clone(), prob_offset);
            }
        }

        // Not found - return null address
        BitAddress::new(Vec::new(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::bit_packing::{read_int57, required_bits, write_int57};

    #[test]
    fn test_bit_packed_base_size() {
        // Test size calculations for various configurations

        // Small vocab, few entries
        let size = BitPacked::base_size(100, 1000, 8);
        assert!(size > 0, "Size should be positive");

        // Large vocab
        let size_large = BitPacked::base_size(10000, 50000, 8);
        assert!(size_large > size, "Larger vocab should need more space");

        // More bits per entry
        let size_more_bits = BitPacked::base_size(100, 1000, 16);
        assert!(size_more_bits > size, "More bits should need more space");
    }

    #[test]
    fn test_bit_packed_base_init() {
        let mut bit_packed = BitPacked::new();
        let buffer = vec![0u8; 1000];

        bit_packed.base_init(buffer, 50000, 8);

        assert_eq!(bit_packed.word_bits, required_bits(50000));
        assert_eq!(bit_packed.total_bits, bit_packed.word_bits + 8);
        assert!(bit_packed.word_mask > 0);
        assert_eq!(bit_packed.insert_index, 0);
    }

    #[test]
    fn test_bit_packed_longest_insert_find() {
        let mut longest = BitPackedLongest::new();
        let buffer_size = BitPackedLongest::size(8, 100, 1000) as usize;
        let buffer = vec![0u8; buffer_size];

        longest.init(buffer, 8, 1000);

        // Insert some words
        let addr1 = longest.insert(100);
        let addr2 = longest.insert(200);
        let addr3 = longest.insert(300);
        let addr4 = longest.insert(500);

        assert!(!addr1.base.is_empty(), "Insert should return valid address");
        assert!(!addr2.base.is_empty(), "Insert should return valid address");
        assert!(!addr3.base.is_empty(), "Insert should return valid address");
        assert!(!addr4.base.is_empty(), "Insert should return valid address");

        // Search for inserted words
        let range = NodeRange { begin: 0, end: 4 };

        let found1 = longest.find(100, &range);
        assert!(!found1.base.is_empty(), "Should find word 100");

        let found2 = longest.find(200, &range);
        assert!(!found2.base.is_empty(), "Should find word 200");

        let found3 = longest.find(300, &range);
        assert!(!found3.base.is_empty(), "Should find word 300");

        let found4 = longest.find(500, &range);
        assert!(!found4.base.is_empty(), "Should find word 500");

        // Search for non-existent word
        let not_found = longest.find(150, &range);
        assert!(not_found.base.is_empty(), "Should not find word 150");
    }

    #[test]
    fn test_bit_packed_longest_empty_range() {
        let mut longest = BitPackedLongest::new();
        let buffer_size = BitPackedLongest::size(8, 10, 100) as usize;
        let buffer = vec![0u8; buffer_size];

        longest.init(buffer, 8, 100);

        // Insert one word
        longest.insert(50);

        // Search in empty range
        let empty_range = NodeRange { begin: 0, end: 0 };
        let not_found = longest.find(50, &empty_range);
        assert!(not_found.base.is_empty(), "Should not find in empty range");
    }

    #[test]
    fn test_bit_packed_longest_large_vocab() {
        let mut longest = BitPackedLongest::new();
        let max_vocab = 100_000;
        let buffer_size = BitPackedLongest::size(8, 1000, max_vocab) as usize;
        let buffer = vec![0u8; buffer_size];

        longest.init(buffer, 8, max_vocab);

        // Insert words with large indices
        longest.insert(10_000);
        longest.insert(50_000);
        longest.insert(99_999);

        let range = NodeRange { begin: 0, end: 3 };

        let found = longest.find(50_000, &range);
        assert!(!found.base.is_empty(), "Should find word with large index");
    }

    #[test]
    fn test_bit_packed_longest_binary_search_correctness() {
        let mut longest = BitPackedLongest::new();
        let buffer_size = BitPackedLongest::size(8, 100, 1000) as usize;
        let buffer = vec![0u8; buffer_size];

        longest.init(buffer, 8, 1000);

        // Insert sorted sequence
        for i in 0..50 {
            longest.insert(i * 10); // 0, 10, 20, 30, ...
        }

        let range = NodeRange { begin: 0, end: 50 };

        // Find all inserted values
        for i in 0..50 {
            let word = i * 10;
            let found = longest.find(word, &range);
            assert!(!found.base.is_empty(), "Should find word {}", word);
        }

        // Verify non-inserted values are not found
        for i in 0..50 {
            let word = i * 10 + 5; // 5, 15, 25, 35, ...
            let not_found = longest.find(word, &range);
            assert!(not_found.base.is_empty(), "Should not find word {}", word);
        }
    }

    #[test]
    fn test_bit_packed_middle_insert_find() {
        let max_vocab = 1000;
        let max_next = 5000;
        let entries = 100;

        let buffer_size = BitPackedMiddle::size(8, entries, max_vocab, max_next) as usize;
        let buffer = vec![0u8; buffer_size];

        let mut middle = BitPackedMiddle::new(buffer, 8, entries, max_vocab, max_next);

        // Insert some words
        let addr1 = middle.insert(100);
        middle.write_next(0, 0); // First entry's next pointer

        let addr2 = middle.insert(200);
        middle.write_next(1, 10); // Second entry's next pointer

        let addr3 = middle.insert(300);
        middle.write_next(2, 20); // Third entry's next pointer

        middle.finished_loading(30); // Final next pointer

        assert!(!addr1.base.is_empty(), "Insert should return valid address");
        assert!(!addr2.base.is_empty(), "Insert should return valid address");
        assert!(!addr3.base.is_empty(), "Insert should return valid address");

        // Search for word
        let mut range = NodeRange { begin: 0, end: 3 };
        let mut pointer = 0;

        let found = middle.find(200, &mut range, &mut pointer);
        assert!(!found.base.is_empty(), "Should find word 200");
        assert_eq!(pointer, 1, "Should return correct pointer");
        assert_eq!(range.begin, 10, "Should update range begin");
        assert_eq!(range.end, 20, "Should update range end");
    }

    #[test]
    fn test_bit_packed_middle_read_entry() {
        let max_vocab = 1000;
        let max_next = 5000;

        let buffer_size = BitPackedMiddle::size(8, 10, max_vocab, max_next) as usize;
        let buffer = vec![0u8; buffer_size];

        let mut middle = BitPackedMiddle::new(buffer, 8, 10, max_vocab, max_next);

        // Insert entries
        middle.insert(100);
        middle.write_next(0, 100);

        middle.insert(200);
        middle.write_next(1, 150);

        middle.finished_loading(200);

        // Read entry by pointer
        let mut range = NodeRange { begin: 0, end: 0 };
        let addr = middle.read_entry(1, &mut range);

        assert!(!addr.base.is_empty(), "Should return valid address");
        assert_eq!(range.begin, 150, "Should read correct next begin");
        assert_eq!(range.end, 200, "Should read correct next end");
    }

    #[test]
    fn test_bit_packed_middle_not_found() {
        let max_vocab = 1000;
        let max_next = 5000;

        let buffer_size = BitPackedMiddle::size(8, 10, max_vocab, max_next) as usize;
        let buffer = vec![0u8; buffer_size];

        let mut middle = BitPackedMiddle::new(buffer, 8, 10, max_vocab, max_next);

        middle.insert(100);
        middle.write_next(0, 0);
        middle.insert(300);
        middle.write_next(1, 10);
        middle.finished_loading(20);

        // Search for non-existent word
        let mut range = NodeRange { begin: 0, end: 2 };
        let mut pointer = 0;

        let not_found = middle.find(200, &mut range, &mut pointer);
        assert!(not_found.base.is_empty(), "Should not find word 200");
    }

    #[test]
    fn test_middle_size_calculation() {
        use crate::quantize::QuantConfig;
        let qc = QuantConfig::default();
        let size1 = Middle::size(8, 1000, 10000, 2000, &qc);
        assert!(size1 > 0, "Size should be positive");

        let size2 = Middle::size(8, 5000, 10000, 10000, &qc);
        assert!(size2 > size1, "More entries should need more space");

        let size3 = Middle::size(16, 1000, 10000, 2000, &qc);
        assert!(size3 > size1, "More quant bits should need more space");
    }

    #[test]
    fn test_longest_size_calculation() {
        let size1 = Longest::size(8, 1000, 10000);
        assert!(size1 > 0, "Size should be positive");

        let size2 = Longest::size(8, 5000, 10000);
        assert!(size2 > size1, "More entries should need more space");

        let size3 = Longest::size(16, 1000, 10000);
        assert!(size3 > size1, "More quant bits should need more space");
    }
}
