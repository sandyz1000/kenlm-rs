use std::{convert::TryInto, fs::File};
use crate::common::NGram;

use super::Discount;


const KBOS: usize = 0;

#[derive(Debug, Clone)]
pub struct OrderStat {
    n: [u64; 5],
    count: u64,
    count_pruned: u64,
}

pub struct StatCollector<'a> {
    orders: Vec<OrderStat>,
    full: OrderStat,
    counts: &'a mut Vec<u64>,
    counts_pruned: &'a mut Vec<u64>,
    discounts: &'a mut Vec<Discount>,
}

#[derive(Clone, Default)]
pub struct DiscountConfig {
    // Overrides discounts for orders [1,discount_override.size()].
    overwrite: Vec<Discount>,
    // If discounting fails for an order, copy them from here.
    fallback: Discount,
    // What to do when discounts are out of range or would trigger division by zero.
    bad_action: WarningAction,
}

#[derive(Clone, Debug)]
pub enum WarningAction {
    ThrowUp,
    Complain,
    Silent,
}

impl<'a> StatCollector<'a> {
    fn new(
        order: usize,
        counts: &'a mut Vec<u64>,
        counts_pruned: &'a mut Vec<u64>,
        discounts: &'a mut Vec<Discount>,
    ) -> Self {
        let orders = vec![
            OrderStat {
                n: [0; 5],
                count: 0,
                count_pruned: 0
            };
            order
        ];
        let full = orders.last().unwrap().clone();
        Self {
            orders,
            full,
            counts,
            counts_pruned,
            discounts,
        }
    }

    fn calculate_discounts(&mut self, config: &DiscountConfig) {
        self.counts.resize(self.orders.len(), 0);
        self.counts_pruned.resize(self.orders.len(), 0);
        for i in 0..self.orders.len() {
            let s = &self.orders[i];
            self.counts[i] = s.count;
            self.counts_pruned[i] = s.count_pruned;
        }

        *self.discounts = config.overwrite.clone();
        self.discounts
            .resize(self.orders.len(), Discount { amount: [0.0; 4] });

        for i in config.overwrite.len()..self.orders.len() {
            let s = &self.orders[i];
            for j in 1..4 {
                let message = format!("BadDiscountException: Could not calculate Kneser-Ney discounts for {}-grams with adjusted count {} because we didn't observe any {}-grams with adjusted count {}; Is this small or artificial data?", i + 1, j + 1, i + 1, j);
                assert!(s.n[j] != 0, message);
            }

            let y = s.n[1] as f32 / (s.n[1] + 2.0 * s.n[2]) as f32;
            for j in 1..4 {
                self.discounts[i].amount[j] =
                    j as f32 - (j + 1) as f32 * y * s.n[j + 1] as f32 / s.n[j] as f32;
                assert!(self.discounts[i].amount[j] >= 0.0 && self.discounts[i].amount[j] <= j as f32, "BadDiscountException: ERROR: {}-gram discount out of range for adjusted count {}: {}. This means modified Kneser-Ney smoothing thinks something is weird about your data.", i + 1, j, self.discounts[i].amount[j]);
            }
        }
    }

    fn add(&mut self, order_minus_1: usize, count: u64, pruned: bool) {
        let stat = &mut self.orders[order_minus_1];
        stat.count += 1;
        if !pruned {
            stat.count_pruned += 1;
        }
        if count < 5 {
            stat.n[count as usize] += 1;
        }
    }

    fn add_full(&mut self, count: u64, pruned: bool) {
        self.full.count += 1;
        if !pruned {
            self.full.count_pruned += 1;
        }
        if count < 5 {
            self.full.n[count as usize] += 1;
        }
    }
}



struct CorpusCount<'a> {
    from: &'a mut File,
    vocab_write: i32,
    dynamic_vocab: bool,
    token_count: &'a mut u64,
    type_count: &'a mut WordIndex,
    prune_words: &'a mut Vec<bool>,
    prune_vocab_filename: String,
    dedupe_mem_size: usize,
    dedupe_mem: Vec<u8>, // Placeholder for util::scoped_malloc equivalent
    disallowed_symbol_action: WarningAction,
}

impl<'a> CorpusCount<'a> {
    fn dedupe_multiplier(order: usize) -> f32 {
        // Implementation of the dedupe_multiplier logic
        unimplemented!()
    }

    fn vocab_usage(vocab_estimate: usize) -> usize {
        // Implementation of the vocab_usage logic
        unimplemented!()
    }

    fn new(
        from: &'a mut File,
        vocab_write: i32,
        dynamic_vocab: bool,
        token_count: &'a mut u64,
        type_count: &'a mut WordIndex,
        prune_words: &'a mut Vec<bool>,
        prune_vocab_filename: &str,
        entries_per_block: usize,
        disallowed_symbol: WarningAction,
    ) -> Self {
        // Initialize other necessary fields as required
        let dedupe_mem_size = Self::dedupe_multiplier(entries_per_block) as usize;
        let dedupe_mem = vec![0; dedupe_mem_size];

        CorpusCount {
            from,
            vocab_write,
            dynamic_vocab,
            token_count,
            type_count,
            prune_words,
            prune_vocab_filename: prune_vocab_filename.to_string(),
            dedupe_mem_size,
            dedupe_mem,
            disallowed_symbol_action: disallowed_symbol,
        }
    }

    fn run(&mut self, position: &ChainPosition) {
        // Logic for the run method
        self.run_with_vocab(position, &mut Vec::new()); // Placeholder for vocab
    }

    fn run_with_vocab<Vocab>(&mut self, position: &ChainPosition, vocab: &mut Vocab) {
        // Logic for the run_with_vocab method
        unimplemented!()
    }
}

// Placeholder types for WordIndex and ChainPosition
pub type WordIndex = usize;

struct ChainPosition {
    // Fields for ChainPosition
}


#[derive(Debug, Clone)]
pub struct CombineCounts;

impl CombineCounts {
    fn combine(&self, first_void: *mut u8, second_void: *const u8, compare: &SuffixOrder) -> bool {
        let order = compare.order();
        let first_data = unsafe { slice::from_raw_parts_mut(first_void, order * size_of::<WordIndex>() + size_of::<BuildingPayload>()) };
        let second_data = unsafe { slice::from_raw_parts(second_void, order * size_of::<WordIndex>() + size_of::<BuildingPayload>()) };
        
        let mut first = NGram::new(first_data, order);
        let second = NGram::new(second_data, order);

        if first.begin() != second.begin() {
            return false;
        }
        
        first.value_mut().count += second.value().count;
        true
    }
}
