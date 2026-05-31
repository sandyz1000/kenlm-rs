use crate::common::ordering::SuffixOrder;
use crate::constant::WarningAction;
use crate::types::WordIndex;
use std::fs::File;

// Add missing types
#[derive(Clone, Debug, Default)]
pub struct Discount {
    pub amount: [f32; 4],
}

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

#[derive(Clone, Default, Debug)]
pub struct DiscountConfig {
    // Overrides discounts for orders [1,discount_override.size()].
    overwrite: Vec<Discount>,
    // If discounting fails for an order, copy them from here.
    fallback: Discount,
    // What to do when discounts are out of range or would trigger division by zero.
    bad_action: WarningAction,
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
                let message = format!(
                    "BadDiscountException: Could not calculate Kneser-Ney discounts for {}-grams with adjusted count {} because we didn't observe any {}-grams with adjusted count {}; Is this small or artificial data?",
                    i + 1,
                    j + 1,
                    i + 1,
                    j
                );
                assert!(s.n[j] != 0, "{}", message);
            }

            let y = (s.n[1] as f32) / ((s.n[1] as f32) + 2.0 * (s.n[2] as f32));
            for j in 1..4 {
                self.discounts[i].amount[j] =
                    (j as f32) - (((j + 1) as f32) * y * (s.n[j + 1] as f32)) / (s.n[j] as f32);
                assert!(
                    self.discounts[i].amount[j] >= 0.0 && self.discounts[i].amount[j] <= (j as f32),
                    "BadDiscountException: ERROR: {}-gram discount out of range for adjusted count {}: {}. This means modified Kneser-Ney smoothing thinks something is weird about your data.",
                    i + 1,
                    j,
                    self.discounts[i].amount[j]
                );
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
    fn dedupe_multiplier(_order: usize) -> f32 {
        // Placeholder implementation
        1.0
    }

    fn vocab_usage(_vocab_estimate: usize) -> usize {
        // Placeholder implementation
        0
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

    fn run(&mut self, _position: &crate::stream::chain::ChainPosition) {
        // Streaming run is implemented via count_ngrams_to_chain(); this stub is kept for API compatibility.
    }

    fn run_with_vocab<Vocab>(
        &mut self,
        _position: &crate::stream::chain::ChainPosition,
        _vocab: &mut Vocab,
    ) {
    }
}

/// Read a plain-text corpus and write counted n-grams to a chain.
///
/// Each chain entry is 16 bytes: `[hash: u64][count: u64]`.
/// N-grams are hashed via murmur so the downstream sort/dedup stage can work without
/// storing the actual word strings. Returns total token count.
///
/// This is the streaming entry point for CorpusCount: it replaces the C-style `run()` stub.
pub fn count_ngrams_to_chain(
    input_path: &str,
    order: usize,
    chain: &crate::stream::chain::Chain,
) -> Result<u64, crate::error::LMError> {
    use crate::utils::pieces::file::FilePiece;
    use std::collections::HashMap;

    let mut fp = FilePiece::open(input_path)?;
    let delims = {
        let mut d = [false; 256];
        d[b' ' as usize] = true;
        d[b'\t' as usize] = true;
        d[b'\n' as usize] = true;
        d[b'\r' as usize] = true;
        d
    };

    let mut counts: HashMap<Vec<u32>, u64> = HashMap::new();
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<s>".to_string(), 1);
    vocab.insert("</s>".to_string(), 2);
    let mut next_id = 3u32;
    let mut tokens: Vec<u32> = Vec::new();
    let mut total_tokens = 0u64;

    while !fp.at_end() {
        let line = match fp.read_line('\n', true) {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.is_empty() {
            continue;
        }
        tokens.clear();
        tokens.push(1); // <s>
        for word in line.split_whitespace() {
            let id = *vocab.entry(word.to_string()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            tokens.push(id);
            total_tokens += 1;
        }
        tokens.push(2); // </s>

        // Count n-grams of each order
        for start in 0..tokens.len() {
            for len in 1..=order.min(tokens.len() - start) {
                let ngram = tokens[start..start + len].to_vec();
                *counts.entry(ngram).or_insert(0) += 1;
            }
        }
    }

    // Write counted n-grams to chain blocks as (hash, count) pairs
    const ENTRY_BYTES: usize = 16;
    let mut block = chain.add();
    for (ngram, count) in &counts {
        let hash = simple_hash(ngram);
        let mut entry = [0u8; ENTRY_BYTES];
        entry[0..8].copy_from_slice(&hash.to_le_bytes());
        entry[8..16].copy_from_slice(&count.to_le_bytes());
        if !block.push(&entry) {
            chain.pass(block);
            block = chain.add();
            block.push(&entry);
        }
    }
    chain.pass(block);
    Ok(total_tokens)
}

fn simple_hash(ngram: &[u32]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &w in ngram {
        for byte in w.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(1099511628211);
        }
    }
    h
}
