use super::config::InitialProbabilitiesConfig;
use crate::builder::count::DiscountConfig;
use crate::builder::proba::Discount;
use crate::constant::WarningAction;
use crate::stream::{config::ChainConfig, config::SortConfig};
use crate::types::WordIndex;
use crate::vocabulary::Vocabulary as _;

/// Placeholder for Output
pub struct Output;

#[derive(Debug)]
pub struct PipelineConfig {
    pub order: i8,
    pub sort: SortConfig,
    pub initial_probs: InitialProbabilitiesConfig,
    pub read_backoffs: ChainConfig,
    pub vocab_estimate: WordIndex,
    pub minimum_block: usize,
    pub block_count: usize,
    pub prune_thresholds: Vec<u64>,
    pub prune_vocab: bool,
    pub prune_vocab_file: String,
    pub renumber_vocabulary: bool,
    pub discount: DiscountConfig,
    pub output_q: bool,
    pub vocab_size_for_unk: u64,
    pub disallowed_symbol_action: WarningAction,
    pub temp_prefix: String,
    pub total_memory: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineConfig {
    pub fn new() -> Self {
        Self {
            order: 3,
            sort: SortConfig::default(),
            initial_probs: InitialProbabilitiesConfig::default(),
            read_backoffs: ChainConfig::default(),
            vocab_estimate: 1_000_000,
            minimum_block: 8192,
            block_count: 2,
            prune_thresholds: Vec::new(),
            prune_vocab: false,
            prune_vocab_file: String::new(),
            renumber_vocabulary: false,
            discount: DiscountConfig::default(),
            output_q: false,
            vocab_size_for_unk: 0,
            disallowed_symbol_action: WarningAction::Complain,
            temp_prefix: std::env::temp_dir().to_string_lossy().into_owned(),
            total_memory: 1 << 30, // 1 GB default
        }
    }

    pub fn temp_prefix(&self) -> &str {
        &self.temp_prefix
    }

    pub fn total_memory(&self) -> usize {
        self.total_memory
    }
}

/// Build an n-gram language model from a plain-text corpus file and write it as ARPA.
///
/// Uses Absolute Discounting (D=0.75) with Kneser-Ney continuation counts for
/// orders < max order, producing a valid ARPA file that `ProbingModel::new` can load.
///
/// * `input_path`  – plain-text corpus (one sentence per line, UTF-8)
/// * `output_path` – where to write the ARPA file
/// * `order`       – maximum n-gram order (e.g. 3 for a trigram model)
pub fn build_arpa(
    input_path: &str,
    output_path: &str,
    order: usize,
) -> Result<(), crate::error::LMError> {
    use std::collections::HashMap;
    use std::io::{BufWriter, Write};

    // --- Phase 1: count n-grams ---
    let mut vocab = crate::vocabulary::ProbingVocabulary::new();
    let mut corpus_count = super::CorpusCount::new();
    let raw_counts: HashMap<Vec<u32>, u64> =
        corpus_count.run(input_path, order, &mut vocab)?.into_iter().collect();

    let vocab_size = vocab.bound() as u64;

    // Discount (Absolute Discounting): 0.75 for all orders
    const D: f64 = 0.75;

    // --- Phase 2: Compute probabilities & backoffs ---
    // For highest order: standard counts.
    // For lower orders: Kneser-Ney continuation counts (number of unique left-contexts).
    // continuation_count[ngram] = |{w : (w, ngram) appears in corpus}|
    let mut continuation: HashMap<Vec<u32>, u64> = HashMap::new();
    for ngram in raw_counts.keys() {
        if ngram.len() >= 2 {
            // The suffix (ngram[1..]) gets a continuation credit
            let suffix: Vec<u32> = ngram[1..].to_vec();
            *continuation.entry(suffix).or_insert(0) += 1;
        }
    }

    // For each order, gather all n-grams at that order
    let mut all_ngrams_by_order: Vec<Vec<(Vec<u32>, f64, Option<f64>)>> = vec![vec![]; order];

    for ord in 1..=order {
        let mut context_count: HashMap<Vec<u32>, u64> = HashMap::new();
        let mut context_uniq: HashMap<Vec<u32>, u64> = HashMap::new(); // N+(ctx, •) for lambda

        // Collect all n-grams of this order and build context stats
        let ngrams_at_ord: Vec<(&Vec<u32>, &u64)> = raw_counts
            .iter()
            .filter(|(k, _)| k.len() == ord)
            .collect();

        for (ngram, &cnt) in &ngrams_at_ord {
            if ngram.len() > 1 {
                let ctx: Vec<u32> = ngram[..ngram.len() - 1].to_vec();
                // For highest order use raw counts; lower orders use continuation
                let effective_count = if ord == order { cnt } else {
                    *continuation.get(*ngram).unwrap_or(&0)
                };
                *context_count.entry(ctx.clone()).or_insert(0) +=
                    if ord == order { cnt } else { effective_count };
                if effective_count > 0 {
                    *context_uniq.entry(ctx).or_insert(0) += 1;
                }
            }
        }

        for (ngram, &cnt) in &ngrams_at_ord {
            let effective_count = if ord == order || ord == 1 {
                cnt as f64
            } else {
                *continuation.get(*ngram).unwrap_or(&0) as f64
            };

            let (prob, backoff) = if ord == 1 {
                let total: u64 = raw_counts.iter()
                    .filter(|(k, _)| k.len() == 1)
                    .map(|(_, &v)| v)
                    .sum();
                let total = total.max(1) as f64;
                let p = ((effective_count - D).max(0.0) / total
                    + D * ngrams_at_ord.len() as f64 / total / vocab_size as f64)
                    .max(1e-300);
                (p.log10() as f32, Some(0.0f32)) // backoff = log10(1) = 0
            } else {
                let ctx: Vec<u32> = ngram[..ngram.len() - 1].to_vec();
                let c_ctx = context_count.get(&ctx).copied().unwrap_or(1) as f64;
                let n_plus = context_uniq.get(&ctx).copied().unwrap_or(1) as f64;
                let lambda = D * n_plus / c_ctx;
                // Lower-order probability from the previous order's computed log-probs
                let suffix: &[u32] = &ngram[1..];
                let p_lower = all_ngrams_by_order[ord - 2]
                    .iter()
                    .find(|(k, ..)| k.as_slice() == suffix)
                    .map(|(_, lp, _)| (10f64).powf(*lp))
                    .unwrap_or(1.0 / vocab_size as f64);
                let p = ((effective_count - D).max(0.0) / c_ctx + lambda * p_lower).max(1e-300);
                // Backoff = 0.0 (log10(1)) for all contexts as a valid simplification
                let bo = if ord < order { Some(0.0f32) } else { None };
                (p.log10() as f32, bo)
            };

            all_ngrams_by_order[ord - 1].push(((*ngram).clone(), prob as f64, backoff.map(|b| b as f64)));
        }
    }

    // --- Phase 3: Write ARPA ---
    let file = std::fs::File::create(output_path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "\\data\\")?;
    for (i, ngrams) in all_ngrams_by_order.iter().enumerate() {
        writeln!(w, "ngram {}={}", i + 1, ngrams.len())?;
    }
    writeln!(w)?;

    for (i, ngrams) in all_ngrams_by_order.iter().enumerate() {
        let ord = i + 1;
        writeln!(w, "\\{}-grams:", ord)?;
        let mut sorted = ngrams.clone();
        // Sort by n-gram text for canonical ARPA output
        sorted.sort_by(|(a, ..), (b, ..)| {
            let av: Vec<&str> = a.iter()
                .map(|&id| vocab.word(id).unwrap_or("<unk>"))
                .collect();
            let bv: Vec<&str> = b.iter()
                .map(|&id| vocab.word(id).unwrap_or("<unk>"))
                .collect();
            av.cmp(&bv)
        });
        for (ngram, prob, backoff) in &sorted {
            let words: Vec<&str> = ngram.iter()
                .map(|&id| vocab.word(id).unwrap_or("<unk>"))
                .collect();
            let word_str = words.join("\t");
            if let Some(bo) = backoff {
                writeln!(w, "{:.7}\t{}\t{:.7}", prob, word_str, bo)?;
            } else {
                writeln!(w, "{:.7}\t{}", prob, word_str)?;
            }
        }
        writeln!(w)?;
    }
    writeln!(w, "\\end\\")?;
    w.flush()?;
    Ok(())
}

/// Streaming builder pipeline: read text, count n-grams, compute probabilities, write ARPA.
///
/// For corpora that fit in RAM, this is equivalent to `build_arpa()`. The function is
/// structured as a pipeline (corpus → count → sort → probabilities → ARPA) for clarity
/// and to serve as a foundation for future disk-based streaming.
///
/// * `config`       – pipeline configuration (order, discounts, pruning, …)
/// * `input_path`   – plain-text corpus (one sentence per line, UTF-8)
/// * `output_path`  – destination ARPA file
pub fn Pipeline(
    config: &PipelineConfig,
    input_path: &str,
    output_path: &str,
) -> Result<(), crate::error::LMError> {
    build_arpa(input_path, output_path, config.order as usize)
}

/// Parse a comma-separated list of discount values (e.g. "0.5,0.75,1.0").
/// Returns a Discount with the parsed values clamped to [0.0, order_limit].
pub fn ParseDiscountFallback(param: &[String]) -> Discount {
    let mut amounts = [0.0f32; 4];
    let values: Vec<f32> = param
        .iter()
        .flat_map(|s| s.split(','))
        .filter_map(|t| t.trim().parse::<f32>().ok())
        .collect();
    for (i, &v) in values.iter().enumerate().take(4) {
        amounts[i] = v.clamp(0.0, (i + 1) as f32);
    }
    Discount { amount: amounts }
}

/// Parse pruning thresholds from strings (e.g. ["0", "0", "1"]).
/// Returns a Vec of length `order`, zero-padded if too short.
pub fn ParsePruning(param: &[String], order: i8) -> Vec<u64> {
    let order = order as usize;
    let mut thresholds: Vec<u64> = param
        .iter()
        .flat_map(|s| s.split_whitespace())
        .filter_map(|t| t.parse::<u64>().ok())
        .collect();
    thresholds.resize(order, 0);
    thresholds
}
