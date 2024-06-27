use crate::stream::{config::SortConfig, config::ChainConfig};
use builder::count::DiscountConfig;
use super::config::InitialProbabilitiesConfig;



#[derive(Debug, Default)]
pub struct PipelineConfig {
    order: i8,
    sort: SortConfig,
    initial_probs: InitialProbabilitiesConfig,
    read_backoffs: ChainConfig,

    // Estimated vocabulary size.  Used for sizing CorpusCount memory and initial probing hash table sizing, also in CorpusCount.
    vocab_estimate: WordIndex,

    // Minimum block size to tolerate.
    minimum_block: i8,

    // Number of blocks to use.  This will be overridden to 1 if everything fits.
    block_count: i8,

    // n-gram count thresholds for pruning. 0 values means no pruning for corresponding n-gram order
    prune_thresholds: Vec<i64>,  // mjd
    prune_vocab: bool,
    prune_vocab_file: String,

    // Renumber the vocabulary the way the trie likes it?
    renumber_vocabulary: bool,

    // What to do with discount failures.
    discount: DiscountConfig,

    // Compute collapsed q values instead of probability and backoff
    output_q: bool,

    // * Computing the perplexity of LMs with different vocabularies is hard.  For
    // * example, the lowest perplexity is attained by a unigram model that
    // * predicts p(<unk>) = 1 and has no other vocabulary.  Also, linearly
    // * interpolated models will sum to more than 1 because <unk> is duplicated
    // * (SRI just pretends p(<unk>) = 0 for these purposes, which makes it sum to
    // * 1 but comes with its own problems).  This option will make the vocabulary
    // * a particular size by replicating <unk> multiple times for purposes of
    // * computing vocabulary size.  It has no effect if the actual vocabulary is
    // * larger.  This parameter serves the same purpose as IRSTLM's "dub".

    vocab_size_for_unk: u64,

    // What to do the first time <s>, </s>, or <unk> appears in the input. If this is
    // anything but THROW_UP, then the symbol will always be treated as whitespace.
    disallowed_symbol_action: WarningAction,

}

impl PipelineConfig {
    fn new() -> Self {
        todo!()
    }
    
    pub fn TempPrefix(self) -> String {
        todo!()
    }
    
    pub fn TotalMemory() -> i8 {
        todo!()
    }
}

pub fn Pipeline(config: &PipelineConfig, text_file: i64, output: &Output);

pub fn ParseDiscountFallback(param: &Vec<String>) -> Discount;

pub fn ParsePruning(param: &Vec<String>, order: i8) -> Vec<u64>;
