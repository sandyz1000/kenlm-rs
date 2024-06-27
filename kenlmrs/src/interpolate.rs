use crate::kenlm::{Error, ModelBuffer, Result, StringPiece};
use crate::util::FixedArray;

use core::option::Option;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub struct SplitWorker;


#[derive(Debug, Clone, Copy)]
pub struct Config {
    lambdas: f64,
    sort: SortConfig,
    // pub fn BufferSize() -> i64;
}

#[derive(Debug, Clone, Copy)]
pub struct InstancesConfig {
    // For batching the model reads.  This is per order.
    model_read_chain_mem: i64,

    // This is being sorted, make it larger.
    extension_write_chain_mem: i64,

    lazy_memory: i64,

    sort: SortConfig,
}

impl SplitWorker {
    // Constructs a split worker for a particular order. It writes the
    // split-off backoff values to the backoff chain and the ngram id and
    // probability to the sort chain for each ngram in the input.
    fn new(order: i64, backoff_chain: &Chain, sort_chain: &Chain) -> Self {}

    // The callback invoked to handle the input from the ngram intermediate files.
    pub fn run(&self, position: &ChainPosition) {}
}

pub fn tune_weights(
    tune_file: i64,
    model_names: &Vec<StringPiece>,
    config: &InstancesConfig,
    weights_out: &Vec<f64>,
) {
}

pub fn pipeline(models: &FixedArray<ModelBuffer>, config: &Config, write_file: i64) {}
