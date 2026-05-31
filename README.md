# kenlm-rs

A Rust port of [KenLM](https://kheafield.com/code/kenlm/) — a fast N-gram language model toolkit originally written in C++ by Kenneth Heafield.

## What is an N-gram Language Model?

A **language model** assigns probabilities to sequences of words. An **N-gram** model estimates the probability of each word conditioned on the *N-1* preceding words:

```text
P(word | context) ≈ P(word | w_{n-N+1}, ..., w_{n-1})
```

KenLM implements **Modified Kneser-Ney** smoothing with a **backoff** mechanism: when a long N-gram is unseen, the model backs off to shorter N-grams and applies a learned penalty weight. Scoring a sentence is the sum of log-probabilities:

```text
log P(w1 w2 … wn) = Σ log P(wi | context)
```

## Where kenlm is Useful

| Use Case | Why N-gram LMs Help |
| --- | --- |
| **ASR rescoring** | Re-rank speech recognition hypotheses using language fluency |
| **Machine translation** | Score translation candidates in SMT decoder |
| **Spelling correction** | Distinguish "their/there/they're" by context probability |
| **Text perplexity** | Evaluate how well a model fits a test corpus |
| **Beam search** | Prune low-probability hypotheses early in decoding |
| **Data filtering** | Remove low-quality text based on domain perplexity |

## Project Status

The core scoring engine is feature-complete. Both `ProbingModel` (hash table) and `TrieModel` (bit-packed suffix trie) load ARPA files and score correctly, numerically verified against C++ KenLM. Binary save/load, memory-mapped file I/O, gzip support, and a functional LM builder are all implemented.

### What Works Today

| Feature | Status |
| --- | --- |
| ARPA loading — `ProbingModel` (hash table) | Complete |
| ARPA loading — `TrieModel` (bit-packed suffix trie) | Complete |
| `full_score` / `score_except_backoff` | Complete |
| `begin_sentence_state` / `empty_context_state` | Complete |
| `get_state` / `full_score_forgot_state` | Complete |
| `base_score` / `short_score` on `Model` trait | Complete |
| `Left` / `ChartState` for chart decoding | Complete |
| Vocabulary (`ProbingVocabulary` + `SortedVocabulary`) | Complete |
| Quantization (`DontQuantize` + `SeparatelyQuantize`) | Complete — parity-tested |
| Binary save/load — `ProbingModel::save` / `load_binary` | Complete — round-trip tested |
| Binary save/load — all `TrieModel` variants | Complete — round-trip tested |
| `load_virtual` binary auto-detection | Complete |
| `QuantTrieModel` / `ArrayTrieModel` variants | Complete — numerically verified |
| Memory-mapped file I/O (`FilePiece`) | Complete — via `memmap2` |
| Gzip / compressed ARPA input | Complete — `.gz` files transparently decompressed |
| Builder — `build_arpa(corpus, output, order)` | Complete (Absolute Discounting) |
| Streaming builder pipeline — `Pipeline()` | Functional — delegates to `build_arpa()` |
| `Chain` / `Block` / `PCQueue` stream infrastructure | Implemented — single-threaded |
| `count_ngrams_to_chain` — streaming n-gram counting | Implemented |
| `ParseDiscountFallback` / `ParsePruning` | Complete |
| `initial_probabilities_direct` | Complete |
| 272 unit + integration tests | All pass |
| Numerical parity with C++ (max Δ = 4.3×10⁻⁷) | Verified |

### Coverage Summary

| C++ Source | Rust Module | Coverage | Notes |
| --- | --- | --- | --- |
| `lm/read_arpa.hh` | `src/arpa_reader.rs` | ~90% | All parse functions ported |
| `util/murmur_hash.hh` | `src/utils/hash.rs` | ~90% | MurmurHash64A complete |
| `util/bit_packing.hh` | `src/utils/bit_packing.rs` | ~80% | 57/25-bit, floats, masks |
| `util/string_piece.hh` | `src/utils/pieces/string.rs` | ~85% | Full find/substr/trim API |
| `util/tokenize_piece.hh` | `src/utils/pieces/tokenize.rs` | ~80% | All delimiter types |
| `lm/state.hh` | `src/types.rs` (`State`) | ~95% | Right context state |
| `lm/return.hh` | `src/types.rs` (`FullScoreReturn`) | ~100% | Left/ChartState complete |
| `lm/config.hh` | `src/types.rs` | ~88% | Config complete + builder methods + RestFunction; `EnumerateVocab` callbacks unused |
| `lm/common/ngram.hh` | `src/common/ngram.rs` | ~90% | NGram, OwnedNGram, Iterator |
| `lm/common/compare.hh` | `src/common/ordering.rs` | ~80% | Suffix/Context/Prefix ordering |
| `util/file_piece.hh` | `src/utils/pieces/file.rs` | ~85% | mmap + gzip; bzip2/xz absent |
| `lm/vocab.hh` | `src/vocabulary.rs` | ~90% | Binary I/O for both vocab types complete |
| `lm/search_hashed.hh` | `src/search.rs` `HashedSearch` | ~85% | Full scoring; uses `HashMap` |
| `lm/model.hh` | `src/model.rs` | ~95% | Full scoring + state + binary I/O |
| `lm/quantize.hh` | `src/quantize.rs` | ~92% | Both quantizers, encode/decode, train |
| `lm/trie.hh` | `src/trie.rs` | ~90% | Bit-packed suffix trie + binary I/O |
| `lm/search_trie.hh` | `src/search.rs` `TrieSearch` | ~88% | Full suffix-order scoring + binary |
| `lm/binary_format.hh` | `src/ngram/binary_format.rs` | ~88% | Read + write paths done; mmap-backed LoadBinary absent |
| `lm/builder/` | `src/builder/` | ~72% | MurmurHash fixed; streaming initial_probabilities; Callback restored; bzip2/xz/pruning absent |
| `util/stream/` | `src/stream/` | ~70% | Chain/Block/PCQueue + sort + producer/consumer threads; multi-stream merge absent |
| `lm/filter/` | `src/filter.rs` | ~88% | All 4 modes + multi-threading + vocab/context/phrase filters; threading uses scoped threads |

Legend: largely complete · partially working · stub / unimplemented

### Not Yet Translated

- bzip2 / xz compressed input (gzip is done)
- Multi-stream merge operators and external sort for billion-token corpora (`lmplz`-scale; in-memory + temp-file sort implemented, but k-way merge not wired into builder pipeline)
- `EnumerateVocab` callbacks during model loading
- Python bindings (`lm/wrappers/`)
- Interpolation of multiple models (`lm/interpolate/`)

## Quick Start

```bash
git clone <this-repo>
cd kenlm-rs/kenlmrs
cargo build
cargo test       # 272 tests pass
```

**Score a sentence from an ARPA file:**

```rust
use kenlmrs::model::ProbingModel;
use kenlmrs::types::{Config, State};
use kenlmrs::vocabulary::Vocabulary;

let model = ProbingModel::new("model.arpa", &Config::default())?;
let vocab = model.vocab();

let mut state = model.begin_sentence_state();  // start after <s>
let mut out = State::default();
let word = vocab.index("hello");
let ret = model.full_score(&state, word, &mut out);
println!("log10 P(hello|<s>) = {:.6}", ret.prob);
```

**Build a language model from raw text:**

```rust
use kenlmrs::builder::pipeline::build_arpa;

// Reads corpus.txt line-by-line, writes trigram model in ARPA format
build_arpa("corpus.txt", "model.arpa", 3)?;

let model = ProbingModel::new("model.arpa", &Config::default())?;
```

**Build via the streaming pipeline (same result, pipeline-structured):**

```rust
use kenlmrs::builder::pipeline::{Pipeline, PipelineConfig};

let mut config = PipelineConfig::new();
config.order = 3;
Pipeline(&config, "corpus.txt", "model.arpa")?;
```

**Save and reload a model as a binary file:**

```rust
// ProbingModel binary — fast to reload, skips ARPA re-parse
model.save("model.bin")?;
let model = ProbingModel::load_binary("model.bin", &Config::default())?;

// TrieModel binary — all four trie variants support round-trip save/load
use kenlmrs::model::TrieModel;
let trie = TrieModel::new("model.arpa", &Config::default())?;
trie.save("trie.bin")?;
let trie = TrieModel::load_binary("trie.bin", &Config::default())?;

// Or auto-detect format (binary magic check, falls back to ARPA)
use kenlmrs::model::{load_virtual, ModelType};
let model = load_virtual("model.bin", &Config::default(), ModelType::Probing)?;
```

**Use the trie model (smaller memory footprint):**

```rust
use kenlmrs::model::TrieModel;

let model = TrieModel::new("model.arpa", &Config::default())?;
let mut out = State::default();
let ret = model.full_score(&model.begin_sentence_state(), vocab.index("hello"), &mut out);
```

**Open a gzip-compressed ARPA file:**

```rust
// FilePiece transparently decompresses .gz — no extra code needed
let model = ProbingModel::new("model.arpa.gz", &Config::default())?;
```

## Model Types

| Model | Search | Memory | Speed | Status |
| --- | --- | --- | --- | --- |
| `ProbingModel` | Hash table | ~1.3× n-gram size | Fastest | Full |
| `RestProbingModel` | Hash table + rest costs | ~1.3× | Fast | Full |
| `TrieModel` | Sorted bit-packed suffix trie | Smallest | Moderate | Full |
| `QuantTrieModel` | Trie + quantized probs | Small | Moderate | Full |
| `ArrayTrieModel` | Trie + Bhiksha compression | Very small | Moderate | Full |
| `QuantArrayTrieModel` | Trie + quant + Bhiksha | Smallest | Moderate | Full |

## ARPA Format

ARPA is the standard interchange format for N-gram language models:

```text
\data\
ngram 1=3
ngram 2=5

\1-grams:
-99     <unk>
-1.5    <s>     -0.3
-1.0    </s>

\2-grams:
-0.8    <s> hello   0.0
-1.2    hello world

\end\
```

Each N-gram line: `log10_prob   w1 w2 … wn   [log10_backoff]`

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed walk-through of the module design, data flow, key design decisions, and a step-by-step explanation of the scoring algorithm.

```text
kenlmrs/src/
├── types.rs           # State, Config, FullScoreReturn, ProbBackoff
├── vocabulary.rs      # ProbingVocabulary, SortedVocabulary
├── model.rs           # GenericModel<Search, Vocabulary>, Model trait, binary I/O
├── search.rs          # HashedSearch (complete), TrieSearch (re-exports from trie.rs)
├── trie.rs            # Suffix-order bit-packed trie with ARPA loading
├── arpa_reader.rs     # ARPA file parser (complete)
├── constant.rs        # KENLM_MAX_ORDER, UNK/BOS/EOS indices, binary magic
├── error.rs           # LMError enum
├── ngram/             # Binary format read/write (ProbingModel)
├── builder/           # CorpusCount, build_arpa, ParseDiscountFallback/Pruning
├── stream/            # Chain/Block/PCQueue — single-threaded streaming infrastructure
├── filter.rs          # Single-threaded n-gram filter
├── common/            # NGram, ordering comparators
└── utils/
    ├── hash.rs        # MurmurHash64A
    ├── bit_packing.rs # Variable-width bit read/write
    └── pieces/        # FilePiece, StringPiece, tokenizer
```

## Building and Testing

```bash
cargo build --release        # Release build
cargo test                   # Run all 272 tests
cargo doc --open             # Generate and open API docs
cargo run --example simple   # Basic example
cargo run --example query    # Query mode example
cargo run --example arpa     # ARPA parsing example
```

## Contributing

Remaining areas to implement:

1. **Multi-threaded streaming sort** — `stream/` Chain/Block infrastructure is in place; the missing piece is an on-disk external merge-sort for billion-token corpora (needed for true `lmplz`-scale pipelines)
2. **bzip2 / xz compressed input** — gzip is done; add `bzip2` / `xz` decompression backends to `FilePiece`
3. **Probing hash table** — optionally replace `HashMap` with open-addressing matching `util/probing_hash_table.hh` for lower memory overhead
4. **TrieModel binary format compatibility** — current binary format is kenlm-rs internal; aligning with C++ KenLM's on-disk layout would enable cross-tool interoperability
5. **Python bindings** — expose the scoring API via `pyo3`

When porting features from C++:

- Use Rust idioms (traits, `Result`, iterators) over direct C++ translation
- Add a `#[cfg(test)]` block with tests covering the new functions
- Document divergences from C++ in inline comments
- Keep `cargo test` green at 272 passed throughout

## References

- [Original KenLM](https://kheafield.com/code/kenlm/)
- [KenLM Paper (Heafield, 2011)](https://kheafield.com/papers/avenue/kenlm.pdf)
- [KenLM GitHub](https://github.com/kpu/kenlm)

## License

Follows the same license as the original KenLM (LGPL). See the original repository: <https://github.com/kpu/kenlm>
