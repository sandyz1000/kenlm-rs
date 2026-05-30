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

🚧 **Work in Progress** — This is an ongoing port of the C++ KenLM to Rust. The core data structures and parsing logic are in place; the end-to-end ARPA loading into trie structures and binary format I/O are not yet complete.

### Coverage Summary

| C++ Source | Rust Module | Status | Notes |
| --- | --- | --- | --- |
| `lm/read_arpa.hh` | `src/arpa_reader.rs` | ~90% | All parse functions ported |
| `util/murmur_hash.hh` | `src/utils/hash.rs` | ~90% | MurmurHash64A complete |
| `util/bit_packing.hh` | `src/utils/bit_packing.rs` | ~80% | 57/25-bit, floats, masks |
| `util/string_piece.hh` | `src/utils/pieces/string.rs` | ~85% | Full find/substr/trim API |
| `util/tokenize_piece.hh` | `src/utils/pieces/tokenize.rs` | ~80% | All delimiter types |
| `lm/state.hh` | `src/types.rs` (`State`) | ~95% | Right context state |
| `lm/return.hh` | `src/types.rs` (`FullScoreReturn`) | ~95% | Left/ChartState absent |
| `lm/config.hh` | `src/types.rs` + `src/config.rs` | ~70% | Duplicate Config structs |
| `lm/common/ngram.hh` | `src/common/ngram.rs` | ~90% | NGram, NGramMut, Iterator |
| `lm/common/compare.hh` | `src/common/ordering.rs` | ~80% | Suffix/Context/Prefix ordering |
| `util/file_piece.hh` | `src/utils/pieces/file.rs` | ~70% | mmap/compressed absent |
| `lm/vocab.hh` | `src/vocabulary.rs` | ~70% | Binary load missing |
| `lm/search_hashed.hh` | `src/search.rs` `HashedSearch` | ~60% | Uses `HashMap` not probing table |
| `lm/model.hh` | `src/model.rs` | ~60% | Scoring logic done; load is stub |
| `lm/trie.hh` | `src/trie.rs` | ~50% | BitPacked structs done; ARPA load stub |
| `lm/search_trie.hh` | `src/search.rs` `TrieSearch` | ~40% | All methods are `todo!()` |
| `lm/binary_format.hh` | `src/ngram/binary_format.rs` | ~15% | Struct only |
| `lm/builder/` | `src/builder/` | ~5% | Placeholder types |
| `util/stream/` | `src/stream/` | ~5% | Placeholder types |
| `lm/filter/` | `src/filter.rs` | ~5% | Types only |

Legend: largely complete · partially working · stub / unimplemented

### Not Yet Translated

- Corpus counting from raw text (`lm/builder/corpus_count.hh` → `lmplz` equivalent)
- Memory-mapped I/O (`util/mmap.hh`)
- Compressed file reading (gzip / bzip2 / xz)
- Python bindings (`lm/wrappers/`)
- Interpolation of multiple models (`lm/interpolate/`)

## Quick Start

```bash
git clone <this-repo>
cd kenlm-rs/kenlmrs
cargo build
cargo test       # 179 tests pass
```

**Example — parse an ARPA file:**

```rust
use kenlmrs::utils::pieces::file::FilePiece;
use kenlmrs::arpa_reader::{read_arpa_counts, read_1grams, read_end, PositiveProbWarn};
use kenlmrs::constant::WarningAction;
use kenlmrs::vocabulary::ProbingVocabulary;
use kenlmrs::types::ProbBackoff;

let mut fp = FilePiece::open("model.arpa")?;
let counts = read_arpa_counts(&mut fp)?;
println!("N-gram counts: {:?}", counts);

let mut vocab = ProbingVocabulary::new();
let warn = PositiveProbWarn::new(WarningAction::Complain);
let mut unigrams = vec![ProbBackoff::default(); counts[0] as usize];
read_1grams(&mut fp, counts[0] as usize, &mut vocab, &mut unigrams, &warn)?;
read_end(&mut fp)?;
```

## Model Types

KenLM supports two families of search structures, each with optional variants:

| Model | Search | Memory | Speed |
| --- | --- | --- | --- |
| `ProbingModel` | Hash table | ~1.3× n-gram size | Fastest |
| `RestProbingModel` | Hash table + rest costs | ~1.3× | Fast |
| `TrieModel` | Sorted trie | Smallest | Moderate |
| `ArrayTrieModel` | Trie + Bhiksha compression | Very small | Moderate |
| `QuantTrieModel` | Trie + quantized probs | Small | Moderate |
| `QuantArrayTrieModel` | Trie + quant + Bhiksha | Smallest | Moderate |

The Rust port has functional `HashedSearch` scaffolding and the trie bit-packed structures built; full end-to-end loading for the trie models is not yet complete.

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

```
kenlmrs/src/
├── types.rs           # State, Config, FullScoreReturn, ProbBackoff
├── vocabulary.rs      # ProbingVocabulary, SortedVocabulary
├── model.rs           # GenericModel<Search, Vocabulary>, Model trait
├── search.rs          # HashedSearch (functional), TrieSearch (stubs)
├── trie.rs            # BitPackedMiddle, BitPackedLongest, Unigram
├── arpa_reader.rs     # ARPA file parser (complete)
├── constant.rs        # KENLM_MAX_ORDER, UNK/BOS/EOS indices
├── error.rs           # LMError enum
├── ngram/             # Binary format, query helpers
├── builder/           # Builder pipeline (stubs)
├── stream/            # Stream chain (stubs)
├── filter.rs          # Filter types (stubs)
├── common/            # NGram, ordering comparators, ModelBuffer
└── utils/
    ├── hash.rs        # MurmurHash64A
    ├── bit_packing.rs # Variable-width bit read/write
    └── pieces/        # FilePiece, StringPiece, tokenizer
```

## Building and Testing

```bash
cargo build --release        # Release build
cargo test                   # Run all 179 unit tests
cargo doc --open             # Generate and open API docs
cargo run --example simple   # Basic example
cargo run --example query    # Query mode example
cargo run --example arpa     # ARPA parsing example
```

## Contributing

The highest-priority areas to implement:

1. **`TrieSearch::initialize_from_arpa`** — `src/search.rs` TrieSearch methods (currently `todo!()`)
2. **Binary format loading** — `src/ngram/binary_format.rs` read/write
3. **Probing hash table** — replace `HashMap` with open-addressing matching `util/probing_hash_table.hh`
4. **Memory-mapped I/O** — large model support

When porting features from C++:

- Use Rust idioms (traits, `Result`, iterators) over direct C++ translation
- Add a `#[cfg(test)]` block with tests covering the new functions
- Document divergences from C++ in inline comments

## References

- [Original KenLM](https://kheafield.com/code/kenlm/)
- [KenLM Paper (Heafield, 2011)](https://kheafield.com/papers/avenue/kenlm.pdf)
- [KenLM GitHub](https://github.com/kpu/kenlm)

## License

Follows the same license as the original KenLM (LGPL). See the original repository: <https://github.com/kpu/kenlm>
