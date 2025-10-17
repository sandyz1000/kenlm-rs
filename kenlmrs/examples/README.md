# KenLM-RS Examples

This directory contains examples demonstrating the usage of the kenlm-rs library, inspired by the original C++ KenLM implementation.

## Examples Overview

### 1. Simple Example (`simple.rs`)
Basic functionality test showing:
- Configuration setup
- Vocabulary operations
- State management

```bash
cargo run --example simple
```

### 2. ARPA Reader Example (`arpa.rs`)
Demonstrates reading and parsing ARPA format language model files:
- Creating test ARPA files
- Reading n-gram counts
- Validating ARPA format

```bash
cargo run --example arpa
```

### 3. Query Example (`query.rs`)
Interactive sentence scoring similar to KenLM's `query` binary:
- Reading sentences from stdin
- Tokenization and vocabulary lookup
- Word-level scoring with n-gram matching
- Perplexity calculation
- OOV detection

```bash
cargo run --example query
# Then type sentences to score (Ctrl+D to exit)
```

### 4. Build Binary Example (`build_binary.rs`)
Memory estimation and binary model building (work in progress):
- Reading ARPA files
- Memory usage estimation for different model types
- Understanding model size requirements

```bash
cargo run --example build_binary
```

### 5. Model Example (`model.rs`)
Comprehensive API demonstration similar to Python kenlm:
- Sentence scoring
- Full scores with n-gram matching
- OOV word detection
- Stateful querying with context

```bash
cargo run --example model
```

## Comparison with C++ KenLM

These examples mirror the functionality of the original KenLM:

| C++ KenLM | kenlm-rs Example | Description |
|-----------|------------------|-------------|
| `query` | `query.rs` | Interactive sentence scoring |
| `build_binary` | `build_binary.rs` | Building binary models from ARPA |
| `example.py` | `model.rs` | Python API equivalent in Rust |
| Test files | `arpa.rs` | ARPA format parsing |

## ARPA Format

The examples work with ARPA format language model files. Here's the structure:

```
\data\
ngram 1=<count>
ngram 2=<count>
...

\1-grams:
<prob> <word> [<backoff>]
...

\2-grams:
<prob> <word1> <word2> [<backoff>]
...

\end\
```

Example ARPA file content:
```
\data\
ngram 1=3
ngram 2=2

\1-grams:
-1.0    <unk>   0.0
-2.5    hello   -0.3
-3.2    world   -0.1

\2-grams:
-0.5    hello world
-1.2    world hello

\end\
```

## API Usage Patterns

### Basic Configuration
```rust
use kenlmrs::types::{Config, LoadMethod};

let mut config = Config::default();
config.load_method = LoadMethod::ReadMethod;
config.probing_multiplier = 1.5;
```

### Vocabulary Operations
```rust
use kenlmrs::vocabulary::{Vocabulary, ProbingVocabulary};

let mut vocab = ProbingVocabulary::new();
let idx = vocab.add_word("hello");
let found = vocab.index("hello");
```

### State Management
```rust
use kenlmrs::types::State;

let mut state = State::new();
state.length = 1;
state.words[0] = word_index;
```

## Current Limitations

These examples demonstrate the API and structure, but some functionality is not yet fully implemented:

- ✅ Vocabulary management
- ✅ Configuration system
- ✅ State management
- ✅ ARPA file parsing (counts)
- 🚧 Complete ARPA probability reading
- 🚧 Trie construction
- 🚧 Hash table construction
- 🚧 Binary model serialization
- 🚧 Actual probability scoring
- 🚧 Quantization support

## Next Steps

To make these examples fully functional:

1. Implement complete ARPA reading (probabilities and backoffs)
2. Build the trie/hash table data structures
3. Implement probability lookup algorithms
4. Add binary model serialization
5. Support quantization for memory efficiency

## Contributing

When adding new examples, please:
1. Follow the existing naming conventions
2. Add comprehensive comments
3. Update this README
4. Include error handling
5. Match the C++ KenLM API where applicable

## References

- [Original KenLM](https://kheafield.com/code/kenlm/)
- [KenLM Paper](https://kheafield.com/papers/avenue/kenlm.pdf)
- [Python kenlm](https://github.com/kpu/kenlm/tree/master/python)
