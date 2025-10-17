# kenlm-rs

A Rust reimplementation of [KenLM Language Model](https://github.com/kpu/kenlm), providing fast and memory-efficient n-gram language modeling.

## Project Status

🚧 **Work in Progress** - This is an ongoing port of the C++ KenLM to Rust.

### Currently Implemented
- ✅ Core types (State, Config, ModelType)
- ✅ Vocabulary management (Sorted and Probing)
- ✅ ARPA file parsing (counts)
- ✅ Trie structure (with generic Quantization and Bhiksha)
- ✅ Configuration system
- ✅ Basic examples

### In Progress
- 🚧 Complete ARPA probability reading
- 🚧 Trie search algorithms
- 🚧 Probing hash table implementation
- 🚧 Binary model serialization

### Planned
- ⏳ Quantization support
- ⏳ Model building from text
- ⏳ Python bindings (PyO3)
- ⏳ Memory mapping for large models
- ⏳ REST costs and interpolation

## Quick Start

```rust
use kenlmrs::types::{Config, State};
use kenlmrs::vocabulary::{Vocabulary, ProbingVocabulary};

fn main() {
    // Create configuration
    let config = Config::default();
    
    // Initialize vocabulary
    let mut vocab = ProbingVocabulary::new();
    vocab.add_word("hello");
    vocab.add_word("world");
    
    // Create state for n-gram context
    let mut state = State::new();
    
    println!("Vocabulary size: {}", vocab.size());
}
```

## Examples

The `kenlmrs/examples/` directory contains comprehensive examples:

1. **simple.rs** - Basic functionality demonstration
2. **arpa.rs** - ARPA file reading and parsing
3. **query.rs** - Interactive sentence scoring (like KenLM's query binary)
4. **build_binary.rs** - Binary model building and memory estimation
5. **model.rs** - Complete API demonstration (similar to Python kenlm)

Run examples with:
```bash
cargo run --example simple
cargo run --example query
cargo run --example model
```

See [examples/README.md](kenlmrs/examples/README.md) for detailed documentation.

## Documentation

- [COMPARISON.md](COMPARISON.md) - Detailed comparison between C++ KenLM and kenlm-rs
- [examples/README.md](kenlmrs/examples/README.md) - Example usage and API patterns
- API docs: `cargo doc --open`

## Architecture

kenlm-rs follows the original KenLM architecture:

```
kenlmrs/
├── types.rs         # Core types (State, Config, etc.)
├── vocabulary.rs    # Vocabulary implementations
├── trie.rs          # Trie search structure
├── ngram/           # N-gram models
│   ├── binary_format.rs
│   ├── query.rs
│   └── mod.rs
├── builder/         # Model building
├── arpa.rs          # ARPA file parsing
└── utils/           # Utilities (file I/O, bit packing, etc.)
```

## Comparison with C++ KenLM

| Feature | C++ KenLM | kenlm-rs | Status |
|---------|-----------|----------|--------|
| ARPA reading | ✅ | 🚧 | Counts done |
| Probing model | ✅ | 🚧 | Structure ready |
| Trie model | ✅ | 🚧 | Structure ready |
| Vocabulary | ✅ | ✅ | Complete |
| Binary format | ✅ | 🚧 | In progress |
| Quantization | ✅ | ⏳ | Planned |
| Python bindings | ✅ | ⏳ | Planned |

See [COMPARISON.md](COMPARISON.md) for detailed comparison.

## Building

```bash
# Build library
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example model

# Generate documentation
cargo doc --open
```

## Performance Goals

kenlm-rs aims to match or exceed C++ KenLM performance while providing:
- Memory safety without garbage collection
- Thread safety via Rust's type system
- Zero-cost abstractions
- Idiomatic Rust APIs

## Contributing

Contributions welcome! When porting features:

1. Maintain API compatibility where possible
2. Use Rust idioms (traits, iterators, Result types)
3. Preserve performance characteristics
4. Add comprehensive tests
5. Document differences from C++ implementation

## References

- [Original KenLM](https://kheafield.com/code/kenlm/)
- [KenLM Paper (Heafield, 2011)](https://kheafield.com/papers/avenue/kenlm.pdf)
- [Python kenlm](https://github.com/kpu/kenlm/tree/master/python)

## License

This project follows the same license as the original KenLM (LGPL).

See the original repository: https://github.com/kpu/kenlm
