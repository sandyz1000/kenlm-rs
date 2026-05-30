# kenlm-rs Architecture

This document describes the design of kenlm-rs, how it maps to the original C++ KenLM, a detailed feature-parity comparison, and the key algorithmic decisions behind the implementation.

---

## N-gram Language Model Background

### Modified Kneser-Ney Smoothing

KenLM uses Modified Kneser-Ney (MKN) smoothing to assign non-zero probabilities to unseen N-grams. The core formula for a word `w` given context `h` is:

```text
P_KN(w | h) = max(c(h,w) - D, 0) / c(h)   +   λ(h) · P_KN(w | h')
```

Where:

- `c(h,w)` is the count of the N-gram
- `D` is a discount (interpolated between `D1`, `D2`, `D3+`)
- `λ(h)` is a normalization constant ensuring probabilities sum to 1
- `h'` is the shorter context (backoff)

This is stored as log-probabilities plus a **backoff weight**. A word's full score is:

```text
full_score(w | h) = log P(w | h[0..n]) + backoff[n] + backoff[n+1] + ...
```

The backoff weights cascade from the matched N-gram length up to the full context length.

### ARPA File Format

ARPA files store the pre-computed log-probabilities and backoff weights:

```text
\data\
ngram 1=<count>
ngram 2=<count>
...

\1-grams:
<log10_prob>  <word>  [<log10_backoff>]

\2-grams:
<log10_prob>  <word1> <word2>  [<log10_backoff>]

\end\
```

Backoff weights for the **longest** N-gram order are absent (there is nothing to back off to).

---

## Data Flow

```text
 ARPA File
     │
     ▼
 FilePiece ──────────────────────── buffered byte reader
     │                              src/utils/pieces/file.rs
     ▼
 read_arpa_counts()                 parse \data\ section
 read_1grams() / read_ngram()       parse N-gram sections
     │                              src/arpa_reader.rs
     ▼
 Vocabulary                         word → index mapping
 (ProbingVocabulary or              src/vocabulary.rs
  SortedVocabulary)
     │
     ▼
 Search::initialize_from_arpa()     load probs into search structure
 (HashedSearch — implemented)       src/search.rs
 (TrieSearch — not yet)
     │
     ▼
 GenericModel<Search, Vocab>        unified query interface
     │                              src/model.rs
     ▼
 full_score(State, word) → FullScoreReturn
 get_state(context) → State
 full_score_forgot_state(context, word) → FullScoreReturn
```

---

## Feature Parity: kenlm-rs vs C++ KenLM

The table below covers every significant component from the C++ codebase. Status codes:

- **Complete** — logic fully ported and tested
- **Partial** — structure/signature exists; some logic missing
- **Stub / Missing** — placeholder only or not present

### Core Scoring (`lm/model.hh`, `lm/facade.hh`, `lm/virtual_interface.hh`)

| C++ | Rust (`src/model.rs`) | Status | Notes |
| --- | --- | --- | --- |
| `GenericModel::FullScore()` | `GenericModel::full_score()` | | Complete with backoff accumulation |
| `GenericModel::FullScoreForgotState()` | `GenericModel::full_score_forgot_state()` | | Stateless scoring with context backoff |
| `GenericModel::GetState()` | `GenericModel::get_state()` | | Reconstructs State from context words |
| `GenericModel::BeginSentenceState()` | `GenericModel::begin_sentence_state()` | | `<s>` seeded State |
| `GenericModel::NullContextState()` | `GenericModel::null_context_state()` | | Zero-length State |
| `GenericModel::ExtendLeft()` | — | | Left-extension for SMT decoding |
| `GenericModel::UnRest()` | — | | Unrest probability computation |
| `GenericModel(file, config)` constructor | `GenericModel::new()` | | Stub — no actual file I/O |
| `LoadVirtual()` | `load_virtual()` | | Always returns ProbingModel; no type detection |
| `RecognizeBinary()` | `recognize_binary()` | | Reads magic bytes + FixedWidthParameters only |
| `base::Model` virtual interface | `Model` trait | | `full_score` works; `base_score`/`short_score` stub |
| `ModelFacade::Score()` | — | | Short-form (f32) score not exposed |

### State Types (`lm/state.hh`)

| C++ | Rust (`src/types.rs`) | Status | Notes |
| --- | --- | --- | --- |
| `State` (words, backoff, length) | `State` | | All fields and methods |
| `State::Compare()` | `State::compare()` | | Three-way ordering |
| `State::ZeroRemaining()` | `State::zero_remaining()` | | |
| `hash_value(State)` | via `State::compare` | | Not a standalone hash function |
| `Left` (pointers, length, full) | `Left` | | All fields and methods |
| `Left::Compare()` | `Left::compare()` | | By length, last pointer, full flag |
| `Left::ZeroRemaining()` | `Left::zero_remaining()` | | |
| `hash_value(Left)` | `Left::hash_value()` | | MurmurHash on [length, full] |
| `ChartState` (left + right) | `ChartState` | | All fields and methods |
| `ChartState::Compare()` | `ChartState::compare()` | | Left-first ordering |
| `hash_value(ChartState)` | `ChartState::hash_value()` | | Combines left and right hashes |
| `FullScoreReturn` struct | `FullScoreReturn` | | All five fields |
| `RuleScore<M>` (SMT decoding) | — | | `lm/left.hh` — no equivalent |
| `RevealBefore/After`, `Subsume` | — | | `lm/partial.hh` — no equivalent |

### Search Structures

#### HashedSearch (`lm/search_hashed.hh/.cc`)

| C++ method | Rust (`src/search.rs` `HashedSearch`) | Status |
| --- | --- | --- |
| `Size(counts, config)` | `size(counts, config)` | |
| `SetupMemory(start, counts, config)` | `setup_memory(start, counts, config)` | |
| `InitializeFromARPA(file, f, counts, config, vocab, backing)` | `initialize_from_arpa(file, counts, config, vocab)` | |
| `Order()` | `order()` | |
| `LookupUnigram(word, node, independent_left, extend_left)` | `lookup_unigram(...)` | |
| `LookupMiddle(order_minus_2, word, node, ...)` | `lookup_middle(...)` | |
| `LookupLongest(word, node)` | `lookup_longest(...)` | |
| `FastMakeNode(begin, end, node)` | `fast_make_node(begin, node)` | |
| `Unpack(extend_pointer, extend_length, node)` | `unpack(...)` | |
| `UnknownUnigram()` | `unknown_unigram()` | |
| `UpdateConfigFromBinary(...)` | — | |
| Probing hash table backend | `HashMap<u64, ProbBackoff>` | Uses `HashMap`; C++ uses custom open-addressing |
| Sign-bit `IndependentLeft` encoding | `is_independent_left()` / `mark_extends_left()` | |
| Context marking (`ActivateUnigram`/`ActivateLowerMiddle`) | `mark_context_as_extending()` | |

#### TrieSearch (`lm/search_trie.hh`)

| C++ method | Rust (`src/search.rs` `TrieSearch`) | Status |
| --- | --- | --- |
| `InitializeFromARPA(...)` | `initialize_from_arpa(...)` | Returns `LoadError` |
| `LookupUnigram(...)` | `lookup_unigram(...)` | Returns not-found |
| `LookupMiddle(...)` | `lookup_middle(...)` | Returns not-found |
| `LookupLongest(...)` | `lookup_longest(...)` | Returns not-found |
| `FastMakeNode(...)` | `fast_make_node(...)` | Returns false |
| `Unpack(...)` | `unpack(...)` | Returns not-found |
| `Order()` | `order()` | Returns 0 |
| `SetupMemory(...)` | `setup_layers(...)` | Allocates `Vec<Middle>` |
| `Size(...)` | `size(...)` | Returns 0 |
| `middles: Vec<Middle>` (safe redesign) | `middles: Vec<Middle>` | No raw pointers |

**Trie bit-packed structures** (`lm/trie.hh`):

| C++ | Rust (`src/trie.rs`) | Status |
| --- | --- | --- |
| `BitPackedMiddle::Insert()` | `BitPackedMiddle::insert()` | |
| `BitPackedMiddle::Find()` | `BitPackedMiddle::find()` | |
| `BitPackedMiddle::ReadEntry()` | `BitPackedMiddle::read_entry()` | |
| `BitPackedMiddle::FinishedLoading()` | `BitPackedMiddle::finished_loading()` | |
| `BitPackedLongest::Insert()` | `BitPackedLongest::insert()` | |
| `BitPackedLongest::Find()` | `BitPackedLongest::find()` | |
| `Unigram::Find()` | `Unigram::find()` | |
| `Unigram::Size()` | `Unigram::size()` | |
| Wiring BitPacked → TrieSearch | — | `Middle` placeholder not connected |

### Vocabulary (`lm/vocab.hh`)

| C++ | Rust (`src/vocabulary.rs`) | Status | Notes |
| --- | --- | --- | --- |
| `SortedVocabulary::Index()` | `SortedVocabulary::index()` | | Binary search after sort |
| `SortedVocabulary::Insert()` | `SortedVocabulary::insert()` | | — |
| `SortedVocabulary::FinishedLoading()` | `SortedVocabulary::finished_loading()` | | Sorts hashes |
| `SortedVocabulary::LoadedBinary()` | — | | Binary vocab loading |
| `SortedVocabulary::SetupMemory()` | — | | C++ in-place memory init |
| `SortedVocabulary::Relocate()` | — | | C++ pointer rebasing |
| `SortedVocabulary::ComputeRenumbering()` | — | | Reorder to hash order |
| `ProbingVocabulary::Index()` | `ProbingVocabulary::index()` | | HashMap lookup |
| `ProbingVocabulary::Insert()` | `ProbingVocabulary::add_word()` | | — |
| `ProbingVocabulary::LoadedBinary()` | — | | Binary vocab loading |
| `ProbingVocabulary::SetupMemory()` | — | | C++ in-place memory init |
| `GrowableVocab<NewWordAction>` | — | | Dynamic vocab for building |
| `EnumerateVocab` callback | `EnumerateVocab` trait (stub) | | Trait defined; not wired |
| `WriteUniqueWords` / `NoOpUniqueWords` | — | | — |

### Binary Format (`lm/binary_format.hh`)

| C++ | Rust (`src/ngram/binary_format.rs`) | Status |
| --- | --- | --- |
| `IsBinaryFormat(fd)` | `recognize_binary()` | Reads magic; returns model type |
| `InitializeBinary(...)` | — | |
| `LoadBinary(size)` | — | |
| `GrowForSearch(...)` | — | |
| `FinishFile(...)` | — | |
| `SetupJustVocab(...)` | — | |
| `WriteVocabWords(...)` | — | |
| `ReadForConfig(...)` | `read_fixed_width()` | Reads header only |
| `FixedWidthParameters` struct | `FixedWidthParameters` | |
| `Parameters` struct | `Parameters` | |
| Memory-mapped loading | — | |

### ARPA Reading (`lm/read_arpa.hh`)

| C++ | Rust (`src/arpa_reader.rs`) | Status |
| --- | --- | --- |
| `ReadARPACounts()` | `read_arpa_counts()` | |
| `ReadNGramHeader()` | `read_ngram_header()` | |
| `Read1Gram<Voc, Weights>()` | `read_1gram()` | |
| `Read1Grams<Voc, Weights>()` | `read_1grams()` | |
| `ReadNGram<Voc, Weights, Iter>()` | `read_ngram()` | |
| `ReadBackoff()` overloads | `read_backoff_probbackoff()` | |
| `ReadEnd()` | `read_end()` | |
| `PositiveProbWarn` | `PositiveProbWarn` | |
| `kARPASpaces` | `ARPA_SPACES` | |

### Quantization (`lm/quantize.hh`)

| C++ | Rust (`src/quantize.rs`) | Status |
| --- | --- | --- |
| `DontQuantize::MiddleBits()` | `DontQuantize::middle_bits()` | |
| `DontQuantize::LongestBits()` | `DontQuantize::longest_bits()` | |
| `DontQuantize::MiddlePointer` | `DontQuantizeMiddlePointer` | |
| `DontQuantize::LongestPointer` | `DontQuantizeLongestPointer` | |
| `SeparatelyQuantize::Train()` | `SeparatelyQuantize::train()` | |
| `SeparatelyQuantize::TrainProb()` | `SeparatelyQuantize::train_prob()` | |
| `SeparatelyQuantize::FinishedLoading()` | `SeparatelyQuantize::finished_loading()` | |
| `SeparatelyQuantize::MiddlePointer` | `QuantizedMiddlePointer` | |
| `SeparatelyQuantize::LongestPointer` | `QuantizedLongestPointer` | |
| `Bins::EncodeProb/Backoff/Decode` | `Bins::encode/decode` | |
| `UpdateConfigFromBinary()` | `update_config_from_binary()` | |

### Bhiksha Compression (`lm/bhiksha.hh`)

| C++ | Rust (`src/bhiksha.rs`) | Status |
| --- | --- | --- |
| `DontBhiksha::ReadNext()` | `DontBhiksha::read_next()` | |
| `DontBhiksha::WriteNext()` | `DontBhiksha::write_next()` | |
| `DontBhiksha::InlineBits()` | `DontBhiksha::inline_bits()` | |
| `ArrayBhiksha::ReadNext()` | `ArrayBhiksha::read_next()` | Binary search + inline bits |
| `ArrayBhiksha::WriteNext()` | `ArrayBhiksha::write_next()` | Grows offset array |
| `ArrayBhiksha::FinishedLoading()` | `ArrayBhiksha::finished_loading()` | |
| `ArrayBhiksha::Size()` | `ArrayBhiksha::size()` | |
| `UpdateConfigFromBinary()` | `update_config_from_binary()` | |

### Builder Pipeline (`lm/builder/`)

| C++ | Rust (`src/builder/`) | Status |
| --- | --- | --- |
| `Pipeline(config, text_fd, output)` | `pipeline.rs` | Stub only |
| `CorpusCount` | `count.rs` | Stub |
| `AdjustCounts` | — | Missing |
| `InitialProbabilities(...)` | `proba.rs` | Stub |
| `OutputHook` / `PrintHook` | `OutputHook` struct | Empty type only |
| `PipelineConfig` | `PipelineConfig` struct | Fields exist, no logic |
| `Discount` struct + methods | `Discount` + `get()`/`apply()` | |
| `BuildingPayload` + mark/unmark | `BuildingPayload` | |
| `HeaderInfo` struct | `HeaderInfo` | |
| Stream chain infrastructure | `src/stream/` | Placeholders |

### Interpolation (`lm/interpolate/`)

| C++ | Rust (`src/interpolate.rs`) | Status |
| --- | --- | --- |
| `Pipeline(models, config, write_fd)` | `pipeline()` | Stub |
| `MergeVocab(...)` | — | Missing |
| `TuneWeights(...)` | `tune_weights()` | Stub |
| `BoundedSequenceEncoding` | — | Missing |

### Filter (`lm/filter/`)

| C++ | Rust (`src/filter.rs`) | Status |
| --- | --- | --- |
| `ARPAOutput` class | — | Missing |
| `BinaryFilter<T>` | — | Missing |
| `ContextFilter<T>` | — | Missing |
| `RunThreadedFilter(...)` | `RunThreadedFilter()` | Empty signature only |

### Utilities (`util/`)

| C++ | Rust (`src/utils/`) | Status |
| --- | --- | --- |
| `MurmurHashNative` | `murmur_hash_64a()` | |
| `bit_packing.hh` (read/write 57-bit) | `bit_packing.rs` | |
| `required_bits()` | `required_bits()` | |
| `FilePiece` (buffered reader) | `FilePiece` | Core path; no mmap/compressed |
| `StringPiece` | `StringPiece<'a>` | |
| `Tokenize` utilities | `tokenize.rs` | |
| `ProbingHashTable<E, H>` | replaced by `HashMap<u64, V>` | Different collision policy |
| `SortedUniform` (binary search table) | — | |
| `ReadCompressed` (gzip/bzip2/xz) | — | Plain text only |
| `mmap.hh` | — | No memory-mapped I/O |
| `PCQueue<T>` (producer-consumer) | `PCQueue<T>` stub | Placeholder |
| `stream/Chain` pipeline | `src/stream/` | Placeholders |

---

## Module Map

| Rust Module | C++ Source | Purpose | Completion |
| --- | --- | --- | --- |
| `src/arpa_reader.rs` | `lm/read_arpa.hh/.cc` | Parse ARPA text format | ~95% |
| `src/vocabulary.rs` | `lm/vocab.hh` | Word↔index mapping | ~75% (no binary load, no GrowableVocab) |
| `src/model.rs` | `lm/model.hh`, `lm/facade.hh` | GenericModel + Model trait | ~70% (scoring complete; file load stub) |
| `src/search.rs` | `lm/search_hashed.hh/.cc`, `lm/search_trie.hh` | HashedSearch (complete) + TrieSearch (stub) | ~55% |
| `src/trie.rs` | `lm/trie.hh/.cc` | Bit-packed trie node structures | ~50% (structures done; not wired to TrieSearch) |
| `src/types.rs` | `lm/state.hh`, `lm/return.hh`, `lm/config.hh` | State, Left, ChartState, Config | ~90% |
| `src/constant.rs` | `lm/model_type.hh`, `lm/blank.hh` | Constants, model type enum | ~95% |
| `src/error.rs` | `util/exception.hh`, `lm/lm_exception.hh` | `LMError` enum | ~80% |
| `src/quantize.rs` | `lm/quantize.hh` | `DontQuantize`, `SeparatelyQuantize` | ~90% |
| `src/bhiksha.rs` | `lm/bhiksha.hh` | `DontBhiksha`, `ArrayBhiksha` | ~90% |
| `src/ngram/binary_format.rs` | `lm/binary_format.hh` | Binary model file I/O | ~15% (header only) |
| `src/common/ngram.rs` | `lm/common/ngram.hh` | `NGram`, `NGramMut`, `OwnedNGram`, iterator | ~90% |
| `src/common/ordering.rs` | `lm/common/compare.hh` | Suffix/Context/Prefix ordering | ~80% |
| `src/common/buffer.rs` | `lm/common/model_buffer.hh` | Intermediate file buffering | ~30% |
| `src/utils/hash.rs` | `util/murmur_hash.hh` | MurmurHash64A | ~90% |
| `src/utils/bit_packing.rs` | `util/bit_packing.hh` | Variable-width integer/float pack | ~85% |
| `src/utils/pieces/file.rs` | `util/file_piece.hh` | Buffered file reader | ~70% (no mmap/compressed) |
| `src/utils/pieces/string.rs` | `util/string_piece.hh` | Non-owning string view | ~85% |
| `src/utils/pieces/tokenize.rs` | `util/tokenize_piece.hh` | Delimiter-based tokenizer | ~80% |
| `src/builder/` | `lm/builder/` | Builder pipeline (`lmplz`) | ~5% |
| `src/stream/` | `util/stream/` | Streaming chain infrastructure | ~5% |
| `src/filter.rs` | `lm/filter/` | Model filtering/pruning | ~5% |
| `src/interpolate.rs` | `lm/interpolate/` | Multi-model interpolation | ~5% |

---

## Key Design Decisions

### 1. Generic Model over Search and Vocabulary

The central type is:

```rust
pub struct GenericModel<S: Search, V: Vocabulary + VocabularyNew + VocabularySize> {
    backing: BinaryFormat,
    vocab: V,
    search: S,
}
```

This mirrors the C++ `GenericModel<Search, VocabularyT>` template. All six concrete model types are created via a macro:

```rust
define_model!(ProbingModel, HashedSearch<BackoffValue>, ProbingVocabulary);
define_model!(TrieModel,    TrieSearch<DontQuantize, DontBhiksha>, SortedVocabulary);
// ...
```

The macro also exposes `null_context_state`, `begin_sentence_state`, `get_state`, and `full_score_forgot_state` on every concrete type — avoiding the need for dynamic dispatch.

**vs C++**: C++ uses `ModelFacade<Child, State, Vocab>` for CRTP-based static dispatch plus a `base::Model` virtual interface for runtime polymorphism. The Rust `Model` trait provides the static interface; `load_virtual()` currently lacks the runtime polymorphism (always returns `ProbingModel`).

### 2. `HashedSearch` uses `HashMap` instead of Probing Hash Table

The C++ uses a custom open-addressing probing hash table (`util/probing_hash_table.hh`) with configurable load factor. The Rust port uses `std::collections::HashMap<u64, ProbBackoff>`:

```rust
struct MiddleTable { data: HashMap<u64, ProbBackoff> }
```

**Impact**: Correct but ~2–4× more memory and slower under high load than the C++ probing table. Replacing with an open-addressing table is on the roadmap.

### 3. Sign-Bit Encoding for `independent_left`

The C++ encodes "this n-gram has no higher-order match" by setting the IEEE 754 sign bit of the stored probability. Rust mirrors this exactly:

```rust
fn mark_independent(prob: f32) -> f32 { f32::from_bits(prob.to_bits() | 0x80000000) }
fn mark_extends_left(prob: f32) -> f32 { f32::from_bits(prob.to_bits() & !0x80000000u32) }
fn is_independent_left(prob: f32) -> bool { prob.to_bits() & 0x80000000 != 0 }
fn scoring_prob(stored_prob: f32) -> f32 { f32::from_bits(stored_prob.to_bits() | 0x80000000) }
```

`Pointer::prob()` forces the sign bit ON before returning (so the value is always a valid negative log-prob). `Pointer::independent_left()` checks the stored sign bit.

### 4. Extension Markers: `-0.0` vs `+0.0`

C++ `blank.hh` uses `kNoExtensionBackoff = -0.0` (bit pattern `0x80000000`) to mean "no higher-order child exists". Everything else — including `+0.0` — means "has a child".

```rust
fn has_extension(backoff: f32) -> bool { backoff.to_bits() != 0x80000000 }
fn set_extension(backoff: &mut f32) {
    if backoff.to_bits() == 0x80000000 { *backoff = 0.0_f32; }
}
```

This is a critical correctness detail: `abs() > epsilon` would be wrong because it cannot distinguish `+0.0` from `-0.0`.

### 5. `TrieSearch` uses `Vec<Middle>` (safe smart pointer)

The C++ `TrieSearch` uses raw `*mut Middle` pointers with manual `alloc`/`dealloc` and a custom `Drop`. The Rust port replaces this with `Vec<Middle>`:

```rust
pub struct TrieSearch<Quant, Bhiksha> {
    middles: Vec<Middle>,   // owns the heap allocation; Drop is automatic
    ...
}
```

`Vec` acts as the owning smart pointer: it manages the allocation lifecycle without any `unsafe`. No `Drop` impl needed. The `order()` method reads `self.middles.len()` instead of subtracting raw pointer addresses.

### 6. `NGramIterator` — Typed Ownership Instead of Raw Bytes

The original NGramIterator transmuted a raw `&[u8]` buffer to typed values (two `unsafe` blocks). The Rust port uses owned typed entries:

```rust
pub struct OwnedNGram<Payload> { pub words: Vec<WordIndex>, pub payload: Payload }
pub struct NGramIterator<Payload> { entries: std::vec::IntoIter<OwnedNGram<Payload>>, ... }
```

Zero `unsafe`. The caller parses binary data into typed entries before constructing the iterator.

### 7. No `unsafe` Code

The entire codebase has **zero `unsafe` blocks**. The C++ uses raw memory arithmetic extensively for performance. The Rust port achieves safety through:

- `Vec<T>` instead of raw `*mut T` pointer pairs
- Safe byte extraction via `to_ne_bytes()` instead of `from_raw_parts`
- Typed entry ownership instead of `&[u8]` transmutation
- `pub unsafe fn` removed (the one `set_from_ptr` C-compat function was deleted)

### 8. Scoring Walk-Through

Given `full_score(in_state, word, out_state)` for a 3-gram model with context `["foo", "bar"]`:

```text
1. lookup_unigram("baz")
   → prob=-1.2, backoff=-0.1   (sign bit cleared if "baz" has bigram children)
   out_state.backoff[0] = -0.1
   out_state.words[0] = index("baz")
   ret.prob = -1.2, ret.ngram_length = 1

2. lookup_middle(order=0, "bar", node)    ← checks 2-gram "baz|bar"
   → found: prob=-0.8, backoff=-0.05
   out_state.backoff[1] = -0.05
   ret.prob = -0.8, ret.ngram_length = 2

3. lookup_longest("foo", node)            ← checks 3-gram "baz|bar|foo"
   → found: prob=-0.5
   ret.prob = -0.5, ret.ngram_length = 3
   ret.independent_left = true            ← full match, stop context search

4. Accumulate backoffs from in_state for unmatched positions:
   ngram_length=3, state.length=2  →  no backoffs needed (full match)

5. Return ret.prob = -0.5
```

---

## What Is Not Implemented

### Blocking: Can't Load Models from Disk

| Gap | C++ Location | Impact |
| --- | --- | --- |
| `GenericModel` constructor file I/O | `lm/model.hh` | No model loads from ARPA/binary |
| Binary format write + mmap read | `lm/binary_format.hh` | No `.bin` file support |
| `TrieSearch::initialize_from_arpa` | `lm/search_trie.hh` | Trie model family non-functional |
| Wiring `BitPackedMiddle/Longest` into `TrieSearch` | `lm/trie.hh` | Bit-packing code exists but unused |

### Blocking: Decoder Integration Missing

| Gap | C++ Location | Impact |
| --- | --- | --- |
| `GenericModel::ExtendLeft()` | `lm/model.hh` | SMT left-context extension |
| `GenericModel::UnRest()` | `lm/model.hh` | Rest-cost computation |
| `RuleScore<M>` | `lm/left.hh` | Chart-based decoding state machine |
| `RevealBefore/After/Subsume` | `lm/partial.hh` | Partial state utilities |
| `base::Model` virtual interface (runtime) | `lm/virtual_interface.hh` | Runtime model type selection |

### Medium Priority

| Gap | Impact |
| --- | --- |
| `GrowableVocab<NewWordAction>` | Building models from raw text |
| `EnumerateVocab` wired to loading | Vocabulary callbacks during model load |
| Probing hash table (replace `HashMap`) | Memory + performance parity with C++ |
| Memory-mapped I/O | Large model loading |
| Compressed input (gz/bz2/xz) | `FilePiece` accepts plain text only |
| `SortedVocabulary::LoadedBinary()` + `Relocate()` | Binary vocabulary loading |

### Low Priority

| Gap | Impact |
| --- | --- |
| Builder pipeline (`lmplz`) | Cannot train models from corpus |
| Interpolation pipeline | Multi-model combination |
| ARPA filter | Vocabulary-restricted model pruning |
| Stream/chain infrastructure | Parallel model building |
| Benchmark utilities | Performance measurement |
