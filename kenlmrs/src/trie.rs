use std::alloc::{ alloc, dealloc, Layout };
use std::mem::{ size_of, MaybeUninit };
use std::ptr::{ null_mut, write };
use std::marker::PhantomData;

use crate::types::{ ModelType, WordIndex };
use crate::utils::pieces::file::FilePiece;
use crate::utils::bit_packing::BitAddress;
use crate::ngram::binary_format::BinaryFormat;

type Config = ();

pub struct SortedVocabulary;
pub struct SortedFiles;

// Placeholder pointer structures
pub struct MiddlePointer {
    // Will contain pointer to quantized middle data
}

impl MiddlePointer {
    pub fn new<Q: Quantization>(_quant: &Q, _order: u8, _address: BitAddress) -> Self {
        MiddlePointer {}
    }

    pub fn found(&self) -> bool {
        // Placeholder - check if pointer is valid
        true
    }
}

pub struct LongestPointer {
    // Will contain pointer to longest n-gram data
}

impl LongestPointer {
    pub fn new<Q: Quantization>(_quant: &Q, _address: BitAddress) -> Self {
        LongestPointer {}
    }
}

const TRIE_SORTED: u8 = 0;
const MODEL_TYPE: u8 = 1;

// Trait for quantization implementations
pub trait Quantization: Default {
    fn update_config_from_binary(file: &BinaryFormat, offset: u64, config: &mut Config);
    fn size(order: usize, config: &Config) -> u64;
    fn middle_bits(config: &Config) -> u8;
    fn longest_bits(config: &Config) -> u8;
}

// Trait for Bhiksha implementations
pub trait BhikshaImpl {
    fn update_config_from_binary(file: &BinaryFormat, offset: u64, config: &mut Config);
}

pub struct TrieSearch<Quant: Quantization, Bhiksha: BhikshaImpl> {
    middle_begin_: *mut Middle,
    middle_end_: *mut Middle,
    quant_: Quant,
    longest_: Longest,
    unigram_: Unigram,
    _phantom: PhantomData<Bhiksha>,
}

impl<Quant: Quantization, Bhiksha: BhikshaImpl> TrieSearch<Quant, Bhiksha> {
    pub const K_DIFFERENT_REST: bool = false;
    pub const K_MODEL_TYPE: ModelType = ModelType::Trie;
    pub const K_VERSION: u8 = 1;

    pub fn new() -> Self {
        TrieSearch {
            middle_begin_: null_mut(),
            middle_end_: null_mut(),
            quant_: Default::default(),
            longest_: Longest::new(),
            unigram_: Unigram::new(),
            _phantom: PhantomData,
        }
    }

    pub fn update_config_from_binary(
        file: &BinaryFormat,
        counts: &Vec<u64>,
        offset: u64,
        config: &mut Config
    ) {
        Quant::update_config_from_binary(file, offset, config);
        if counts.len() > 2 {
            Bhiksha::update_config_from_binary(
                file,
                offset + Quant::size(counts.len(), config) + Unigram::size(counts[0]),
                config
            );
        }
    }

    pub fn size(counts: &Vec<u64>, config: &Config) -> u64 {
        let mut ret = Quant::size(counts.len(), config) + Unigram::size(counts[0]);
        for i in 1..counts.len() - 1 {
            ret += Middle::size(
                Quant::middle_bits(config),
                counts[i],
                counts[0],
                counts[i + 1],
                config
            );
        }
        ret + Longest::size(Quant::longest_bits(config), counts[counts.len() - 1], counts[0])
    }

    pub fn setup_memory(&mut self, start: *mut u8, counts: &Vec<u64>, config: &Config) -> *mut u8 {
        // Memory layout:
        // [Quantization tables] [Unigram array] [Middle[0]] [Middle[1]] ... [Longest]

        let mut position = start;

        // Step 1: Setup quantization tables (probability/backoff compression)
        // The quantization object manages its own memory layout
        // For DontQuantize, this returns position unchanged
        // For SeparatelyQuantize, this allocates space for lookup tables
        // position = self.quant_.setup_memory(position, counts.len(), config);
        // TODO: Uncomment above when Quantization trait has setup_memory method

        // Step 2: Setup unigram table
        // Unigrams are stored as an array of UnigramValue structs
        let unigram_size = Unigram::size(counts[0]);
        unsafe {
            // Initialize unigram with the allocated memory
            let unigram_slice = std::slice::from_raw_parts_mut(
                position as *mut UnigramValue,
                (counts[0] + 2) as usize // +1 for unknown, +1 for end marker
            );
            // Zero out the memory
            for item in unigram_slice.iter_mut() {
                *item = UnigramValue::default();
            }
            // self.unigram_.init() would go here, but it takes Vec, not slice
            position = position.add(unigram_size as usize);
        }

        // Step 3: Setup middle layers
        // Each middle layer is a bit-packed array with word indices, probabilities, and next pointers
        let middle_count = if counts.len() > 2 { counts.len() - 2 } else { 0 };

        if middle_count > 0 {
            self.middle_begin_ = position as *mut Middle;

            for i in 0..middle_count {
                let middle_bits = Quant::middle_bits(config);
                let middle_size = Middle::size(
                    middle_bits,
                    counts[i + 1], // Number of n-grams in this layer
                    counts[0], // Vocab size (for word bits)
                    counts[i + 2], // Next layer size (for next pointer bits)
                    config
                );

                // TODO: Actually initialize Middle structure here
                // This requires implementing BitPackedMiddle properly
                // For now, just allocate the space
                unsafe {
                    position = position.add(middle_size as usize);
                }
            }

            self.middle_end_ = position as *mut Middle;
        } else {
            // No middle layers (bigram model or less)
            self.middle_begin_ = null_mut();
            self.middle_end_ = null_mut();
        }

        // Step 4: Setup longest layer
        // The longest layer stores final n-grams with just word indices and probabilities
        // No next pointers needed since there's nothing after
        let longest_bits = Quant::longest_bits(config);
        let longest_size = Longest::size(
            longest_bits,
            counts[counts.len() - 1], // Number of longest n-grams
            counts[0] // Vocab size
        );

        // TODO: Actually initialize Longest structure here
        // For now, just allocate the space
        unsafe {
            position = position.add(longest_size as usize);
        }

        position
    }

    pub fn initialize_from_arpa(
        &mut self,
        file: &str,
        f: &mut FilePiece,
        counts: &mut Vec<u64>,
        config: &Config,
        vocab: &mut SortedVocabulary,
        backing: &mut BinaryFormat
    ) {
        // Implementation for initializing from ARPA file
        unimplemented!()
    }

    pub fn order(&self) -> u8 {
        // Calculate the distance between pointers
        let distance = unsafe {
            (self.middle_end_ as usize).wrapping_sub(self.middle_begin_ as usize) /
                size_of::<Middle>()
        };
        (distance as u8) + 2
    }

    pub fn unknown_unigram(&mut self) -> &mut ProbBackoff {
        self.unigram_.unknown()
    }

    pub fn lookup_unigram(
        &self,
        word: WordIndex,
        next: &mut NodeRange,
        independent_left: &mut bool,
        extend_left: &mut u64
    ) -> UnigramPointer {
        *extend_left = word as u64;
        let ret = self.unigram_.find(word as usize, next);
        *independent_left = next.begin == next.end;
        ret
    }

    pub fn unpack(
        &self,
        extend_pointer: u64,
        extend_length: u8,
        node: &mut NodeRange
    ) -> MiddlePointer {
        MiddlePointer::new(&self.quant_, extend_length - 2, unsafe {
            (*self.middle_begin_.add((extend_length - 2) as usize)).read_entry(extend_pointer, node)
        })
    }

    pub fn lookup_middle(
        &self,
        order_minus_2: u8,
        word: WordIndex,
        node: &mut NodeRange,
        independent_left: &mut bool,
        extend_left: &mut u64
    ) -> MiddlePointer {
        let address = unsafe {
            (*self.middle_begin_.add(order_minus_2 as usize)).find(word, node, extend_left)
        };
        *independent_left = address.base.is_empty() || node.begin == node.end;
        MiddlePointer::new(&self.quant_, order_minus_2, address)
    }

    pub fn lookup_longest(&self, word: WordIndex, node: &NodeRange) -> LongestPointer {
        LongestPointer::new(&self.quant_, self.longest_.find(word, node))
    }

    pub fn fast_make_node(
        &self,
        begin: &[WordIndex],
        end: &[WordIndex],
        node: &mut NodeRange
    ) -> bool {
        assert!(!begin.is_empty());
        let mut independent_left = false;
        let mut ignored = 0;
        self.lookup_unigram(begin[0], node, &mut independent_left, &mut ignored);
        for (idx, word) in begin.iter().enumerate().skip(1) {
            if idx >= end.len() {
                break;
            }
            if
                independent_left ||
                !self
                    .lookup_middle(idx as u8, *word, node, &mut independent_left, &mut ignored)
                    .found()
            {
                return false;
            }
        }
        true
    }

    fn free_middles(&mut self) {
        unsafe {
            let middle_count =
                (self.middle_end_ as usize).wrapping_sub(self.middle_begin_ as usize) /
                size_of::<Middle>();
            for i in 0..middle_count {
                std::ptr::drop_in_place(self.middle_begin_.add(i));
            }
            if !self.middle_begin_.is_null() {
                dealloc(
                    self.middle_begin_ as *mut u8,
                    Layout::array::<Middle>(middle_count).unwrap()
                );
            }
        }
    }
}

impl<Quant: Quantization, Bhiksha: BhikshaImpl> Drop for TrieSearch<Quant, Bhiksha> {
    fn drop(&mut self) {
        self.free_middles();
    }
}

// Placeholder structs and functions for `Quant`, `Bhiksha`, `NodeRange`, `Unigram`, `Middle`, and `Longest` that must be implemented
mod util {
    pub struct FilePiece;
    impl FilePiece {
        pub fn new() -> Self {
            FilePiece
        }
    }
}

pub struct Middle;

impl Middle {
    /// Calculate memory size for a middle trie layer
    ///
    /// # Arguments
    /// * `middle_bits` - Bits for quantized probability/backoff
    /// * `count` - Number of n-grams in this layer
    /// * `base_count` - Number of unigrams (for Bhiksha compression)
    /// * `next_count` - Number of n-grams in next layer (for next pointers)
    /// * `config` - Model configuration
    pub fn size(
        middle_bits: u8,
        count: u64,
        base_count: u64,
        next_count: u64,
        config: &Config
    ) -> u64 {
        // For now, use simple calculation without Bhiksha
        // TODO: Add Bhiksha compression support

        // Size = word_bits + middle_bits + next_pointer_bits
        // For MVP, we'll estimate next pointer bits
        use crate::utils::bit_packing::required_bits;

        let word_bits = required_bits(base_count);
        let next_bits = required_bits(next_count);
        let total_bits = word_bits + middle_bits + next_bits;

        // Similar to BitPacked::base_size
        ((count + 1) * (total_bits as u64) + 7) / 8 + 8
    }

    pub fn read_entry(&self, _extend_pointer: u64, _node: &mut NodeRange) -> BitAddress {
        // Placeholder - will implement with full Middle structure
        BitAddress::new(Vec::new(), 0)
    }

    pub fn find(&self, word: WordIndex, node: &mut NodeRange, extend_left: &mut u64) -> BitAddress {
        // Placeholder - will implement with full Middle structure
        *extend_left = word as u64;
        *node = NodeRange { begin: 0, end: 0 };
        BitAddress::new(Vec::new(), 0)
    }
}

pub struct Longest;
impl Longest {
    pub fn new() -> Self {
        Longest
    }

    /// Calculate memory size for longest n-gram layer
    pub fn size(longest_bits: u8, count: u64, base_count: u64) -> u64 {
        // Longest layer is just BitPackedLongest
        // base_count is the max_vocab (number of unigrams)
        BitPackedLongest::size(longest_bits, count, base_count)
    }

    pub fn find(&self, _word: WordIndex, _node: &NodeRange) -> BitAddress {
        // Placeholder - will be implemented with actual BitPackedLongest
        BitAddress::new(Vec::new(), 0)
    }
}

impl<Quant: Quantization, Bhiksha: BhikshaImpl> TrieSearch<Quant, Bhiksha> {
    pub fn middle_bits(config: &Config) -> u8 {
        // Placeholder implementation
        unimplemented!()
    }

    pub fn longest_bits(config: &Config) -> u8 {
        // Placeholder implementation
        unimplemented!()
    }
}

#[derive(Default, Clone, Copy)]
pub struct NodeRange {
    pub begin: u64,
    pub end: u64,
}

// Definition of ProbBackoff struct
#[derive(Default, Clone, Copy)]
pub struct ProbBackoff {
    pub prob: f32,
    pub backoff: f32,
}

// Definition of UnigramValue struct
#[derive(Default, Clone, Copy)]
struct UnigramValue {
    weights: ProbBackoff,
    next: u64,
}

impl UnigramValue {
    fn next(&self) -> u64 {
        self.next
    }
}

// Definition of UnigramPointer struct
#[derive(Default, Clone, Copy)]
pub struct UnigramPointer<'a> {
    to: Option<&'a ProbBackoff>,
}

impl<'a> UnigramPointer<'a> {
    pub fn new(to: &'a ProbBackoff) -> Self {
        UnigramPointer { to: Some(to) }
    }

    pub fn found(&self) -> bool {
        self.to.is_some()
    }

    pub fn prob(&self) -> f32 {
        self.to.map_or(0.0, |p| p.prob)
    }

    pub fn backoff(&self) -> f32 {
        self.to.map_or(0.0, |p| p.backoff)
    }

    pub fn rest(&self) -> f32 {
        self.prob()
    }
}

// Definition of Unigram struct
struct Unigram {
    unigram: Vec<UnigramValue>,
}

impl Unigram {
    pub fn new() -> Self {
        Unigram {
            unigram: Vec::new(),
        }
    }

    pub fn init(&mut self, start: Vec<UnigramValue>) {
        self.unigram = start;
    }

    pub fn size(count: u64) -> u64 {
        (count + 2) * (std::mem::size_of::<UnigramValue>() as u64)
    }

    pub fn lookup(&self, index: usize) -> &ProbBackoff {
        &self.unigram[index].weights
    }

    pub fn unknown(&mut self) -> &mut ProbBackoff {
        &mut self.unigram[0].weights
    }

    pub fn raw(&mut self) -> &mut [UnigramValue] {
        &mut self.unigram
    }

    pub fn find(&self, word: usize, next: &mut NodeRange) -> UnigramPointer {
        let val = &self.unigram[word];
        next.begin = val.next;
        next.end = self.unigram.get(word + 1).map_or(0, |v| v.next);
        UnigramPointer::new(&val.weights)
    }
}

// Definition of BitPacked struct
#[derive(Debug, Default)]
pub struct BitPacked {
    word_bits: u8,
    total_bits: u8,
    word_mask: u64,
    base: Vec<u8>,
    insert_index: u64,
    max_vocab: u64,
}

impl BitPacked {
    pub fn new() -> Self {
        BitPacked {
            word_bits: 0,
            total_bits: 0,
            word_mask: 0,
            base: Vec::new(),
            insert_index: 0,
            max_vocab: 0,
        }
    }

    pub fn insert_index(&self) -> u64 {
        self.insert_index
    }

    /// Calculate memory size needed for bit-packed storage
    ///
    /// # Arguments
    /// * `entries` - Number of n-gram entries to store
    /// * `max_vocab` - Maximum vocabulary size (determines bits needed per word)
    /// * `remaining_bits` - Additional bits per entry (for probabilities, next pointers, etc.)
    ///
    /// # Returns
    /// Size in bytes needed for the bit-packed array
    pub fn base_size(entries: u64, max_vocab: u64, remaining_bits: u8) -> u64 {
        use crate::utils::bit_packing::required_bits;

        let word_bits = required_bits(max_vocab);
        let total_bits = word_bits + remaining_bits;

        // +1 for extra entry at the end (for end pointer)
        // +7 to round up when converting bits to bytes
        // +8 for safety margin to prevent ReadInt57 from going out of bounds
        ((entries + 1) * (total_bits as u64) + 7) / 8 + 8
    }

    /// Initialize bit-packed array
    ///
    /// # Arguments
    /// * `base` - Pre-allocated byte buffer (must be zeroed)
    /// * `max_vocab` - Maximum vocabulary size
    /// * `remaining_bits` - Additional bits per entry beyond word index
    pub fn base_init(&mut self, base: Vec<u8>, max_vocab: u64, remaining_bits: u8) {
        use crate::utils::bit_packing::required_bits;

        // Calculate bits needed for word indices
        self.word_bits = required_bits(max_vocab);

        // Create mask for extracting word bits
        self.word_mask = if self.word_bits > 0 { (1u64 << self.word_bits) - 1 } else { 0 };

        // Total bits = word bits + probability/next pointer bits
        self.total_bits = self.word_bits + remaining_bits;

        // Store the buffer
        self.base = base;

        // Start inserting at index 0
        self.insert_index = 0;

        // Store max vocab for validation
        self.max_vocab = max_vocab;
    }
}

// Definition of BitPackedMiddle struct
#[derive(Debug)]
struct BitPackedMiddle {
    bit_packed: BitPacked,
    quant_bits: u8,
    next_bits: u8,
    next_mask: u64,
}

impl BitPackedMiddle {
    /// Create a new middle layer
    ///
    /// # Arguments
    /// * `base` - Pre-allocated zero-filled byte buffer
    /// * `quant_bits` - Bits for quantized probability/backoff
    /// * `entries` - Number of n-grams in this layer
    /// * `max_vocab` - Maximum vocabulary size (determines word bits)
    /// * `max_next` - Maximum next pointer value (determines next bits)
    pub fn new(base: Vec<u8>, quant_bits: u8, entries: u64, max_vocab: u64, max_next: u64) -> Self {
        use crate::utils::bit_packing::required_bits;

        // Calculate bits needed for next pointers
        let next_bits = required_bits(max_next);
        let next_mask = if next_bits > 0 { (1u64 << next_bits) - 1 } else { 0 };

        // Total bits per entry = word_bits + quant_bits + next_bits
        let remaining_bits = quant_bits + next_bits;

        let mut bit_packed = BitPacked::new();
        bit_packed.base_init(base, max_vocab, remaining_bits);

        BitPackedMiddle {
            bit_packed,
            quant_bits,
            next_bits,
            next_mask,
        }
    }

    /// Calculate memory size for middle layer
    ///
    /// For MVP, we don't use Bhiksha compression, so this is simpler
    pub fn size(
        quant_bits: u8,
        entries: u64,
        max_vocab: u64,
        max_next: u64,
        _config: &Config
    ) -> u64 {
        use crate::utils::bit_packing::required_bits;

        let next_bits = required_bits(max_next);
        let remaining_bits = quant_bits + next_bits;

        BitPacked::base_size(entries, max_vocab, remaining_bits)
    }

    /// Insert a word during trie construction
    ///
    /// Returns the BitAddress where probability should be written
    /// The next pointer will be written by the caller after determining the next index
    pub fn insert(&mut self, word: WordIndex) -> BitAddress {
        use crate::utils::bit_packing::write_int57;

        assert!(
            (word as u64) <= self.bit_packed.word_mask,
            "Word index {} exceeds max vocab",
            word
        );

        // Calculate bit offset for this entry
        let at_pointer = self.bit_packed.insert_index * (self.bit_packed.total_bits as u64);

        // Write the word index
        write_int57(&mut self.bit_packed.base, at_pointer, self.bit_packed.word_bits, word as u64);

        // Probability will be written at: at_pointer + word_bits
        let prob_offset = at_pointer + (self.bit_packed.word_bits as u64);

        // Next pointer will be at: prob_offset + quant_bits
        // (written later by write_next)

        // Increment insert index
        self.bit_packed.insert_index += 1;

        BitAddress::new(self.bit_packed.base.clone(), prob_offset)
    }

    /// Write next pointer for an entry
    ///
    /// Call this after insert() to set the next pointer value
    pub fn write_next(&mut self, entry_index: u64, next_value: u64) {
        use crate::utils::bit_packing::write_int57;

        let at_pointer = entry_index * (self.bit_packed.total_bits as u64);
        let next_offset =
            at_pointer + (self.bit_packed.word_bits as u64) + (self.quant_bits as u64);

        write_int57(&mut self.bit_packed.base, next_offset, self.next_bits, next_value);
    }

    /// Finalize after all entries loaded
    pub fn finished_loading(&mut self, next_end: u64) {
        // Write the final next pointer (points past the end)
        if self.bit_packed.insert_index > 0 {
            self.write_next(self.bit_packed.insert_index, next_end);
        }
    }

    /// Binary search to find a word in the given range
    ///
    /// Returns BitAddress of probability and updates pointer/range for next layer
    pub fn find(&self, word: WordIndex, range: &mut NodeRange, pointer: &mut u64) -> BitAddress {
        use crate::utils::bit_packing::read_int57;

        // Binary search in range [range.begin, range.end)
        let mut left = range.begin;
        let mut right = range.end;

        while left < right {
            let mid = (left + right) / 2;

            // Read word at mid index
            let mid_offset = mid * (self.bit_packed.total_bits as u64);
            let mid_word = read_int57(
                &self.bit_packed.base,
                mid_offset,
                self.bit_packed.word_bits,
                self.bit_packed.word_mask
            ) as WordIndex;

            if mid_word < word {
                left = mid + 1;
            } else if mid_word > word {
                right = mid;
            } else {
                // Found! Read the next pointer to determine range for next layer
                *pointer = mid;

                let prob_offset = mid_offset + (self.bit_packed.word_bits as u64);
                let next_offset = prob_offset + (self.quant_bits as u64);

                // Read this entry's next pointer
                let next_begin = read_int57(
                    &self.bit_packed.base,
                    next_offset,
                    self.next_bits,
                    self.next_mask
                );

                // Read next entry's next pointer (for end of range)
                let next_entry_offset = (mid + 1) * (self.bit_packed.total_bits as u64);
                let next_entry_next_offset =
                    next_entry_offset +
                    (self.bit_packed.word_bits as u64) +
                    (self.quant_bits as u64);
                let next_end = read_int57(
                    &self.bit_packed.base,
                    next_entry_next_offset,
                    self.next_bits,
                    self.next_mask
                );

                // Update range for next layer lookup
                range.begin = next_begin;
                range.end = next_end;

                return BitAddress::new(self.bit_packed.base.clone(), prob_offset);
            }
        }

        // Not found - return null address
        BitAddress::new(Vec::new(), 0)
    }

    /// Read an entry by direct pointer access
    ///
    /// Used when we already know the index (from unpacking a pointer)
    pub fn read_entry(&self, pointer: u64, range: &mut NodeRange) -> BitAddress {
        use crate::utils::bit_packing::read_int57;

        let at_pointer = pointer * (self.bit_packed.total_bits as u64);
        let prob_offset = at_pointer + (self.bit_packed.word_bits as u64);
        let next_offset = prob_offset + (self.quant_bits as u64);

        // Read this entry's next pointer
        let next_begin = read_int57(
            &self.bit_packed.base,
            next_offset,
            self.next_bits,
            self.next_mask
        );

        // Read next entry's next pointer (for end of range)
        let next_entry_offset = (pointer + 1) * (self.bit_packed.total_bits as u64);
        let next_entry_next_offset =
            next_entry_offset + (self.bit_packed.word_bits as u64) + (self.quant_bits as u64);
        let next_end = read_int57(
            &self.bit_packed.base,
            next_entry_next_offset,
            self.next_bits,
            self.next_mask
        );

        // Update range
        range.begin = next_begin;
        range.end = next_end;

        BitAddress::new(self.bit_packed.base.clone(), prob_offset)
    }
}

// Definition of BitPackedLongest struct
pub struct BitPackedLongest {
    bit_packed: BitPacked,
}

impl BitPackedLongest {
    pub fn new() -> Self {
        BitPackedLongest {
            bit_packed: BitPacked::new(),
        }
    }

    /// Calculate memory size for longest n-gram layer
    pub fn size(quant_bits: u8, entries: u64, max_vocab: u64) -> u64 {
        BitPacked::base_size(entries, max_vocab, quant_bits)
    }

    /// Initialize the longest layer
    ///
    /// # Arguments
    /// * `base` - Pre-allocated zero-filled byte buffer
    /// * `quant_bits` - Bits needed for probability/quantization value
    /// * `max_vocab` - Maximum vocabulary size
    pub fn init(&mut self, base: Vec<u8>, quant_bits: u8, max_vocab: u64) {
        self.bit_packed.base_init(base, max_vocab, quant_bits);
    }

    /// Insert a word during trie construction
    /// Returns the BitAddress where the probability should be written
    pub fn insert(&mut self, word: WordIndex) -> BitAddress {
        use crate::utils::bit_packing::write_int57;

        assert!(
            (word as u64) <= self.bit_packed.word_mask,
            "Word index {} exceeds max vocab",
            word
        );

        // Calculate bit offset for this entry
        let at_pointer = self.bit_packed.insert_index * (self.bit_packed.total_bits as u64);

        // Write the word index
        write_int57(&mut self.bit_packed.base, at_pointer, self.bit_packed.word_bits, word as u64);

        // Probability will be written at: at_pointer + word_bits
        let prob_offset = at_pointer + (self.bit_packed.word_bits as u64);

        // Increment insert index for next entry
        self.bit_packed.insert_index += 1;

        BitAddress::new(self.bit_packed.base.clone(), prob_offset)
    }

    /// Binary search to find a word in the given node range
    /// Returns BitAddress of the probability if found
    pub fn find(&self, word: WordIndex, node: &NodeRange) -> BitAddress {
        use crate::utils::bit_packing::read_int57;

        // Binary search in range [node.begin, node.end)
        let mut left = node.begin;
        let mut right = node.end;

        while left < right {
            let mid = (left + right) / 2;

            // Read word at mid index
            let mid_offset = mid * (self.bit_packed.total_bits as u64);
            let mid_word = read_int57(
                &self.bit_packed.base,
                mid_offset,
                self.bit_packed.word_bits,
                self.bit_packed.word_mask
            ) as WordIndex;

            if mid_word < word {
                left = mid + 1;
            } else if mid_word > word {
                right = mid;
            } else {
                // Found! Return address of probability
                let prob_offset = mid_offset + (self.bit_packed.word_bits as u64);
                return BitAddress::new(self.bit_packed.base.clone(), prob_offset);
            }
        }

        // Not found - return null address
        BitAddress::new(Vec::new(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::bit_packing::{ read_int57, write_int57, required_bits };

    #[test]
    fn test_bit_packed_base_size() {
        // Test size calculations for various configurations

        // Small vocab, few entries
        let size = BitPacked::base_size(100, 1000, 8);
        assert!(size > 0, "Size should be positive");

        // Large vocab
        let size_large = BitPacked::base_size(10000, 50000, 8);
        assert!(size_large > size, "Larger vocab should need more space");

        // More bits per entry
        let size_more_bits = BitPacked::base_size(100, 1000, 16);
        assert!(size_more_bits > size, "More bits should need more space");
    }

    #[test]
    fn test_bit_packed_base_init() {
        let mut bit_packed = BitPacked::new();
        let buffer = vec![0u8; 1000];

        bit_packed.base_init(buffer, 50000, 8);

        assert_eq!(bit_packed.word_bits, required_bits(50000));
        assert_eq!(bit_packed.total_bits, bit_packed.word_bits + 8);
        assert!(bit_packed.word_mask > 0);
        assert_eq!(bit_packed.insert_index, 0);
    }

    #[test]
    fn test_bit_packed_longest_insert_find() {
        let mut longest = BitPackedLongest::new();
        let buffer_size = BitPackedLongest::size(8, 100, 1000) as usize;
        let buffer = vec![0u8; buffer_size];

        longest.init(buffer, 8, 1000);

        // Insert some words
        let addr1 = longest.insert(100);
        let addr2 = longest.insert(200);
        let addr3 = longest.insert(300);
        let addr4 = longest.insert(500);

        assert!(!addr1.base.is_empty(), "Insert should return valid address");
        assert!(!addr2.base.is_empty(), "Insert should return valid address");
        assert!(!addr3.base.is_empty(), "Insert should return valid address");
        assert!(!addr4.base.is_empty(), "Insert should return valid address");

        // Search for inserted words
        let range = NodeRange { begin: 0, end: 4 };

        let found1 = longest.find(100, &range);
        assert!(!found1.base.is_empty(), "Should find word 100");

        let found2 = longest.find(200, &range);
        assert!(!found2.base.is_empty(), "Should find word 200");

        let found3 = longest.find(300, &range);
        assert!(!found3.base.is_empty(), "Should find word 300");

        let found4 = longest.find(500, &range);
        assert!(!found4.base.is_empty(), "Should find word 500");

        // Search for non-existent word
        let not_found = longest.find(150, &range);
        assert!(not_found.base.is_empty(), "Should not find word 150");
    }

    #[test]
    fn test_bit_packed_longest_empty_range() {
        let mut longest = BitPackedLongest::new();
        let buffer_size = BitPackedLongest::size(8, 10, 100) as usize;
        let buffer = vec![0u8; buffer_size];

        longest.init(buffer, 8, 100);

        // Insert one word
        longest.insert(50);

        // Search in empty range
        let empty_range = NodeRange { begin: 0, end: 0 };
        let not_found = longest.find(50, &empty_range);
        assert!(not_found.base.is_empty(), "Should not find in empty range");
    }

    #[test]
    fn test_bit_packed_longest_large_vocab() {
        let mut longest = BitPackedLongest::new();
        let max_vocab = 100_000;
        let buffer_size = BitPackedLongest::size(8, 1000, max_vocab) as usize;
        let buffer = vec![0u8; buffer_size];

        longest.init(buffer, 8, max_vocab);

        // Insert words with large indices
        longest.insert(10_000);
        longest.insert(50_000);
        longest.insert(99_999);

        let range = NodeRange { begin: 0, end: 3 };

        let found = longest.find(50_000, &range);
        assert!(!found.base.is_empty(), "Should find word with large index");
    }

    #[test]
    fn test_bit_packed_longest_binary_search_correctness() {
        let mut longest = BitPackedLongest::new();
        let buffer_size = BitPackedLongest::size(8, 100, 1000) as usize;
        let buffer = vec![0u8; buffer_size];

        longest.init(buffer, 8, 1000);

        // Insert sorted sequence
        for i in 0..50 {
            longest.insert(i * 10); // 0, 10, 20, 30, ...
        }

        let range = NodeRange { begin: 0, end: 50 };

        // Find all inserted values
        for i in 0..50 {
            let word = i * 10;
            let found = longest.find(word, &range);
            assert!(!found.base.is_empty(), "Should find word {}", word);
        }

        // Verify non-inserted values are not found
        for i in 0..50 {
            let word = i * 10 + 5; // 5, 15, 25, 35, ...
            let not_found = longest.find(word, &range);
            assert!(not_found.base.is_empty(), "Should not find word {}", word);
        }
    }

    #[test]
    fn test_bit_packed_middle_insert_find() {
        let max_vocab = 1000;
        let max_next = 5000;
        let entries = 100;

        let buffer_size = BitPackedMiddle::size(8, entries, max_vocab, max_next, &()) as usize;
        let buffer = vec![0u8; buffer_size];

        let mut middle = BitPackedMiddle::new(buffer, 8, entries, max_vocab, max_next);

        // Insert some words
        let addr1 = middle.insert(100);
        middle.write_next(0, 0); // First entry's next pointer

        let addr2 = middle.insert(200);
        middle.write_next(1, 10); // Second entry's next pointer

        let addr3 = middle.insert(300);
        middle.write_next(2, 20); // Third entry's next pointer

        middle.finished_loading(30); // Final next pointer

        assert!(!addr1.base.is_empty(), "Insert should return valid address");
        assert!(!addr2.base.is_empty(), "Insert should return valid address");
        assert!(!addr3.base.is_empty(), "Insert should return valid address");

        // Search for word
        let mut range = NodeRange { begin: 0, end: 3 };
        let mut pointer = 0;

        let found = middle.find(200, &mut range, &mut pointer);
        assert!(!found.base.is_empty(), "Should find word 200");
        assert_eq!(pointer, 1, "Should return correct pointer");
        assert_eq!(range.begin, 10, "Should update range begin");
        assert_eq!(range.end, 20, "Should update range end");
    }

    #[test]
    fn test_bit_packed_middle_read_entry() {
        let max_vocab = 1000;
        let max_next = 5000;

        let buffer_size = BitPackedMiddle::size(8, 10, max_vocab, max_next, &()) as usize;
        let buffer = vec![0u8; buffer_size];

        let mut middle = BitPackedMiddle::new(buffer, 8, 10, max_vocab, max_next);

        // Insert entries
        middle.insert(100);
        middle.write_next(0, 100);

        middle.insert(200);
        middle.write_next(1, 150);

        middle.finished_loading(200);

        // Read entry by pointer
        let mut range = NodeRange { begin: 0, end: 0 };
        let addr = middle.read_entry(1, &mut range);

        assert!(!addr.base.is_empty(), "Should return valid address");
        assert_eq!(range.begin, 150, "Should read correct next begin");
        assert_eq!(range.end, 200, "Should read correct next end");
    }

    #[test]
    fn test_bit_packed_middle_not_found() {
        let max_vocab = 1000;
        let max_next = 5000;

        let buffer_size = BitPackedMiddle::size(8, 10, max_vocab, max_next, &()) as usize;
        let buffer = vec![0u8; buffer_size];

        let mut middle = BitPackedMiddle::new(buffer, 8, 10, max_vocab, max_next);

        middle.insert(100);
        middle.write_next(0, 0);
        middle.insert(300);
        middle.write_next(1, 10);
        middle.finished_loading(20);

        // Search for non-existent word
        let mut range = NodeRange { begin: 0, end: 2 };
        let mut pointer = 0;

        let not_found = middle.find(200, &mut range, &mut pointer);
        assert!(not_found.base.is_empty(), "Should not find word 200");
    }

    #[test]
    fn test_middle_size_calculation() {
        // Test that Middle::size() returns reasonable values
        let size1 = Middle::size(8, 1000, 10000, 2000, &());
        assert!(size1 > 0, "Size should be positive");

        let size2 = Middle::size(8, 5000, 10000, 10000, &());
        assert!(size2 > size1, "More entries should need more space");

        let size3 = Middle::size(16, 1000, 10000, 2000, &());
        assert!(size3 > size1, "More quant bits should need more space");
    }

    #[test]
    fn test_longest_size_calculation() {
        let size1 = Longest::size(8, 1000, 10000);
        assert!(size1 > 0, "Size should be positive");

        let size2 = Longest::size(8, 5000, 10000);
        assert!(size2 > size1, "More entries should need more space");

        let size3 = Longest::size(16, 1000, 10000);
        assert!(size3 > size1, "More quant bits should need more space");
    }
}
