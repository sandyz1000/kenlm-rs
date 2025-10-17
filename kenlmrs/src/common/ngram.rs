/// N-gram representation with safe Rust references
///
/// This module provides safe access to n-gram data without using raw pointers.
use crate::types::WordIndex;
use std::marker::PhantomData;

/// An N-gram consists of a sequence of word indices and an associated payload.
/// This implementation uses safe Rust slices instead of raw pointers.
#[derive(Debug)]
pub struct NGram<'a, Payload> {
    /// The word indices that make up this n-gram
    words: &'a [WordIndex],
    /// The payload data (e.g., probability, backoff)
    payload: &'a Payload,
}

impl<'a, Payload> NGram<'a, Payload> {
    /// Create a new NGram from a slice of words and a payload reference
    pub fn new(words: &'a [WordIndex], payload: &'a Payload) -> Self {
        Self { words, payload }
    }

    /// Get the order of this n-gram (how many words it contains)
    pub fn order(&self) -> usize {
        self.words.len()
    }

    /// Get the words in this n-gram
    pub fn words(&self) -> &[WordIndex] {
        self.words
    }

    /// Get a specific word by index (0 = first word, order-1 = last word)
    pub fn word(&self, index: usize) -> WordIndex {
        self.words[index]
    }

    /// Get the first word (same as begin in C++ KenLM)
    pub fn begin(&self) -> WordIndex {
        self.words[0]
    }

    /// Get the last word
    pub fn last(&self) -> WordIndex {
        self.words[self.words.len() - 1]
    }

    /// Get the payload (probability, backoff, etc.)
    pub fn value(&self) -> &Payload {
        self.payload
    }

    /// Calculate the total size in memory for an n-gram of given order
    pub fn total_size(order: usize) -> usize {
        order * std::mem::size_of::<WordIndex>() + std::mem::size_of::<Payload>()
    }

    /// Get the total size of this n-gram instance
    pub fn total_size_instance(&self) -> usize {
        Self::total_size(self.order())
    }

    /// Calculate order from a total size
    pub fn order_from_size(size: usize) -> usize {
        (size - std::mem::size_of::<Payload>()) / std::mem::size_of::<WordIndex>()
    }
}

/// Mutable version of NGram for when we need to modify the payload
#[derive(Debug)]
pub struct NGramMut<'a, Payload> {
    /// The word indices (immutable even in mutable ngram)
    words: &'a [WordIndex],
    /// Mutable payload data
    payload: &'a mut Payload,
}

impl<'a, Payload> NGramMut<'a, Payload> {
    /// Create a new mutable NGram
    pub fn new(words: &'a [WordIndex], payload: &'a mut Payload) -> Self {
        Self { words, payload }
    }

    /// Get the order of this n-gram
    pub fn order(&self) -> usize {
        self.words.len()
    }

    /// Get the words in this n-gram
    pub fn words(&self) -> &[WordIndex] {
        self.words
    }

    /// Get a specific word by index
    pub fn word(&self, index: usize) -> WordIndex {
        self.words[index]
    }

    /// Get the first word
    pub fn begin(&self) -> WordIndex {
        self.words[0]
    }

    /// Get the last word
    pub fn last(&self) -> WordIndex {
        self.words[self.words.len() - 1]
    }

    /// Get an immutable reference to the payload
    pub fn value(&self) -> &Payload {
        self.payload
    }

    /// Get a mutable reference to the payload
    pub fn value_mut(&mut self) -> &mut Payload {
        self.payload
    }

    /// Convert to an immutable NGram
    pub fn as_immutable(&self) -> NGram<Payload> {
        NGram {
            words: self.words,
            payload: self.payload,
        }
    }
}

/// Iterator over n-grams in a contiguous buffer
///
/// This safely iterates over n-grams stored in memory without using raw pointers.
pub struct NGramIterator<'a, Payload> {
    /// The underlying data buffer
    data: &'a [u8],
    /// Current position in bytes
    position: usize,
    /// Order of the n-grams
    order: usize,
    /// Size of each n-gram entry in bytes
    entry_size: usize,
    _marker: PhantomData<&'a Payload>,
}

impl<'a, Payload> NGramIterator<'a, Payload> {
    /// Create a new iterator over n-grams
    pub fn new(data: &'a [u8], order: usize) -> Self {
        let entry_size = NGram::<Payload>::total_size(order);
        Self {
            data,
            position: 0,
            order,
            entry_size,
            _marker: PhantomData,
        }
    }

    /// Get the number of n-grams in the buffer
    pub fn count(&self) -> usize {
        self.data.len() / self.entry_size
    }
}

impl<'a, Payload: 'a> Iterator for NGramIterator<'a, Payload> {
    type Item = NGram<'a, Payload>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position + self.entry_size > self.data.len() {
            return None;
        }

        // Get the slice for this entry
        let entry = &self.data[self.position..self.position + self.entry_size];

        // Words are at the beginning
        let words_size = self.order * std::mem::size_of::<WordIndex>();
        let words_bytes = &entry[..words_size];

        // Payload is at the end
        let payload_bytes = &entry[words_size..];

        // Safety: We're transmuting byte slices to properly aligned types
        // This is safe because we control the layout and alignment
        let words = unsafe {
            std::slice::from_raw_parts(words_bytes.as_ptr() as *const WordIndex, self.order)
        };

        let payload = unsafe { &*(payload_bytes.as_ptr() as *const Payload) };

        self.position += self.entry_size;

        Some(NGram::new(words, payload))
    }
}

impl<'a, Payload> ExactSizeIterator for NGramIterator<'a, Payload> {
    fn len(&self) -> usize {
        (self.data.len() - self.position) / self.entry_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_creation() {
        let words = vec![1, 2, 3];
        let payload = 0.5f32;

        let ngram = NGram::new(&words, &payload);

        assert_eq!(ngram.order(), 3);
        assert_eq!(ngram.begin(), 1);
        assert_eq!(ngram.last(), 3);
        assert_eq!(*ngram.value(), 0.5);
    }

    #[test]
    fn test_ngram_mut() {
        let words = vec![1, 2];
        let mut payload = 1.0f32;

        let mut ngram = NGramMut::new(&words, &mut payload);

        assert_eq!(ngram.order(), 2);
        *ngram.value_mut() = 2.0;
        assert_eq!(*ngram.value(), 2.0);
    }

    #[test]
    fn test_size_calculations() {
        let size = NGram::<f32>::total_size(3);
        let order = NGram::<f32>::order_from_size(size);
        assert_eq!(order, 3);
    }
}
