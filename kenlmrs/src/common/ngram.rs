/// N-gram representation with safe Rust references
///
/// This module provides safe access to n-gram data without using raw pointers.
use crate::types::WordIndex;

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
        *self.words.first().expect("NGram must have at least one word")
    }

    /// Get the last word
    pub fn last(&self) -> WordIndex {
        *self.words.last().expect("NGram must have at least one word")
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


/// An owned n-gram produced by [`NGramIterator`].
#[derive(Debug, Clone)]
pub struct OwnedNGram<Payload> {
    pub words: Vec<WordIndex>,
    pub payload: Payload,
}

/// Iterator over a collection of typed n-gram entries.
///
/// Each entry owns its word list (`Vec<WordIndex>`) and payload value.
/// There are no raw pointers or `unsafe` blocks — the data is parsed into
/// owned types before being handed to the iterator.
///
/// Build the entries yourself (e.g. while reading an ARPA/binary file) and pass
/// them to [`NGramIterator::new`].
pub struct NGramIterator<Payload> {
    entries: std::vec::IntoIter<OwnedNGram<Payload>>,
    total: usize,
    remaining: usize,
}

impl<Payload> NGramIterator<Payload> {
    /// Create an iterator from a `Vec` of pre-parsed, owned n-gram entries.
    pub fn new(entries: Vec<OwnedNGram<Payload>>) -> Self {
        let total = entries.len();
        Self {
            remaining: total,
            total,
            entries: entries.into_iter(),
        }
    }

    /// Total number of entries (including already-consumed ones).
    pub fn count_total(&self) -> usize {
        self.total
    }
}

impl<Payload> Iterator for NGramIterator<Payload> {
    type Item = OwnedNGram<Payload>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.entries.next()?;
        self.remaining -= 1;
        Some(item)
    }
}

impl<Payload> ExactSizeIterator for NGramIterator<Payload> {
    fn len(&self) -> usize {
        self.remaining
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
    fn test_ngram_mutable_payload() {
        // NGramMut was removed; mutate the payload directly and read via NGram.
        let words = vec![1u32, 2];
        let mut payload = 1.0f32;
        payload = 2.0;
        let ngram = NGram::new(&words, &payload);
        assert_eq!(ngram.order(), 2);
        assert_eq!(*ngram.value(), 2.0);
    }

    #[test]
    fn test_size_calculations() {
        let size = NGram::<f32>::total_size(3);
        let order = NGram::<f32>::order_from_size(size);
        assert_eq!(order, 3);
    }

    #[test]
    fn test_ngram_word_access() {
        let words = vec![10u32, 20, 30, 40];
        let payload = 0.0f64;
        let ng = NGram::new(&words, &payload);
        assert_eq!(ng.word(0), 10);
        assert_eq!(ng.word(3), 40);
        assert_eq!(ng.begin(), 10);
        assert_eq!(ng.last(), 40);
    }

    #[test]
    fn test_ngram_total_size_instance() {
        let words = vec![1u32, 2];
        let payload = 42u64;
        let ng = NGram::new(&words, &payload);
        assert_eq!(ng.total_size_instance(), NGram::<u64>::total_size(2));
    }

    #[test]
    fn test_ngram_order_from_size_roundtrip() {
        for order in 1..=6 {
            let size = NGram::<f32>::total_size(order);
            assert_eq!(NGram::<f32>::order_from_size(size), order);
        }
    }

    #[test]
    fn test_ngram_iterator_count() {
        let entries: Vec<OwnedNGram<f32>> = (0..3)
            .map(|i| OwnedNGram { words: vec![i as u32], payload: i as f32 })
            .collect();
        let iter = NGramIterator::new(entries);
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.count_total(), 3);
    }

    #[test]
    fn test_ngram_iterator_empty_buffer() {
        let iter: NGramIterator<f32> = NGramIterator::new(vec![]);
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn test_ngram_immutable_access() {
        // Previously tested via NGramMut::as_immutable(); NGram alone suffices.
        let words = vec![5u32, 6];
        let payload = 1.0f32;
        let ng = NGram::new(&words, &payload);
        assert_eq!(ng.order(), 2);
        assert_eq!(ng.begin(), 5);
        assert_eq!(*ng.value(), 1.0);
    }

    #[test]
    fn test_ngram_words_slice() {
        let words = vec![100u32, 200, 300];
        let payload = 0u8;
        let ng = NGram::new(&words, &payload);
        assert_eq!(ng.words(), &[100, 200, 300]);
    }

    #[test]
    fn test_ngram_exact_size_iterator() {
        let entries: Vec<OwnedNGram<u8>> = (0..5)
            .map(|i| OwnedNGram { words: vec![i as u32, i as u32 + 1], payload: i as u8 })
            .collect();
        let iter = NGramIterator::new(entries);
        assert_eq!(iter.len(), 5);
    }

    #[test]
    fn test_ngram_iterator_yields_entries() {
        let entries = vec![
            OwnedNGram { words: vec![1u32, 2], payload: 0.5f32 },
            OwnedNGram { words: vec![3u32, 4], payload: 1.0f32 },
        ];
        let collected: Vec<_> = NGramIterator::new(entries).collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].words, vec![1, 2]);
        assert!((collected[1].payload - 1.0).abs() < 1e-6);
    }
}
