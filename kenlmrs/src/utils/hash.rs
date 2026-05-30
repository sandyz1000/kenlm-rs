/// MurmurHash2 64-bit (Austin Appleby). Beware alignment/endian-ness across platforms.
const M: u64 = 0xc6a4a7935bd1e995;
const R: i32 = 47;

/// MurmurHash64A for strings
/// This is the version used in KenLM for vocabulary hashing
pub fn murmur_hash_64a(data: &[u8], seed: u64) -> u64 {
    let len = data.len();
    let mut h = seed ^ ((len as u64).wrapping_mul(M));

    let n_blocks = len / 8;

    // Process 8-byte blocks
    for i in 0..n_blocks {
        let mut k = read_u64_le(&data[i * 8..]);

        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);

        h ^= k;
        h = h.wrapping_mul(M);
    }

    // Handle remaining bytes
    let remaining = len & 7;
    if remaining > 0 {
        let start = n_blocks * 8;
        let tail = &data[start..];

        let mut k: u64 = 0;
        for (i, &byte) in tail.iter().enumerate() {
            k |= (byte as u64) << (i * 8);
        }

        h ^= k;
        h = h.wrapping_mul(M);
    }

    // Final mixing
    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;

    h
}

/// Read a little-endian u64 from bytes
#[inline]
fn read_u64_le(bytes: &[u8]) -> u64 {
    debug_assert!(bytes.len() >= 8);
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Hash a string for vocabulary lookup
/// This matches the KenLM vocab hash function
pub fn hash_for_vocab(s: &str) -> u64 {
    murmur_hash_64a(s.as_bytes(), 0)
}

/// Known hash values for special tokens (murmur_hash_64a with seed=0)
pub const UNKNOWN_HASH: u64 = 0xea91_e756_1f5b_392b; // hash of "<unk>"
pub const UNKNOWN_CAP_HASH: u64 = 0x7e16_3219_007d_2f54; // hash of "<UNK>"

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_murmur_hash() {
        // Empty string with seed 0 produces 0 (by construction of the algorithm)
        let h_empty = murmur_hash_64a(b"", 0);
        assert_eq!(h_empty, 0);

        // Non-empty strings produce non-zero hashes
        let h1 = murmur_hash_64a(b"hello", 0);
        assert_ne!(h1, 0);

        // Same input → same output (deterministic)
        let h2 = murmur_hash_64a(b"hello", 0);
        assert_eq!(h1, h2);

        // Different strings produce different hashes
        let h3 = murmur_hash_64a(b"world", 0);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_murmur_hash_seed_matters() {
        let h1 = murmur_hash_64a(b"test", 0);
        let h2 = murmur_hash_64a(b"test", 42);
        assert_ne!(h1, h2, "different seeds should produce different hashes");
    }

    #[test]
    fn test_murmur_hash_long_string() {
        // Exercises the 8-byte block path (len >= 8)
        let h = murmur_hash_64a(b"hello world!", 0);
        assert_ne!(h, 0);
        assert_eq!(h, murmur_hash_64a(b"hello world!", 0));
    }

    #[test]
    fn test_hash_for_vocab_unk_constants() {
        // Verify the pre-computed constants match runtime values
        assert_eq!(hash_for_vocab("<unk>"), UNKNOWN_HASH,
            "UNKNOWN_HASH constant must match runtime hash of '<unk>'");
        assert_eq!(hash_for_vocab("<UNK>"), UNKNOWN_CAP_HASH,
            "UNKNOWN_CAP_HASH constant must match runtime hash of '<UNK>'");
    }

    #[test]
    fn test_hash_for_vocab() {
        let h1 = hash_for_vocab("test");
        let h2 = hash_for_vocab("test");
        assert_eq!(h1, h2);

        let h3 = hash_for_vocab("other");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_special_token_hashes() {
        // These should match the C++ KenLM hashes
        let unk_hash = hash_for_vocab("<unk>");
        let unk_cap_hash = hash_for_vocab("<UNK>");
        
        // Just verify they're different
        assert_ne!(unk_hash, unk_cap_hash);
    }
}

