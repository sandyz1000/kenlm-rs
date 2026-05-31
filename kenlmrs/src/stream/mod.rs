pub mod chain;
pub mod config;
pub mod error;

#[cfg(test)]
mod tests {
    use super::chain::*;
    use std::sync::Arc;

    #[test]
    fn block_push_and_entries() {
        let mut b = Block::new(64, 8);
        let entry = [1u8, 2, 3, 4, 5, 6, 7, 8];
        assert!(b.push(&entry));
        assert!(!b.is_empty());
        let entries: Vec<&[u8]> = b.as_entries().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], &entry);
    }

    #[test]
    fn block_is_full_when_capacity_exhausted() {
        let mut b = Block::new(16, 8);
        assert!(b.push(&[0u8; 8]));
        assert!(b.push(&[0u8; 8]));
        assert!(b.is_full());
        assert!(!b.push(&[0u8; 8])); // should fail
    }

    #[test]
    fn chain_add_pass_pop_round_trip() {
        let chain = Chain::new(256, 8, 4);
        let mut block = chain.add();
        block.push(&[42u8; 8]);
        chain.pass(block);
        let received = chain.pop().expect("should have block");
        let entries: Vec<&[u8]> = received.as_entries().collect();
        assert_eq!(entries[0], &[42u8; 8]);
    }

    #[test]
    fn sort_chain_sorts_entries() {
        let chain = Chain::new(256, 4, 4);
        let mut block = chain.add();
        block.push(&[3, 0, 0, 0u8]);
        block.push(&[1, 0, 0, 0u8]);
        block.push(&[2, 0, 0, 0u8]);
        chain.pass(block);
        let sorted = sort_chain(&chain, "/tmp");
        let out_block = sorted.pop().expect("sorted output");
        let entries: Vec<Vec<u8>> = out_block.as_entries().map(|e| e.to_vec()).collect();
        assert_eq!(entries[0], &[1, 0, 0, 0]);
        assert_eq!(entries[1], &[2, 0, 0, 0]);
        assert_eq!(entries[2], &[3, 0, 0, 0]);
    }

    #[test]
    fn sort_chain_deduplicates() {
        let chain = Chain::new(256, 4, 4);
        let mut block = chain.add();
        block.push(&[1, 0, 0, 0u8]);
        block.push(&[1, 0, 0, 0u8]); // duplicate
        block.push(&[2, 0, 0, 0u8]);
        chain.pass(block);
        let sorted = sort_chain(&chain, "/tmp");
        let out_block = sorted.pop().expect("sorted output");
        let entries: Vec<&[u8]> = out_block.as_entries().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn pcqueue_push_pop_ordering() {
        let q = PCQueue::new(4);
        q.push(1u32);
        q.push(2u32);
        q.push(3u32);
        assert_eq!(q.try_pop(), Some(1));
        assert_eq!(q.try_pop(), Some(2));
        assert_eq!(q.try_pop(), Some(3));
        assert_eq!(q.try_pop(), None);
    }
}
