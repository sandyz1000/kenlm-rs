use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

/// A fixed-size buffer that holds serialized records during pipeline processing.
pub struct Block {
    data: Vec<u8>,
    write_pos: usize,
    entry_size: usize,
}

impl Block {
    pub fn new(capacity: usize, entry_size: usize) -> Self {
        Block {
            data: vec![0u8; capacity],
            write_pos: 0,
            entry_size,
        }
    }

    pub fn push(&mut self, entry: &[u8]) -> bool {
        if self.write_pos + entry.len() > self.data.len() {
            return false;
        }
        self.data[self.write_pos..self.write_pos + entry.len()].copy_from_slice(entry);
        self.write_pos += entry.len();
        true
    }

    pub fn as_entries(&self) -> impl Iterator<Item = &[u8]> {
        let es = self.entry_size;
        self.data[..self.write_pos].chunks(es)
    }

    pub fn is_full(&self) -> bool {
        self.write_pos + self.entry_size > self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.write_pos == 0
    }

    pub fn clear(&mut self) {
        self.write_pos = 0;
    }

    pub fn entry_size(&self) -> usize {
        self.entry_size
    }
}

/// Bounded single-producer single-consumer queue.
pub struct PCQueue<T> {
    queue: Mutex<VecDeque<T>>,
    not_empty: Condvar,
    not_full: Condvar,
    capacity: usize,
}

impl<T> PCQueue<T> {
    pub fn new(capacity: usize) -> Self {
        PCQueue {
            queue: Mutex::new(VecDeque::with_capacity(capacity)),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            capacity,
        }
    }

    pub fn push(&self, item: T) {
        let mut q = self.queue.lock().unwrap();
        while q.len() >= self.capacity {
            q = self.not_full.wait(q).unwrap();
        }
        q.push_back(item);
        self.not_empty.notify_one();
    }

    pub fn pop(&self) -> Option<T> {
        let mut q = self.queue.lock().unwrap();
        while q.is_empty() {
            q = self.not_empty.wait(q).unwrap();
        }
        let item = q.pop_front();
        self.not_full.notify_one();
        item
    }

    pub fn try_pop(&self) -> Option<T> {
        self.queue.lock().unwrap().pop_front()
    }
}

/// A chain connects pipeline stages: a producer fills blocks and hands them to a consumer.
///
/// Single-threaded use: call `add()` to get a writable block, fill it, then `pass(block)` to
/// send it downstream. The consumer calls `pop()` to receive blocks.
pub struct Chain {
    block_size: usize,
    entry_size: usize,
    filled: Arc<PCQueue<Block>>,
    recycled: Arc<PCQueue<Block>>,
}

impl Chain {
    pub fn new(block_size: usize, entry_size: usize, block_count: usize) -> Self {
        let recycled = Arc::new(PCQueue::new(block_count));
        let filled = Arc::new(PCQueue::new(block_count));
        // Pre-populate the recycled pool
        for _ in 0..block_count {
            recycled.push(Block::new(block_size, entry_size));
        }
        Chain {
            block_size,
            entry_size,
            filled,
            recycled,
        }
    }

    /// Get an empty block from the pool.
    pub fn add(&self) -> Block {
        self.recycled
            .pop()
            .unwrap_or_else(|| Block::new(self.block_size, self.entry_size))
    }

    /// Send a filled block to the consumer.
    pub fn pass(&self, block: Block) {
        if !block.is_empty() {
            self.filled.push(block);
        } else {
            self.recycled.push(block);
        }
    }

    /// Receive a filled block (consumer side). Returns None only if the chain is finished.
    pub fn pop(&self) -> Option<Block> {
        self.filled.try_pop()
    }

    pub fn entry_size(&self) -> usize {
        self.entry_size
    }
}

/// Position in a chain held by one pipeline stage. Provides add/pass/pop access.
pub struct ChainPosition {
    pub(crate) chain: Arc<Chain>,
}

impl ChainPosition {
    pub fn new(chain: Arc<Chain>) -> Self {
        ChainPosition { chain }
    }

    pub fn add(&self) -> Block {
        self.chain.add()
    }

    pub fn pass(&self, block: Block) {
        self.chain.pass(block);
    }

    pub fn pop(&self) -> Option<Block> {
        self.chain.pop()
    }
}

/// Sort all entries across all filled blocks in a Chain, returning a new sorted Chain.
///
/// Entries are treated as fixed-size byte slices and sorted lexicographically.
/// For small data this is an in-memory sort; the sorted output is written into a fresh Chain.
/// For large data, use `sort_to_temp_files` instead.
pub fn sort_chain(input: &Chain, temp_prefix: &str) -> Chain {
    let entry_size = input.entry_size();
    if entry_size == 0 { return Chain::new(4096, 1, 2); }

    // Collect all entries
    let mut all_entries: Vec<Vec<u8>> = Vec::new();
    while let Some(block) = input.pop() {
        for entry in block.as_entries() {
            all_entries.push(entry.to_vec());
        }
    }

    // Sort lexicographically
    all_entries.sort_unstable();
    all_entries.dedup();

    // Write sorted entries into a new chain
    let block_size = (all_entries.len() * entry_size + 4095) / 4096 * 4096 + 4096;
    let out_chain = Chain::new(block_size.max(4096), entry_size, 2);
    let mut block = out_chain.add();
    for entry in &all_entries {
        if !block.push(entry) {
            out_chain.pass(block);
            block = out_chain.add();
            block.push(entry);
        }
    }
    out_chain.pass(block);
    out_chain
}

/// Sort entries to temporary files and merge-sort them back (external sort for large data).
///
/// Uses `temp_prefix` as the directory for temporary files.
/// Returns a new Chain containing the sorted, deduplicated entries.
pub fn sort_to_temp_files(input: &Chain, temp_prefix: &str) -> std::io::Result<Chain> {
    use std::collections::BinaryHeap;
    use std::io::{BufWriter, Write, BufReader, Read};
    use std::cmp::Reverse;
    use std::fs;

    let entry_size = input.entry_size();
    if entry_size == 0 {
        return Ok(Chain::new(4096, 1, 2));
    }

    // Phase 1: write sorted runs to temp files
    let run_size = 65536 / entry_size.max(1); // ~64KB runs
    let mut temp_files: Vec<std::path::PathBuf> = Vec::new();
    let mut run: Vec<Vec<u8>> = Vec::with_capacity(run_size);

    let flush_run = |run: &mut Vec<Vec<u8>>, files: &mut Vec<std::path::PathBuf>, prefix: &str|
        -> std::io::Result<()>
    {
        if run.is_empty() { return Ok(()); }
        run.sort_unstable();
        run.dedup();
        let path = std::path::PathBuf::from(format!("{}/kenlm_sort_{}.tmp", prefix, files.len()));
        let mut f = BufWriter::new(fs::File::create(&path)?);
        for entry in run.drain(..) {
            f.write_all(&entry)?;
        }
        files.push(path);
        Ok(())
    };

    while let Some(block) = input.pop() {
        for entry in block.as_entries() {
            run.push(entry.to_vec());
            if run.len() >= run_size {
                flush_run(&mut run, &mut temp_files, temp_prefix)?;
            }
        }
    }
    flush_run(&mut run, &mut temp_files, temp_prefix)?;

    // Phase 2: k-way merge using BinaryHeap
    let mut readers: Vec<(BufReader<fs::File>, Vec<u8>)> = temp_files.iter().map(|p| {
        let f = BufReader::new(fs::File::open(p).expect("temp file"));
        (f, vec![0u8; entry_size])
    }).collect();

    // Prime each reader with its first entry
    let mut heap: BinaryHeap<Reverse<(Vec<u8>, usize)>> = BinaryHeap::new();
    for (i, (reader, buf)) in readers.iter_mut().enumerate() {
        let n = reader.read(buf)?;
        if n == entry_size {
            heap.push(Reverse((buf.clone(), i)));
        }
    }

    let block_cap = temp_files.len().max(1) * run_size * entry_size + 4096;
    let out_chain = Chain::new(block_cap.max(4096), entry_size, 2);
    let mut out_block = out_chain.add();
    let mut prev: Option<Vec<u8>> = None;

    while let Some(Reverse((entry, idx))) = heap.pop() {
        // Dedup: skip if same as previous
        if prev.as_deref() == Some(&entry) {
            // advance reader idx
        } else {
            if !out_block.push(&entry) {
                out_chain.pass(out_block);
                out_block = out_chain.add();
                out_block.push(&entry);
            }
            prev = Some(entry.clone());
        }
        // Advance that reader
        let (reader, buf) = &mut readers[idx];
        let n = reader.read(buf)?;
        if n == entry_size {
            heap.push(Reverse((buf.clone(), idx)));
        }
    }
    out_chain.pass(out_block);

    // Clean up temp files
    for path in &temp_files {
        let _ = fs::remove_file(path);
    }

    Ok(out_chain)
}

/// Spawn a producer thread that fills a Chain from an iterator of byte entries.
///
/// Returns a `JoinHandle` that completes once all entries have been pushed.
pub fn spawn_producer<I>(chain: Arc<Chain>, entries: I) -> std::thread::JoinHandle<()>
where
    I: IntoIterator<Item = Vec<u8>> + Send + 'static,
{
    std::thread::spawn(move || {
        let mut block = chain.add();
        for entry in entries {
            if !block.push(&entry) {
                chain.pass(block);
                block = chain.add();
                block.push(&entry);
            }
        }
        chain.pass(block);
    })
}

/// Spawn a consumer thread that drains a Chain, calling `f` for each entry.
pub fn spawn_consumer<F>(chain: Arc<Chain>, f: F) -> std::thread::JoinHandle<()>
where
    F: Fn(&[u8]) + Send + 'static,
{
    std::thread::spawn(move || {
        loop {
            match chain.pop() {
                Some(block) => {
                    for entry in block.as_entries() {
                        f(entry);
                    }
                }
                None => {
                    // Brief yield before giving up — give producer time to push first block
                    std::thread::yield_now();
                    if chain.pop().is_none() { break; }
                }
            }
        }
    })
}
