

#[derive(Clone, Debug, Default)]
/// Represents how a chain should be configured.
pub struct ChainConfig {
    /// Number of bytes in each record.
    entry_size: usize,

    /// Number of blocks in the chain.
    block_count: usize,

    /// Total number of bytes available to the chain.
    /// This value will be divided amongst the blocks in the chain.
    /// Chain's constructor will make this a multiple of entry_size.
    total_memory: usize,
}

impl ChainConfig {
    /// Constructs a configuration with underspecified (or default) parameters.
    pub fn new() -> Self {
        Self {
            entry_size: 0,
            block_count: 0,
            total_memory: 0,
        }
    }

    /// Constructs a chain configuration object.
    ///
    /// # Arguments
    ///
    /// * `entry_size` - Number of bytes in each record.
    /// * `block_count` - Number of blocks in the chain.
    /// * `total_memory` - Total number of bytes available to the chain.
    ///                     This value will be divided amongst the blocks in the chain.
    pub fn with_params(entry_size: usize, block_count: usize, total_memory: usize) -> Self {
        Self {
            entry_size,
            block_count,
            total_memory,
        }
    }
}

/// Represents how a sorter should be configured.
#[derive(Debug, Clone, Default)]
pub struct SortConfig {
    /// Filename prefix where temporary files should be placed.
    temp_prefix: String,

    /// Size of each input/output buffer.
    buffer_size: usize,

    /// Total memory to use when running alone.
    total_memory: usize,
}

impl SortConfig {
    /// Constructs a new sort configuration object.
    ///
    /// # Arguments
    ///
    /// * `temp_prefix` - Filename prefix where temporary files should be placed.
    /// * `buffer_size` - Size of each input/output buffer.
    /// * `total_memory` - Total memory to use when running alone.
    pub fn new(temp_prefix: String, buffer_size: usize, total_memory: usize) -> Self {
        Self {
            temp_prefix,
            buffer_size,
            total_memory,
        }
    }
}
