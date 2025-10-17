use crate::constant::{BINARY_MAGIC, MODEL_VERSION};
use crate::error::LMError;
use crate::types::{Config, ModelType};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem;
use std::path::Path;

/// Model type names for error messages
pub const MODEL_NAMES: [&str; 6] = [
    "probing hash table",
    "probing hash table with rest costs",
    "trie",
    "quantized trie",
    "array-compressed trie",
    "quantized array-compressed trie",
];

/// Fixed-width parameters stored at the beginning of a binary file
#[derive(Debug, Clone)]
pub struct FixedWidthParameters {
    pub order: u8,
    pub probing_multiplier: f32,
    pub model_type: ModelType,
    pub has_vocabulary: bool,
    pub search_version: u32,
}

/// All parameters from the binary file header
#[derive(Debug, Clone)]
pub struct Parameters {
    pub fixed: FixedWidthParameters,
    pub counts: Vec<u64>,
}

/// Handles binary format reading and writing for KenLM models
pub struct BinaryFormat {
    file: Option<File>,
    header_size: usize,
    vocab_size: usize,
    vocab_pad: usize,
    vocab_string_offset: u64,
}

impl BinaryFormat {
    /// Create a new BinaryFormat handler
    pub fn new() -> Self {
        Self {
            file: None,
            header_size: 0,
            vocab_size: 0,
            vocab_pad: 0,
            vocab_string_offset: u64::MAX,
        }
    }

    /// Check if a file is in KenLM binary format
    pub fn recognize_binary<P: AsRef<Path>>(path: P) -> Result<Option<ModelType>, LMError> {
        let mut file = File::open(path)?;
        let mut magic = vec![0u8; BINARY_MAGIC.len()];

        match file.read_exact(&mut magic) {
            Ok(_) => {
                if &magic == BINARY_MAGIC.as_bytes() {
                    // Read the model type
                    file.seek(SeekFrom::Start(BINARY_MAGIC.len() as u64))?;
                    let mut params = FixedWidthParameters {
                        order: 0,
                        probing_multiplier: 0.0,
                        model_type: ModelType::Probing,
                        has_vocabulary: false,
                        search_version: 0,
                    };
                    Self::read_fixed_width(&mut file, &mut params)?;
                    Ok(Some(params.model_type))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None),
        }
    }

    /// Read the fixed-width header parameters
    fn read_fixed_width(file: &mut File, params: &mut FixedWidthParameters) -> Result<(), LMError> {
        // Read order
        let mut buf = [0u8; 1];
        file.read_exact(&mut buf)?;
        params.order = buf[0];

        // Read probing multiplier
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        params.probing_multiplier = f32::from_le_bytes(buf);

        // Read model type
        let mut buf = [0u8; 1];
        file.read_exact(&mut buf)?;
        params.model_type = match buf[0] {
            0 => ModelType::Probing,
            1 => ModelType::RestProbing,
            2 => ModelType::Trie,
            3 => ModelType::QuantTrie,
            4 => ModelType::ArrayTrie,
            5 => ModelType::QuantArrayTrie,
            _ => return Err(LMError::UnsupportedModelType),
        };

        // Read has_vocabulary flag
        let mut buf = [0u8; 1];
        file.read_exact(&mut buf)?;
        params.has_vocabulary = buf[0] != 0;

        // Read search version
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        params.search_version = u32::from_le_bytes(buf);

        Ok(())
    }

    /// Initialize from a binary file
    pub fn initialize_binary<P: AsRef<Path>>(
        &mut self,
        path: P,
        model_type: ModelType,
        search_version: u32,
    ) -> Result<Parameters, LMError> {
        let mut file = File::open(path)?;

        // Read and verify magic string
        let mut magic = vec![0u8; BINARY_MAGIC.len()];
        file.read_exact(&mut magic)?;

        if &magic != BINARY_MAGIC.as_bytes() {
            return Err(LMError::FormatError("Not a KenLM binary file".to_string()));
        }

        // Read fixed-width parameters
        let mut fixed = FixedWidthParameters {
            order: 0,
            probing_multiplier: 0.0,
            model_type: ModelType::Probing,
            has_vocabulary: false,
            search_version: 0,
        };
        Self::read_fixed_width(&mut file, &mut fixed)?;

        // Verify model type matches
        if fixed.model_type != model_type {
            return Err(LMError::FormatError(format!(
                "Model type mismatch: expected {:?}, got {:?}",
                model_type, fixed.model_type
            )));
        }

        // Verify version matches
        if fixed.search_version != search_version {
            return Err(LMError::VersionMismatch {
                expected: search_version,
                actual: fixed.search_version,
            });
        }

        // Read n-gram counts
        let mut counts = Vec::with_capacity(fixed.order as usize);
        for _ in 0..fixed.order {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            counts.push(u64::from_le_bytes(buf));
        }

        self.file = Some(file);
        self.header_size = Self::calculate_header_size(fixed.order);

        Ok(Parameters { fixed, counts })
    }

    /// Calculate the size of the header based on order
    fn calculate_header_size(order: u8) -> usize {
        // Magic string + fixed params + counts
        BINARY_MAGIC.len()
            + 1  // order
            + 4  // probing_multiplier
            + 1  // model_type
            + 1  // has_vocabulary
            + 4  // search_version
            + (order as usize * 8) // counts
    }

    /// Read data from the file at a specific offset (excluding header)
    pub fn read_for_config(
        &self,
        to: &mut [u8],
        offset_excluding_header: u64,
    ) -> Result<(), LMError> {
        if let Some(ref file) = self.file {
            let mut file = file;
            let absolute_offset = self.header_size as u64 + offset_excluding_header;
            file.seek(SeekFrom::Start(absolute_offset))?;
            file.read_exact(to)?;
            Ok(())
        } else {
            Err(LMError::LoadError("No file loaded".to_string()))
        }
    }

    /// Get the offset where vocabulary strings begin
    pub fn vocab_string_reading_offset(&self) -> u64 {
        self.vocab_string_offset
    }

    /// Set the vocabulary string offset
    pub fn set_vocab_string_offset(&mut self, offset: u64) {
        self.vocab_string_offset = offset;
    }
}

impl Default for BinaryFormat {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper module for utility functions
pub mod util {
    use super::*;

    /// Get file size
    pub fn file_size(file: &File) -> Result<u64, LMError> {
        Ok(file.metadata()?.len())
    }

    /// Read integer with specific bit width
    pub fn read_int57(base: &[u8], bit_offset: u64, bits: u8, mask: u64) -> u64 {
        let byte_offset = (bit_offset / 8) as usize;
        let bit_remainder = (bit_offset % 8) as u8;

        if byte_offset + 8 > base.len() {
            return 0;
        }

        // Read 8 bytes as u64
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&base[byte_offset..byte_offset + 8]);
        let value = u64::from_le_bytes(bytes);

        // Shift and mask
        ((value >> bit_remainder) & mask)
    }

    /// Write integer with specific bit width
    pub fn write_int57(base: &mut [u8], bit_offset: u64, bits: u8, value: u64) {
        let byte_offset = (bit_offset / 8) as usize;
        let bit_remainder = (bit_offset % 8) as u8;

        if byte_offset + 8 > base.len() {
            return;
        }

        // Read existing value
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&base[byte_offset..byte_offset + 8]);
        let mut existing = u64::from_le_bytes(bytes);

        // Create mask for the bits we're writing
        let write_mask = ((1u64 << bits) - 1) << bit_remainder;

        // Clear the bits and set new value
        existing = (existing & !write_mask) | ((value << bit_remainder) & write_mask);

        // Write back
        base[byte_offset..byte_offset + 8].copy_from_slice(&existing.to_le_bytes());
    }

    /// Required bits to represent a value
    pub fn required_bits(value: u64) -> u8 {
        if value == 0 {
            return 1;
        }
        (64 - value.leading_zeros()) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_bits() {
        assert_eq!(util::required_bits(0), 1);
        assert_eq!(util::required_bits(1), 1);
        assert_eq!(util::required_bits(2), 2);
        assert_eq!(util::required_bits(3), 2);
        assert_eq!(util::required_bits(255), 8);
        assert_eq!(util::required_bits(256), 9);
    }

    #[test]
    fn test_read_write_int57() {
        let mut buffer = vec![0u8; 16];

        util::write_int57(&mut buffer, 0, 8, 0x42);
        assert_eq!(util::read_int57(&buffer, 0, 8, 0xFF), 0x42);

        util::write_int57(&mut buffer, 8, 16, 0x1234);
        assert_eq!(util::read_int57(&buffer, 8, 16, 0xFFFF), 0x1234);
    }
}
