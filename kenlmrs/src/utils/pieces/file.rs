use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::str::FromStr;

use crate::error::LMError;

const DEFAULT_BUFFER_SIZE: usize = 1048576; // 1 MB

/// FilePiece reads text files, handling memory-mapped IO efficiently
/// and parsing numbers and delimited strings
pub struct FilePiece {
    file: File,
    file_name: String,
    buffer: Vec<u8>,
    position: usize,      // Current position in buffer
    buffer_end: usize,    // End of valid data in buffer
    total_size: u64,      // Total file size
    file_position: u64,   // Current position in file
    at_end: bool,
}

impl FilePiece {
    /// Create a new FilePiece from a file path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, LMError> {
        let file = File::open(&path)?;
        let file_name = path.as_ref().to_string_lossy().to_string();
        let total_size = file.metadata()?.len();
        
        let mut fp = Self {
            file,
            file_name,
            buffer: vec![0u8; DEFAULT_BUFFER_SIZE],
            position: 0,
            buffer_end: 0,
            total_size,
            file_position: 0,
            at_end: false,
        };
        
        fp.shift()?;
        Ok(fp)
    }

    /// Get the file name
    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Peek at the current character without consuming it
    pub fn peek(&mut self) -> Result<char, LMError> {
        if self.position >= self.buffer_end {
            self.shift()?;
            if self.at_end {
                return Err(LMError::IoError(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "End of file",
                )));
            }
        }
        Ok(self.buffer[self.position] as char)
    }

    /// Get and consume the current character
    pub fn get(&mut self) -> Result<char, LMError> {
        let c = self.peek()?;
        self.position += 1;
        Ok(c)
    }

    /// Read a float from the current position
    pub fn read_float(&mut self) -> Result<f32, LMError> {
        self.skip_spaces();
        let s = self.read_until_space()?;
        s.parse::<f32>()
            .map_err(|_| LMError::ParseError(format!("Failed to parse float: {}", s)))
    }

    /// Read a double from the current position
    pub fn read_double(&mut self) -> Result<f64, LMError> {
        self.skip_spaces();
        let s = self.read_until_space()?;
        s.parse::<f64>()
            .map_err(|_| LMError::ParseError(format!("Failed to parse double: {}", s)))
    }

    /// Read an unsigned long from the current position
    pub fn read_ulong(&mut self) -> Result<u64, LMError> {
        self.skip_spaces();
        let s = self.read_until_space()?;
        s.parse::<u64>()
            .map_err(|_| LMError::ParseError(format!("Failed to parse ulong: {}", s)))
    }

    /// Read delimited string based on delimiter table
    pub fn read_delimited(&mut self, delimiters: &[bool; 256]) -> Result<String, LMError> {
        self.skip_delimiters(delimiters);
        let mut result = String::new();
        
        loop {
            if self.position >= self.buffer_end {
                self.shift()?;
                if self.at_end {
                    break;
                }
            }
            
            let c = self.buffer[self.position];
            if delimiters[c as usize] {
                break;
            }
            
            result.push(c as char);
            self.position += 1;
        }
        
        Ok(result)
    }

    /// Read a line until delimiter (typically '\n')
    pub fn read_line(&mut self, delim: char, strip_cr: bool) -> Result<String, LMError> {
        let mut result = String::new();
        
        loop {
            if self.position >= self.buffer_end {
                self.shift()?;
                if self.at_end && result.is_empty() {
                    return Err(LMError::IoError(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "End of file",
                    )));
                }
                if self.at_end {
                    break;
                }
            }
            
            let c = self.buffer[self.position] as char;
            self.position += 1;
            
            if c == delim {
                break;
            }
            
            result.push(c);
        }
        
        if strip_cr && result.ends_with('\r') {
            result.pop();
        }
        
        Ok(result)
    }

    /// Skip whitespace characters
    fn skip_spaces(&mut self) {
        loop {
            if self.position >= self.buffer_end {
                if self.shift().is_err() || self.at_end {
                    break;
                }
            }
            
            let c = self.buffer[self.position];
            if !c.is_ascii_whitespace() {
                break;
            }
            
            self.position += 1;
        }
    }

    /// Skip characters based on delimiter table
    fn skip_delimiters(&mut self, delimiters: &[bool; 256]) {
        loop {
            if self.position >= self.buffer_end {
                if self.shift().is_err() || self.at_end {
                    break;
                }
            }
            
            let c = self.buffer[self.position];
            if !delimiters[c as usize] {
                break;
            }
            
            self.position += 1;
        }
    }

    /// Read until whitespace
    fn read_until_space(&mut self) -> Result<String, LMError> {
        let mut result = String::new();
        
        loop {
            if self.position >= self.buffer_end {
                self.shift()?;
                if self.at_end {
                    break;
                }
            }
            
            let c = self.buffer[self.position];
            if c.is_ascii_whitespace() {
                break;
            }
            
            result.push(c as char);
            self.position += 1;
        }
        
        if result.is_empty() {
            return Err(LMError::ParseError("Empty token".to_string()));
        }
        
        Ok(result)
    }

    /// Refill the buffer from the file
    fn shift(&mut self) -> Result<(), LMError> {
        if self.at_end {
            return Ok(());
        }

        // Move any remaining data to the beginning of the buffer
        if self.position < self.buffer_end {
            let remaining = self.buffer_end - self.position;
            self.buffer.copy_within(self.position..self.buffer_end, 0);
            self.buffer_end = remaining;
            self.position = 0;
        } else {
            self.buffer_end = 0;
            self.position = 0;
        }

        // Read more data
        let bytes_read = self.file.read(&mut self.buffer[self.buffer_end..])?;
        self.buffer_end += bytes_read;
        self.file_position += bytes_read as u64;

        if bytes_read == 0 {
            self.at_end = true;
        }

        Ok(())
    }

    /// Check if at end of file
    pub fn at_end(&self) -> bool {
        self.at_end && self.position >= self.buffer_end
    }

    /// Get current offset in file
    pub fn offset(&self) -> u64 {
        self.file_position - (self.buffer_end - self.position) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_float() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "3.14159").unwrap();
        file.flush().unwrap();

        let mut fp = FilePiece::open(file.path()).unwrap();
        let val = fp.read_float().unwrap();
        assert!((val - 3.14159).abs() < 0.0001);
    }

    #[test]
    fn test_read_line() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "line1\nline2\nline3").unwrap();
        file.flush().unwrap();

        let mut fp = FilePiece::open(file.path()).unwrap();
        assert_eq!(fp.read_line('\n', false).unwrap(), "line1");
        assert_eq!(fp.read_line('\n', false).unwrap(), "line2");
        assert_eq!(fp.read_line('\n', false).unwrap(), "line3");
    }
}
