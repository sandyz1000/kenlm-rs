use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

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
            if self.position >= self.buffer_end && (self.shift().is_err() || self.at_end) {
                break;
            }
            if self.position >= self.buffer_end {
                continue;
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
            if self.position >= self.buffer_end && (self.shift().is_err() || self.at_end) {
                break;
            }
            if self.position >= self.buffer_end {
                continue;
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

    fn make_fp(content: &str) -> (NamedTempFile, FilePiece) {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{}", content).unwrap();
        f.flush().unwrap();
        let path = f.path().to_owned();
        let fp = FilePiece::open(&path).unwrap();
        (f, fp)
    }

    #[test]
    fn test_file_name_is_set() {
        let (_f, fp) = make_fp("hello");
        assert!(!fp.file_name().is_empty());
    }

    #[test]
    fn test_get_and_peek() {
        let (_f, mut fp) = make_fp("AB");
        assert_eq!(fp.peek().unwrap(), 'A');
        assert_eq!(fp.get().unwrap(), 'A');
        assert_eq!(fp.peek().unwrap(), 'B');
        assert_eq!(fp.get().unwrap(), 'B');
    }

    #[test]
    fn test_read_float_negative() {
        let (_f, mut fp) = make_fp("-3.14");
        let v = fp.read_float().unwrap();
        assert!((v - (-3.14)).abs() < 1e-4);
    }

    #[test]
    fn test_read_float_with_leading_spaces() {
        let (_f, mut fp) = make_fp("  42.0  ");
        let v = fp.read_float().unwrap();
        assert!((v - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_read_delimited() {
        let delims = {
            let mut d = [false; 256];
            d[b' ' as usize] = true;
            d[b'\t' as usize] = true;
            d[b'\n' as usize] = true;
            d
        };
        let (_f, mut fp) = make_fp("hello world");
        let token = fp.read_delimited(&delims).unwrap();
        assert_eq!(token, "hello");
    }

    #[test]
    fn test_read_multiple_delimited_tokens() {
        let delims = {
            let mut d = [false; 256];
            d[b' ' as usize] = true;
            d[b'\t' as usize] = true;
            d[b'\n' as usize] = true;
            d
        };
        let (_f, mut fp) = make_fp("foo bar baz");
        assert_eq!(fp.read_delimited(&delims).unwrap(), "foo");
        assert_eq!(fp.read_delimited(&delims).unwrap(), "bar");
        assert_eq!(fp.read_delimited(&delims).unwrap(), "baz");
    }

    #[test]
    fn test_read_line_strips_cr() {
        let (_f, mut fp) = make_fp("line1\r\nline2\r\n");
        let line = fp.read_line('\n', true).unwrap();
        assert_eq!(line, "line1"); // \r stripped
    }

    #[test]
    fn test_at_end_false_at_start() {
        let (_f, fp) = make_fp("data");
        assert!(!fp.at_end());
    }

    #[test]
    fn test_read_double() {
        let (_f, mut fp) = make_fp("1.23456789");
        let v = fp.read_double().unwrap();
        assert!((v - 1.23456789).abs() < 1e-7);
    }

    #[test]
    fn test_read_ulong() {
        let (_f, mut fp) = make_fp("12345678");
        let v = fp.read_ulong().unwrap();
        assert_eq!(v, 12345678);
    }
}
