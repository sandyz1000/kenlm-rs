use std::fs::File;
use std::io;
use std::path::Path;

use crate::error::LMError;

enum Backing {
    // Safety: the Mmap is created read-only; the underlying file is not truncated while held.
    Mmap(memmap2::Mmap),
    Buffer(Vec<u8>),
}

impl Backing {
    fn as_slice(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => m,
            Backing::Buffer(v) => v,
        }
    }
}

/// FilePiece reads text files via memory-mapped or buffered I/O and parses
/// numbers and delimited strings.  Files ending in `.gz` are transparently
/// decompressed via flate2; all other files are memory-mapped for zero-copy access.
pub struct FilePiece {
    backing: Backing,
    file_name: String,
    position: usize,
}

impl FilePiece {
    /// Open a file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, LMError> {
        let file_name = path.as_ref().to_string_lossy().to_string();
        let is_gz = file_name.ends_with(".gz");
        let file = File::open(&path)?;

        let backing = if is_gz {
            use std::io::Read;
            let mut decoder = flate2::read::GzDecoder::new(file);
            let mut buf = Vec::new();
            decoder.read_to_end(&mut buf)?;
            Backing::Buffer(buf)
        } else {
            // Safety: file is opened read-only and not truncated while the Mmap is held.
            match unsafe { memmap2::MmapOptions::new().map(&file) } {
                Ok(mmap) => Backing::Mmap(mmap),
                Err(_) => {
                    // Fallback for platforms where mmap is unavailable (e.g., pipes, WASI).
                    use std::io::{Read, Seek, SeekFrom};
                    let mut file = file;
                    let _ = file.seek(SeekFrom::Start(0));
                    let mut buf = Vec::new();
                    file.read_to_end(&mut buf)?;
                    Backing::Buffer(buf)
                }
            }
        };

        Ok(Self {
            backing,
            file_name,
            position: 0,
        })
    }

    fn len(&self) -> usize {
        self.backing.as_slice().len()
    }

    fn byte_at(&self, idx: usize) -> u8 {
        self.backing.as_slice()[idx]
    }

    fn slice(&self, start: usize, end: usize) -> &[u8] {
        &self.backing.as_slice()[start..end]
    }

    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    pub fn peek(&mut self) -> Result<char, LMError> {
        if self.position >= self.len() {
            return Err(LMError::IoError(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "End of file",
            )));
        }
        Ok(self.byte_at(self.position) as char)
    }

    pub fn get(&mut self) -> Result<char, LMError> {
        let c = self.peek()?;
        self.position += 1;
        Ok(c)
    }

    pub fn read_float(&mut self) -> Result<f32, LMError> {
        self.skip_spaces();
        let s = self.read_until_space()?;
        s.parse::<f32>()
            .map_err(|_| LMError::ParseError(format!("Failed to parse float: {}", s)))
    }

    pub fn read_double(&mut self) -> Result<f64, LMError> {
        self.skip_spaces();
        let s = self.read_until_space()?;
        s.parse::<f64>()
            .map_err(|_| LMError::ParseError(format!("Failed to parse double: {}", s)))
    }

    pub fn read_ulong(&mut self) -> Result<u64, LMError> {
        self.skip_spaces();
        let s = self.read_until_space()?;
        s.parse::<u64>()
            .map_err(|_| LMError::ParseError(format!("Failed to parse ulong: {}", s)))
    }

    pub fn read_delimited(&mut self, delimiters: &[bool; 256]) -> Result<String, LMError> {
        self.skip_delimiters(delimiters);
        let start = self.position;
        let end = {
            let data = self.backing.as_slice();
            let mut e = start;
            while e < data.len() && !delimiters[data[e] as usize] {
                e += 1;
            }
            e
        };
        let result = std::str::from_utf8(self.slice(start, end))
            .unwrap_or("")
            .to_string();
        self.position = end;
        Ok(result)
    }

    pub fn read_line(&mut self, delim: char, strip_cr: bool) -> Result<String, LMError> {
        if self.position >= self.len() {
            return Err(LMError::IoError(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "End of file",
            )));
        }
        let start = self.position;
        let delim_byte = delim as u8;
        let (end, advance) = {
            let data = self.backing.as_slice();
            match data[start..].iter().position(|&b| b == delim_byte) {
                Some(i) => (start + i, 1),
                None => (data.len(), 0),
            }
        };
        let mut result = std::str::from_utf8(self.slice(start, end))
            .unwrap_or("")
            .to_string();
        self.position = end + advance;
        if strip_cr && result.ends_with('\r') {
            result.pop();
        }
        Ok(result)
    }

    fn skip_spaces(&mut self) {
        while self.position < self.len() {
            let c = self.byte_at(self.position);
            if !(c as char).is_ascii_whitespace() {
                break;
            }
            self.position += 1;
        }
    }

    fn skip_delimiters(&mut self, delimiters: &[bool; 256]) {
        while self.position < self.len() {
            let c = self.byte_at(self.position);
            if !delimiters[c as usize] {
                break;
            }
            self.position += 1;
        }
    }

    fn read_until_space(&mut self) -> Result<String, LMError> {
        let start = self.position;
        let end = {
            let data = self.backing.as_slice();
            let mut e = start;
            while e < data.len() && !(data[e] as char).is_ascii_whitespace() {
                e += 1;
            }
            e
        };
        if end == start {
            return Err(LMError::ParseError("Empty token".to_string()));
        }
        let result = std::str::from_utf8(self.slice(start, end))
            .unwrap_or("")
            .to_string();
        self.position = end;
        Ok(result)
    }

    pub fn at_end(&self) -> bool {
        self.position >= self.len()
    }

    pub fn offset(&self) -> u64 {
        self.position as u64
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

    #[test]
    fn reads_gzip_file() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let tmp_dir = tempfile::tempdir().unwrap();
        let gz_path = tmp_dir.path().join("test.gz");
        {
            let f = File::create(&gz_path).unwrap();
            let mut enc = GzEncoder::new(f, Compression::default());
            enc.write_all(b"hello gzip world\n").unwrap();
            enc.finish().unwrap();
        }

        let mut fp = FilePiece::open(&gz_path).unwrap();
        let line = fp.read_line('\n', false).unwrap();
        assert_eq!(line, "hello gzip world");
    }
}
