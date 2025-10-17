use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;
use std::str::FromStr;

use crate::error::LMError as Error;

// Define the LineIterator struct
pub struct LineIterator<'a> {
    backing: Option<&'a FilePiece>,
    lines: Box<dyn Iterator<Item = io::Result<String>> + 'a>,
}

impl<'a> LineIterator<'a> {
    fn new(file_piece: &'a FilePiece, delim: char) -> Self {
        let reader = BufReader::new(&file_piece.file);
        let lines = Box::new(reader.lines());
        Self {
            backing: Some(file_piece),
            lines,
        }
    }
}

impl<'a> Iterator for LineIterator<'a> {
    type Item = io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.lines.next()
    }
}

// Define the FilePiece struct
pub struct FilePiece {
    file: File,
    // data: String,
    // offset: usize,
    file_name: String,
    buffer_size: usize,
}

impl FilePiece {
    fn new<P: AsRef<Path>>(path: P, buffer_size: usize) -> io::Result<Self> {
        let file = File::open(&path)?;
        let file_name = path.as_ref().to_string_lossy().to_string();
        Ok(Self {
            file,
            file_name,
            buffer_size,
        })
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn get(&self) -> char {
        self.data.chars().nth(self.offset).unwrap()
    }

    fn begin(&self) -> LineIterator {
        LineIterator::new(self, '\n')
    }

    pub fn read_float(&mut self) -> f32 {
        let mut chars = self.data[self.offset..].chars();
        let mut result = String::new();
        while let Some(c) = chars.next() {
            if c.is_whitespace() || c == '\t' {
                break;
            }
            result.push(c);
            self.offset += 1;
        }
        f32::from_str(&result).unwrap()
    }

    pub fn read_delimited(&mut self, delimiters: &[bool; 256]) -> String {
        let mut result = String::new();
        while self.offset < self.data.len() {
            let c = self.data.chars().nth(self.offset).unwrap();
            if delimiters[c as usize] {
                break;
            }
            result.push(c);
            self.offset += 1;
        }
        result
    }

    pub fn read_line(&self, delim: char, strip_cr: bool) -> io::Result<String> {
        let mut reader = BufReader::new(&self.file);
        let mut buffer = String::new();
        reader.read_line(&mut buffer)?;
        if strip_cr {
            buffer = buffer.trim_end_matches('\r').to_string();
        }
        Ok(buffer)
    }

    fn read_double(&self) -> Result<f64, Error> {
        // ParseNumberException
        self.read_number::<f64>()
    }

    fn read_long(&self) -> Result<i64, Error> {
        self.read_number::<i64>()
    }

    fn read_ulong(&self) -> Result<u64, Error> {
        // ParseNumberException
        self.read_number::<u64>()
    }

    fn read_number<T: FromStr>(&self) -> Result<T, Error> {
        // ParseNumberException
        let line = self
            .read_line('\n', true)
            .map_err(|e| Error::ParseNumberException(e.to_string()))?;
        line.parse::<T>()
            .map_err(|_| Error::ParseNumberException(line))
    }
}
