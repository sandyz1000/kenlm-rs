use std::cmp;
use std::fmt;
use std::str;
use std::usize;

#[derive(Clone, Copy)]
pub struct StringPiece<'a> {
    ptr: Option<&'a str>,
}

impl<'a> StringPiece<'a> {
    pub fn new() -> Self {
        StringPiece { ptr: None }
    }

    pub fn from_str(s: &'a str) -> Self {
        StringPiece { ptr: Some(s) }
    }

    pub fn from_slice(offset: &'a str, len: usize) -> Self {
        StringPiece {
            ptr: offset.get(0..len),
        }
    }

    pub fn data(&self) -> Option<&'a str> {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.ptr.map_or(0, |s| s.len())
    }

    pub fn length(&self) -> usize {
        self.size()
    }

    pub fn empty(&self) -> bool {
        self.size() == 0
    }

    pub fn clear(&mut self) {
        self.ptr = None;
    }

    pub fn set(&mut self, data: &'a str, len: usize) {
        self.ptr = data.get(0..len);
    }

    pub fn set_from_str(&mut self, str: &'a str) {
        self.ptr = Some(str);
    }

    pub fn remove_prefix(&mut self, n: usize) {
        if let Some(s) = self.ptr {
            self.ptr = s.get(n..);
        }
    }

    pub fn remove_suffix(&mut self, n: usize) {
        if let Some(s) = self.ptr {
            self.ptr = s.get(0..(s.len() - n));
        }
    }

    pub fn compare(&self, other: &StringPiece) -> i32 {
        match (self.ptr, other.ptr) {
            (Some(s1), Some(s2)) => {
                let min_len = cmp::min(s1.len(), s2.len());
                let cmp_result = s1[..min_len].cmp(&s2[..min_len]);
                if cmp_result == cmp::Ordering::Equal {
                    s1.len().cmp(&s2.len()) as i32
                } else {
                    cmp_result as i32
                }
            }
            (None, Some(_)) => -1,
            (Some(_), None) => 1,
            (None, None) => 0,
        }
    }

    pub fn as_string(&self) -> String {
        self.ptr.unwrap_or("").to_string()
    }

    pub fn as_str(&self) -> &str {
        self.ptr.unwrap_or("")
    }

    pub fn starts_with(&self, x: &StringPiece) -> bool {
        match (self.ptr, x.ptr) {
            (Some(s1), Some(s2)) => s1.starts_with(s2),
            _ => false,
        }
    }

    pub fn ends_with(&self, x: &StringPiece) -> bool {
        match (self.ptr, x.ptr) {
            (Some(s1), Some(s2)) => s1.ends_with(s2),
            _ => false,
        }
    }

    pub fn substr(&self, pos: usize, n: Option<usize>) -> Self {
        match self.ptr {
            Some(s) => {
                let end = cmp::min(s.len(), pos + n.unwrap_or(s.len()));
                StringPiece::from_slice(s, end - pos)
            }
            None => StringPiece::new(),
        }
    }

    pub fn copy(&self, buf: &mut [u8], n: usize, pos: usize) -> usize {
        if let Some(s) = self.ptr {
            let end = cmp::min(s.len(), pos + n);
            buf[..end - pos].copy_from_slice(&s.as_bytes()[pos..end]);
            end - pos
        } else {
            0
        }
    }

    pub fn find(&self, s: &StringPiece, pos: usize) -> Option<usize> {
        if let Some(self_str) = self.ptr {
            if pos > self_str.len() {
                return None;
            }
            self_str[pos..]
                .find(s.ptr.unwrap_or(""))
                .map(|index| index + pos)
        } else {
            None
        }
    }

    pub fn find_char(&self, c: char, pos: usize) -> Option<usize> {
        if let Some(self_str) = self.ptr {
            if pos >= self_str.len() {
                return None;
            }
            self_str[pos..].find(c).map(|index| index + pos)
        } else {
            None
        }
    }

    pub fn rfind(&self, s: &StringPiece, pos: Option<usize>) -> Option<usize> {
        if let Some(self_str) = self.ptr {
            let pos = pos.unwrap_or(self_str.len());
            if s.size() == 0 {
                return Some(cmp::min(self_str.len(), pos));
            }
            let last = &self_str[..cmp::min(self_str.len() - s.size(), pos) + s.size()];
            last.rfind(s.ptr.unwrap_or(""))
        } else {
            None
        }
    }

    pub fn rfind_char(&self, c: char, pos: Option<usize>) -> Option<usize> {
        if let Some(self_str) = self.ptr {
            let pos = pos.unwrap_or(self_str.len());
            self_str[..pos].rfind(c)
        } else {
            None
        }
    }

    fn build_lookup_table(s: &StringPiece) -> [bool; 256] {
        let mut table = [false; 256];
        if let Some(data) = s.ptr {
            for &byte in data.as_bytes() {
                table[byte as usize] = true;
            }
        }
        table
    }

    pub fn find_first_of(&self, s: &StringPiece, pos: usize) -> Option<usize> {
        if self.size() == 0 || s.size() == 0 {
            return None;
        }

        if s.size() == 1 {
            return self.find_char(s.ptr.unwrap().chars().next().unwrap(), pos);
        }

        let lookup = Self::build_lookup_table(s);
        if let Some(self_str) = self.ptr {
            for (i, &byte) in self_str.as_bytes()[pos..].iter().enumerate() {
                if lookup[byte as usize] {
                    return Some(pos + i);
                }
            }
        }
        None
    }

    pub fn find_first_not_of(&self, s: &StringPiece, pos: usize) -> Option<usize> {
        if self.size() == 0 {
            return None;
        }

        if s.size() == 0 {
            return Some(0);
        }

        if s.size() == 1 {
            return self
                .ptr?
                .chars()
                .skip(pos)
                .position(|c| c != s.ptr.unwrap().chars().next().unwrap());
        }

        let lookup = Self::build_lookup_table(s);
        if let Some(self_str) = self.ptr {
            for (i, &byte) in self_str.as_bytes()[pos..].iter().enumerate() {
                if !lookup[byte as usize] {
                    return Some(pos + i);
                }
            }
        }
        None
    }

    pub fn find_first_not_of_char(&self, c: char, pos: usize) -> Option<usize> {
        if self.size() == 0 {
            return None;
        }

        self.ptr?.chars().skip(pos).position(|ch| ch != c)
    }

    pub fn find_last_of(&self, s: &StringPiece, pos: Option<usize>) -> Option<usize> {
        if self.size() == 0 || s.size() == 0 {
            return None;
        }

        if s.size() == 1 {
            return self.rfind_char(s.ptr.unwrap().chars().next().unwrap(), pos);
        }

        let lookup = Self::build_lookup_table(s);
        if let Some(self_str) = self.ptr {
            for (i, &byte) in self_str.as_bytes()[..pos.unwrap_or_else(|| self.size())]
                .iter()
                .enumerate()
                .rev()
            {
                if lookup[byte as usize] {
                    return Some(i);
                }
            }
        }
        None
    }

    pub fn find_last_not_of(&self, s: &StringPiece, pos: Option<usize>) -> Option<usize> {
        if self.size() == 0 {
            return None;
        }

        let pos = pos.unwrap_or_else(|| self.size() - 1);
        if s.size() == 0 {
            return Some(pos);
        }

        if s.size() == 1 {
            return self.find_last_not_of_char(s.ptr.unwrap().chars().next().unwrap(), Some(pos));
        }

        let lookup = Self::build_lookup_table(s);
        if let Some(self_str) = self.ptr {
            for (i, &byte) in self_str.as_bytes()[..pos].iter().enumerate().rev() {
                if !lookup[byte as usize] {
                    return Some(i);
                }
            }
        }
        None
    }

    pub fn find_last_not_of_char(&self, c: char, pos: Option<usize>) -> Option<usize> {
        if self.size() == 0 {
            return None;
        }

        let s = self.ptr?;
        let pos = pos.unwrap_or_else(|| self.size() - 1);

        // Collect chars up to pos and search backwards
        let chars: Vec<(usize, char)> = s.chars().take(pos + 1).enumerate().collect();

        // Search from the end
        for (i, ch) in chars.iter().rev() {
            if *ch != c {
                return Some(*i);
            }
        }
        None
    }
}

impl<'a> PartialEq for StringPiece<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<'a> Eq for StringPiece<'a> {}

impl<'a> PartialOrd for StringPiece<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for StringPiece<'a> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match (self.ptr, other.ptr) {
            (Some(s1), Some(s2)) => s1.cmp(s2),
            (None, Some(_)) => cmp::Ordering::Less,
            (Some(_), None) => cmp::Ordering::Greater,
            (None, None) => cmp::Ordering::Equal,
        }
    }
}

impl<'a> fmt::Debug for StringPiece<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.ptr.unwrap_or(""))
    }
}

impl<'a> fmt::Display for StringPiece<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ptr.unwrap_or(""))
    }
}

impl<'a> StringPiece<'a> {
    pub const NPOS: usize = usize::MAX;
}

// fn main() {
//     let sp = StringPiece::from_str("Hello, World!");
//     let sp2 = StringPiece::from_str("World");

//     println!("sp: {:?}", sp);
//     println!("sp2: {:?}", sp2);
//     println!("sp starts with 'Hello': {}", sp.starts_with(&StringPiece::from_str("Hello")));
//     println!("sp ends with 'World!': {}", sp.ends_with(&StringPiece::from_str("World!")));
//     println!("sp contains 'World': {}", sp.find(&sp2, 0).is_some());
// }

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(s: &str) -> StringPiece<'_> { StringPiece::from_str(s) }

    #[test]
    fn test_basic_size_and_empty() {
        assert!(StringPiece::new().empty());
        assert_eq!(StringPiece::new().size(), 0);
        assert_eq!(sp("hello").size(), 5);
        assert!(!sp("hello").empty());
    }

    #[test]
    fn test_as_str_as_string() {
        assert_eq!(sp("world").as_str(), "world");
        assert_eq!(sp("world").as_string(), "world".to_string());
        assert_eq!(StringPiece::new().as_str(), "");
    }

    #[test]
    fn test_clear() {
        let mut s = sp("data");
        s.clear();
        assert!(s.empty());
        assert!(s.data().is_none());
    }

    #[test]
    fn test_set_from_str() {
        let mut s = StringPiece::new();
        s.set_from_str("new");
        assert_eq!(s.as_str(), "new");
    }

    #[test]
    fn test_starts_with_ends_with() {
        let s = sp("Hello, World!");
        assert!(s.starts_with(&sp("Hello")));
        assert!(s.ends_with(&sp("World!")));
        assert!(!s.starts_with(&sp("World")));
        assert!(!s.ends_with(&sp("Hello")));
    }

    #[test]
    fn test_find_char() {
        let s = sp("hello");
        assert_eq!(s.find_char('l', 0), Some(2));
        assert_eq!(s.find_char('l', 3), Some(3));
        assert_eq!(s.find_char('z', 0), None);
    }

    #[test]
    fn test_find_substring() {
        let s = sp("abcabc");
        let pat = sp("bc");
        assert_eq!(s.find(&pat, 0), Some(1));
        assert_eq!(s.find(&pat, 2), Some(4));
        assert_eq!(s.find(&sp("xyz"), 0), None);
    }

    #[test]
    fn test_rfind() {
        let s = sp("abcabc");
        let pat = sp("bc");
        let result = s.rfind(&pat, None);
        assert_eq!(result, Some(4));
    }

    #[test]
    fn test_remove_prefix() {
        let mut s = sp("Hello, World!");
        s.remove_prefix(7); // removes "Hello, "
        assert_eq!(s.as_str(), "World!");
    }

    #[test]
    fn test_remove_suffix() {
        let mut s = sp("Hello!");
        s.remove_suffix(1); // removes "!"
        assert_eq!(s.as_str(), "Hello");
    }

    #[test]
    fn test_from_slice() {
        let s = StringPiece::from_slice("hello world", 5);
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn test_compare_equal() {
        assert_eq!(sp("abc").compare(&sp("abc")), 0);
    }

    #[test]
    fn test_compare_less() {
        // "abc" < "abd" lexicographically
        assert!(sp("abc").compare(&sp("abd")) < 0 || sp("abc").compare(&sp("abd")) <= 0);
    }

    #[test]
    fn test_eq_ord_traits() {
        assert_eq!(sp("foo"), sp("foo"));
        assert_ne!(sp("foo"), sp("bar"));
        assert!(sp("abc") < sp("abd"));
        assert!(sp("abd") > sp("abc"));
    }

    #[test]
    fn test_copy() {
        let s = sp("hello");
        let mut buf = vec![0u8; 5];
        let n = s.copy(&mut buf, 3, 0);
        assert_eq!(n, 3);
        assert_eq!(&buf[..3], b"hel");
    }

    #[test]
    fn test_find_first_of() {
        let s = sp("hello world");
        let chars = sp("aeiou");
        let pos = s.find_first_of(&chars, 0);
        assert_eq!(pos, Some(1)); // 'e' at index 1
    }

    #[test]
    fn test_find_last_not_of() {
        let s = sp("hello   ");
        let spaces = sp(" ");
        let pos = s.find_last_not_of(&spaces, None);
        assert_eq!(pos, Some(4)); // 'o' at index 4
    }
}
