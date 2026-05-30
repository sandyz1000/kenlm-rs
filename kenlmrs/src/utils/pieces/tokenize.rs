use std::error::Error;
use std::fmt;
use std::str;

#[derive(Debug, Clone)]
struct OutOfTokens;

impl fmt::Display for OutOfTokens {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Out of tokens")
    }
}

impl Error for OutOfTokens {}

#[derive(Debug, Clone)]
pub struct StringPiece<'a> {
    data: &'a str,
}

impl<'a> StringPiece<'a> {
    fn new(data: &'a str) -> Self {
        StringPiece { data }
    }
}

#[derive(Debug, Clone)]
pub struct SingleCharacter {
    delim: char,
}

impl SingleCharacter {
    fn new(delim: char) -> Self {
        SingleCharacter { delim }
    }

    fn find<'a>(&self, input: &'a str) -> StringPiece<'a> {
        match input.find(self.delim) {
            Some(pos) => StringPiece::new(&input[pos..pos + 1]),
            None => StringPiece::new(&input[input.len()..]),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiCharacter<'a> {
    delimiter: StringPiece<'a>,
}

impl<'a> MultiCharacter<'a> {
    fn new(delimiter: StringPiece<'a>) -> Self {
        MultiCharacter { delimiter }
    }

    fn find<'b>(&self, input: &'b str) -> StringPiece<'b> {
        match input.find(self.delimiter.data) {
            Some(pos) => StringPiece::new(&input[pos..pos + self.delimiter.data.len()]),
            None => StringPiece::new(&input[input.len()..]),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnyCharacter<'a> {
    chars: StringPiece<'a>,
}

impl<'a> AnyCharacter<'a> {
    fn new(chars: StringPiece<'a>) -> Self {
        AnyCharacter { chars }
    }

    fn find<'b>(&self, input: &'b str) -> StringPiece<'b> {
        match input.find(&self.chars.data) {
            Some(pos) => StringPiece::new(&input[pos..pos + 1]),
            None => StringPiece::new(&input[input.len()..]),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BoolCharacter<'a> {
    delimiter: &'a [bool; 256],
}

impl<'a> BoolCharacter<'a> {
    fn new(delimiter: &'a [bool; 256]) -> Self {
        BoolCharacter { delimiter }
    }

    fn find<'b>(&self, input: &'b str) -> StringPiece<'b> {
        for (i, &c) in input.as_bytes().iter().enumerate() {
            if self.delimiter[c as usize] {
                return StringPiece::new(&input[i..i + 1]);
            }
        }
        StringPiece::new(&input[input.len()..])
    }

    fn build(characters: &str, out: &mut [bool; 256]) {
        for c in characters.bytes() {
            out[c as usize] = true;
        }
    }
}

#[derive(Debug, Clone)]
struct AnyCharacterLast<'a> {
    chars: StringPiece<'a>,
}

impl<'a> AnyCharacterLast<'a> {
    fn new(chars: StringPiece<'a>) -> Self {
        AnyCharacterLast { chars }
    }

    fn find<'b>(&self, input: &'b str) -> StringPiece<'b> {
        match input.rfind(&self.chars.data) {
            Some(pos) => StringPiece::new(&input[pos..pos + 1]),
            None => StringPiece::new(&input[input.len()..]),
        }
    }
}

struct TokenIter<'a, Find> {
    current: StringPiece<'a>,
    after: StringPiece<'a>,
    finder: Find,
    skip_empty: bool,
}

impl<'a, Find> TokenIter<'a, Find>
where
    Find: Fn(&'a str) -> StringPiece<'a>,
{
    fn new<S>(str: &'a str, construct: Find, skip_empty: bool) -> Self {
        let after = StringPiece::new(str);
        let mut iter = TokenIter {
            current: StringPiece::new(""),
            after,
            finder: construct,
            skip_empty,
        };
        iter.advance();
        iter
    }

    fn advance(&mut self) {
        // Loop terminates when: not skipping empties, current is non-empty, or after is exhausted.
        while self.skip_empty && self.current.data.is_empty() && !self.after.data.is_empty() {
            let found = (self.finder)(self.after.data);
            if found.data.is_empty() {
                // No delimiter found — everything remaining is the final token.
                self.current = self.after.clone();
                self.after = StringPiece::new("");
            } else {
                // Delimiter found at some offset within after. Extract token before it.
                let offset = found.data.as_ptr() as usize - self.after.data.as_ptr() as usize;
                self.current = StringPiece::new(&self.after.data[..offset]);
                self.after = StringPiece::new(&self.after.data[offset + found.data.len()..]);
            }
        }
    }
}

impl<'a, Find> Iterator for TokenIter<'a, Find>
where
    Find: Fn(&'a str) -> StringPiece<'a>,
{
    type Item = StringPiece<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.data.is_empty() {
            None
        } else {
            let item = self.current.clone();
            // Reset current so advance() will search for the next token.
            self.current = StringPiece::new("");
            self.advance();
            Some(item)
        }
    }
}

fn trim<'a>(mut str: &'a str, spaces: &[bool; 256]) -> &'a str {
    while !str.is_empty() && spaces[str.as_bytes()[0] as usize] {
        str = &str[1..];
    }
    while !str.is_empty() && spaces[str.as_bytes()[str.len() - 1] as usize] {
        str = &str[..str.len() - 1];
    }
    str
}

// fn main() { ... }

#[cfg(test)]
mod tests {
    use super::*;

    fn space_table() -> [bool; 256] {
        let mut t = [false; 256];
        for &c in b" \t\n\r" { t[c as usize] = true; }
        t
    }

    #[test]
    fn test_trim_basic() {
        let spaces = space_table();
        assert_eq!(trim("  hello  ", &spaces), "hello");
        assert_eq!(trim("hello", &spaces), "hello");
        assert_eq!(trim("", &spaces), "");
        assert_eq!(trim("  ", &spaces), "");
    }

    #[test]
    fn test_trim_tabs_and_newlines() {
        let spaces = space_table();
        assert_eq!(trim("\thello\n", &spaces), "hello");
    }

    #[test]
    fn test_single_character_find_present() {
        let finder = SingleCharacter::new(' ');
        let result = finder.find("hello world");
        assert_eq!(result.data, " ");
    }

    #[test]
    fn test_single_character_find_absent() {
        let finder = SingleCharacter::new('x');
        let result = finder.find("hello");
        assert_eq!(result.data, ""); // returns empty slice at end
    }

    #[test]
    fn test_multi_character_find_present() {
        let finder = MultiCharacter::new(StringPiece::new("or"));
        let result = finder.find("hello world");
        assert_eq!(result.data, "or");
    }

    #[test]
    fn test_multi_character_find_absent() {
        let finder = MultiCharacter::new(StringPiece::new("xyz"));
        let result = finder.find("hello");
        assert_eq!(result.data, "");
    }

    #[test]
    fn test_any_character_find_first_match() {
        // AnyCharacter searches for the `chars` string as a substring
        // (not individual characters — that's the C++ AnyCharacter semantic here)
        let finder = AnyCharacter::new(StringPiece::new("ell")); // look for "ell" in input
        let result = finder.find("hello world");
        // "ell" found at pos 1 → returns 1-char slice starting there
        assert_eq!(result.data, "e");
    }

    #[test]
    fn test_any_character_find_absent() {
        let finder = AnyCharacter::new(StringPiece::new("xyz"));
        let result = finder.find("hello");
        assert_eq!(result.data, "", "not found → empty slice at end");
    }

    #[test]
    fn test_bool_character_find() {
        let mut delims = [false; 256];
        BoolCharacter::build(" ,", &mut delims);
        let finder = BoolCharacter::new(&delims);

        let result = finder.find("hello, world");
        assert_eq!(result.data, ",");
    }

    #[test]
    fn test_bool_character_find_absent() {
        let delims = [false; 256];
        let finder = BoolCharacter::new(&delims);
        let result = finder.find("hello");
        assert_eq!(result.data, "");
    }

    #[test]
    fn test_any_character_last_find() {
        let finder = AnyCharacterLast::new(StringPiece::new("l"));
        let result = finder.find("hello");
        // rfind of 'l' in "hello" → index 3
        assert_eq!(result.data, "l");
    }

    #[test]
    fn test_token_iter_basic() {
        static INPUT: &str = "hello world foo";
        let finder = SingleCharacter::new(' ');
        let f = move |s: &'static str| finder.find(s);
        let tokens: Vec<_> = TokenIter::new::<()>(INPUT, f, true).collect();
        let words: Vec<&str> = tokens.iter().map(|t| t.data).collect();
        assert_eq!(words, vec!["hello", "world", "foo"]);
    }
}
