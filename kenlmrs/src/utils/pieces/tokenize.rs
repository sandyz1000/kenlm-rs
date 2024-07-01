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
        while self.skip_empty && self.current.data.is_empty() {
            let found = (self.finder)(self.after.data);
            self.current = StringPiece::new(&self.after.data[..found.data.len()]);
            if found.data.is_empty() {
                self.after = StringPiece::new("");
            } else {
                self.after = StringPiece::new(&self.after.data[found.data.len()..]);
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
            self.advance();
            Some(item)
        }
    }
}

fn trim(mut str: &str, spaces: &[bool; 256]) -> &str {
    while !str.is_empty() && spaces[str.as_bytes()[0] as usize] {
        str = &str[1..];
    }
    while !str.is_empty() && spaces[str.as_bytes()[str.len() - 1] as usize] {
        str = &str[..str.len() - 1];
    }
    str
}

// fn main() {
//     let spaces: [bool; 256] = {
//         let mut array = [false; 256];
//         for &c in b" \t\n\r" {
//             array[c as usize] = true;
//         }
//         array
//     };

//     let input = "hello world";
//     let single_char_finder = SingleCharacter::new(' ');
//     let trimmed = trim(input, &spaces);
//     println!("Trimmed: {:?}", trimmed);
// }
