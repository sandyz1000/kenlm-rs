use std::slice;
use std::cmp::Ordering;
use std::mem::size_of;
use std::marker::PhantomData;
use std::mem;


type WordIndex = u32;

pub trait Comparator {
    fn order(&self) -> usize;
    fn compare(&self, lhs: &[WordIndex], rhs: &[WordIndex]) -> bool;
}

pub struct SuffixOrder {
    order: usize,
}

impl Comparator for SuffixOrder {
    fn order(&self) -> usize {
        self.order
    }

    fn compare(&self, lhs: &[WordIndex], rhs: &[WordIndex]) -> bool {
        for i in (0..self.order).rev() {
            if lhs[i] != rhs[i] {
                return lhs[i] < rhs[i];
            }
        }
        false
    }
}

pub struct ContextOrder {
    order: usize,
}

impl Comparator for ContextOrder {
    fn order(&self) -> usize {
        self.order
    }

    fn compare(&self, lhs: &[WordIndex], rhs: &[WordIndex]) -> bool {
        for i in (0..self.order - 1).rev() {
            if lhs[i] != rhs[i] {
                return lhs[i] < rhs[i];
            }
        }
        lhs[self.order - 1] < rhs[self.order - 1]
    }
}

pub struct PrefixOrder {
    order: usize,
}

impl Comparator for PrefixOrder {
    fn order(&self) -> usize {
        self.order
    }

    fn compare(&self, lhs: &[WordIndex], rhs: &[WordIndex]) -> bool {
        for i in 0..self.order {
            if lhs[i] != rhs[i] {
                return lhs[i] < rhs[i];
            }
        }
        false
    }
}

pub struct SuffixLexicographicLess;

impl SuffixLexicographicLess {
    fn compare<T: AsRef<[WordIndex]>>(&self, first: T, second: T) -> bool {
        let first = first.as_ref();
        let second = second.as_ref();
        let mut f_iter = first.iter().rev();
        let mut s_iter = second.iter().rev();

        loop {
            match (f_iter.next(), s_iter.next()) {
                (Some(f), Some(s)) => {
                    if f < s {
                        return true;
                    }
                    if f > s {
                        return false;
                    }
                }
                (Some(_), None) => return false,
                (None, Some(_)) => return true,
                (None, None) => return first.len() < second.len(),
            }
        }
    }
}


#[derive(Debug, Clone, Copy)]
pub struct BuildingPayload {
    count: u64,
}

pub struct ChainPositions<'a> {
    positions: Vec<&'a [u8]>,
}

impl<'a> ChainPositions<'a> {
    pub fn size(&self) -> usize {
        self.positions.len()
    }

    pub fn get(&self, index: usize) -> &'a [u8] {
        self.positions[index]
    }
}

pub struct ProxyStream<'a, T> {
    data: Option<&'a [u8]>,
    header: T,
}

impl<'a, T> ProxyStream<'a, T> {
    pub fn new(data: Option<&'a [u8]>, header: T) -> Self {
        ProxyStream { data, header }
    }

    pub fn begin(&self) -> &[u8] {
        self.data.unwrap_or(&[])
    }

    pub fn get(&self) -> &T {
        &self.header
    }

    pub fn is_some(&self) -> bool {
        self.data.is_some()
    }

    pub fn advance(&mut self) -> bool {
        // Implement logic to advance the stream
        // Return true if advancement is successful, false otherwise
        true
    }
}

pub struct NGramHeader {
    pub dummy: Option<*const ()>,
    pub value: usize,
}

impl NGramHeader {
    pub fn new(dummy: Option<*const ()>, value: usize) -> Self {
        NGramHeader { dummy, value }
    }
}

pub trait Callback {
    fn enter(&mut self, current: usize, header: &NGramHeader);
    fn exit(&mut self, current: usize, header: &NGramHeader);
}

pub trait Compare {
    const K_MATCH_OFFSET: usize;

    fn compare(a: &[u8], b: &[u8]) -> Ordering;
}

pub fn joint_order<Cb, Cmp>(positions: &ChainPositions, callback: &mut Cb)
where
    Cb: Callback,
    Cmp: Compare,
{
    let mut streams_with_dummy: Vec<ProxyStream<NGramHeader>> = Vec::with_capacity(positions.size() + 1);
    streams_with_dummy.push(ProxyStream::new(None, NGramHeader::new(None, 0)));

    for i in 0..positions.size() {
        streams_with_dummy.push(ProxyStream::new(Some(positions.get(i)), NGramHeader::new(None, i + 1)));
    }

    let streams: &mut [ProxyStream<NGramHeader>] = &mut streams_with_dummy[1..];
    let mut order = 0;

    while order < positions.size() && streams[order].is_some() {
        order += 1;
    }

    assert!(order > 0, "Should always have <unk>.");

    let mut current = 0;
    loop {
        if current > 0 && streams[current - 1].begin() == &streams[current].begin()[Cmp::K_MATCH_OFFSET..] {
            callback.enter(current, streams[current].get());
            if current + 1 < order {
                current += 1;
                continue;
            }
        }

        loop {
            assert!(current > 0);
            current -= 1;
            callback.exit(current, streams[current].get());

            if streams[current].advance() {
                break;
            }

            if order != current + 1 {
                panic!("Detected n-gram without matching suffix");
            }

            order = current;
            if order == 0 {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*; 
    
    #[test]
    fn test_ordering() {
        // Example usage
        let positions = ChainPositions {
            positions: vec![b"data1", b"data2"],
        };
    
        struct MyCallback;
        impl Callback for MyCallback {
            fn enter(&mut self, _current: usize, _header: &NGramHeader) {
                // Implement logic for entering callback
            }
    
            fn exit(&mut self, _current: usize, _header: &NGramHeader) {
                // Implement logic for exiting callback
            }
        }
    
        struct MyCompare;
        impl Compare for MyCompare {
            const K_MATCH_OFFSET: usize = 0;
    
            fn compare(a: &[u8], b: &[u8]) -> Ordering {
                a.cmp(b)
            }
        }
    
        let mut callback = MyCallback;
        joint_order::<_, MyCompare>(&positions, &mut callback);
    }
}
