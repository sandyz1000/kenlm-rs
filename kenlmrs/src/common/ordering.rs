use std::slice;
use std::cmp::Ordering;
use std::mem::size_of;


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
