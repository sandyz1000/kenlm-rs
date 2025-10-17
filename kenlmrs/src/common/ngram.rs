use crate::types::WordIndex;
use std::mem::size_of;

#[derive(Debug, Clone)]
pub struct NGramHeader<'a> {
    begin: *mut WordIndex,
    end: *mut WordIndex,
    _marker: std::marker::PhantomData<&'a mut [WordIndex]>,
}

impl<'a> NGramHeader<'a> {
    fn new(begin: *mut WordIndex, order: usize) -> Self {
        unsafe {
            NGramHeader {
                begin,
                end: begin.add(order),
                _marker: std::marker::PhantomData,
            }
        }
    }

    fn default() -> Self {
        NGramHeader {
            begin: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            _marker: std::marker::PhantomData,
        }
    }

    fn base(&self) -> *const u8 {
        self.begin as *const u8
    }

    fn base_mut(&self) -> *mut u8 {
        self.begin as *mut u8
    }

    fn rebase(&mut self, to: *mut WordIndex) {
        let difference = unsafe { self.end.offset_from(self.begin) };
        self.begin = to;
        self.end = unsafe { self.begin.add(difference as usize) };
    }

    fn begin(&self) -> *const WordIndex {
        self.begin
    }

    fn begin_mut(&self) -> *mut WordIndex {
        self.begin
    }

    fn end(&self) -> *const WordIndex {
        self.end
    }

    fn end_mut(&self) -> *mut WordIndex {
        self.end
    }

    fn size(&self) -> usize {
        unsafe { self.end.offset_from(self.begin) as usize }
    }

    fn order(&self) -> usize {
        self.size()
    }
}

#[derive(Debug, Clone)]
pub struct NGram<'a, Payload> {
    header: NGramHeader<'a>,
    _marker: std::marker::PhantomData<Payload>,
}

impl<'a, Payload> NGram<'a, Payload> {
    pub fn new(begin: *mut WordIndex, order: usize) -> Self {
        NGram {
            header: NGramHeader::new(begin, order),
            _marker: std::marker::PhantomData,
        }
    }

    fn default() -> Self {
        NGram {
            header: NGramHeader::default(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn next_in_memory(&mut self) {
        let value_size = size_of::<Payload>();
        unsafe {
            let next_base = (self.header.end as *mut u8).add(value_size);
            self.header.rebase(next_base as *mut WordIndex);
        }
    }

    pub fn total_size(order: usize) -> usize {
        order * size_of::<WordIndex>() + size_of::<Payload>()
    }

    pub fn total_size_instance(&self) -> usize {
        Self::total_size(self.header.order())
    }

    pub fn order_from_size(size: usize) -> usize {
        let ret = (size - size_of::<Payload>()) / size_of::<WordIndex>();
        assert!(size == Self::total_size(ret));
        ret
    }

    pub fn value(&self) -> &Payload {
        unsafe { &*(self.header.end() as *const Payload) }
    }

    pub fn value_mut(&mut self) -> &mut Payload {
        unsafe { &mut *(self.header.end_mut() as *mut Payload) }
    }
}
