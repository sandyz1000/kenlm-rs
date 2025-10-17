use crate::constant::WriteMethod;
use crate::types::ModelType;
use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::mem;
use std::os::fd::RawFd;

// Placeholder types
pub struct Mmap;
pub struct LoadException;
pub struct FormatLoadException;
pub struct Sanity;
pub struct Config;
pub struct MmapOptions;
pub struct OldSanity;

// Placeholder functions
fn read_header(_fd: RawFd, _params: &mut FixedWidthParameters) -> Result<(), LoadException> {
    todo!()
}
fn match_check(
    _model_type: ModelType,
    _search_version: u32,
    _params: &FixedWidthParameters,
) -> Result<(), LoadException> {
    todo!()
}
fn total_header_size(_order: u8) -> u64 {
    todo!()
}

// Placeholder constants
const k_bad_size: u64 = 0;
const k_magic_incomplete: &str = "incomplete";
const k_magic_before_version: &str = "version";
const k_magic_version: u32 = 1;

// Placeholder util module types and functions
pub mod util {
    use std::os::fd::RawFd;

    pub enum LoadMethod {
        Lazy,
        Populate,
    }

    pub const k_bad_size: u64 = 0;

    pub fn size_file(_fd: RawFd) -> u64 {
        0
    }
    pub fn map_read(
        _memory: &mut super::Mmap,
        _fd: RawFd,
        _method: LoadMethod,
        _offset: u64,
        _size: u64,
    ) -> Result<(), std::io::Error> {
        Ok(())
    }
    pub fn open_read_or_throw(_file: &str) -> Result<RawFd, std::io::Error> {
        Ok(0)
    }
}

#[derive(Debug)]
pub struct FixedWidthParameters {
    order: u8,
    probing_multiplier: f32,
    model_type: ModelType,
    has_vocabulary: bool,
    search_version: u32,
}

const ALIGN8: usize = 8;

#[derive(Debug)]
pub struct Parameters {
    fixed: FixedWidthParameters,
    counts: Vec<u64>,
}

#[derive(Debug)]
pub struct BinaryFormat {
    write_method: WriteMethod,
    write_mmap: Option<String>,
    load_method: util::LoadMethod,
    file: Option<File>,
    mapping: Option<Mmap>,
    memory_vocab: Option<Vec<u8>>,
    memory_search: Option<Vec<u8>>,
    header_size: usize,
    vocab_size: usize,
    vocab_pad: usize,
    vocab_string_offset: u64,
}

impl BinaryFormat {
    fn new(config: &Config) -> Self {
        Self {
            write_method: config.write_method.clone(),
            write_mmap: config.write_mmap.clone(),
            load_method: config.load_method.clone(),
            file: None,
            mapping: None,
            memory_vocab: None,
            memory_search: None,
            header_size: usize::MAX,
            vocab_size: usize::MAX,
            vocab_pad: 0,
            vocab_string_offset: u64::MAX,
        }
    }

    fn initialize_binary(
        &mut self,
        fd: RawFd,
        model_type: ModelType,
        search_version: u32,
        params: &mut Parameters,
    ) -> Result<(), LoadException> {
        self.file = Some(unsafe { File::from_raw_fd(fd) });
        self.write_mmap = None; // Ignore write requests; this is already in binary format.
        read_header(fd, params)?;
        match_check(model_type, search_version, params)?;
        self.header_size = total_header_size(params.counts.len() as u8);
        Ok(())
    }

    fn read_for_config(
        &self,
        to: &mut [u8],
        amount: usize,
        offset_excluding_header: u64,
    ) -> Result<(), LoadException> {
        assert!(self.header_size != usize::MAX);
        let file = self.file.as_ref().ok_or(LoadException)?;
        file.seek(SeekFrom::Start(
            offset_excluding_header + self.header_size as u64,
        ))?;
        file.read_exact(to)?;
        Ok(())
    }

    fn load_binary(&mut self, size: usize) -> Result<*mut u8, LoadException> {
        assert!(self.header_size != usize::MAX);
        let file = self.file.as_ref().ok_or(LoadException)?;
        let file_size = file.metadata()?.len();
        let total_map = self.header_size as u64 + size as u64;
        if file_size < total_map {
            return Err(FormatLoadException.into());
        }
        let mmap = unsafe {
            MmapOptions::new()
                .offset(0)
                .len(total_map as usize)
                .map(file)?
        };
        self.mapping = Some(mmap);
        self.vocab_string_offset = total_map;
        Ok(unsafe {
            self.mapping
                .as_mut()
                .unwrap()
                .as_mut_ptr()
                .add(self.header_size)
        })
    }

    // ... other methods
}

// Other utility functions and types
// ...

fn is_binary_format(fd: RawFd) -> bool {
    let size = util::size_file(fd);
    if size == util::k_bad_size || size <= mem::size_of::<Sanity>() as u64 {
        return false;
    }

    let mut memory = Vec::with_capacity(mem::size_of::<Sanity>());
    unsafe { memory.set_len(mem::size_of::<Sanity>()) };
    if let Err(_) = util::map_read(
        util::LoadMethod::Lazy,
        fd,
        0,
        mem::size_of::<Sanity>(),
        &mut memory,
    ) {
        return false;
    }

    let reference_header = Sanity::reference();
    if memory == reference_header.as_bytes() {
        return true;
    }

    if memory.starts_with(k_magic_incomplete.as_bytes()) {
        panic!("This binary file did not finish building");
    }

    if memory.starts_with(k_magic_before_version.as_bytes()) {
        let version_str = &memory[k_magic_before_version.len()..];
        let version = str::from_utf8(version_str).unwrap().parse::<i32>().unwrap();
        if version != k_magic_version {
            panic!(
                "Binary file has version {} but this implementation expects version {}",
                version, k_magic_version
            );
        }

        let old_sanity = OldSanity::reference();
        if memory == old_sanity.as_bytes() {
            panic!("Looks like this is an old 32-bit format. The old 32-bit format has been removed so that 64-bit and 32-bit files are exchangeable.");
        }
        panic!("File looks like it should be loaded with mmap, but the test values don't match. Try rebuilding the binary format LM using the same code revision, compiler, and architecture");
    }

    false
}

fn recognize_binary(file: &str, recognized: &mut ModelType) -> bool {
    let fd = util::open_read_or_throw(file).unwrap();
    if !is_binary_format(fd) {
        return false;
    }

    let mut params = Parameters {
        fixed: FixedWidthParameters {
            order: 0,
            probing_multiplier: 0.0,
            model_type: ModelType::ProbingHashTables,
            has_vocabulary: false,
            search_version: 0,
        },
        counts: vec![],
    };
    read_header(fd, &mut params).unwrap();
    *recognized = params.fixed.model_type;
    true
}
