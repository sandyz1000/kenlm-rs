use crate::model::Model;
use std::io::Write;

pub fn convert_to_bytes<M: Model>(model: &M, fd_in: i64, width: i8) {
    todo!()
}

// Determine how much physical memory there is.  Return 0 on failure.
pub fn guess_physical_memory() -> u64 {
    todo!()
}
// Parse a size like unix sort.  Sadly, this means the default multiplier is K.
pub fn parse_size(arg: &str) -> u64 {
    todo!()
}

// If it's a directory, add a /.  This lets users say -T /tmp without creating /tmpAAAAAA
pub fn normalize_temp_prefix(base: &str) {
    todo!()
}

pub fn create_or_throw(name: &str) -> i64 {
    todo!()
}

pub fn open_read_or_throw(name: &str) -> i64 {
    todo!()
}

pub fn print_usage(out: &mut dyn Write) {
    todo!()
}

// Time in seconds since process started.  Zero on unsupported platforms.
pub fn wall_time() -> f64 {
    todo!()
}

// User + system time, process-wide.
pub fn cputime() -> f64 {
    todo!()
}

// User + system time, thread-specific.
pub fn thread_time() -> f64 {
    todo!()
}

// Resident usage in bytes.
pub fn rssmax() -> u64 {
    todo!()
}
