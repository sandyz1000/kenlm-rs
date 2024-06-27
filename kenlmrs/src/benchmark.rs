pub fn QueryFromBytes(model: &Model, config: BenchmarkConfig, Width: i8) {
    todo!()
}

pub fn ConvertToBytes(model: &Model, fd_in: i64, width: i8) {
    todo!()
}

// Determine how much physical memory there is.  Return 0 on failure.
pub fn GuessPhysicalMemory() -> u64 {
    todo!()
}
// Parse a size like unix sort.  Sadly, this means the default multiplier is K.
pub fn ParseSize(arg: &str) -> u64 {
    todo!()
}

// If it's a directory, add a /.  This lets users say -T /tmp without creating /tmpAAAAAA
pub fn NormalizeTempPrefix(base: &str) {
    todo!()
}
pub fn CreateOrThrow(name: &str) -> i64 {
    todo!()
}
pub fn OpenReadOrThrow(name: &str) -> i64 {
    todo!()
}
pub fn PrintUsage(out: &OStream) {
    todo!()
}

// Time in seconds since process started.  Zero on unsupported platforms.
pub fn WallTime() -> f64 {
    todo!()
}

// User + system time, process-wide.
pub fn CPUTime() -> f64 {
    todo!()
}

// User + system time, thread-specific.
pub fn ThreadTime() -> f64 {
    todo!()
}

// Resident usage in bytes.
pub fn RSSMax() -> u64 {
    todo!()
}
