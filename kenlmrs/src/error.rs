#[derive(Debug, thiserror::Error)]
pub enum LMError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Error loading language model: {0}")]
    LoadError(String),

    #[error("Format error: {0}")]
    FormatError(String),

    #[error("Error loading vocabulary: {0}")]
    VocabError(String),

    #[error("Special word missing: {0}")]
    SpecialWordMissingError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Unsupported model type")]
    UnsupportedModelType,

    #[error("Binary format version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },

    #[error("Invalid ARPA file: {0}")]
    InvalidArpa(String),

    #[error("Positive log probability encountered: {0}")]
    PositiveLogProbability(f32),
}
