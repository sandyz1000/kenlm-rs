

#[derive(thiserror::Error)]
pub enum LMError {
    #[error("Configuration error in Langauge Model")]
    ConfigError,

    #[error("Error while while Ken Language model")]
    LoadError,
    
    #[error("Formatting error!!")]
    FormatLoadError,
    
    #[error("Error while loading vocabulary file")]
    VocabLoadError,

    #[error("Special word missing error")]
    SpecialWordMissingError,

}
