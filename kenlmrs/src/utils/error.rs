
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    ParseNumberException(String)
}
