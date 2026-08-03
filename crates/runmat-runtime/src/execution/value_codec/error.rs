use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValueCodecError {
    #[error("value kind `{0}` is not portable")]
    Unsupported(&'static str),
    #[error("invalid value payload: {0}")]
    Invalid(String),
}
