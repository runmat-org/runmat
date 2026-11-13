use thiserror::Error;

use crate::loader::LoaderError;

#[derive(Debug, Error)]
pub enum InvocationError {
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error("native call `{symbol}` expected {expected} arguments but received {actual}")]
    Arity {
        symbol: String,
        expected: usize,
        actual: usize,
    },
    #[error("native call `{symbol}` argument {argument} ({name}) is invalid: {message}")]
    Argument {
        symbol: String,
        argument: usize,
        name: String,
        message: String,
    },
    #[error("native call `{symbol}` has unsupported ABI metadata: {message}")]
    Abi { symbol: String, message: String },
    #[error("native call `{symbol}` returned an invalid value: {message}")]
    Output { symbol: String, message: String },
    #[error("native callback in `{symbol}` failed: {message}")]
    Callback { symbol: String, message: String },
}
