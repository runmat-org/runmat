use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NativeCacheError {
    #[error("cache database failed: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("cache filesystem operation failed for `{path}`: {source}")]
    Filesystem {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cache state serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("native cache configuration is invalid: {0}")]
    Config(String),
}

impl NativeCacheError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Filesystem {
            path: path.into(),
            source,
        }
    }
}
