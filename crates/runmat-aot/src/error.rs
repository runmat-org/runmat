#[derive(Debug, thiserror::Error)]
pub enum AotError {
    #[error("{code}: {message}")]
    Contract { code: &'static str, message: String },
    #[error("{operation} `{path}` failed: {source}")]
    Io {
        operation: &'static str,
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("native linker `{driver}` failed with status {status}: {diagnostic}")]
    Linker {
        driver: std::path::PathBuf,
        status: String,
        diagnostic: String,
    },
}

impl AotError {
    pub(crate) fn contract(code: &'static str, message: impl Into<String>) -> Self {
        Self::Contract {
            code,
            message: message.into(),
        }
    }

    pub(crate) fn io(
        operation: &'static str,
        path: impl Into<std::path::PathBuf>,
        source: std::io::Error,
    ) -> Self {
        Self::Io {
            operation,
            path: path.into(),
            source,
        }
    }
}

pub type AotResult<T> = Result<T, AotError>;
