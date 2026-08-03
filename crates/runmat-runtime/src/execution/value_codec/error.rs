use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValueCodecError {
    #[error("value at `{path}` is not portable: {rule}")]
    Unsupported { path: String, rule: &'static str },
    #[error("invalid value payload at `{path}`: {message}")]
    Invalid { path: String, message: String },
}

impl ValueCodecError {
    pub(super) fn unsupported(path: &str, rule: &'static str) -> Self {
        Self::Unsupported {
            path: path.to_owned(),
            rule,
        }
    }

    pub(super) fn invalid(path: &str, message: impl Into<String>) -> Self {
        Self::Invalid {
            path: path.to_owned(),
            message: message.into(),
        }
    }
}
