use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValueCodecError {
    #[error("value at `{path}` is not portable: {rule}")]
    Unsupported { path: String, rule: &'static str },
    #[error("invalid value payload at `{path}`: {message}")]
    Invalid { path: String, message: String },
    #[error("TransientSequenceNotPortable: transient output sequence at `{path}` cannot cross a persistence or transport boundary")]
    TransientSequenceNotPortable { path: String },
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

    pub(super) fn transient_not_portable(path: &str) -> Self {
        Self::TransientSequenceNotPortable {
            path: path.to_owned(),
        }
    }
}
