use std::borrow::Cow;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValueCodecError {
    #[error("could not root decoded value at `{path}`: {source}")]
    RootRegistration {
        path: String,
        #[source]
        source: runmat_gc::GcError,
    },
    #[error("allocation for value at `{path}` failed: {source}")]
    Allocation {
        path: String,
        #[source]
        source: crate::context::host_allocation::HostAllocationError,
    },
    #[error("value at `{path}` is not portable: {rule}")]
    Unsupported {
        path: String,
        rule: Cow<'static, str>,
    },
    #[error("invalid value payload at `{path}`: {message}")]
    Invalid { path: String, message: String },
    #[error("TransientSequenceNotPortable: transient output sequence at `{path}` cannot cross a persistence or transport boundary")]
    TransientSequenceNotPortable { path: String },
}

impl ValueCodecError {
    /// Preserve operational allocation/root causes when a canonical Runtime
    /// constructor is used by the codec. Semantic construction failures remain
    /// invalid payloads; classification never depends on diagnostic strings.
    pub(super) fn runtime(path: &str, mut error: crate::RuntimeError) -> Self {
        let mut cause = std::error::Error::source(&error);
        while let Some(source) = cause {
            if let Some(allocation) =
                source.downcast_ref::<crate::context::host_allocation::HostAllocationError>()
            {
                return Self::allocation(path, *allocation);
            }
            cause = source.source();
        }
        if let Some(source) = error.source.take() {
            match source.downcast::<runmat_gc::GcError>() {
                Ok(source) => return Self::root_registration(path, *source),
                Err(source) => error.source = Some(source),
            }
        }
        Self::invalid(path, error.to_string())
    }
    pub(super) fn root_registration(path: &str, source: runmat_gc::GcError) -> Self {
        Self::RootRegistration {
            path: path.to_owned(),
            source,
        }
    }

    pub(super) fn allocation(
        path: &str,
        source: crate::context::host_allocation::HostAllocationError,
    ) -> Self {
        Self::Allocation {
            path: path.to_owned(),
            source,
        }
    }

    pub(in crate::execution) fn unsupported(
        path: &str,
        rule: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self::Unsupported {
            path: path.to_owned(),
            rule: rule.into(),
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
