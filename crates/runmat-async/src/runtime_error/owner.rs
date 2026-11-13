//! The original caught error has one known-layout immutable shared owner.
use super::{DiagnosticConstructionError, RetainedDiagnosticAllocation, RuntimeError};
use runmat_execution::resource::ExactShared;
use std::{fmt, ops::Deref};

#[derive(Debug)]
struct Payload {
    error: RuntimeError,
    // ExactShared frees its own header before dropping these fields.
    header_claim: Option<RetainedDiagnosticAllocation>,
}

#[derive(Clone, Debug)]
pub struct RuntimeErrorOwner(ExactShared<Payload>);

impl RuntimeErrorOwner {
    pub fn new(error: RuntimeError) -> Self {
        Self(ExactShared::new(Payload {
            error,
            header_claim: None,
        }))
    }

    pub fn requested_header_bytes() -> usize {
        ExactShared::<Payload>::requested_bytes()
    }

    /// Calls the real allocating domain's admission before creating this
    /// original header. Child diagnostics retain their separate allocations.
    pub fn try_new_exact<E>(
        error: RuntimeError,
        admit: impl FnOnce(usize) -> Result<RetainedDiagnosticAllocation, E>,
    ) -> Result<Self, DiagnosticConstructionError<E>> {
        let claim = admit(Self::requested_header_bytes())
            .map_err(DiagnosticConstructionError::Admission)?;
        ExactShared::try_new(Payload {
            error,
            header_claim: Some(claim),
        })
        .map(Self)
        .map_err(|_| DiagnosticConstructionError::AllocationFailed)
    }

    pub fn ptr_eq(first: &Self, second: &Self) -> bool {
        first.0.ptr_eq(&second.0)
    }

    pub fn try_unwrap(owner: Self) -> Result<RuntimeError, Self> {
        match owner.0.try_unwrap() {
            Ok(Payload {
                error,
                header_claim,
            }) => {
                drop(header_claim);
                Ok(error)
            }
            Err(shared) => Err(Self(shared)),
        }
    }
}
impl From<RuntimeError> for RuntimeErrorOwner {
    fn from(error: RuntimeError) -> Self {
        Self::new(error)
    }
}
impl Deref for RuntimeErrorOwner {
    type Target = RuntimeError;
    fn deref(&self) -> &Self::Target {
        &self.0.get().error
    }
}
impl AsRef<RuntimeError> for RuntimeErrorOwner {
    fn as_ref(&self) -> &RuntimeError {
        self
    }
}
impl fmt::Display for RuntimeErrorOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, formatter)
    }
}
impl std::error::Error for RuntimeErrorOwner {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&**self)
    }
}
