//! Native Krylov mechanisms. Allocation, operation authority and publication
//! remain with the caller. An original operation/control error is never erased.
pub mod gmres;
pub mod pcg;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplyStatus {
    Applied,
    NumericalFailure,
}

#[derive(Debug)]
pub enum KrylovFailure<E> {
    Invalid(&'static str),
    Operation(E),
}
impl<E: std::fmt::Display> std::fmt::Display for KrylovFailure<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Operation(original) => std::fmt::Display::fmt(original, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for KrylovFailure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Operation(original) => Some(original),
            Self::Invalid(_) => None,
        }
    }
}

/// The selected execution owner implements both original-authority checks and
/// actual-work scheduling. Ready callers explicitly choose ready turns; async
/// callers can suspend after each bounded vector/reduction block.
pub trait Control {
    type Error;
    fn checkpoint(&mut self) -> Result<(), Self::Error>;
    fn cooperate(
        &mut self,
        completed: u64,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>>;
}
