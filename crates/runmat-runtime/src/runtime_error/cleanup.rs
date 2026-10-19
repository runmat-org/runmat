//! Preserve both execution and cleanup failures at an owning call boundary.
use super::{GpuGatherRetry, RuntimeError};
use runmat_async::runtime_error::RuntimeErrorSource;
use std::{error::Error, fmt};

pub fn complete_with_cleanup<T>(
    result: Result<T, RuntimeError>,
    cleanup: Result<(), RuntimeError>,
) -> Result<T, RuntimeError> {
    match (result, cleanup) {
        (result, Ok(())) => result,
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(mut primary), Err(cleanup)) => {
            primary.source = Some(Box::new(CombinedFailure {
                primary_source: primary.source.take(),
                cleanup,
            }));
            primary.gpu_gather_retry = GpuGatherRetry::Never;
            Err(primary)
        }
    }
}

#[derive(Debug)]
struct CombinedFailure {
    primary_source: Option<Box<RuntimeErrorSource>>,
    cleanup: RuntimeError,
}
impl fmt::Display for CombinedFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "cleanup also failed: {}", self.cleanup)
    }
}
impl Error for CombinedFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.primary_source
            .as_ref()
            .map(|error| error.as_ref() as _)
            .or(Some(&self.cleanup))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "wasm32")]
    use std::rc::Rc as Owner;
    #[cfg(not(target_arch = "wasm32"))]
    use std::sync::Arc as Owner;

    #[derive(Debug)]
    struct OwnedFailure(Owner<()>);
    impl fmt::Display for OwnedFailure {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "retained source ({})",
                Owner::strong_count(&self.0)
            )
        }
    }
    impl Error for OwnedFailure {}

    #[test]
    fn simultaneous_failures_preserve_primary_contract_and_both_owners() {
        let first = Owner::new(());
        let second = Owner::new(());
        let first_weak = Owner::downgrade(&first);
        let second_weak = Owner::downgrade(&second);
        let primary = crate::build_runtime_error("primary")
            .with_identifier("RunMat:Primary")
            .with_source(OwnedFailure(first))
            .build();
        let cleanup = crate::build_runtime_error("cleanup")
            .with_source(OwnedFailure(second))
            .build();
        let error = complete_with_cleanup::<()>(Err(primary), Err(cleanup)).unwrap_err();
        assert_eq!(error.message(), "primary");
        assert_eq!(error.identifier(), Some("RunMat:Primary"));
        assert!(first_weak.upgrade().is_some());
        assert!(second_weak.upgrade().is_some());
        drop(error);
        assert!(first_weak.upgrade().is_none());
        assert!(second_weak.upgrade().is_none());
    }

    #[test]
    fn failed_cleanup_drops_successful_owned_result_and_keeps_failure() {
        let value = Owner::new(());
        let value_weak = Owner::downgrade(&value);
        let failure = Owner::new(());
        let failure_weak = Owner::downgrade(&failure);
        let cleanup = crate::build_runtime_error("cleanup")
            .with_source(OwnedFailure(failure))
            .build();
        let error = complete_with_cleanup(Ok(value), Err(cleanup)).unwrap_err();
        assert!(value_weak.upgrade().is_none());
        assert!(failure_weak.upgrade().is_some());
        drop(error);
        assert!(failure_weak.upgrade().is_none());
    }
}
