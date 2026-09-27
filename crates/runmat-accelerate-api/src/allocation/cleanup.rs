use std::panic::{catch_unwind, AssertUnwindSafe};

/// An allocation-owner cleanup callback unwound. This is not proof of physical
/// retirement and must not authorize retrying an uncertain release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationCleanupPanicked;

/// Contain exceptional unwinds at an allocation owner's teardown boundary.
/// The caller must establish conservative retirement state before invoking a
/// fallible callback. Normal results (including errors) are returned unchanged.
///
/// A caught payload is dropped normally. If its destructor also unwinds, only
/// that secondary panic payload is forgotten to avoid recursive destructor
/// failure. Ordinary allocations and owner resources are never forgotten.
/// This cannot catch panic-abort or make the owner's own destructor safe.
pub fn catch_allocation_cleanup_unwind<T>(
    operation: impl FnOnce() -> T,
) -> Result<T, AllocationCleanupPanicked> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|payload| {
        if let Err(secondary) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
            std::mem::forget(secondary);
        }
        AllocationCleanupPanicked
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_cleanup_preserves_normal_results_and_errors() {
        assert_eq!(catch_allocation_cleanup_unwind(|| 42), Ok(42));
        assert_eq!(
            catch_allocation_cleanup_unwind(|| Err::<(), _>("failure")),
            Ok(Err("failure"))
        );
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn allocation_cleanup_contains_payload_destructor_unwind() {
        struct Payload;
        impl Drop for Payload {
            fn drop(&mut self) {
                panic!("payload destructor");
            }
        }
        assert_eq!(
            catch_allocation_cleanup_unwind::<()>(|| std::panic::panic_any(Payload)),
            Err(AllocationCleanupPanicked)
        );
    }
}
