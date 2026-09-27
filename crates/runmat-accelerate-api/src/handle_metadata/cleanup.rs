//! Checked, separately staged cleanup for allocation owners. These operations
//! never change the handle-owned physical descriptor or assert device release.

use crate::allocation::{catch_allocation_cleanup_unwind, AllocationCleanupPanicked};
use crate::*;

/// The annotation lock that prevented cleanup. No annotations are removed when
/// acquiring any one of these locks fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleAnnotationLock {
    Logical,
    LogicalHits,
    Class,
    Transpose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandleAnnotationLockPoisoned(pub HandleAnnotationLock);

impl std::fmt::Display for HandleAnnotationLockPoisoned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPU handle {:?} annotation lock poisoned", self.0)
    }
}

impl std::error::Error for HandleAnnotationLockPoisoned {}

/// Invoke the existing residency callback without holding annotation locks.
/// An unwind is uncertain callback completion, not a retry authorization. An
/// allocation owner must record this phase before invocation, independently of
/// subsequent annotation cleanup, so later failure cannot repeat the callback.
pub fn clear_residency_checked(handle: &GpuTensorHandle) -> Result<(), AllocationCleanupPanicked> {
    clear_residency_with(handle, RESIDENCY_CLEAR.get().copied())
}

fn clear_residency_with(
    handle: &GpuTensorHandle,
    callback: Option<ResidencyClearFn>,
) -> Result<(), AllocationCleanupPanicked> {
    catch_allocation_cleanup_unwind(|| {
        if let Some(callback) = callback {
            callback(handle);
        }
    })
}

/// Remove mutable annotations only after acquiring every required lock. This
/// does not invoke residency callbacks and does not erase physical metadata.
/// Keeping callback cleanup separate permits an owner to persist each phase.
pub fn clear_handle_annotations_checked(
    handle: &GpuTensorHandle,
) -> Result<(), HandleAnnotationLockPoisoned> {
    AnnotationLocks {
        logical: &LOGICAL_HANDLES,
        hits: &LOGICAL_HANDLE_HITS,
        classes: &HANDLE_CLASS_IDENTITIES,
        transpose: &TRANSPOSED_HANDLES,
    }
    .clear(handle_identity(handle))
}

struct AnnotationLocks<'a> {
    logical: &'a RwLock<HashSet<GpuHandleIdentity>>,
    hits: &'a RwLock<HashMap<GpuHandleIdentity, u64>>,
    classes: &'a RwLock<HashMap<GpuHandleIdentity, ClassIdentity>>,
    transpose: &'a RwLock<HashMap<GpuHandleIdentity, TransposeInfo>>,
}

impl AnnotationLocks<'_> {
    fn clear(&self, identity: GpuHandleIdentity) -> Result<(), HandleAnnotationLockPoisoned> {
        // Existing nested acquisition is logical -> hits. Other annotation
        // users acquire a single lock at a time; retain that ordering here.
        let detached_class = {
            let mut logical = self
                .logical
                .write()
                .map_err(|_| HandleAnnotationLockPoisoned(HandleAnnotationLock::Logical))?;
            let mut hits = self
                .hits
                .write()
                .map_err(|_| HandleAnnotationLockPoisoned(HandleAnnotationLock::LogicalHits))?;
            let mut classes = self
                .classes
                .write()
                .map_err(|_| HandleAnnotationLockPoisoned(HandleAnnotationLock::Class))?;
            let mut transpose = self
                .transpose
                .write()
                .map_err(|_| HandleAnnotationLockPoisoned(HandleAnnotationLock::Transpose))?;
            logical.remove(&identity);
            hits.remove(&identity);
            transpose.remove(&identity);
            classes.remove(&identity)
        };
        // Even the class annotation is destroyed outside all owner locks.
        drop(detached_class);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
