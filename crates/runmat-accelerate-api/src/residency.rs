//! External view residency, independent of physical allocation retirement.
use crate::allocation::catch_allocation_cleanup_unwind;
use crate::{
    mark_residency, GpuHandleIdentity, GpuTensorHandle, ResidencyClearFn, RESIDENCY_CLEAR,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyCleanupError {
    Unavailable,
    CallbackPanicked,
}
impl std::fmt::Display for ResidencyCleanupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPU view residency cleanup failed: {self:?}")
    }
}
impl std::error::Error for ResidencyCleanupError {}

pub fn clear_residency_checked(handle: &GpuTensorHandle) -> Result<(), ResidencyCleanupError> {
    clear_identity(handle.view_identity(), RESIDENCY_CLEAR.get().copied())
}
fn clear_identity(
    identity: GpuHandleIdentity,
    callback: Option<ResidencyClearFn>,
) -> Result<(), ResidencyCleanupError> {
    catch_allocation_cleanup_unwind(|| callback.map_or(Ok(()), |callback| callback(identity)))
        .map_err(|_| ResidencyCleanupError::CallbackPanicked)?
}

/// Field-owned terminal cleanup, declared before the physical lease so final
/// view teardown always precedes allocation retirement. No handle is forged.
pub(crate) struct ViewCleanup {
    identity: GpuHandleIdentity,
    attempted: bool,
}
impl ViewCleanup {
    pub(crate) fn new(identity: GpuHandleIdentity) -> Self {
        Self {
            identity,
            attempted: false,
        }
    }
    pub(crate) fn finish(&mut self) -> Result<(), ResidencyCleanupError> {
        self.finish_with(RESIDENCY_CLEAR.get().copied())
    }
    fn finish_with(
        &mut self,
        callback: Option<ResidencyClearFn>,
    ) -> Result<(), ResidencyCleanupError> {
        if self.attempted {
            return Ok(());
        }
        self.attempted = true;
        clear_identity(self.identity, callback)
    }
}
impl Drop for ViewCleanup {
    fn drop(&mut self) {
        if let Err(error) = self.finish() {
            let _ = catch_allocation_cleanup_unwind(
                || tracing::warn!(?error, view = ?self.identity, "GPU view cleanup failed; not retried"),
            );
        }
    }
}

/// Reassert borrowed residency only; immutable tensor facts need no restore.
pub struct ResidencyRestoreGuard<'a> {
    handle: &'a GpuTensorHandle,
}
impl<'a> ResidencyRestoreGuard<'a> {
    pub fn new(handle: &'a GpuTensorHandle) -> Self {
        Self { handle }
    }
    pub fn restore(&self) {
        mark_residency(self.handle);
    }
}
impl Drop for ResidencyRestoreGuard<'_> {
    fn drop(&mut self) {
        let _ = catch_allocation_cleanup_unwind(|| self.restore());
    }
}

/// Restore every borrowed operand without allocating one guard per handle.
/// Each restoration retains the ordinary guard's independent unwind barrier.
pub(crate) struct ResidencyRestoreSliceGuard<'a> {
    handles: &'a [&'a GpuTensorHandle],
}
impl<'a> ResidencyRestoreSliceGuard<'a> {
    pub(crate) fn new(handles: &'a [&'a GpuTensorHandle]) -> Self {
        Self { handles }
    }
}
impl Drop for ResidencyRestoreSliceGuard<'_> {
    fn drop(&mut self) {
        for handle in self.handles {
            drop(ResidencyRestoreGuard::new(handle));
        }
    }
}

#[cfg(test)]
mod tests;
