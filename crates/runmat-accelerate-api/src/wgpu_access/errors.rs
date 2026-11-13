use std::{error::Error, fmt, sync::Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuScopeErrorKind {
    Validation,
    OutOfMemory,
    Internal,
}

/// Original WGPU errors are retained, not reduced to strings or relabeled as
/// DeviceLost. A mutex protects inspection of retained backend errors across
/// the existing native provider error boundary.
/// Inspect original error/source chains with with_errors. They cannot be lent
/// through std::error::Error::source across the inspection lock's lifetime.
#[derive(Debug)]
pub struct WgpuScopeErrors {
    kinds: Vec<WgpuScopeErrorKind>,
    errors: Mutex<Vec<wgpu::Error>>,
}
impl WgpuScopeErrors {
    pub(super) fn new(errors: Vec<wgpu::Error>) -> Self {
        let kinds = errors
            .iter()
            .map(|error| match error {
                wgpu::Error::Validation { .. } => WgpuScopeErrorKind::Validation,
                wgpu::Error::OutOfMemory { .. } => WgpuScopeErrorKind::OutOfMemory,
                wgpu::Error::Internal { .. } => WgpuScopeErrorKind::Internal,
            })
            .collect();
        Self {
            kinds,
            errors: Mutex::new(errors),
        }
    }
    pub fn kinds(&self) -> &[WgpuScopeErrorKind] {
        &self.kinds
    }
    pub fn with_errors<T>(&self, inspect: impl FnOnce(&[wgpu::Error]) -> T) -> T {
        // A panicking observer cannot corrupt the immutable error payload.
        // Recover this inspection lock only; never a device/allocation lock.
        let errors = self
            .errors
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        inspect(&errors)
    }
}
impl fmt::Display for WgpuScopeErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGPU scoped emission failed: {:?}", self.kinds)
    }
}
impl Error for WgpuScopeErrors {}
