//! Call-owned upload references; returned aliases retain their own leases.
use super::*;
use runmat_accelerate_api::allocation::GpuHandleReleaseError;
use runmat_accelerate_api::GpuTensorHandle;

#[derive(Debug)]
pub(super) struct UploadCleanupFailure {
    failures: Vec<GpuHandleReleaseError>,
}
impl UploadCleanupFailure {
    pub(super) fn report(&self, phase: &'static str) {
        // Fixed schema and bounded scalar fields, not arbitrary backend text.
        log::warn!(target: "runmat::placement::cleanup",
            "event=prepared_operand_retirement_failed phase={} failed_allocations={}",
            phase, self.failures.len());
    }
}
impl std::fmt::Display for UploadCleanupFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} prepared operand retirement failure(s)",
            self.failures.len()
        )
    }
}
impl std::error::Error for UploadCleanupFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.failures.first().map(|error| error as _)
    }
}

pub(super) struct OwnedUploads {
    borrowed_handles: Vec<GpuTensorHandle>,
    owned: Vec<GpuTensorHandle>,
}

impl OwnedUploads {
    pub(super) fn new(args: &[Value]) -> PreparationResult<Self> {
        let mut borrowed_handles: Vec<GpuTensorHandle> = Vec::new();
        for value in args {
            runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
                if !borrowed_handles
                    .iter()
                    .any(|existing| existing.shares_allocation_with(handle))
                {
                    borrowed_handles.push(handle.clone());
                }
            })?;
        }
        Ok(Self {
            borrowed_handles,
            owned: Vec::new(),
        })
    }

    pub(super) fn protected(&self) -> Vec<&GpuTensorHandle> {
        self.borrowed_handles.iter().chain(&self.owned).collect()
    }

    pub(super) fn has_uploads(&self) -> bool {
        !self.owned.is_empty()
    }

    pub(super) fn track(&mut self, handle: &GpuTensorHandle) {
        if !self
            .borrowed_handles
            .iter()
            .any(|borrowed| borrowed.shares_allocation_with(handle))
            && !self
                .owned
                .iter()
                .any(|owned| owned.shares_allocation_with(handle))
        {
            self.owned.push(handle.clone());
        }
    }

    pub(super) fn transfer_all(mut self) {
        self.owned.clear();
    }

    /// Release this call's references. Returned aliases retain their own lease;
    /// only the last reference attempts physical retirement, exactly once.
    pub(super) fn release(&mut self) -> std::result::Result<(), UploadCleanupFailure> {
        let mut failures = Vec::new();
        for handle in self.owned.drain(..) {
            if let Err(error) = handle.release() {
                failures.push(error);
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(UploadCleanupFailure { failures })
        }
    }
}

impl Drop for OwnedUploads {
    fn drop(&mut self) {
        // Invocation observation is already unsuccessful on abandonment. Never
        // mutate an ambient recorder here: it may belong to another invocation.
        if let Err(error) = self.release() {
            error.report("abandonment");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };

    #[test]
    fn explicit_release_preserves_output_alias_and_retires_unused_upload() {
        let owner = TestAllocationOwner::new();
        for retain_output in [false, true] {
            let allocation = owner
                .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let value = Value::GpuTensor(allocation.handle);
            let output = retain_output.then(|| value.clone());
            let mut uploads = OwnedUploads::new(&[]).unwrap();
            assert!(!uploads.has_uploads());
            let Value::GpuTensor(handle) = &value else {
                unreachable!()
            };
            uploads.track(handle);
            uploads.track(handle);
            assert!(uploads.has_uploads());
            drop(value);
            uploads.release().unwrap();
            assert!(!uploads.has_uploads());
            assert_eq!(allocation.probe.retirements(), usize::from(!retain_output));
            drop(uploads);
            drop(output);
            assert_eq!(allocation.probe.retirements(), 1);
        }
    }

    #[test]
    fn borrowed_view_is_not_tracked_as_a_fresh_upload() {
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(vec![2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let other_view = allocation
            .handle
            .with_view(runmat_accelerate_api::GpuTensorView::contiguous(vec![1, 2]))
            .unwrap();
        let mut uploads =
            OwnedUploads::new(&[Value::GpuTensor(allocation.handle.clone())]).unwrap();
        uploads.track(&other_view);
        assert!(!uploads.has_uploads());
        drop(other_view);
        assert!(uploads.owned.is_empty());
        uploads.release().unwrap();
        drop(uploads);
        assert_eq!(allocation.probe.retirements(), 0);
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
