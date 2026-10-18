//! One admitted executor write, composed from neutral workspace owners.
//!
//! Collection growth and GC-root admission remain separate publication
//! stages; this borrowed alias path never certifies an unsupported Value.

use runmat_value::{ExactHostAllocationPermit, ExactHostNumericAllocationError, Value};

use crate::workspace::retained::{
    is_allocation_free_workspace_alias, WorkspaceName, WorkspaceRetainedValue,
};
use crate::workspace::WorkspaceSlotReadLease;

/// An admitted name/value alias. This is deliberately not a collection-slot
/// permit and does not authorize insertion into projection/delta/origin maps.
#[derive(Clone, Debug)]
pub(crate) struct PreparedWorkspaceWrite {
    slot: usize,
    name: WorkspaceName,
    value: WorkspaceRetainedValue,
}

impl PreparedWorkspaceWrite {
    pub(crate) const fn retained_claim_count() -> usize {
        2
    }

    /// A retained source already owns its Value header and roots. This write
    /// allocates only a new admitted name; no second Value owner is minted.
    pub(crate) fn retained_generation_bytes(
        name: &str,
        source: &WorkspaceSlotReadLease,
    ) -> Result<Option<u64>, crate::context::host_allocation::HostAllocationError> {
        if !source.shared_generation().has_retained_claim() {
            return Ok(None);
        }
        let raw = WorkspaceName::requested_bytes(name)?;
        crate::context::host_allocation::HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw)
                .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?,
            1,
        )
        .map(Some)
    }

    pub(crate) fn prepare_retained(
        slot: usize,
        name: &str,
        source: WorkspaceSlotReadLease,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Option<Self>, ExactHostNumericAllocationError> {
        let generation = source.shared_generation();
        if !generation.has_retained_claim() {
            return Ok(None);
        }
        let roots = source.shared_roots();
        let name = WorkspaceName::prepare(name, permit)?;
        Ok(Some(Self {
            slot,
            name,
            value: WorkspaceRetainedValue::from_shared_generation(generation, roots),
        }))
    }

    /// Exact backing bytes excluding the two permit claim controls.
    pub(crate) fn requested_bytes(
        name: &str,
        value: &Value,
    ) -> Result<Option<usize>, ExactHostNumericAllocationError> {
        if !is_allocation_free_workspace_alias(value) {
            return Ok(None);
        }
        WorkspaceName::requested_bytes(name)?
            .checked_add(WorkspaceRetainedValue::requested_bytes())
            .map(Some)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub(super) fn retained_bytes(
        name: &str,
        value: &Value,
    ) -> Result<Option<u64>, crate::context::host_allocation::HostAllocationError> {
        let Some(raw) = Self::requested_bytes(name, value)? else {
            return Ok(None);
        };
        crate::context::host_allocation::HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw)
                .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?,
            Self::retained_claim_count(),
        )
        .map(Some)
    }

    pub(crate) fn prepare(
        slot: usize,
        name: &str,
        value: &Value,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Option<Self>, ExactHostNumericAllocationError> {
        if !is_allocation_free_workspace_alias(value) {
            return Ok(None);
        }
        let name = WorkspaceName::prepare(name, permit)?;
        let value = WorkspaceRetainedValue::prepare(value, permit)?
            .expect("value eligibility was checked before admission");
        Ok(Some(Self { slot, name, value }))
    }

    pub(crate) fn slot(&self) -> usize {
        self.slot
    }

    pub(crate) fn name(&self) -> &WorkspaceName {
        &self.name
    }

    pub(crate) fn value(&self) -> &WorkspaceRetainedValue {
        &self.value
    }

    pub(crate) fn into_parts(self) -> (usize, WorkspaceName, WorkspaceRetainedValue) {
        (self.slot, self.name, self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{ExactHostAllocationReservation, Tensor};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct Claim {
        live: Arc<AtomicUsize>,
        bytes: usize,
    }

    impl Drop for Claim {
        fn drop(&mut self) {
            self.live.fetch_sub(self.bytes, Ordering::SeqCst);
        }
    }

    impl ExactHostAllocationReservation for Claim {}

    #[derive(Debug)]
    struct Permit {
        live: Arc<AtomicUsize>,
        claims: Arc<AtomicUsize>,
    }

    impl ExactHostAllocationPermit for Permit {
        fn reserve_exact(
            &self,
            bytes: usize,
        ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError>
        {
            self.live.fetch_add(bytes, Ordering::SeqCst);
            self.claims.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(Claim {
                live: self.live.clone(),
                bytes,
            }))
        }
    }

    #[test]
    fn array_write_prepares_two_independent_shared_owners() {
        let gpu_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let gpu_allocation = gpu_owner
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("fixture allocation");
        let values = [
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::SparseTensor(
                runmat_value::SparseTensor::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 2.0])
                    .unwrap(),
            ),
            Value::GpuTensor(gpu_allocation.handle.clone()),
        ];
        for value in values {
            let live = Arc::new(AtomicUsize::new(0));
            let claims = Arc::new(AtomicUsize::new(0));
            let permit = Permit {
                live: live.clone(),
                claims: claims.clone(),
            };
            let expected = PreparedWorkspaceWrite::requested_bytes("state", &value)
                .unwrap()
                .expect("supported value");
            let prepared = PreparedWorkspaceWrite::prepare(7, "state", &value, &permit)
                .unwrap()
                .expect("prepared write");
            assert_eq!(claims.load(Ordering::SeqCst), 2);
            assert_eq!(live.load(Ordering::SeqCst), expected);
            assert_eq!(prepared.slot(), 7);
            assert_eq!(prepared.name().as_str(), "state");
            assert_eq!(prepared.value().value(), &value);
            match (&value, prepared.value().value()) {
                (Value::Tensor(source), Value::Tensor(published)) => {
                    assert_eq!(
                        source.as_f64_slice().unwrap().as_ptr(),
                        published.as_f64_slice().unwrap().as_ptr()
                    );
                }
                (Value::SparseTensor(source), Value::SparseTensor(published)) => {
                    assert!(source.col_ptrs.shares_allocation_with(&published.col_ptrs));
                    assert!(source
                        .row_indices
                        .shares_allocation_with(&published.row_indices));
                    assert!(source
                        .numeric_host_buffer()
                        .unwrap()
                        .shares_allocation_with(published.numeric_host_buffer().unwrap()));
                }
                (Value::GpuTensor(source), Value::GpuTensor(published)) => {
                    assert_eq!(source.view_identity(), published.view_identity());
                }
                _ => {}
            }
            let alias = prepared.clone();
            assert!(prepared.name.shares_owner_with(&alias.name));
            assert!(prepared.value.shares_owner_with(&alias.value));
            assert_eq!(claims.load(Ordering::SeqCst), 2);
            drop(prepared);
            assert_eq!(live.load(Ordering::SeqCst), expected);
            drop(alias);
            assert_eq!(live.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn unsupported_aggregate_declines_before_claiming_or_cloning() {
        let live = Arc::new(AtomicUsize::new(0));
        let claims = Arc::new(AtomicUsize::new(0));
        let permit = Permit {
            live: live.clone(),
            claims: claims.clone(),
        };
        let value = Value::String("requires an owned clone".into());
        assert_eq!(
            PreparedWorkspaceWrite::requested_bytes("text", &value).unwrap(),
            None
        );
        assert!(PreparedWorkspaceWrite::prepare(0, "text", &value, &permit)
            .unwrap()
            .is_none());
        assert_eq!(claims.load(Ordering::SeqCst), 0);
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }
}
