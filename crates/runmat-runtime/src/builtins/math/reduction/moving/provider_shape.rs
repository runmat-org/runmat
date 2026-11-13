//! One exactly claimed, non-growing request-shape owner through provider await.
use super::admission::MovingCall;
use super::*;
use crate::context::host_allocation::{
    HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};

enum ShapeStorage {
    Managed(HostTypedWorkspace<usize>),
    Ordinary(Vec<usize>),
}
pub(super) struct ProviderShape {
    storage: ShapeStorage,
    receipt: Option<HostAllocationReceipt>,
}
impl ProviderShape {
    pub(super) fn new(
        work: &MovingCall<'_>,
        name: &'static str,
        input: &[usize],
        dim: usize,
        parsed: &ParsedMoving,
    ) -> BuiltinResult<Self> {
        work.check()?;
        let rank = input.len().max(dim);
        let scratch =
            HostTypedWorkspace::<usize>::quoted_bytes(rank).map_err(|cause| work.error(cause))?;
        let plan = HostAllocationPlan::new(scratch, 0).map_err(|cause| work.error(cause))?;
        let receipt =
            HostAllocationReceipt::admit(work.runtime, plan).map_err(|cause| work.error(cause))?;
        work.check()?;
        let storage = match (&receipt, work.runtime) {
            (Some(receipt), Some(runtime)) => ShapeStorage::Managed(
                receipt
                    .temporary_typed(runtime, rank)
                    .map_err(|cause| work.error(cause))?,
            ),
            (None, _) => ShapeStorage::Ordinary(vec![0; rank]),
            _ => {
                return Err(work.error(
                    crate::context::host_allocation::HostAllocationError::MissingManagedOwner,
                ))
            }
        };
        let mut owner = Self { storage, receipt };
        let shape = owner.as_mut_slice();
        for (index, output) in shape.iter_mut().enumerate() {
            *output = input.get(index).copied().unwrap_or(1);
        }
        if matches!(parsed.endpoints, Endpoints::Discard) {
            let (before, after) = parsed.window.count_radius().ok_or_else(|| {
                invalid_argument(
                    name,
                    "Endpoints 'discard' is not supported with SamplePoints",
                )
            })?;
            let trim = before
                .checked_add(after)
                .ok_or_else(|| invalid_argument(name, "window length overflow"))?;
            shape[dim - 1] = shape[dim - 1].saturating_sub(trim);
        }
        owner.check(work)?;
        Ok(owner)
    }
    fn as_mut_slice(&mut self) -> &mut [usize] {
        match &mut self.storage {
            ShapeStorage::Managed(values) => values.as_mut_slice(),
            ShapeStorage::Ordinary(values) => values,
        }
    }
    pub(super) fn as_slice(&self) -> &[usize] {
        match &self.storage {
            ShapeStorage::Managed(values) => values.as_slice(),
            ShapeStorage::Ordinary(values) => values,
        }
    }
    pub(super) fn check(&self, work: &MovingCall<'_>) -> BuiltinResult<()> {
        work.check()?;
        if let (Some(receipt), Some(runtime)) = (&self.receipt, work.runtime) {
            receipt
                .checkpoint(runtime)
                .map_err(|cause| work.error(cause))?;
        }
        Ok(())
    }
}
