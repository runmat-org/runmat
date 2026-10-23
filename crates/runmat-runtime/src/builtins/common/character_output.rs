//! One exact retained character output for selected finite Host shape work.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{
    CharArray, ExactArrayInitializationError, ExactArrayStorage, SharedShape, Utf16CodeUnit,
};

pub(crate) const CHECKPOINT_CHUNK: usize = 4096;

pub(crate) struct CharacterBuildErrors {
    pub admission: fn(&RuntimeContext, HostAllocationError) -> RuntimeError,
    pub invalid: fn(String) -> RuntimeError,
}

pub(crate) fn build(
    context: &RuntimeContext,
    rank: usize,
    count: usize,
    extra_scratch_bytes: u64,
    mut shape_at: impl FnMut(usize) -> BuiltinResult<usize>,
    fill: impl FnOnce(&mut [Utf16CodeUnit], &[usize], &HostAllocationReceipt) -> BuiltinResult<()>,
    errors: CharacterBuildErrors,
) -> BuiltinResult<CharArray> {
    let error = errors.admission;
    let invalid = errors.invalid;
    let payload_bytes = ExactArrayStorage::<Utf16CodeUnit>::requested_bytes(count)
        .map_err(|cause| error(context, HostAllocationError::Numeric(cause)))?;
    let shape_bytes = SharedShape::requested_bytes_for_rank(rank)
        .map_err(|cause| error(context, HostAllocationError::Numeric(cause)))?;
    let physical_bytes = payload_bytes
        .checked_add(shape_bytes)
        .ok_or_else(|| error(context, HostAllocationError::Overflow))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical_bytes).map_err(|_| error(context, HostAllocationError::Overflow))?,
        usize::from(count != 0) + 1,
    )
    .map_err(|cause| error(context, cause))?;
    let shape_scratch =
        HostTypedWorkspace::<usize>::quoted_bytes(rank).map_err(|cause| error(context, cause))?;
    let scratch_bytes = shape_scratch
        .checked_add(extra_scratch_bytes)
        .ok_or_else(|| error(context, HostAllocationError::Overflow))?;
    let plan =
        HostAllocationPlan::new(scratch_bytes, retained).map_err(|cause| error(context, cause))?;
    let receipt = HostAllocationReceipt::admit(Some(context), plan)
        .map_err(|cause| error(context, cause))?
        .ok_or_else(|| error(context, HostAllocationError::MissingManagedOwner))?;
    let mut shape = receipt
        .temporary_typed::<usize>(context, rank)
        .map_err(|cause| error(context, cause))?;
    for axis in 0..rank {
        if axis.is_multiple_of(CHECKPOINT_CHUNK) {
            receipt
                .checkpoint(context)
                .map_err(|cause| error(context, cause))?;
        }
        shape.as_mut_slice()[axis] = shape_at(axis)?;
    }
    let shape = receipt
        .with_output_allocation(context, |permit| {
            SharedShape::new_admitted_from_slice(shape.as_slice(), permit)
        })
        .map_err(|cause| error(context, cause))?;
    let mut data = receipt
        .with_output_construction(context, |permit| {
            ExactArrayStorage::try_initialize(count, permit, |index| {
                if index.is_multiple_of(CHECKPOINT_CHUNK) {
                    receipt.checkpoint(context)?;
                }
                Ok::<Utf16CodeUnit, HostAllocationError>(Utf16CodeUnit::from_raw(b' ' as u16))
            })
        })
        .map_err(|cause| error(context, cause))?
        .map_err(|cause| match cause {
            ExactArrayInitializationError::Allocation(cause) => {
                error(context, HostAllocationError::Numeric(cause))
            }
            ExactArrayInitializationError::Element(cause) => error(context, cause),
        })?;
    let slice = data
        .try_as_mut_slice_unique()
        .ok_or_else(|| invalid("Character output became aliased before assembly".into()))?;
    fill(slice, &shape, &receipt)?;
    receipt
        .checkpoint(context)
        .map_err(|cause| error(context, cause))?;
    CharArray::from_admitted(data, shape).map_err(invalid)
}
