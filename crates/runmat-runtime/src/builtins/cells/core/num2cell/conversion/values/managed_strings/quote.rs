use runmat_value::{ExactArrayStorage, OutputListStorage, SharedShape, StringArray, StringElement};

use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

use super::{admission, numeric};
use crate::builtins::cells::core::num2cell::plan::PartitionGeometry;

fn add(
    runtime: &RuntimeContext,
    physical: &mut u64,
    claims: &mut usize,
    bytes: usize,
    owner_count: usize,
) -> BuiltinResult<()> {
    *physical = physical
        .checked_add(
            u64::try_from(bytes).map_err(|_| admission(runtime, HostAllocationError::Overflow))?,
        )
        .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
    *claims = claims
        .checked_add(owner_count)
        .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
    Ok(())
}

pub(super) fn retained_bytes(
    runtime: &RuntimeContext,
    input: &StringArray,
    plan: &impl PartitionGeometry,
    unique: bool,
) -> BuiltinResult<u64> {
    let output_shape: &[usize] = if plan.output_shape().is_empty() && plan.output_count() == 1 {
        &[1, 1]
    } else {
        plan.output_shape()
    };
    let mut physical = 0_u64;
    let mut claims = 0_usize;
    let list = OutputListStorage::requested_bytes(plan.output_count())
        .map_err(|cause| numeric(runtime, cause))?;
    add(
        runtime,
        &mut physical,
        &mut claims,
        list,
        usize::from(plan.output_count() != 0),
    )?;
    let shape = SharedShape::requested_bytes_for_rank(output_shape.len())
        .map_err(|cause| numeric(runtime, cause))?;
    add(runtime, &mut physical, &mut claims, shape, 1)?;
    if plan.slice_count() != 1 {
        let group_list = ExactArrayStorage::<StringElement>::requested_bytes(plan.slice_count())
            .map_err(|cause| numeric(runtime, cause))?;
        let group_shape = SharedShape::requested_bytes_for_rank(plan.slice_shape().len())
            .map_err(|cause| numeric(runtime, cause))?;
        let groups = plan.output_count();
        let group_physical = u64::try_from(group_list)
            .ok()
            .and_then(|bytes| bytes.checked_add(u64::try_from(group_shape).ok()?))
            .and_then(|bytes| bytes.checked_mul(u64::try_from(groups).ok()?))
            .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
        physical = physical
            .checked_add(group_physical)
            .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
        let per_group_claims = usize::from(plan.slice_count() != 0) + 1;
        claims = claims
            .checked_add(
                per_group_claims
                    .checked_mul(groups)
                    .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?,
            )
            .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
    }
    for text in input.data().iter() {
        if text.has_retained_claim() || text.capacity() == 0 {
            continue;
        }
        let capacity = if unique {
            text.capacity()
        } else {
            text.text_ref().map_or(0, |text| text.len())
        };
        let bytes = StringElement::requested_bytes_for_capacity(capacity)
            .map_err(|cause| numeric(runtime, cause))?;
        add(
            runtime,
            &mut physical,
            &mut claims,
            bytes,
            usize::from(capacity != 0),
        )?;
    }
    HostAllocationPlan::retained_exact_output_bytes(physical, claims)
        .map_err(|cause| admission(runtime, cause))
}
