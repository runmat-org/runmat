//! Exact retained shape ownership for finite host metadata-only shifts.

use super::{shiftdim_error, SHIFTDIM_ERROR_INVALID_SHIFT};
use crate::context::host_allocation::{
    AdmittedCollectionAllocator, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use allocator_api2::vec::Vec as AdmittedVec;
use runmat_value::{SharedShape, Value};
use std::alloc::Layout;

pub(super) struct NegativeShiftShape {
    dimensions: AdmittedVec<usize, AdmittedCollectionAllocator>,
    receipt: HostAllocationReceipt,
}

fn admission_error(context: &RuntimeContext, cause: HostAllocationError) -> crate::RuntimeError {
    crate::build_runtime_error(format!("shiftdim: shape admission: {cause}"))
        .with_builtin("shiftdim")
        .with_identifier(format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

impl NegativeShiftShape {
    pub(super) fn prepare(
        context: &RuntimeContext,
        source_shape: &[usize],
        shift: isize,
    ) -> BuiltinResult<Option<Self>> {
        if !context.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        }) {
            return Ok(None);
        }
        let count = shift
            .checked_abs()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| {
                shiftdim_error(
                    &SHIFTDIM_ERROR_INVALID_SHIFT,
                    "shiftdim: negative n is too large to represent",
                )
            })?;
        let effective_rank = match source_shape {
            [] | [_] => 2,
            _ => {
                let mut end = source_shape.len();
                while end > 2 && source_shape[end - 1] == 1 {
                    end -= 1;
                }
                end
            }
        };
        let rank = effective_rank.checked_add(count).ok_or_else(|| {
            shiftdim_error(
                &SHIFTDIM_ERROR_INVALID_SHIFT,
                "shiftdim: shifted rank exceeds platform limits",
            )
        })?;
        let layout = Layout::array::<usize>(rank).map_err(|_| {
            shiftdim_error(
                &SHIFTDIM_ERROR_INVALID_SHIFT,
                "shiftdim: shifted rank exceeds platform limits",
            )
        })?;
        let dimensions = AdmittedCollectionAllocator::requested_bytes(layout).map_err(|_| {
            shiftdim_error(
                &SHIFTDIM_ERROR_INVALID_SHIFT,
                "shiftdim: shifted rank exceeds platform limits",
            )
        })?;
        let shape = SharedShape::requested_bytes_for_rank(rank).map_err(|cause| {
            shiftdim_error(
                &SHIFTDIM_ERROR_INVALID_SHIFT,
                format!("shiftdim: shifted rank: {cause}"),
            )
        })?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(dimensions)
                .ok()
                .and_then(|bytes| bytes.checked_add(u64::try_from(shape).ok()?))
                .ok_or_else(|| {
                    shiftdim_error(
                        &SHIFTDIM_ERROR_INVALID_SHIFT,
                        "shiftdim: shifted rank exceeds platform limits",
                    )
                })?,
            AdmittedCollectionAllocator::retained_claim_count() + 1,
        )
        .map_err(|cause| admission_error(context, cause))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| admission_error(context, cause))?;
        let Some(receipt) = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|cause| admission_error(context, cause))?
        else {
            return Ok(None);
        };
        let allocator = AdmittedCollectionAllocator::new(
            receipt
                .output_permit(context)
                .map_err(|cause| admission_error(context, cause))?,
        );
        let mut dimensions = AdmittedVec::new_in(allocator);
        dimensions.try_reserve_exact(rank).map_err(|cause| {
            shiftdim_error(
                &SHIFTDIM_ERROR_INVALID_SHIFT,
                format!("shiftdim: shifted rank storage: {cause}"),
            )
        })?;
        dimensions.resize(count, 1);
        match source_shape {
            [] => {
                dimensions.push(1);
                dimensions.push(1);
            }
            [length] => {
                dimensions.push(1);
                dimensions.push(*length);
            }
            _ => {
                for &dimension in &source_shape[..effective_rank] {
                    dimensions.push(dimension);
                }
            }
        }
        receipt
            .checkpoint(context)
            .map_err(|cause| admission_error(context, cause))?;
        Ok(Some(Self {
            dimensions,
            receipt,
        }))
    }

    pub(super) fn publish(self, value: Value, context: &RuntimeContext) -> BuiltinResult<Value> {
        let shape = self
            .receipt
            .with_output_allocation(context, |permit| {
                SharedShape::new_admitted_from_slice(&self.dimensions, permit)
            })
            .map_err(|cause| admission_error(context, cause))?;
        match value {
            Value::Tensor(mut tensor) => {
                tensor
                    .reshape_with_shared_shape(shape)
                    .map_err(|cause| shiftdim_error(&SHIFTDIM_ERROR_INVALID_SHIFT, cause))?;
                Ok(Value::Tensor(tensor))
            }
            Value::ComplexTensor(mut tensor) => {
                tensor
                    .reshape_with_shared_shape(shape)
                    .map_err(|cause| shiftdim_error(&SHIFTDIM_ERROR_INVALID_SHIFT, cause))?;
                Ok(Value::ComplexTensor(tensor))
            }
            Value::LogicalArray(mut logical) => {
                logical
                    .reshape_with_shared_shape(shape)
                    .map_err(|cause| shiftdim_error(&SHIFTDIM_ERROR_INVALID_SHIFT, cause))?;
                Ok(Value::LogicalArray(logical))
            }
            _ => unreachable!("admitted shiftdim applies only to shared host carriers"),
        }
    }
}
