//! Move a settled provider-owned native Vec into one exact retained tensor.
use super::{admission_error, invalid, CpuReadbackFailure};
use runmat_accelerate_api::{HostNumericDataOwned, HostNumericTensorOwned};
use runmat_runtime::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use runmat_runtime::context::RuntimeContext;
use runmat_value::{ExactHostTensorAllocation, NumericDType, Value};
use std::mem::size_of;

pub(super) fn adopt_settled_download(
    host: HostNumericTensorOwned,
    source: &super::super::Source,
    runtime: &RuntimeContext,
    receipt: &HostAllocationReceipt,
) -> Result<Value, Box<CpuReadbackFailure>> {
    let payload_capacity_bytes = match &host.data {
        HostNumericDataOwned::F32(values) => values.capacity().checked_mul(size_of::<f32>()),
        HostNumericDataOwned::F64(values) => values.capacity().checked_mul(size_of::<f64>()),
        _ => {
            return Err(Box::new(CpuReadbackFailure::AfterEffect(invalid(
                "selected host provider returned a non-floating payload",
            ))))
        }
    }
    .and_then(|bytes| u64::try_from(bytes).ok())
    .ok_or_else(|| {
        Box::new(CpuReadbackFailure::AfterEffect(invalid(
            "selected host capacity overflows",
        )))
    })?;
    let shape_capacity_bytes = host
        .shape
        .capacity()
        .checked_mul(size_of::<usize>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            Box::new(CpuReadbackFailure::AfterEffect(invalid(
                "selected host shape overflows",
            )))
        })?;
    if payload_capacity_bytes
        .checked_add(shape_capacity_bytes)
        .is_none_or(|bytes| bytes > source.host_workspace)
    {
        return Err(Box::new(CpuReadbackFailure::AfterEffect(invalid(
            "selected provider exceeded its host workspace bound",
        ))));
    }
    let (dtype, capacity) = match &host.data {
        HostNumericDataOwned::F32(values) => (NumericDType::F32, values.capacity()),
        HostNumericDataOwned::F64(values) => (NumericDType::F64, values.capacity()),
        _ => unreachable!("checked native floating payload"),
    };
    let raw =
        ExactHostTensorAllocation::adopted_requested_bytes(dtype, capacity, host.shape.capacity())
            .ok()
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                Box::new(CpuReadbackFailure::AfterEffect(invalid(
                    "selected host adoption overflows",
                )))
            })?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        raw,
        ExactHostTensorAllocation::retained_claim_count(),
    )
    .map_err(|cause| Box::new(admission_error(runtime, cause, true)))?;
    // Successful staged return settles both the callback and awaiting stage
    // leases. Reassign only this exact returned backing, never provider scratch
    // still reachable from a callback.
    receipt
        .transfer_settled_scratch_to_output(runtime, retained)
        .map_err(|cause| Box::new(admission_error(runtime, cause, true)))?;
    let tensor = match host.data {
        HostNumericDataOwned::F32(values) => receipt.with_output_allocation(runtime, |permit| {
            ExactHostTensorAllocation::adopt_f32_vec(values, host.shape, permit)
        }),
        HostNumericDataOwned::F64(values) => receipt.with_output_allocation(runtime, |permit| {
            ExactHostTensorAllocation::adopt_f64_vec(values, host.shape, permit)
        }),
        _ => unreachable!("checked native floating payload"),
    }
    .map_err(|cause| Box::new(admission_error(runtime, cause, true)))?;
    Ok(Value::Tensor(tensor))
}
