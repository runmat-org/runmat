//! Settled native host Vec ownership and exact layout-conversion exceptions.
use super::{admission_error, invalid, ExactNativeDownloadQuote};
#[path = "managed_download_conversion.rs"]
mod conversion;
use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::RuntimeError;
use conversion::{convert_integer_complex, convert_logical};
use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned};
use runmat_value::{
    ExactHostComplexTensorAllocation, ExactHostIntegerComplexTensorAllocation,
    ExactHostLogicalTensorAllocation, ExactHostTensorAllocation, NumericDType, NumericScalar,
    Value,
};
use std::mem::size_of;

/// The stage and its callback must have physically settled before this call.
/// The caller separately authenticates the source descriptor and quotes its
/// provider workspace; this boundary checks actual capacities before moving
/// either original Vec into the exact retained tensor.
pub fn adopt_settled_native_download(
    mut host: HostNumericTensorOwned,
    quote: ExactNativeDownloadQuote,
    runtime: &RuntimeContext,
    receipt: &HostAllocationReceipt,
) -> Result<Value, RuntimeError> {
    host.validate()
        .map_err(|_| invalid("selected provider returned malformed native storage"))?;
    if host.storage != quote.storage {
        return Err(invalid("selected provider changed native storage layout"));
    }
    let payload_capacity_bytes = match &host.data {
        HostNumericDataOwned::F32(values) => values.capacity().checked_mul(size_of::<f32>()),
        HostNumericDataOwned::F64(values) => values.capacity().checked_mul(size_of::<f64>()),
        HostNumericDataOwned::I8(values) => values.capacity().checked_mul(size_of::<i8>()),
        HostNumericDataOwned::I16(values) => values.capacity().checked_mul(size_of::<i16>()),
        HostNumericDataOwned::I32(values) => values.capacity().checked_mul(size_of::<i32>()),
        HostNumericDataOwned::I64(values) => values.capacity().checked_mul(size_of::<i64>()),
        HostNumericDataOwned::U8(values) => values.capacity().checked_mul(size_of::<u8>()),
        HostNumericDataOwned::U16(values) => values.capacity().checked_mul(size_of::<u16>()),
        HostNumericDataOwned::U32(values) => values.capacity().checked_mul(size_of::<u32>()),
        HostNumericDataOwned::U64(values) => values.capacity().checked_mul(size_of::<u64>()),
    }
    .and_then(|bytes| u64::try_from(bytes).ok())
    .ok_or_else(|| invalid("selected host capacity overflows"))?;
    let shape_capacity_bytes = host
        .shape
        .capacity()
        .checked_mul(size_of::<usize>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid("selected host shape overflows"))?;
    if payload_capacity_bytes
        .checked_add(shape_capacity_bytes)
        .is_none_or(|bytes| bytes > quote.host_workspace)
    {
        return Err(invalid(
            "selected provider exceeded its host workspace bound",
        ));
    }
    let (dtype, capacity) = match &host.data {
        HostNumericDataOwned::F32(values) => (NumericDType::F32, values.capacity()),
        HostNumericDataOwned::F64(values) => (NumericDType::F64, values.capacity()),
        HostNumericDataOwned::I8(values) => (NumericDType::I8, values.capacity()),
        HostNumericDataOwned::I16(values) => (NumericDType::I16, values.capacity()),
        HostNumericDataOwned::I32(values) => (NumericDType::I32, values.capacity()),
        HostNumericDataOwned::I64(values) => (NumericDType::I64, values.capacity()),
        HostNumericDataOwned::U8(values) => (NumericDType::U8, values.capacity()),
        HostNumericDataOwned::U16(values) => (NumericDType::U16, values.capacity()),
        HostNumericDataOwned::U32(values) => (NumericDType::U32, values.capacity()),
        HostNumericDataOwned::U64(values) => (NumericDType::U64, values.capacity()),
    };
    if dtype != quote.dtype {
        return Err(invalid("selected provider changed native precision"));
    }
    let complex_integer =
        quote.storage == GpuTensorStorage::ComplexInterleaved && dtype.integer_class().is_some();
    let converting = complex_integer || (quote.logical && dtype != NumericDType::U8);
    let raw = match (quote.storage, quote.logical) {
        (GpuTensorStorage::Real, true) if converting => {
            let count = host
                .shape
                .iter()
                .try_fold(1usize, |count, &extent| count.checked_mul(extent))
                .ok_or_else(|| invalid("selected logical extent overflows"))?;
            ExactHostLogicalTensorAllocation::requested_bytes_for_dimensions(
                count,
                host.shape.len(),
            )
        }
        (GpuTensorStorage::Real, true) => {
            ExactHostLogicalTensorAllocation::adopted_requested_bytes(
                capacity,
                host.shape.capacity(),
            )
        }
        (GpuTensorStorage::Real, false) => ExactHostTensorAllocation::adopted_requested_bytes(
            dtype,
            capacity,
            host.shape.capacity(),
        ),
        (GpuTensorStorage::ComplexInterleaved, false) if complex_integer => {
            let count = host
                .shape
                .iter()
                .try_fold(1usize, |count, &extent| count.checked_mul(extent))
                .ok_or_else(|| invalid("selected integer complex extent overflows"))?;
            ExactHostIntegerComplexTensorAllocation::requested_bytes_for_dimensions(
                dtype,
                count,
                host.shape.len(),
            )
        }
        (GpuTensorStorage::ComplexInterleaved, false) => {
            ExactHostComplexTensorAllocation::adopted_requested_bytes(
                dtype,
                capacity,
                host.shape.capacity(),
            )
        }
        _ => return Err(invalid("selected native layout is unsupported")),
    }
    .ok()
    .and_then(|bytes| u64::try_from(bytes).ok())
    .ok_or_else(|| invalid("selected host adoption overflows"))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        raw,
        if complex_integer {
            ExactHostIntegerComplexTensorAllocation::retained_claim_count()
        } else {
            ExactHostTensorAllocation::retained_claim_count()
        },
    )
    .map_err(|cause| admission_error(runtime, cause))?;
    if converting {
        if retained > quote.precharged_output_bytes {
            return Err(invalid(
                "selected integer complex output exceeded its precharge",
            ));
        }
        if quote.logical {
            return match host.data {
                HostNumericDataOwned::F32(values) => {
                    convert_logical(runtime, receipt, &host.shape, &values, |value| value != 0.0)
                }
                HostNumericDataOwned::F64(values) => {
                    convert_logical(runtime, receipt, &host.shape, &values, |value| value != 0.0)
                }
                _ => Err(invalid("selected logical payload changed class")),
            };
        }
        macro_rules! convert {
            ($values:expr, $variant:ident) => {
                convert_integer_complex(
                    runtime,
                    receipt,
                    dtype,
                    &host.shape,
                    &$values,
                    NumericScalar::$variant,
                )
            };
        }
        return match host.data {
            HostNumericDataOwned::I8(values) => convert!(values, I8),
            HostNumericDataOwned::I16(values) => convert!(values, I16),
            HostNumericDataOwned::I32(values) => convert!(values, I32),
            HostNumericDataOwned::I64(values) => convert!(values, I64),
            HostNumericDataOwned::U8(values) => convert!(values, U8),
            HostNumericDataOwned::U16(values) => convert!(values, U16),
            HostNumericDataOwned::U32(values) => convert!(values, U32),
            HostNumericDataOwned::U64(values) => convert!(values, U64),
            _ => Err(invalid("selected integer complex payload changed class")),
        };
    }
    if quote.logical {
        if let HostNumericDataOwned::U8(values) = &mut host.data {
            for (index, value) in values.iter_mut().enumerate() {
                if index % 4096 == 0 {
                    receipt
                        .checkpoint(runtime)
                        .map_err(|cause| admission_error(runtime, cause))?;
                }
                *value = u8::from(*value != 0);
            }
        }
    }
    receipt
        .transfer_settled_scratch_to_output(runtime, retained)
        .map_err(|cause| admission_error(runtime, cause))?;
    macro_rules! adopt_real {
        ($values:expr, $method:ident) => {
            receipt
                .with_output_allocation(runtime, |permit| {
                    ExactHostTensorAllocation::$method($values, host.shape, permit)
                })
                .map(Value::Tensor)
        };
    }
    let value = match (quote.storage, quote.logical, host.data) {
        (GpuTensorStorage::Real, true, HostNumericDataOwned::U8(values)) => receipt
            .with_output_allocation(runtime, |permit| {
                ExactHostLogicalTensorAllocation::adopt_u8_vec(values, host.shape, permit)
            })
            .map(Value::LogicalArray),
        (GpuTensorStorage::Real, false, HostNumericDataOwned::F32(values)) => {
            adopt_real!(values, adopt_f32_vec)
        }
        (GpuTensorStorage::Real, false, HostNumericDataOwned::F64(values)) => {
            adopt_real!(values, adopt_f64_vec)
        }
        (GpuTensorStorage::Real, false, HostNumericDataOwned::I8(values)) => {
            adopt_real!(values, adopt_i8_vec)
        }
        (GpuTensorStorage::Real, false, HostNumericDataOwned::I16(values)) => {
            adopt_real!(values, adopt_i16_vec)
        }
        (GpuTensorStorage::Real, false, HostNumericDataOwned::I32(values)) => {
            adopt_real!(values, adopt_i32_vec)
        }
        (GpuTensorStorage::Real, false, HostNumericDataOwned::I64(values)) => {
            adopt_real!(values, adopt_i64_vec)
        }
        (GpuTensorStorage::Real, false, HostNumericDataOwned::U8(values)) => {
            adopt_real!(values, adopt_u8_vec)
        }
        (GpuTensorStorage::Real, false, HostNumericDataOwned::U16(values)) => {
            adopt_real!(values, adopt_u16_vec)
        }
        (GpuTensorStorage::Real, false, HostNumericDataOwned::U32(values)) => {
            adopt_real!(values, adopt_u32_vec)
        }
        (GpuTensorStorage::Real, false, HostNumericDataOwned::U64(values)) => {
            adopt_real!(values, adopt_u64_vec)
        }
        (GpuTensorStorage::ComplexInterleaved, false, HostNumericDataOwned::F32(values)) => receipt
            .with_output_allocation(runtime, |permit| {
                ExactHostComplexTensorAllocation::adopt_f32_interleaved_vec(
                    values, host.shape, permit,
                )
            })
            .map(Value::ComplexTensor),
        (GpuTensorStorage::ComplexInterleaved, false, HostNumericDataOwned::F64(values)) => receipt
            .with_output_allocation(runtime, |permit| {
                ExactHostComplexTensorAllocation::adopt_f64_interleaved_vec(
                    values, host.shape, permit,
                )
            })
            .map(Value::ComplexTensor),
        _ => {
            return Err(invalid(
                "selected provider returned an incompatible native payload",
            ))
        }
    }
    .map_err(|cause| admission_error(runtime, cause))?;
    Ok(value)
}
