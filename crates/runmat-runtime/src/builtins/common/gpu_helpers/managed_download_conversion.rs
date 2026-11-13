//! Exact, overlap-funded conversions where the provider and Value layouts differ.
use super::super::{admission_error, invalid};
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::RuntimeError;
use runmat_value::{
    ExactHostIntegerComplexTensorAllocation, ExactHostLogicalTensorAllocation, NumericDType,
    NumericScalar, Value,
};

pub(super) fn convert_integer_complex<T: Copy>(
    runtime: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    dtype: NumericDType,
    shape: &[usize],
    values: &[T],
    scalar: impl Fn(T) -> NumericScalar,
) -> Result<Value, RuntimeError> {
    receipt
        .with_output_construction(runtime, |permit| {
            let mut output = ExactHostIntegerComplexTensorAllocation::new(dtype, shape, permit)
                .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))?;
            for (index, pair) in values.chunks_exact(2).enumerate() {
                if index % 4096 == 0 {
                    receipt
                        .checkpoint(runtime)
                        .map_err(|cause| admission_error(runtime, cause))?;
                }
                output
                    .set(index, scalar(pair[0]), scalar(pair[1]))
                    .map_err(|_| invalid("selected integer complex assignment failed"))?;
            }
            output
                .into_tensor()
                .map(Value::ComplexTensor)
                .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))
        })
        .map_err(|cause| admission_error(runtime, cause))?
}

pub(super) fn convert_logical<T: Copy>(
    runtime: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    shape: &[usize],
    values: &[T],
    truthy: impl Fn(T) -> bool,
) -> Result<Value, RuntimeError> {
    receipt
        .with_output_construction(runtime, |permit| {
            let mut output = ExactHostLogicalTensorAllocation::new(shape, permit)
                .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))?;
            for (index, (destination, source)) in
                output.as_mut_slice().iter_mut().zip(values).enumerate()
            {
                if index % 4096 == 0 {
                    receipt
                        .checkpoint(runtime)
                        .map_err(|cause| admission_error(runtime, cause))?;
                }
                *destination = u8::from(truthy(*source));
            }
            output
                .into_array()
                .map(Value::LogicalArray)
                .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))
        })
        .map_err(|cause| admission_error(runtime, cause))?
}
