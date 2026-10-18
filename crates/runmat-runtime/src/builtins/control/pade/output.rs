use runmat_value::{
    exact_capacity_vec, ExactHostTensorAllocation, NumericDType, OutputListInitializationError,
    OutputListStorage, Tensor, Value,
};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};

use super::{
    admission, coefficients as recurrence, invalid, numerical, BuiltinResult, RuntimeContext,
};

pub(super) fn coefficients(
    delay: f64,
    order: usize,
    dtype: NumericDType,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let length = if delay == 0.0 {
        1
    } else {
        order
            .checked_add(1)
            .ok_or_else(|| invalid("order is too large"))?
    };
    let shape = [1, length];
    let tensor_bytes = HostAllocationPlan::retained_tensor_bytes(dtype, &shape)
        .map_err(|error| admission(runtime, error))?;
    let list_physical_bytes = u64::try_from(
        OutputListStorage::requested_bytes(2)
            .map_err(|error| admission(runtime, HostAllocationError::Numeric(error)))?,
    )
    .map_err(|_| admission(runtime, HostAllocationError::Overflow))?;
    let list_bytes = HostAllocationPlan::retained_exact_output_bytes(list_physical_bytes, 1)
        .map_err(|error| admission(runtime, error))?;
    let retained = tensor_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(list_bytes))
        .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
    let plan = HostAllocationPlan::new(0, retained).map_err(|error| admission(runtime, error))?;
    let receipt =
        HostAllocationReceipt::admit(runtime, plan).map_err(|error| admission(runtime, error))?;
    let mut outputs = if let Some(receipt) = receipt.as_ref() {
        let context =
            runtime.ok_or_else(|| admission(runtime, HostAllocationError::MissingManagedOwner))?;
        receipt
            .with_output_allocation(context, |permit| {
                OutputListStorage::try_initialize(2, permit, |_| {
                    ExactHostTensorAllocation::new(dtype, &shape, permit)?
                        .into_tensor()
                        .map(Value::Tensor)
                })
                .map_err(|error| match error {
                    OutputListInitializationError::Allocation(error)
                    | OutputListInitializationError::Element(error) => error,
                })
            })
            .map_err(|error| admission(runtime, error))?
    } else {
        let allocate = || -> BuiltinResult<Value> {
            match dtype {
                NumericDType::F32 => {
                    let mut values = exact_capacity_vec::<f32>(length)
                        .map_err(|error| admission(runtime, HostAllocationError::Numeric(error)))?;
                    values.resize(length, 0.0);
                    Tensor::from_f32(values, shape.to_vec())
                        .map(Value::Tensor)
                        .map_err(|error| numerical(format!("Padé tensor shape: {error}")))
                }
                NumericDType::F64 => {
                    let mut values = exact_capacity_vec::<f64>(length)
                        .map_err(|error| admission(runtime, HostAllocationError::Numeric(error)))?;
                    values.resize(length, 0.0);
                    Tensor::new(values, shape.to_vec())
                        .map(Value::Tensor)
                        .map_err(|error| numerical(format!("Padé tensor shape: {error}")))
                }
                _ => Err(invalid("delay must be single or double")),
            }
        };
        OutputListStorage::unbounded(vec![allocate()?, allocate()?])
    };
    let values = outputs
        .try_as_mut_slice_unique()
        .ok_or_else(|| numerical("Padé output list lost unique ownership"))?;
    let [numerator, denominator] = values else {
        return Err(numerical("Padé output list cardinality changed"));
    };
    match (numerator, denominator) {
        (Value::Tensor(num), Value::Tensor(den)) if dtype == NumericDType::F64 => {
            let num = num
                .try_as_f64_slice_mut_unique()
                .ok_or_else(|| numerical("Padé numerator is not unique double storage"))?;
            let den = den
                .try_as_f64_slice_mut_unique()
                .ok_or_else(|| numerical("Padé denominator is not unique double storage"))?;
            recurrence::visit(delay, order, runtime, |index, n, d| {
                num[index] = n;
                den[index] = d;
                Ok(())
            })?;
        }
        (Value::Tensor(num), Value::Tensor(den)) if dtype == NumericDType::F32 => {
            let num = num
                .try_as_f32_slice_mut_unique()
                .ok_or_else(|| numerical("Padé numerator is not unique single storage"))?;
            let den = den
                .try_as_f32_slice_mut_unique()
                .ok_or_else(|| numerical("Padé denominator is not unique single storage"))?;
            recurrence::visit(delay, order, runtime, |index, n, d| {
                let single = d as f32;
                if !single.is_finite() {
                    return Err(numerical("Padé coefficient exceeds single precision"));
                }
                num[index] = n as f32;
                den[index] = single;
                Ok(())
            })?;
        }
        _ => return Err(numerical("Padé output class changed")),
    }
    if let Some(receipt) = receipt.as_ref() {
        receipt
            .checkpoint_captured()
            .map_err(|error| admission(runtime, error))?;
    }
    Ok(Value::output_list(outputs))
}
