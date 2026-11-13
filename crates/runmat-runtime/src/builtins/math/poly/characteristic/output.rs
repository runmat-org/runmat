use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, ExactHostComplexTensorAllocation,
    ExactHostNumericAllocationError, ExactHostTensorAllocation, NumericDType, Tensor, Value,
};

use crate::builtins::math::linalg::factor::schur::SchurForm;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

use super::input::VectorRoots;
use super::{admission, checkpoint, invalid, numerical, recurrence};

pub(super) fn from_vector(
    roots: VectorRoots<'_>,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let length = roots
        .len()
        .checked_add(1)
        .ok_or_else(|| invalid("poly degree is too large"))?;
    let (dtype, complex) = roots.class();
    let (mut value, receipt) = allocate(dtype, complex, length, context)?;
    recurrence::vector(&mut value, roots, context)?;
    finish(value, receipt.as_ref(), context)
}

pub(super) fn from_real_schur(
    form: &SchurForm,
    dtype: NumericDType,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let SchurForm::Real { triangular, .. } = form else {
        unreachable!("real Schur form")
    };
    let length = triangular
        .nrows()
        .checked_add(1)
        .ok_or_else(|| invalid("poly degree is too large"))?;
    let (mut value, receipt) = allocate(dtype, false, length, context)?;
    recurrence::real_schur(&mut value, triangular, dtype, context)?;
    finish(value, receipt.as_ref(), context)
}

pub(super) fn from_complex_schur(
    form: &SchurForm,
    dtype: NumericDType,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let SchurForm::Complex { triangular, .. } = form else {
        unreachable!("complex Schur form")
    };
    let length = triangular
        .nrows()
        .checked_add(1)
        .ok_or_else(|| invalid("poly degree is too large"))?;
    let (mut value, receipt) = allocate(dtype, true, length, context)?;
    recurrence::complex_schur(&mut value, triangular, dtype, context)?;
    finish(value, receipt.as_ref(), context)
}

fn finish(
    value: Value,
    receipt: Option<&HostAllocationReceipt>,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    if let Some(receipt) = receipt {
        receipt
            .checkpoint_captured()
            .map_err(|error| admission(context, error))?;
    } else {
        checkpoint(context)?;
    }
    Ok(value)
}

fn temporary_allocation(context: Option<&RuntimeContext>) -> crate::RuntimeError {
    admission(
        context,
        HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed),
    )
}

fn zeros<T: Copy>(
    length: usize,
    zero: T,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| temporary_allocation(context))?;
    while values.len() < length {
        checkpoint(context)?;
        values.resize(values.len().saturating_add(4096).min(length), zero);
    }
    Ok(values)
}

fn allocate(
    dtype: NumericDType,
    complex: bool,
    length: usize,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<(Value, Option<HostAllocationReceipt>)> {
    let shape = [1, length];
    let bytes = if complex {
        HostAllocationPlan::retained_complex_tensor_bytes(dtype, &shape)
    } else {
        HostAllocationPlan::retained_tensor_bytes(dtype, &shape)
    }
    .map_err(|error| admission(context, error))?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(|error| admission(context, error))?;
    let receipt =
        HostAllocationReceipt::admit(context, plan).map_err(|error| admission(context, error))?;
    let value = if let Some(receipt) = receipt.as_ref() {
        let runtime =
            context.ok_or_else(|| admission(context, HostAllocationError::MissingManagedOwner))?;
        receipt
            .with_output_allocation(runtime, |permit| {
                if complex {
                    ExactHostComplexTensorAllocation::new(dtype, &shape, permit)?
                        .into_tensor()
                        .map(Value::ComplexTensor)
                } else {
                    ExactHostTensorAllocation::new(dtype, &shape, permit)?
                        .into_tensor()
                        .map(Value::Tensor)
                }
            })
            .map_err(|error| admission(context, error))?
    } else {
        match (dtype, complex) {
            (NumericDType::F32, false) => {
                Tensor::from_f32(zeros(length, 0.0_f32, context)?, shape.to_vec())
                    .map(Value::Tensor)
            }
            (NumericDType::F64, false) => {
                Tensor::new(zeros(length, 0.0_f64, context)?, shape.to_vec()).map(Value::Tensor)
            }
            (NumericDType::F32, true) => ComplexTensor::from_complex_storage(
                ComplexStorage::F32(
                    zeros(length, ComplexElement(0.0_f32, 0.0_f32), context)?.into(),
                ),
                shape.to_vec(),
            )
            .map(Value::ComplexTensor),
            (NumericDType::F64, true) => ComplexTensor::from_complex_storage(
                ComplexStorage::F64(
                    zeros(length, ComplexElement(0.0_f64, 0.0_f64), context)?.into(),
                ),
                shape.to_vec(),
            )
            .map(Value::ComplexTensor),
            _ => return Err(invalid("poly requires single or double output")),
        }
        .map_err(numerical)?
    };
    if let Some(receipt) = receipt.as_ref() {
        receipt
            .checkpoint_captured()
            .map_err(|error| admission(context, error))?;
    }
    Ok((value, receipt))
}

pub(super) fn real_f32(value: &mut Value) -> BuiltinResult<&mut [f32]> {
    let Value::Tensor(tensor) = value else {
        return Err(numerical("poly output class mismatch"));
    };
    tensor
        .try_as_f32_slice_mut_unique()
        .ok_or_else(|| numerical("poly output is not unique"))
}
pub(super) fn real_f64(value: &mut Value) -> BuiltinResult<&mut [f64]> {
    let Value::Tensor(tensor) = value else {
        return Err(numerical("poly output class mismatch"));
    };
    tensor
        .try_as_f64_slice_mut_unique()
        .ok_or_else(|| numerical("poly output is not unique"))
}
pub(super) fn complex_f32(value: &mut Value) -> BuiltinResult<&mut [ComplexElement<f32>]> {
    let Value::ComplexTensor(tensor) = value else {
        return Err(numerical("poly output class mismatch"));
    };
    tensor
        .try_as_f32_slice_mut_unique()
        .ok_or_else(|| numerical("poly output is not unique"))
}
pub(super) fn complex_f64(value: &mut Value) -> BuiltinResult<&mut [ComplexElement<f64>]> {
    let Value::ComplexTensor(tensor) = value else {
        return Err(numerical("poly output class mismatch"));
    };
    tensor
        .try_as_f64_slice_mut_unique()
        .ok_or_else(|| numerical("poly output is not unique"))
}
