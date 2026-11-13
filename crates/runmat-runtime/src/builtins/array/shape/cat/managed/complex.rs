//! Exact floating and paired-integer complex concatenation.

use super::*;
use crate::builtins::common::random_args::complex_tensor_into_value;
use runmat_value::{
    ExactHostComplexTensorAllocation, ExactHostIntegerComplexTensorAllocation, NumericDType,
};

enum ComplexOutput {
    Floating(ExactHostComplexTensorAllocation),
    Paired(ExactHostIntegerComplexTensorAllocation, IntegerClass),
}

impl ComplexOutput {
    fn set(&mut self, index: usize, real: NumericScalar, imag: NumericScalar) -> BuiltinResult<()> {
        match self {
            Self::Floating(output) => output
                .set(index, real.materialize_f64(), imag.materialize_f64())
                .map_err(|error| cat_err(format!("cat: {error}"))),
            Self::Paired(output, target) => {
                let real = convert_scalar(real, NumericDType::from(*target), Some(*target));
                let imag = convert_scalar(imag, NumericDType::from(*target), Some(*target));
                output.set(index, real, imag).map_err(cat_err)
            }
        }
    }

    fn into_value(self) -> BuiltinResult<Value> {
        match self {
            Self::Floating(output) => output
                .into_tensor()
                .map(complex_tensor_into_value)
                .map_err(|error| cat_err(format!("cat: {error}"))),
            Self::Paired(output, _) => output
                .into_tensor()
                .map(Value::ComplexTensor)
                .map_err(|error| cat_err(format!("cat: {error}"))),
        }
    }
}

fn source_pair(value: &Value, index: usize) -> BuiltinResult<(NumericScalar, NumericScalar)> {
    let zero = NumericScalar::F64(0.0);
    match value {
        Value::ComplexTensor(tensor) => tensor
            .numeric_value_at(index)
            .ok_or_else(|| cat_err("cat: complex source geometry changed during assembly")),
        Value::Complex(real, imag) if index == 0 => {
            Ok((NumericScalar::F64(*real), NumericScalar::F64(*imag)))
        }
        Value::Tensor(tensor) => tensor
            .host_buffer()
            .value_at(index)
            .map(|real| (real, zero))
            .ok_or_else(|| cat_err("cat: numeric source geometry changed during assembly")),
        Value::LogicalArray(array) => array
            .data
            .get(index)
            .copied()
            .map(|real| (NumericScalar::F64(f64::from(real)), zero))
            .ok_or_else(|| cat_err("cat: logical source geometry changed during assembly")),
        Value::Num(real) if index == 0 => Ok((NumericScalar::F64(*real), zero)),
        Value::Int(real) if index == 0 => Ok((NumericScalar::from(real.clone()), zero)),
        Value::Bool(real) if index == 0 => {
            Ok((NumericScalar::F64(f64::from(u8::from(*real))), zero))
        }
        _ => Err(cat_err(
            "cat: selected complex owner received a nonnumerical input",
        )),
    }
}

pub(in super::super) fn complex(
    context: &RuntimeContext,
    dim_zero: usize,
    values: FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let geometry = inspect_geometry(dim_zero, values)?;
    let integer = leftmost_complex_integer_target(values);
    let dtype = integer.map(NumericDType::from).unwrap_or_else(|| {
        if values.iter().all(|value| {
            matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        }) {
            NumericDType::F32
        } else {
            NumericDType::F64
        }
    });
    let (raw, claims) = if integer.is_some() {
        (
            ExactHostIntegerComplexTensorAllocation::requested_bytes_for_dimensions(
                dtype,
                geometry.len,
                geometry.output_rank,
            ),
            ExactHostIntegerComplexTensorAllocation::retained_claim_count(),
        )
    } else {
        (
            ExactHostComplexTensorAllocation::requested_bytes_for_dimensions(
                dtype,
                geometry.len,
                geometry.output_rank,
            ),
            ExactHostComplexTensorAllocation::retained_claim_count(),
        )
    };
    let raw = raw.map_err(|cause| admission_error(context, HostAllocationError::Numeric(cause)))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| admission_error(context, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|cause| admission_error(context, cause))?;
    let scratch = HostTypedWorkspace::<usize>::quoted_bytes(geometry.rank)
        .map_err(|cause| admission_error(context, cause))?;
    let plan = HostAllocationPlan::new(scratch, retained)
        .map_err(|cause| admission_error(context, cause))?;
    let receipt = HostAllocationReceipt::admit(Some(context), plan)
        .map_err(|cause| admission_error(context, cause))?
        .ok_or_else(|| admission_error(context, HostAllocationError::MissingManagedOwner))?;
    let mut shape = receipt
        .temporary_typed::<usize>(context, geometry.rank)
        .map_err(|cause| admission_error(context, cause))?;
    for axis in 0..geometry.rank {
        shape.as_mut_slice()[axis] = output_extent(values, axis, dim_zero, geometry.has_nonempty)?;
    }
    let mut output = receipt
        .with_output_allocation(context, |permit| match integer {
            Some(target) => ExactHostIntegerComplexTensorAllocation::new(
                dtype,
                &shape.as_slice()[..geometry.output_rank],
                permit,
            )
            .map(|output| ComplexOutput::Paired(output, target)),
            None => ExactHostComplexTensorAllocation::new(
                dtype,
                &shape.as_slice()[..geometry.output_rank],
                permit,
            )
            .map(ComplexOutput::Floating),
        })
        .map_err(|cause| admission_error(context, cause))?;
    fill_elements(
        context,
        &receipt,
        dim_zero,
        values,
        shape.as_slice(),
        &geometry,
        |destination, value, source_index| {
            let (real, imag) = source_pair(value, source_index)?;
            output.set(destination, real, imag)
        },
    )?;
    output.into_value()
}
