mod character;
use character::character;

use super::*;
use runmat_value::{
    ComplexElement, ExactHostComplexTensorAllocation, ExactHostIntegerComplexTensorAllocation,
    ExactHostLogicalTensorAllocation, ExactHostTensorAllocation, NumericDType, NumericScalar,
    Value,
};

pub(super) fn finish(source: &Managed<'_>, shape: &[usize]) -> Result<Value, UniformScalarError> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or(UniformScalarError::SIZE_OVERFLOW)?;
    let values = source.values.initialized();
    if count != values.len() && !(values.is_empty() && source.class.is_none()) {
        return Err(UniformScalarError::invalid_storage("callback shape"));
    }
    let class = source.class.unwrap_or(Class::F64);
    if class == Class::Char {
        return character(source, shape);
    }
    let dtype = match class {
        Class::F32 | Class::ComplexF32 => NumericDType::F32,
        Class::Integer(dtype) | Class::IntegerComplex(dtype) => dtype,
        _ => NumericDType::F64,
    };
    let (bytes, claims) = match class {
        Class::Logical => (
            ExactHostLogicalTensorAllocation::requested_bytes(shape),
            ExactHostLogicalTensorAllocation::retained_claim_count(),
        ),
        Class::ComplexF64 | Class::ComplexF32 => (
            ExactHostComplexTensorAllocation::requested_bytes(dtype, shape),
            ExactHostComplexTensorAllocation::retained_claim_count(),
        ),
        Class::IntegerComplex(_) => (
            ExactHostIntegerComplexTensorAllocation::requested_bytes_for_dimensions(
                dtype,
                count,
                shape.len(),
            ),
            ExactHostIntegerComplexTensorAllocation::retained_claim_count(),
        ),
        _ => (
            ExactHostTensorAllocation::requested_bytes(dtype, shape),
            ExactHostTensorAllocation::retained_claim_count(),
        ),
    };
    let receipt = admit(source, bytes, claims)?;
    let permit = receipt
        .output_permit(source.runtime)
        .map_err(|e| allocation(source.runtime, e))?;
    macro_rules! exact {
        ($e:expr) => {{
            let result = $e;
            checkpoint(source, &receipt)?;
            result.map_err(|e: ExactHostNumericAllocationError| {
                allocation(source.runtime, e.into())
            })?
        }};
    }
    let result = match class {
        Class::Logical => {
            let mut out = exact!(ExactHostLogicalTensorAllocation::new(shape, &permit));
            for (index, value) in values.iter().enumerate() {
                checkpoint(source, &receipt)?;
                let ClassifiedValue::Logical(value) = value else {
                    return Err(invalid());
                };
                out.as_mut_slice()[index] = u8::from(*value);
            }
            Value::LogicalArray(exact!(out.into_array()))
        }
        Class::ComplexF64 | Class::ComplexF32 => {
            let mut out = exact!(ExactHostComplexTensorAllocation::new(dtype, shape, &permit));
            for (index, value) in values.iter().enumerate() {
                checkpoint(source, &receipt)?;
                match value {
                    ClassifiedValue::ComplexF32((r, i)) => {
                        out.set_f32(index, ComplexElement(*r, *i))
                    }
                    ClassifiedValue::F32(r) => out.set_f32(index, ComplexElement(*r, 0.0)),
                    ClassifiedValue::ComplexF64((r, i)) => out.set(index, *r, *i),
                    ClassifiedValue::F64(r) => out.set(index, *r, 0.0),
                    ClassifiedValue::Logical(r) => out.set(index, if *r { 1.0 } else { 0.0 }, 0.0),
                    _ => return Err(invalid()),
                }
                .map_err(|e| allocation(source.runtime, e.into()))?;
            }
            Value::ComplexTensor(exact!(out.into_tensor()))
        }
        Class::IntegerComplex(_) => {
            let mut out = exact!(ExactHostIntegerComplexTensorAllocation::new(
                dtype, shape, &permit
            ));
            for (index, value) in values.iter().enumerate() {
                checkpoint(source, &receipt)?;
                let ClassifiedValue::IntegerComplex(r, i) = value else {
                    return Err(invalid());
                };
                out.set(index, r.clone().into(), i.clone().into())
                    .map_err(UniformScalarError::materialization)?;
            }
            Value::ComplexTensor(exact!(out.into_tensor()))
        }
        _ => {
            let mut out = exact!(ExactHostTensorAllocation::new(dtype, shape, &permit));
            for (index, value) in values.iter().enumerate() {
                checkpoint(source, &receipt)?;
                let scalar = match value {
                    ClassifiedValue::F64(v) => NumericScalar::F64(*v),
                    ClassifiedValue::F32(v) => NumericScalar::F32(*v),
                    ClassifiedValue::Integer(v) => v.clone().into(),
                    ClassifiedValue::Logical(v) => NumericScalar::F64(if *v { 1.0 } else { 0.0 }),
                    _ => return Err(invalid()),
                };
                out.set_value(index, scalar)
                    .map_err(UniformScalarError::materialization)?;
            }
            Value::Tensor(exact!(out.into_tensor()))
        }
    };
    checkpoint(source, &receipt)?;
    Ok(result)
}
fn invalid() -> UniformScalarError {
    UniformScalarError::invalid_storage("classified scalar")
}
fn checkpoint(
    source: &Managed<'_>,
    receipt: &HostAllocationReceipt,
) -> Result<(), UniformScalarError> {
    source.checkpoint()?;
    receipt
        .checkpoint(source.runtime)
        .map_err(|e| allocation(source.runtime, e))
}
fn admit(
    source: &Managed<'_>,
    bytes: Result<usize, ExactHostNumericAllocationError>,
    claims: usize,
) -> Result<HostAllocationReceipt, UniformScalarError> {
    source.checkpoint()?;
    let bytes = bytes.map_err(|e| allocation(source.runtime, e.into()))?;
    let bytes = u64::try_from(bytes)
        .map_err(|_| allocation(source.runtime, HostAllocationError::Overflow))?;
    let bytes = HostAllocationPlan::retained_exact_output_bytes(bytes, claims)
        .map_err(|e| allocation(source.runtime, e))?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(|e| allocation(source.runtime, e))?;
    let receipt = HostAllocationReceipt::admit(Some(source.runtime), plan)
        .map_err(|e| allocation(source.runtime, e))?
        .ok_or_else(|| allocation(source.runtime, HostAllocationError::StaleAuthority))?;
    checkpoint(source, &receipt)?;
    Ok(receipt)
}
