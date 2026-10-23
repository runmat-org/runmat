//! One final native payload allocation, shared by read, delete and scatter.
use super::{allocation, map_slice_shape_error, receipt};
use crate::context::RuntimeContext;
use crate::indexing::plan::admission::Owner;
use crate::RuntimeError;
use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, ExactHostComplexTensorAllocation,
    ExactHostIntegerComplexTensorAllocation, ExactHostNumericAllocation, IntegerComplexStorage,
    NumericDType, NumericScalar, NumericStorage, SharedShape, Tensor,
};

pub(in crate::indexing) enum RealDestination<'a> {
    Ordinary(&'a mut NumericStorage),
    Exact(&'a mut ExactHostNumericAllocation, NumericDType),
}
impl RealDestination<'_> {
    pub(in crate::indexing) fn set(
        &mut self,
        index: usize,
        value: NumericScalar,
    ) -> Result<(), RuntimeError> {
        let dtype = match self {
            Self::Ordinary(storage) => storage.numeric_dtype(),
            Self::Exact(_, dtype) => *dtype,
        };
        let value = floating_assignment(dtype, value);
        match self {
            Self::Ordinary(storage) => storage.set_value(index, value),
            Self::Exact(storage, _) => storage.set_value(index, value),
        }
        .map_err(map_slice_shape_error)
    }
}

// Preserve native floating bits on same-class assignment, while retaining
// the original mathematical conversion for genuinely different classes.
fn floating_assignment(dtype: NumericDType, value: NumericScalar) -> NumericScalar {
    match (dtype, value) {
        (NumericDType::F32, NumericScalar::F32(_)) | (NumericDType::F64, NumericScalar::F64(_)) => {
            value
        }
        (NumericDType::F32, _) => NumericScalar::F32(value.materialize_f32()),
        (NumericDType::F64, _) => NumericScalar::F64(value.materialize_f64()),
        _ => value,
    }
}

pub(in crate::indexing) fn real(
    context: Option<&RuntimeContext>,
    dtype: NumericDType,
    shape: SharedShape,
    fill: impl FnOnce(&mut RealDestination<'_>) -> Result<(), RuntimeError>,
) -> Result<Tensor, RuntimeError> {
    let owner = Owner::new(context)?;
    let count = elements(&shape)?;
    let raw = ExactHostNumericAllocation::requested_bytes(dtype, count)
        .map_err(|source| allocation(context, source.into()))?;
    let receipt = receipt(context, raw, 1)?;
    let output = if let Some(receipt) = &receipt {
        let runtime = context.expect("admitted native output has an owner");
        receipt
            .with_output_construction(runtime, |permit| {
                let mut payload = ExactHostNumericAllocation::new(dtype, count, permit)
                    .map_err(|source| allocation(context, source.into()))?;
                fill(&mut RealDestination::Exact(&mut payload, dtype))?;
                owner.check()?;
                let buffer = payload
                    .try_into_host_buffer()
                    .map_err(|source| allocation(context, source.into()))?;
                Tensor::from_shared_host_buffer(buffer, shape).map_err(map_slice_shape_error)
            })
            .map_err(|source| allocation(context, source))??
    } else {
        let mut payload = NumericStorage::zeros(dtype, count);
        fill(&mut RealDestination::Ordinary(&mut payload))?;
        owner.check()?;
        Tensor::from_numeric_storage(payload, shape).map_err(map_slice_shape_error)?
    };
    owner.check()?;
    Ok(output)
}

pub(in crate::indexing) enum ComplexDestination<'a> {
    Ordinary(&'a mut ComplexStorage),
    Floating(&'a mut ExactHostComplexTensorAllocation),
    Integer(&'a mut ExactHostIntegerComplexTensorAllocation),
}
impl ComplexDestination<'_> {
    pub(in crate::indexing) fn set(
        &mut self,
        index: usize,
        real: NumericScalar,
        imag: NumericScalar,
    ) -> Result<(), RuntimeError> {
        match self {
            Self::Ordinary(ComplexStorage::F32(values)) => {
                let NumericScalar::F32(real) = floating_assignment(NumericDType::F32, real) else {
                    unreachable!("single class")
                };
                let NumericScalar::F32(imag) = floating_assignment(NumericDType::F32, imag) else {
                    unreachable!("single class")
                };
                *values.get_mut(index).ok_or_else(|| {
                    map_slice_shape_error("complex destination index out of bounds")
                })? = ComplexElement(real, imag);
                Ok(())
            }
            Self::Ordinary(ComplexStorage::F64(values)) => {
                *values.get_mut(index).ok_or_else(|| {
                    map_slice_shape_error("complex destination index out of bounds")
                })? = ComplexElement(real.materialize_f64(), imag.materialize_f64());
                Ok(())
            }
            Self::Ordinary(ComplexStorage::Integer(values)) => {
                let real = real
                    .into_int_value()
                    .ok_or_else(|| map_slice_shape_error("integer component class mismatch"))?;
                let imag = imag
                    .into_int_value()
                    .ok_or_else(|| map_slice_shape_error("integer component class mismatch"))?;
                values
                    .set_exact_assignment(index, real, imag)
                    .map_err(map_slice_shape_error)
            }
            Self::Floating(values) => {
                let result = if values.numeric_dtype() == NumericDType::F32 {
                    let NumericScalar::F32(real) = floating_assignment(NumericDType::F32, real)
                    else {
                        unreachable!("single class")
                    };
                    let NumericScalar::F32(imag) = floating_assignment(NumericDType::F32, imag)
                    else {
                        unreachable!("single class")
                    };
                    values.set_f32(index, ComplexElement(real, imag))
                } else {
                    values.set(index, real.materialize_f64(), imag.materialize_f64())
                };
                result.map_err(map_slice_shape_error)
            }
            Self::Integer(values) => values.set(index, real, imag).map_err(map_slice_shape_error),
        }
    }
}

pub(in crate::indexing) fn complex(
    context: Option<&RuntimeContext>,
    dtype: NumericDType,
    shape: &[usize],
    fill: impl FnOnce(&mut ComplexDestination<'_>) -> Result<(), RuntimeError>,
) -> Result<ComplexTensor, RuntimeError> {
    let owner = Owner::new(context)?;
    let count = elements(shape)?;
    let integer = dtype.integer_class().is_some();
    let (raw, claims) = if integer {
        (
            ExactHostIntegerComplexTensorAllocation::requested_bytes_for_dimensions(
                dtype,
                count,
                shape.len(),
            ),
            ExactHostIntegerComplexTensorAllocation::retained_claim_count(),
        )
    } else {
        (
            ExactHostComplexTensorAllocation::requested_bytes(dtype, shape),
            ExactHostComplexTensorAllocation::retained_claim_count(),
        )
    };
    let receipt = receipt(
        context,
        raw.map_err(|source| allocation(context, source.into()))?,
        claims,
    )?;
    let output = if let Some(receipt) = &receipt {
        let runtime = context.expect("admitted native output has an owner");
        receipt
            .with_output_construction(runtime, |permit| {
                if integer {
                    let mut payload =
                        ExactHostIntegerComplexTensorAllocation::new(dtype, shape, permit)
                            .map_err(|source| allocation(context, source.into()))?;
                    fill(&mut ComplexDestination::Integer(&mut payload))?;
                    owner.check()?;
                    payload
                        .into_tensor()
                        .map_err(|source| allocation(context, source.into()))
                } else {
                    let mut payload = ExactHostComplexTensorAllocation::new(dtype, shape, permit)
                        .map_err(|source| allocation(context, source.into()))?;
                    fill(&mut ComplexDestination::Floating(&mut payload))?;
                    owner.check()?;
                    payload
                        .into_tensor()
                        .map_err(|source| allocation(context, source.into()))
                }
            })
            .map_err(|source| allocation(context, source))??
    } else {
        let mut payload = match dtype {
            NumericDType::F32 => ComplexStorage::F32(vec![ComplexElement(0.0, 0.0); count].into()),
            NumericDType::F64 => ComplexStorage::F64(vec![ComplexElement(0.0, 0.0); count].into()),
            _ => ComplexStorage::Integer(
                IntegerComplexStorage::new(
                    NumericStorage::zeros(dtype, count)
                        .into_integer_storage()
                        .expect("integer class"),
                    NumericStorage::zeros(dtype, count)
                        .into_integer_storage()
                        .expect("integer class"),
                )
                .map_err(map_slice_shape_error)?,
            ),
        };
        fill(&mut ComplexDestination::Ordinary(&mut payload))?;
        owner.check()?;
        ComplexTensor::from_complex_storage(payload, shape.to_vec())
            .map_err(map_slice_shape_error)?
    };
    owner.check()?;
    Ok(output)
}

fn elements(shape: &[usize]) -> Result<usize, RuntimeError> {
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| map_slice_shape_error("numeric output shape overflow"))
}
