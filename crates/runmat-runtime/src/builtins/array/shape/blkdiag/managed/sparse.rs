//! Claimed CSC assembly directly from borrowed block storage.

use super::*;
use runmat_value::{ExactHostSparseClass, ExactHostSparseTensorAllocation};
use std::num::NonZeroUsize;

fn dimensions(value: &Value) -> BuiltinResult<(usize, usize)> {
    match value {
        Value::SparseTensor(sparse) => Ok((sparse.rows, sparse.cols)),
        other => shape(other),
    }
}

fn real_value(value: &Value, index: usize) -> BuiltinResult<NumericScalar> {
    match value {
        Value::SparseTensor(sparse) => sparse.numeric_value_at(index),
        Value::Tensor(tensor) => tensor.numeric_value_at(index),
        Value::LogicalArray(array) => array
            .data
            .get(index)
            .map(|&value| NumericScalar::F64(f64::from(value))),
        Value::Num(value) if index == 0 => Some(NumericScalar::F64(*value)),
        Value::Int(value) if index == 0 => Some(NumericScalar::from(value.clone())),
        Value::Bool(value) if index == 0 => Some(NumericScalar::F64(f64::from(u8::from(*value)))),
        Value::Complex(real, imag) if index == 0 && *imag == 0.0 => Some(NumericScalar::F64(*real)),
        Value::ComplexTensor(tensor) => tensor
            .numeric_value_at(index)
            .and_then(|(re, im)| im.is_zero().then_some(re)),
        _ => None,
    }
    .ok_or_else(|| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, "nonreal sparse block"))
}

fn output_class(
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ExactHostSparseClass> {
    let integer = values.iter().find_map(|value| match value {
        Value::SparseTensor(sparse) => sparse
            .integer_storage()
            .map(|_| sparse.numeric_dtype().unwrap()),
        _ => None,
    });
    if let Some(dtype) = integer {
        if values.iter().all(|value| match value {
            Value::SparseTensor(sparse) => sparse.numeric_dtype() == Some(dtype),
            Value::Tensor(tensor) => tensor.numeric_dtype() == dtype,
            Value::Int(value) => {
                IntegerStorage::from_scalar(value.clone()).numeric_dtype() == dtype
            }
            _ => false,
        }) {
            return Ok(ExactHostSparseClass::Numeric(dtype));
        }
        return Err(error_with_detail(
            &BLKDIAG_ERROR_INVALID_INPUT,
            "typed sparse integer blocks require the same integer class",
        ));
    }
    if values
        .iter()
        .any(|value| matches!(value, Value::SparseTensor(sparse) if sparse.is_complex()))
    {
        let single = values.iter().any(|value| match value {
            Value::SparseTensor(sparse) => sparse.numeric_dtype() == Some(NumericDType::F32),
            Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
            Value::ComplexTensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
            _ => false,
        });
        return Ok(ExactHostSparseClass::Complex(if single {
            NumericDType::F32
        } else {
            NumericDType::F64
        }));
    }
    if values.iter().all(|value| {
        matches!(value,
            Value::SparseTensor(sparse) if sparse.is_logical()
        ) || matches!(value, Value::LogicalArray(_) | Value::Bool(_))
    }) {
        return Ok(ExactHostSparseClass::Logical);
    }
    let single = values.iter().any(|value| match value {
        Value::SparseTensor(sparse) => sparse.numeric_dtype() == Some(NumericDType::F32),
        Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        Value::ComplexTensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        _ => false,
    });
    Ok(ExactHostSparseClass::Numeric(if single {
        NumericDType::F32
    } else {
        NumericDType::F64
    }))
}

fn stored_value(class: ExactHostSparseClass, value: NumericScalar) -> BuiltinResult<NumericScalar> {
    match class {
        ExactHostSparseClass::Logical => Ok(value),
        ExactHostSparseClass::Numeric(dtype) if dtype.integer_class().is_some() => {
            if value.class_name() != dtype.class_name() {
                return Err(error_with_detail(
                    &BLKDIAG_ERROR_INVALID_INPUT,
                    "typed sparse integer source class changed",
                ));
            }
            Ok(value)
        }
        ExactHostSparseClass::Numeric(NumericDType::F32) => {
            Ok(NumericScalar::F32(value.materialize_f64() as f32))
        }
        ExactHostSparseClass::Numeric(NumericDType::F64) => {
            Ok(NumericScalar::F64(value.materialize_f64()))
        }
        _ => Err(error_with_detail(
            &BLKDIAG_ERROR_INVALID_INPUT,
            "unsupported sparse output class",
        )),
    }
}

fn complex_value(value: &Value, index: usize) -> BuiltinResult<(f64, f64)> {
    match value {
        Value::SparseTensor(sparse) => sparse
            .complex_value_at(index)
            .map(|value| (value.0, value.1))
            .or_else(|| {
                sparse
                    .numeric_value_at(index)
                    .map(|real| (real.materialize_f64(), 0.0))
            }),
        Value::ComplexTensor(tensor) => tensor
            .numeric_value_at(index)
            .map(|(real, imag)| (real.materialize_f64(), imag.materialize_f64())),
        Value::Complex(real, imag) if index == 0 => Some((*real, *imag)),
        _ => real_value(value, index)
            .ok()
            .map(|real| (real.materialize_f64(), 0.0)),
    }
    .ok_or_else(|| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, "complex sparse value missing"))
}

pub(super) fn assemble(
    context: &RuntimeContext,
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let class = output_class(values)?;
    let mut rows = 0usize;
    let mut cols = 0usize;
    let mut nnz = 0usize;
    let mut inspected = 0usize;
    for value in values {
        let (height, width) = dimensions(value)?;
        rows = rows.checked_add(height).ok_or_else(size_overflow)?;
        cols = cols.checked_add(width).ok_or_else(size_overflow)?;
        if let Value::SparseTensor(sparse) = value {
            nnz = nnz.checked_add(sparse.nnz()).ok_or_else(size_overflow)?;
            continue;
        }
        let len = height.checked_mul(width).ok_or_else(size_overflow)?;
        for index in 0..len {
            if inspected.is_multiple_of(CHECKPOINT_CHUNK) && context.is_cancelled() {
                return Err(admission(context, HostAllocationError::Cancelled));
            }
            let nonzero = if matches!(class, ExactHostSparseClass::Complex(_)) {
                let (real, imag) = complex_value(value, index)?;
                real != 0.0 || imag != 0.0
            } else {
                !real_value(value, index)?.is_zero()
            };
            if nonzero {
                nnz = nnz.checked_add(1).ok_or_else(size_overflow)?;
            }
            inspected += 1;
        }
    }
    let raw =
        ExactHostSparseTensorAllocation::requested_bytes_for_dimensions(class, rows, cols, nnz)
            .map_err(|cause| admission(context, HostAllocationError::Numeric(cause)))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| admission(context, HostAllocationError::Overflow))?,
        class.retained_claim_count(),
    )
    .map_err(|cause| admission(context, cause))?;
    let receipt = HostAllocationReceipt::admit(
        Some(context),
        HostAllocationPlan::new(0, retained).map_err(|cause| admission(context, cause))?,
    )
    .map_err(|cause| admission(context, cause))?
    .ok_or_else(|| admission(context, HostAllocationError::MissingManagedOwner))?;
    let mut output = receipt
        .with_output_allocation(context, |permit| {
            ExactHostSparseTensorAllocation::new(class, rows, cols, nnz, permit)
        })
        .map_err(|cause| admission(context, cause))?;
    output
        .set_col_ptr(0, 0)
        .map_err(|detail| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail))?;
    let mut row_offset = 0usize;
    let mut column = 0usize;
    let mut written = 0usize;
    for value in values {
        let (height, width) = dimensions(value)?;
        for col in 0..width {
            if column.is_multiple_of(CHECKPOINT_CHUNK) {
                receipt
                    .checkpoint(context)
                    .map_err(|cause| admission(context, cause))?;
            }
            match value {
                Value::SparseTensor(sparse) => {
                    for index in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
                        let row = row_offset
                            .checked_add(sparse.row_indices[index])
                            .ok_or_else(size_overflow)?;
                        output.set_row_index(written, row).map_err(|detail| {
                            error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail)
                        })?;
                        if matches!(class, ExactHostSparseClass::Complex(_)) {
                            let (real, imag) = complex_value(value, index)?;
                            output
                                .set_complex_value(written, real, imag)
                                .map_err(|detail| {
                                    error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail)
                                })?;
                        } else if class != ExactHostSparseClass::Logical {
                            let scalar = sparse.numeric_value_at(index).ok_or_else(|| {
                                error_with_detail(
                                    &BLKDIAG_ERROR_INVALID_INPUT,
                                    "sparse value missing",
                                )
                            })?;
                            output
                                .set_numeric_value(written, stored_value(class, scalar)?)
                                .map_err(|detail| {
                                    error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail)
                                })?;
                        }
                        written += 1;
                        if written.is_multiple_of(CHECKPOINT_CHUNK) {
                            receipt
                                .checkpoint(context)
                                .map_err(|cause| admission(context, cause))?;
                        }
                    }
                }
                _ => {
                    for row in 0..height {
                        let index = row + col * height;
                        let pair = if matches!(class, ExactHostSparseClass::Complex(_)) {
                            Some(complex_value(value, index)?)
                        } else {
                            None
                        };
                        let scalar = if pair.is_some() {
                            None
                        } else {
                            Some(real_value(value, index)?)
                        };
                        let nonzero = pair.is_some_and(|(real, imag)| real != 0.0 || imag != 0.0)
                            || scalar.is_some_and(|value| !value.is_zero());
                        if nonzero {
                            output
                                .set_row_index(written, row_offset + row)
                                .map_err(|detail| {
                                    error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail)
                                })?;
                            if let Some((real, imag)) = pair {
                                output.set_complex_value(written, real, imag).map_err(
                                    |detail| {
                                        error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail)
                                    },
                                )?;
                            } else if class != ExactHostSparseClass::Logical {
                                output
                                    .set_numeric_value(
                                        written,
                                        stored_value(class, scalar.expect("real output class"))?,
                                    )
                                    .map_err(|detail| {
                                        error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail)
                                    })?;
                            }
                            written += 1;
                        }
                        if row % CHECKPOINT_CHUNK == 0 {
                            receipt
                                .checkpoint(context)
                                .map_err(|cause| admission(context, cause))?;
                        }
                    }
                }
            }
            column += 1;
            output
                .set_col_ptr(column, written)
                .map_err(|detail| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail))?;
        }
        row_offset = row_offset.checked_add(height).ok_or_else(size_overflow)?;
    }
    receipt
        .checkpoint(context)
        .map_err(|cause| admission(context, cause))?;
    let mut validation = output
        .into_validation()
        .map_err(|detail| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail))?;
    let quantum = NonZeroUsize::new(CHECKPOINT_CHUNK).expect("nonzero validation quantum");
    while !validation
        .advance(quantum)
        .map_err(|detail| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail))?
    {
        receipt
            .checkpoint(context)
            .map_err(|cause| admission(context, cause))?;
    }
    receipt
        .checkpoint(context)
        .map_err(|cause| admission(context, cause))?;
    let result = validation
        .into_tensor()
        .map(Value::SparseTensor)
        .map_err(|detail| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail))?;
    receipt
        .checkpoint(context)
        .map_err(|cause| admission(context, cause))?;
    Ok(result)
}
