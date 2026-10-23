//! Direct, exact retained dense outputs for finite Host block assembly.

use super::*;
use crate::builtins::common::integer_conversion::{IntegerClass, IntegerClassExt};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use runmat_value::{
    ExactHostComplexTensorAllocation, ExactHostIntegerComplexTensorAllocation,
    ExactHostLogicalTensorAllocation, ExactHostTensorAllocation,
};

const CHECKPOINT_CHUNK: usize = 4096;

mod sparse;
mod character;

pub(super) fn character(
    context: &RuntimeContext,
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    character::assemble(context, values)
}

pub(super) fn sparse(
    context: &RuntimeContext,
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    sparse::assemble(context, values)
}

pub(super) fn has_nonreal_dense_complex(
    context: &RuntimeContext,
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<bool> {
    for value in values {
        match value {
            Value::Complex(_, imag) if *imag != 0.0 => return Ok(true),
            Value::ComplexTensor(tensor) => {
                for index in 0..tensor.len() {
                    if index % CHECKPOINT_CHUNK == 0 && context.is_cancelled() {
                        return Err(admission(context, HostAllocationError::Cancelled));
                    }
                    let (_, imag) = tensor.numeric_value_at(index).ok_or_else(|| {
                        error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, "complex payload missing")
                    })?;
                    if !imag.is_zero() {
                        return Ok(true);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(false)
}

pub(super) fn admission(context: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    build_runtime_error(format!("blkdiag: host allocation: {cause}"))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

#[derive(Clone, Copy)]
enum DenseClass {
    Real(NumericDType),
    Logical,
    Complex(NumericDType),
    Paired(NumericDType),
}

fn shape(value: &Value) -> BuiltinResult<(usize, usize)> {
    match value {
        Value::Tensor(tensor) => matrix_dims_from_shape(tensor.shape()),
        Value::ComplexTensor(tensor) => matrix_dims_from_shape(tensor.shape()),
        Value::LogicalArray(array) => matrix_dims_from_shape(&array.shape),
        Value::SparseTensor(sparse) => Some((sparse.rows, sparse.cols)),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Some((1, 1)),
        _ => None,
    }
    .ok_or_else(|| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, "unsupported dense block"))
}

fn class(values: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<DenseClass> {
    if values.is_empty() {
        return Ok(DenseClass::Real(NumericDType::F64));
    }
    let paired = values.iter().find_map(|value| match value {
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
            Some(tensor.numeric_dtype())
        }
        _ => None,
    });
    if let Some(dtype) = paired {
        if values.iter().all(|value| {
            matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() && tensor.numeric_dtype() == dtype)
        }) {
            return Ok(DenseClass::Paired(dtype));
        }
        return Err(error_with_detail(
            &BLKDIAG_ERROR_INVALID_INPUT,
            "typed complex integer blocks must all use the same integer class",
        ));
    }
    let integer = values.iter().find_map(|value| match value {
        Value::Int(value) => Some(IntegerClass::from_int_value(value)),
        Value::Tensor(tensor) => tensor.integer_storage().map(IntegerClass::from_storage),
        Value::SparseTensor(sparse) => sparse.integer_storage().map(IntegerClass::from_storage),
        _ => None,
    });
    if let Some(integer) = integer {
        if values.iter().any(|value| {
            matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
                || matches!(value, Value::SparseTensor(sparse) if sparse.is_complex())
        }) {
            return Err(error_with_detail(
                &BLKDIAG_ERROR_INVALID_INPUT,
                "integer-dominant blocks must be real numeric or logical matrices",
            ));
        }
        return Ok(DenseClass::Real(NumericDType::from(integer)));
    }
    if values
        .iter()
        .all(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
    {
        return Ok(DenseClass::Logical);
    }
    if values.iter().any(|value| {
        matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
            || matches!(value, Value::SparseTensor(sparse) if sparse.is_complex())
    }) {
        let single = values.iter().any(|value| match value {
            Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
            Value::ComplexTensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
            Value::SparseTensor(sparse) => sparse.numeric_dtype() == Some(NumericDType::F32),
            _ => false,
        });
        return Ok(DenseClass::Complex(if single {
            NumericDType::F32
        } else {
            NumericDType::F64
        }));
    }
    let mut real_dtype = None;
    for value in values {
        let input_dtype = match value {
            Value::Tensor(tensor) => Some(tensor.numeric_dtype()),
            Value::SparseTensor(sparse) => sparse.numeric_dtype(),
            Value::Num(_) => Some(NumericDType::F64),
            _ => None,
        };
        if let Some(input_dtype) = input_dtype {
            real_dtype = match real_dtype {
                None => Some(input_dtype),
                Some(previous) if previous == input_dtype => Some(previous),
                Some(_) => Some(NumericDType::F64),
            };
        }
    }
    Ok(DenseClass::Real(real_dtype.unwrap_or(NumericDType::F64)))
}

fn pair(value: &Value, index: usize) -> BuiltinResult<(NumericScalar, NumericScalar)> {
    let zero = NumericScalar::F64(0.0);
    match value {
        Value::Tensor(tensor) => tensor.numeric_value_at(index).map(|real| (real, zero)),
        Value::ComplexTensor(tensor) => tensor.numeric_value_at(index),
        Value::LogicalArray(array) => array
            .data
            .get(index)
            .map(|&real| (NumericScalar::F64(f64::from(real)), zero)),
        Value::Num(real) if index == 0 => Some((NumericScalar::F64(*real), zero)),
        Value::Int(real) if index == 0 => Some((NumericScalar::from(real.clone()), zero)),
        Value::Bool(real) if index == 0 => {
            Some((NumericScalar::F64(f64::from(u8::from(*real))), zero))
        }
        Value::Complex(real, imag) if index == 0 => {
            Some((NumericScalar::F64(*real), NumericScalar::F64(*imag)))
        }
        Value::SparseTensor(sparse) => {
            let row = index % sparse.rows;
            let col = index / sparse.rows;
            if let Some(value) = sparse.complex_at(row, col) {
                Some((NumericScalar::F64(value.0), NumericScalar::F64(value.1)))
            } else if sparse.is_complex() {
                Some((zero, zero))
            } else {
                sparse
                    .real_at(row, col)
                    .ok()
                    .map(|value| (NumericScalar::F64(value.unwrap_or(0.0)), zero))
            }
        }
        _ => None,
    }
    .ok_or_else(|| {
        error_with_detail(
            &BLKDIAG_ERROR_INVALID_INPUT,
            "block payload does not match its shape",
        )
    })
}

enum Output {
    Real(ExactHostTensorAllocation, NumericDType),
    Logical(ExactHostLogicalTensorAllocation),
    Complex(ExactHostComplexTensorAllocation),
    Paired(ExactHostIntegerComplexTensorAllocation),
}

impl Output {
    fn set(&mut self, index: usize, real: NumericScalar, imag: NumericScalar) -> BuiltinResult<()> {
        match self {
            Self::Real(output, dtype) => {
                let value = if let Some(class) = dtype.integer_class() {
                    let cast = match real.into_int_value() {
                        Some(value) => class.cast_int(&value),
                        None => class.cast_scalar(real.materialize_f64()),
                    };
                    NumericScalar::from(cast)
                } else if *dtype == NumericDType::F32 {
                    NumericScalar::F32(real.materialize_f64() as f32)
                } else {
                    NumericScalar::F64(real.materialize_f64())
                };
                output
                    .set_value(index, value)
                    .map_err(|error| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, error))
            }
            Self::Logical(output) => {
                output.as_mut_slice()[index] = u8::from(real.materialize_f64() != 0.0);
                Ok(())
            }
            Self::Complex(output) => output
                .set(index, real.materialize_f64(), imag.materialize_f64())
                .map_err(|error| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, error)),
            Self::Paired(output) => output
                .set(index, real, imag)
                .map_err(|error| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, error)),
        }
    }

    fn into_value(self) -> BuiltinResult<Value> {
        match self {
            Self::Real(output, dtype) => output
                .into_tensor()
                .map(|tensor| {
                    if dtype.integer_class().is_some() {
                        Value::Tensor(tensor)
                    } else {
                        tensor_into_blkdiag_value(tensor)
                    }
                })
                .map_err(|error| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, error)),
            Self::Logical(output) => output
                .into_array()
                .map(Value::LogicalArray)
                .map_err(|error| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, error)),
            Self::Complex(output) => output
                .into_tensor()
                .map(complex_tensor_into_blkdiag_value)
                .map_err(|error| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, error)),
            Self::Paired(output) => output
                .into_tensor()
                .map(Value::ComplexTensor)
                .map_err(|error| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, error)),
        }
    }
}

pub(super) fn dense(
    context: &RuntimeContext,
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let class = class(values)?;
    let mut rows = 0usize;
    let mut cols = 0usize;
    for value in values {
        let (height, width) = shape(value)?;
        rows = rows.checked_add(height).ok_or_else(size_overflow)?;
        cols = cols.checked_add(width).ok_or_else(size_overflow)?;
        if !matches!(value, Value::SparseTensor(_))
            && height.checked_mul(width).ok_or_else(size_overflow)?
                != match value {
                    Value::Tensor(tensor) => tensor.len(),
                    Value::ComplexTensor(tensor) => tensor.len(),
                    Value::LogicalArray(array) => array.data.len(),
                    _ => 1,
                }
        {
            return Err(error_with_detail(
                &BLKDIAG_ERROR_INVALID_INPUT,
                "block payload does not match its shape",
            ));
        }
    }
    let len = rows.checked_mul(cols).ok_or_else(size_overflow)?;
    let (raw, claims) = match class {
        DenseClass::Real(dtype) => (
            ExactHostTensorAllocation::requested_bytes_for_dimensions(dtype, len, 2),
            ExactHostTensorAllocation::retained_claim_count(),
        ),
        DenseClass::Logical => (
            ExactHostLogicalTensorAllocation::requested_bytes_for_dimensions(len, 2),
            ExactHostLogicalTensorAllocation::retained_claim_count(),
        ),
        DenseClass::Complex(dtype) => (
            ExactHostComplexTensorAllocation::requested_bytes_for_dimensions(dtype, len, 2),
            ExactHostComplexTensorAllocation::retained_claim_count(),
        ),
        DenseClass::Paired(dtype) => (
            ExactHostIntegerComplexTensorAllocation::requested_bytes_for_dimensions(dtype, len, 2),
            ExactHostIntegerComplexTensorAllocation::retained_claim_count(),
        ),
    };
    let raw = raw.map_err(|cause| admission(context, HostAllocationError::Numeric(cause)))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| admission(context, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|cause| admission(context, cause))?;
    let receipt = HostAllocationReceipt::admit(
        Some(context),
        HostAllocationPlan::new(0, retained).map_err(|cause| admission(context, cause))?,
    )
    .map_err(|cause| admission(context, cause))?
    .ok_or_else(|| admission(context, HostAllocationError::MissingManagedOwner))?;
    let output_shape = [rows, cols];
    let mut output = receipt
        .with_output_allocation(context, |permit| match class {
            DenseClass::Real(dtype) => ExactHostTensorAllocation::new(dtype, &output_shape, permit)
                .map(|output| Output::Real(output, dtype)),
            DenseClass::Logical => {
                ExactHostLogicalTensorAllocation::new(&output_shape, permit).map(Output::Logical)
            }
            DenseClass::Complex(dtype) => {
                ExactHostComplexTensorAllocation::new(dtype, &output_shape, permit)
                    .map(Output::Complex)
            }
            DenseClass::Paired(dtype) => {
                ExactHostIntegerComplexTensorAllocation::new(dtype, &output_shape, permit)
                    .map(Output::Paired)
            }
        })
        .map_err(|cause| admission(context, cause))?;
    let mut row_offset = 0usize;
    let mut col_offset = 0usize;
    let mut counter = 0usize;
    for value in values {
        let (height, width) = shape(value)?;
        for col in 0..width {
            for row in 0..height {
                if counter.is_multiple_of(CHECKPOINT_CHUNK) {
                    receipt
                        .checkpoint(context)
                        .map_err(|cause| admission(context, cause))?;
                }
                let source = row + col * height;
                let destination = (row_offset + row) + (col_offset + col) * rows;
                let (real, imag) = pair(value, source)?;
                output.set(destination, real, imag)?;
                counter += 1;
            }
        }
        row_offset += height;
        col_offset += width;
    }
    receipt
        .checkpoint(context)
        .map_err(|cause| admission(context, cause))?;
    output.into_value()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use runmat_value::{IntegerStorage, SparseTensor, Tensor};
    use std::rc::Rc;

    struct ManagedParallel(ManagedHostEffectLedger);
    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    fn runtime(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
            )
    }

    #[test]
    fn exact_integer_blocks_zero_fill_and_refund() {
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let context = runtime(&ledger);
        let left = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 3]), vec![2, 1]).unwrap(),
        );
        let right = Value::Int(runmat_value::IntValue::U64(7));
        let result = dense(
            &context,
            crate::call::FunctionArgumentView::from_slice(&[left, right]),
        )
        .unwrap();
        let Value::Tensor(tensor) = &result else {
            panic!("integer tensor output")
        };
        assert_eq!(tensor.shape(), &[3, 2]);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 3, 0, 0, 0, 7,]))
        );
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(result);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        let chars = character(
            &context,
            crate::call::FunctionArgumentView::from_slice(&[
                Value::CharArray(CharArray::new_row("ab")),
                Value::CharArray(CharArray::new_row("c")),
            ]),
        )
        .unwrap();
        let Value::CharArray(output) = &chars else {
            panic!("character output")
        };
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(
            runmat_value::decode_utf16_code_units(output.data()).expect("valid UTF-16 fixture"),
            "ab   c"
        );
        let selected = futures::executor::block_on(super::super::blkdiag_builtin(
            Some(runtime(&ledger)),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::CharArray(CharArray::new_row("ab")),
                Value::CharArray(CharArray::new_row("c")),
            ]),
        ))
        .unwrap();
        assert!(
            matches!(&selected, Value::CharArray(chars) if runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture") == "ab   c")
        );
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop((chars, selected));
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

        let denied = ManagedHostEffectLedger::new(0);
        assert!(character(
            &runtime(&denied),
            crate::call::FunctionArgumentView::from_slice(&[Value::CharArray(CharArray::new_row(
                "a"
            ))])
        )
        .is_err());

        assert!(dense(
            &runtime(&denied),
            crate::call::FunctionArgumentView::from_slice(&[Value::Num(1.0)])
        )
        .is_err());
        assert_eq!(denied.snapshot().live_reserved_bytes, 0);

        let sparse_input = Value::SparseTensor(
            SparseTensor::new_f32(2, 1, vec![0, 1], vec![1], vec![4.25]).unwrap(),
        );
        let dense = Value::Tensor(Tensor::from_f32(vec![0.0, 7.5], vec![2, 1]).unwrap());
        let result = futures::executor::block_on(super::super::blkdiag_builtin(
            Some(runtime(&ledger)),
            crate::builtins::common::test_support::builtin_arguments(vec![
                sparse_input.clone(),
                dense.clone(),
            ]),
        ))
        .unwrap();
        let Value::SparseTensor(output) = &result else {
            panic!("sparse output");
        };
        assert_eq!((output.rows, output.cols), (4, 2));
        assert_eq!(&output.col_ptrs[..], &[0, 1, 2]);
        assert_eq!(&output.row_indices[..], &[1, 3]);
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(output.materialize_f64(), vec![4.25, 7.5]);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(result);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert!(sparse(
            &runtime(&denied),
            crate::call::FunctionArgumentView::from_slice(&[sparse_input, dense])
        )
        .is_err());
        assert_eq!(denied.snapshot().live_reserved_bytes, 0);

        let complex_input = Value::SparseTensor(
            SparseTensor::new_complex_f32(2, 1, vec![0, 1], vec![0], vec![(2.0, -3.0)]).unwrap(),
        );
        let result = futures::executor::block_on(super::super::blkdiag_builtin(
            Some(runtime(&ledger)),
            crate::builtins::common::test_support::builtin_arguments(vec![
                complex_input.clone(),
                Value::Num(5.0),
            ]),
        ))
        .unwrap();
        let Value::SparseTensor(output) = &result else {
            panic!("complex sparse output");
        };
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(output.complex_at(0, 0), Some((2.0, -3.0)));
        assert_eq!(output.complex_at(2, 1), Some((5.0, 0.0)));
        drop(result);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        let result = futures::executor::block_on(super::super::blkdiag_builtin(
            Some(runtime(&ledger)),
            crate::builtins::common::test_support::builtin_arguments(vec![
                complex_input,
                Value::Complex(7.0, 1.0),
            ]),
        ))
        .unwrap();
        let Value::ComplexTensor(output) = &result else {
            panic!("dense complex output");
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(output.materialize_f64()[0], (2.0, -3.0));
        assert_eq!(output.materialize_f64()[5], (7.0, 1.0));
        drop(result);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
