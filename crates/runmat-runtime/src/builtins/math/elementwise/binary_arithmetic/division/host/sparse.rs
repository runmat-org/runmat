//! CSC-preserving scalar division. A dense boundary is explicit when an
//! implicit zero would acquire a nonzero or nonfinite value.
use runmat_value::{
    ComplexStorage, ComplexTensor, HostNumericBuffer, IntegerStorage, NumericStorage, SparseTensor,
    Tensor, Value,
};

#[path = "sparse/prune.rs"]
mod prune;

use super::DivisionContext;
use crate::BuiltinResult;

pub(super) fn execute(
    context: DivisionContext,
    numerator: Value,
    denominator: Value,
) -> BuiltinResult<Value> {
    if let Value::SparseTensor(sparse) = &numerator {
        if let Some(divisor) = scalar_divisor(&denominator) {
            // Structural zeros remain unstored when the divisor is nonzero,
            // including a negative or infinite scalar. Zero and NaN divisors
            // make those positions NaN and require a full result.
            if divisor {
                return divide_stored(context, sparse, denominator);
            }
        }
    }
    if let (Some(left), Some(right)) = (operand_shape(&numerator), operand_shape(&denominator)) {
        let shape = crate::builtins::common::broadcast::broadcast_shapes(
            context.identity.name,
            &left,
            &right,
        )
        .map_err(|error| context.error_with_detail(context.size_mismatch, error))?;
        crate::builtins::math::elementwise::sparse::checked_sparse_result_len(
            &shape,
            context.identity.name,
        )?;
    }
    let numerator = densify_operand(context, numerator)?;
    let denominator = densify_operand(context, denominator)?;
    super::execute(context, numerator, denominator)
}

fn operand_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::SparseTensor(sparse) => Some(vec![sparse.rows, sparse.cols]),
        Value::Tensor(tensor) => Some(tensor.shape().to_vec()),
        Value::ComplexTensor(tensor) => Some(tensor.shape().to_vec()),
        Value::LogicalArray(array) => Some(array.shape.to_vec()),
        Value::CharArray(array) => Some(vec![array.rows, array.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Some(vec![1, 1]),
        _ => None,
    }
}

fn scalar_divisor(value: &Value) -> Option<bool> {
    match value {
        Value::Num(value) => Some(*value != 0.0 && !value.is_nan()),
        Value::Int(value) => Some(!value.is_zero()),
        Value::Bool(value) => Some(*value),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            let scalar = crate::builtins::common::tensor::tensor_value_f64(tensor, 0);
            Some(scalar != 0.0 && !scalar.is_nan())
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Some(array.data[0] != 0),
        Value::CharArray(array) if array.data().len() == 1 => Some(array.data()[0].raw() != 0),
        Value::Complex(real, imag) => {
            Some(real.is_finite() && imag.is_finite() && (*real != 0.0 || *imag != 0.0))
        }
        Value::ComplexTensor(tensor) if tensor.len() == 1 => {
            let (real, imag) = tensor.materialize_f64()[0];
            Some(real.is_finite() && imag.is_finite() && (real != 0.0 || imag != 0.0))
        }
        _ => None,
    }
}

fn densify_operand(context: DivisionContext, value: Value) -> BuiltinResult<Value> {
    let Value::SparseTensor(sparse) = value else {
        return Ok(value);
    };
    crate::builtins::math::elementwise::sparse::checked_sparse_result_len(
        &[sparse.rows, sparse.cols],
        context.identity.name,
    )?;
    if sparse.is_complex() {
        sparse.to_dense_complex().map(Value::ComplexTensor)
    } else if sparse.is_logical() {
        sparse.to_dense_logical().map(Value::LogicalArray)
    } else {
        sparse.to_dense().map(Value::Tensor)
    }
    .map_err(|error| context.internal_error(error))
}

fn divide_stored(
    context: DivisionContext,
    sparse: &SparseTensor,
    denominator: Value,
) -> BuiltinResult<Value> {
    let shape = vec![1, sparse.nnz()];
    let stored = if sparse.is_complex() {
        let storage = if let Some(values) = sparse.complex_host_buffer() {
            ComplexStorage::F64(values.clone())
        } else {
            ComplexStorage::F32(
                sparse
                    .complex_f32_host_buffer()
                    .expect("validated complex sparse storage")
                    .clone(),
            )
        };
        Value::ComplexTensor(
            ComplexTensor::from_complex_storage(storage, shape)
                .map_err(|error| context.internal_error(error))?,
        )
    } else {
        let storage = if let Some(values) = sparse.numeric_host_buffer() {
            values.clone().into_numeric_storage()
        } else {
            let mut values = Vec::new();
            values
                .try_reserve_exact(sparse.nnz())
                .map_err(|error| context.internal_error(error))?;
            values.resize(sparse.nnz(), 1.0);
            NumericStorage::F64(values)
        };
        Value::Tensor(
            Tensor::from_numeric_storage(storage, shape)
                .map_err(|error| context.internal_error(error))?,
        )
    };
    let divided = match super::execute(context, stored, denominator)? {
        Value::Num(value) => Value::Tensor(
            Tensor::new(vec![value], vec![1, 1]).map_err(|error| context.internal_error(error))?,
        ),
        Value::Int(value) => Value::Tensor(
            Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|error| context.internal_error(error))?,
        ),
        Value::Complex(real, imag) => Value::ComplexTensor(
            ComplexTensor::new(vec![(real, imag)], vec![1, 1])
                .map_err(|error| context.internal_error(error))?,
        ),
        value => value,
    };
    let pointers = sparse.col_ptrs.clone();
    let rows = sparse.row_indices.clone();
    let output = match divided {
        Value::Tensor(tensor) => SparseTensor::from_host_numeric_buffers(
            sparse.rows,
            sparse.cols,
            pointers,
            rows,
            HostNumericBuffer::from_numeric_storage(
                tensor
                    .into_numeric_storage()
                    .map_err(|error| context.internal_error(error))?,
            ),
        ),
        Value::ComplexTensor(tensor) => match tensor.into_complex_storage() {
            ComplexStorage::F64(values) => SparseTensor::from_host_complex_buffers(
                sparse.rows,
                sparse.cols,
                pointers,
                rows,
                values,
            ),
            ComplexStorage::F32(values) => SparseTensor::from_host_complex_f32_buffers(
                sparse.rows,
                sparse.cols,
                pointers,
                rows,
                values,
            ),
            ComplexStorage::Integer(_) => {
                return Err(context
                    .internal_error("complex integer storage reached sparse floating division"))
            }
        },
        _ => return Err(context.internal_error("scalar division did not return a typed array")),
    }
    .map_err(|error| context.internal_error(error))?;
    Ok(Value::SparseTensor(prune::prune_result_zeros(
        context, output,
    )?))
}
