//! Exact arithmetic for RunMat's typed-integer sparse extension.
//!
//! MATLAB sparse values are floating or logical, but RunMat preserves exact
//! integer buffers when converting an integer tensor to sparse storage. Those
//! buffers must never flow through the legacy f64 sparse arithmetic helpers.

use runmat_builtins::{IntValue, IntegerStorage, SparseTensor, Tensor, Value};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::math::elementwise::integer_arithmetic::{
    integer_binary_scalar, IntegerBinaryOp,
};
use crate::builtins::math::elementwise::sparse::SparseBinaryOp;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const MAX_DENSE_RESULT_ELEMENTS: usize = 10_000_000;

pub(crate) fn try_typed_sparse_integer_binary(
    lhs: &Value,
    rhs: &Value,
    operation: SparseBinaryOp,
    builtin: &'static str,
) -> Option<BuiltinResult<Value>> {
    if !has_typed_sparse(lhs) && !has_typed_sparse(rhs) {
        return None;
    }
    Some(typed_sparse_integer_binary(lhs, rhs, operation, builtin))
}

fn has_typed_sparse(value: &Value) -> bool {
    matches!(value, Value::SparseTensor(sparse) if sparse.integer_storage().is_some())
}

fn typed_sparse_integer_binary(
    lhs: &Value,
    rhs: &Value,
    operation: SparseBinaryOp,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let prototype = typed_sparse_storage(lhs)
        .or_else(|| typed_sparse_storage(rhs))
        .expect("typed sparse input establishes the integer class");
    let lhs = ExactOperand::from_value(lhs, prototype, builtin)?;
    let rhs = ExactOperand::from_value(rhs, prototype, builtin)?;

    if let (ExactOperand::Sparse(lhs), ExactOperand::Sparse(rhs)) = (&lhs, &rhs) {
        if lhs.rows == rhs.rows && lhs.cols == rhs.cols {
            return sparse_sparse_exact(lhs, rhs, prototype, operation, builtin);
        }
    }

    if let Some(result) = sparse_scalar_fast_path(&lhs, &rhs, prototype, operation, builtin)? {
        return Ok(result);
    }
    if let Some(result) = sparse_dense_times_fast_path(&lhs, &rhs, prototype, operation, builtin)? {
        return Ok(result);
    }

    exact_dense_result(&lhs, &rhs, prototype, operation, builtin)
}

fn typed_sparse_storage(value: &Value) -> Option<&IntegerStorage> {
    match value {
        Value::SparseTensor(sparse) => sparse.integer_storage(),
        _ => None,
    }
}

enum ExactOperand<'a> {
    Sparse(&'a SparseTensor),
    Tensor(&'a Tensor),
    ScalarInt(&'a IntValue),
    ScalarReal(&'a Value),
}

impl<'a> ExactOperand<'a> {
    fn from_value(
        value: &'a Value,
        prototype: &IntegerStorage,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        match value {
            Value::SparseTensor(sparse) => {
                let Some(storage) = sparse.integer_storage() else {
                    return Err(unsupported_error(
                        builtin,
                        "typed sparse arithmetic requires exact integer sparse operands",
                    ));
                };
                ensure_same_class(storage, prototype, builtin)?;
                Ok(Self::Sparse(sparse))
            }
            Value::Tensor(tensor) => {
                let Some(storage) = tensor.integer_storage() else {
                    if tensor.data.len() == 1 {
                        return Ok(Self::ScalarReal(value));
                    }
                    return Err(unsupported_error(
                        builtin,
                        "typed sparse arithmetic accepts only scalar double operands or same-class integer arrays",
                    ));
                };
                ensure_same_class(storage, prototype, builtin)?;
                Ok(Self::Tensor(tensor))
            }
            Value::Int(integer) => {
                ensure_same_class_name(integer.class_name(), prototype, builtin)?;
                Ok(Self::ScalarInt(integer))
            }
            Value::Num(_) | Value::Bool(_) => Ok(Self::ScalarReal(value)),
            Value::LogicalArray(logical) if logical.data.len() == 1 => Ok(Self::ScalarReal(value)),
            _ => Err(unsupported_error(
                builtin,
                "typed sparse arithmetic accepts only scalar double or logical values and same-class integer arrays",
            )),
        }
    }

    fn value_at(&self, index: usize, prototype: &IntegerStorage) -> BuiltinResult<Value> {
        match self {
            Self::Sparse(sparse) => Ok(Value::Int(sparse_value(sparse, index, prototype)?)),
            Self::Tensor(tensor) => Ok(Value::Int(
                tensor
                    .integer_storage()
                    .and_then(|storage| storage.value_at(index))
                    .ok_or_else(|| {
                        internal_error("typed sparse operand storage is inconsistent")
                    })?,
            )),
            Self::ScalarInt(value) => Ok(Value::Int((*value).clone())),
            Self::ScalarReal(value) => Ok((*value).clone()),
        }
    }

    fn sparse(&self) -> Option<&'a SparseTensor> {
        match self {
            Self::Sparse(sparse) => Some(sparse),
            _ => None,
        }
    }
}

fn ensure_same_class(
    storage: &IntegerStorage,
    prototype: &IntegerStorage,
    builtin: &'static str,
) -> BuiltinResult<()> {
    ensure_same_class_name(storage.class_name(), prototype, builtin)
}

fn ensure_same_class_name(
    class_name: &str,
    prototype: &IntegerStorage,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if class_name == prototype.class_name() {
        Ok(())
    } else {
        Err(unsupported_error(
            builtin,
            "typed sparse integer operands must have the same integer class",
        ))
    }
}

fn sparse_sparse_exact(
    lhs: &SparseTensor,
    rhs: &SparseTensor,
    prototype: &IntegerStorage,
    operation: SparseBinaryOp,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let lhs_values = lhs
        .integer_storage()
        .ok_or_else(|| internal_error("typed sparse lhs storage is missing"))?;
    let rhs_values = rhs
        .integer_storage()
        .ok_or_else(|| internal_error("typed sparse rhs storage is missing"))?;
    let mut col_ptrs = Vec::with_capacity(lhs.cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);

    for col in 0..lhs.cols {
        let mut left = lhs.col_ptrs[col];
        let left_end = lhs.col_ptrs[col + 1];
        let mut right = rhs.col_ptrs[col];
        let right_end = rhs.col_ptrs[col + 1];
        while left < left_end || right < right_end {
            if matches!(operation, SparseBinaryOp::Mul) && (left >= left_end || right >= right_end)
            {
                break;
            }
            let (row, left_value, right_value) = if right >= right_end
                || (left < left_end && lhs.row_indices[left] < rhs.row_indices[right])
            {
                if matches!(operation, SparseBinaryOp::Mul) {
                    left += 1;
                    continue;
                }
                let row = lhs.row_indices[left];
                let value = lhs_values
                    .value_at(left)
                    .ok_or_else(|| internal_error("typed sparse lhs storage is inconsistent"))?;
                left += 1;
                (row, value, zero_value(prototype))
            } else if left >= left_end || rhs.row_indices[right] < lhs.row_indices[left] {
                if matches!(operation, SparseBinaryOp::Mul) {
                    right += 1;
                    continue;
                }
                let row = rhs.row_indices[right];
                let value = rhs_values
                    .value_at(right)
                    .ok_or_else(|| internal_error("typed sparse rhs storage is inconsistent"))?;
                right += 1;
                (row, zero_value(prototype), value)
            } else {
                let row = lhs.row_indices[left];
                let left_value = lhs_values
                    .value_at(left)
                    .ok_or_else(|| internal_error("typed sparse lhs storage is inconsistent"))?;
                let right_value = rhs_values
                    .value_at(right)
                    .ok_or_else(|| internal_error("typed sparse rhs storage is inconsistent"))?;
                left += 1;
                right += 1;
                (row, left_value, right_value)
            };
            let value = apply_exact(left_value, right_value, operation, builtin)?;
            if !value.is_zero() {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new_integer_like(lhs.rows, lhs.cols, col_ptrs, row_indices, values, prototype)
        .map(Value::SparseTensor)
        .map_err(internal_error)
}

fn sparse_scalar_fast_path(
    lhs: &ExactOperand<'_>,
    rhs: &ExactOperand<'_>,
    prototype: &IntegerStorage,
    operation: SparseBinaryOp,
    builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    let (sparse, scalar, sparse_is_lhs) = match (lhs.sparse(), rhs.sparse()) {
        (Some(sparse), None)
            if matches!(
                rhs,
                ExactOperand::ScalarInt(_) | ExactOperand::ScalarReal(_)
            ) =>
        {
            (sparse, rhs, true)
        }
        (None, Some(sparse))
            if matches!(
                lhs,
                ExactOperand::ScalarInt(_) | ExactOperand::ScalarReal(_)
            ) =>
        {
            (sparse, lhs, false)
        }
        _ => return Ok(None),
    };
    let scalar_value = scalar.value_at(0, prototype)?;
    let scalar_is_zero = matches!(&scalar_value, Value::Int(value) if value.is_zero())
        || matches!(&scalar_value, Value::Num(value) if *value == 0.0)
        || matches!(&scalar_value, Value::Bool(false));

    let preserve_sparse = match operation {
        SparseBinaryOp::Mul => true,
        SparseBinaryOp::Add => scalar_is_zero,
        SparseBinaryOp::Sub => scalar_is_zero,
    };
    if !preserve_sparse {
        return Ok(None);
    }

    let storage = sparse
        .integer_storage()
        .ok_or_else(|| internal_error("typed sparse storage is missing"))?;
    let mut col_ptrs = Vec::with_capacity(sparse.cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..sparse.cols {
        for entry in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let value = storage
                .value_at(entry)
                .ok_or_else(|| internal_error("typed sparse storage is inconsistent"))?;
            let result = if sparse_is_lhs {
                apply_value_pair(Value::Int(value), scalar_value.clone(), operation, builtin)?
            } else {
                apply_value_pair(scalar_value.clone(), Value::Int(value), operation, builtin)?
            };
            if !result.is_zero() {
                row_indices.push(sparse.row_indices[entry]);
                values.push(result);
            }
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new_integer_like(
        sparse.rows,
        sparse.cols,
        col_ptrs,
        row_indices,
        values,
        prototype,
    )
    .map(Value::SparseTensor)
    .map(Some)
    .map_err(internal_error)
}

fn sparse_dense_times_fast_path(
    lhs: &ExactOperand<'_>,
    rhs: &ExactOperand<'_>,
    prototype: &IntegerStorage,
    operation: SparseBinaryOp,
    builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    if !matches!(operation, SparseBinaryOp::Mul) {
        return Ok(None);
    }
    let (sparse, dense, sparse_is_lhs) = match (lhs.sparse(), rhs.sparse()) {
        (Some(sparse), None) => (sparse, rhs, true),
        (None, Some(sparse)) => (sparse, lhs, false),
        _ => return Ok(None),
    };
    let ExactOperand::Tensor(dense) = dense else {
        return Ok(None);
    };
    if dense.shape != sparse.shape() {
        return Ok(None);
    }
    let sparse_values = sparse
        .integer_storage()
        .ok_or_else(|| internal_error("typed sparse storage is missing"))?;
    let dense_values = dense
        .integer_storage()
        .ok_or_else(|| internal_error("typed dense storage is missing"))?;
    let mut col_ptrs = Vec::with_capacity(sparse.cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..sparse.cols {
        for entry in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let row = sparse.row_indices[entry];
            let sparse_value = sparse_values
                .value_at(entry)
                .ok_or_else(|| internal_error("typed sparse storage is inconsistent"))?;
            let dense_value = dense_values
                .value_at(row + col * sparse.rows)
                .ok_or_else(|| internal_error("typed dense storage is inconsistent"))?;
            let result = if sparse_is_lhs {
                apply_exact(sparse_value, dense_value, operation, builtin)?
            } else {
                apply_exact(dense_value, sparse_value, operation, builtin)?
            };
            if !result.is_zero() {
                row_indices.push(row);
                values.push(result);
            }
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new_integer_like(
        sparse.rows,
        sparse.cols,
        col_ptrs,
        row_indices,
        values,
        prototype,
    )
    .map(Value::SparseTensor)
    .map(Some)
    .map_err(internal_error)
}

fn exact_dense_result(
    lhs: &ExactOperand<'_>,
    rhs: &ExactOperand<'_>,
    prototype: &IntegerStorage,
    operation: SparseBinaryOp,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let lhs_shape = operand_shape(lhs);
    let rhs_shape = operand_shape(rhs);
    let plan =
        BroadcastPlan::new(&lhs_shape, &rhs_shape).map_err(|error| size_error(builtin, error))?;
    if plan.len() > MAX_DENSE_RESULT_ELEMENTS {
        return Err(unsupported_error(
            builtin,
            format!(
                "typed sparse arithmetic would materialize {} elements; limit is {MAX_DENSE_RESULT_ELEMENTS}",
                plan.len()
            ),
        ));
    }
    let mut values = Vec::with_capacity(plan.len());
    for (_, lhs_index, rhs_index) in plan.iter() {
        values.push(apply_value_pair(
            lhs.value_at(lhs_index, prototype)?,
            rhs.value_at(rhs_index, prototype)?,
            operation,
            builtin,
        )?);
    }
    Tensor::new_integer(
        prototype
            .from_same_class_values(values)
            .map_err(internal_error)?,
        plan.output_shape().to_vec(),
    )
    .map(Value::Tensor)
    .map_err(internal_error)
}

fn operand_shape(operand: &ExactOperand<'_>) -> Vec<usize> {
    match operand {
        ExactOperand::Sparse(sparse) => sparse.shape(),
        ExactOperand::Tensor(tensor) => tensor.shape.clone(),
        ExactOperand::ScalarInt(_) | ExactOperand::ScalarReal(_) => vec![1, 1],
    }
}

fn sparse_value(
    sparse: &SparseTensor,
    linear: usize,
    prototype: &IntegerStorage,
) -> BuiltinResult<IntValue> {
    let row = linear % sparse.rows;
    let col = linear / sparse.rows;
    Ok(sparse
        .integer_at(row, col)
        .unwrap_or_else(|| zero_value(prototype)))
}

fn zero_value(prototype: &IntegerStorage) -> IntValue {
    prototype
        .zeros_like(1)
        .value_at(0)
        .expect("one typed integer zero")
}

fn apply_value_pair(
    lhs: Value,
    rhs: Value,
    operation: SparseBinaryOp,
    builtin: &'static str,
) -> BuiltinResult<IntValue> {
    integer_binary_scalar(&lhs, &rhs, integer_operation(operation), builtin).map_err(internal_error)
}

fn apply_exact(
    lhs: IntValue,
    rhs: IntValue,
    operation: SparseBinaryOp,
    builtin: &'static str,
) -> BuiltinResult<IntValue> {
    apply_value_pair(Value::Int(lhs), Value::Int(rhs), operation, builtin)
}

fn integer_operation(operation: SparseBinaryOp) -> IntegerBinaryOp {
    match operation {
        SparseBinaryOp::Add => IntegerBinaryOp::Add,
        SparseBinaryOp::Sub => IntegerBinaryOp::Subtract,
        SparseBinaryOp::Mul => IntegerBinaryOp::Multiply,
    }
}

fn size_error(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    build_runtime_error(format!(
        "{builtin}: sparse operand sizes are not compatible: {}",
        detail.into()
    ))
    .with_builtin(builtin)
    .with_identifier(format!("RunMat:{builtin}:SparseSizeMismatch"))
    .build()
}

fn unsupported_error(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    build_runtime_error(format!("{builtin}: {}", detail.into()))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:SparseUnsupportedOperand"))
        .build()
}

fn internal_error(detail: impl Into<String>) -> RuntimeError {
    build_runtime_error(detail.into()).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed_sparse(storage: IntegerStorage) -> SparseTensor {
        SparseTensor::new_integer(2, 2, vec![0, 1, 2], vec![0, 1], storage).unwrap()
    }

    #[test]
    fn same_class_sparse_union_preserves_uint64_exact_values() {
        let lhs = typed_sparse(IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]));
        let rhs = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![1, 0],
            IntegerStorage::U64(vec![7, 1]),
        )
        .unwrap();
        let Some(result) = try_typed_sparse_integer_binary(
            &Value::SparseTensor(lhs),
            &Value::SparseTensor(rhs),
            SparseBinaryOp::Add,
            "plus",
        ) else {
            panic!("typed sparse route");
        };
        let Value::SparseTensor(result) = result.unwrap() else {
            panic!("typed sparse output");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                7,
                1,
                u64::MAX
            ]))
        );
    }

    #[test]
    fn typed_sparse_times_typed_dense_keeps_sparse_exact_storage() {
        let sparse = typed_sparse(IntegerStorage::I64(vec![i64::MIN, i64::MAX]));
        let dense =
            Tensor::new_integer(IntegerStorage::I64(vec![1, 3, -2, 2]), vec![2, 2]).unwrap();
        let Some(result) = try_typed_sparse_integer_binary(
            &Value::SparseTensor(sparse),
            &Value::Tensor(dense),
            SparseBinaryOp::Mul,
            "times",
        ) else {
            panic!("typed sparse route");
        };
        let Value::SparseTensor(result) = result.unwrap() else {
            panic!("typed sparse output");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))
        );
    }

    #[test]
    fn typed_sparse_add_nonzero_scalar_materializes_exact_dense_tensor() {
        let sparse = typed_sparse(IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]));
        let Some(result) = try_typed_sparse_integer_binary(
            &Value::SparseTensor(sparse),
            &Value::Int(IntValue::U64(1)),
            SparseBinaryOp::Add,
            "plus",
        ) else {
            panic!("typed sparse route");
        };
        let Value::Tensor(result) = result.unwrap() else {
            panic!("dense exact output");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_994,
                1,
                1,
                u64::MAX
            ]))
        );
    }

    #[test]
    fn typed_sparse_mixed_classes_reject_before_f64_conversion() {
        let sparse = typed_sparse(IntegerStorage::U64(vec![1, 2]));
        let err = try_typed_sparse_integer_binary(
            &Value::SparseTensor(sparse),
            &Value::Int(IntValue::U32(1)),
            SparseBinaryOp::Add,
            "plus",
        )
        .expect("typed sparse route")
        .expect_err("mixed integer classes");
        assert_eq!(
            err.identifier(),
            Some("RunMat:plus:SparseUnsupportedOperand")
        );
    }
}
