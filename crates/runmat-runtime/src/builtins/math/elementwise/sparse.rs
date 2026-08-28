use runmat_value::{
    CharArray, ComplexTensor, LogicalArray, NumericDType, SparseTensor, Tensor, Value,
};

use crate::builtins::common::broadcast::{broadcast_shapes, BroadcastPlan};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const MAX_SPARSE_FULL_RESULT_ELEMENTS: usize = 10_000_000;

#[derive(Clone, Copy)]
pub(crate) enum SparseBinaryOp {
    Add,
    Sub,
    Mul,
}

pub(crate) fn try_sparse_binary(
    lhs: &Value,
    rhs: &Value,
    op: SparseBinaryOp,
    builtin: &'static str,
) -> Option<BuiltinResult<Value>> {
    if !matches!(lhs, Value::SparseTensor(_)) && !matches!(rhs, Value::SparseTensor(_)) {
        return None;
    }
    Some(sparse_binary(lhs, rhs, op, builtin))
}

fn sparse_binary(
    lhs: &Value,
    rhs: &Value,
    op: SparseBinaryOp,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match (lhs, rhs) {
        (Value::SparseTensor(a), Value::SparseTensor(b)) => sparse_sparse(a, b, op, builtin),
        (Value::SparseTensor(sparse), other) => sparse_other(sparse, other, op, true, builtin),
        (other, Value::SparseTensor(sparse)) => sparse_other(sparse, other, op, false, builtin),
        _ => unreachable!("caller only invokes sparse_binary for sparse operands"),
    }
}

fn sparse_error(
    builtin: &'static str,
    suffix: &'static str,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:Sparse{suffix}"))
        .build()
}

fn shape_error(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    sparse_error(builtin, "SizeMismatch", detail)
}

fn unsupported_error(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    sparse_error(builtin, "UnsupportedOperand", detail)
}

fn densify_error(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    sparse_error(builtin, "DensifyTooLarge", detail)
}

fn map_internal_error(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    sparse_error(builtin, "Internal", detail)
}

fn checked_len(shape: &[usize], builtin: &'static str) -> BuiltinResult<usize> {
    let mut len = 1usize;
    for &dim in shape {
        len = len.checked_mul(dim).ok_or_else(|| {
            densify_error(
                builtin,
                format!("sparse arithmetic result shape {shape:?} overflows usize"),
            )
        })?;
    }
    if len > MAX_SPARSE_FULL_RESULT_ELEMENTS {
        return Err(densify_error(
            builtin,
            format!(
                "sparse arithmetic would materialize {len} elements; limit is {MAX_SPARSE_FULL_RESULT_ELEMENTS}"
            ),
        ));
    }
    Ok(len)
}

/// Validate a sparse operation which may need to visit or materialize every
/// element of its result. Kept here so sparse-aware builtins share the same
/// resource bound and error contract.
pub(crate) fn checked_sparse_result_len(
    shape: &[usize],
    builtin: &'static str,
) -> BuiltinResult<usize> {
    checked_len(shape, builtin)
}

fn sparse_shape(sparse: &SparseTensor) -> [usize; 2] {
    [sparse.rows, sparse.cols]
}

fn checked_broadcast_shape(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    builtin: &'static str,
    label: &'static str,
) -> BuiltinResult<Vec<usize>> {
    broadcast_shapes(builtin, lhs_shape, rhs_shape)
        .map_err(|err| shape_error(builtin, format!("{label} are not compatible: {err}")))
}

fn is_sparse_scalar(sparse: &SparseTensor) -> bool {
    sparse.rows == 1 && sparse.cols == 1
}

#[derive(Clone, Copy)]
struct RealScalar {
    value: f64,
    dtype: NumericDType,
}

fn floating_dtype(dtype: NumericDType) -> NumericDType {
    if dtype == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn sparse_floating_dtype(dtype: Option<NumericDType>) -> NumericDType {
    dtype.map(floating_dtype).unwrap_or(NumericDType::F64)
}

fn combined_floating_dtype(lhs: NumericDType, rhs: NumericDType) -> NumericDType {
    if lhs == NumericDType::F32 || rhs == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn sparse_from_f64_values(
    rows: usize,
    cols: usize,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<f64>,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let sparse = match dtype {
        NumericDType::F32 => SparseTensor::new_f32(
            rows,
            cols,
            col_ptrs,
            row_indices,
            values.into_iter().map(|value| value as f32).collect(),
        ),
        NumericDType::F64 => SparseTensor::new(rows, cols, col_ptrs, row_indices, values),
        _ => unreachable!("floating sparse arithmetic requested an integer output dtype"),
    }
    .map_err(|err| map_internal_error(builtin, err))?;
    Ok(Value::SparseTensor(sparse))
}

fn sparse_zeros_with_dtype(rows: usize, cols: usize, dtype: NumericDType) -> SparseTensor {
    match dtype {
        NumericDType::F32 => SparseTensor::zeros_f32(rows, cols),
        NumericDType::F64 => SparseTensor::zeros(rows, cols),
        _ => unreachable!("floating sparse arithmetic requested an integer output dtype"),
    }
}

fn preserve_sparse_with_dtype(
    sparse: &SparseTensor,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    if sparse_floating_dtype(sparse.numeric_dtype()) == dtype && !sparse.is_logical() {
        return Ok(Value::SparseTensor(sparse.clone()));
    }
    sparse_from_f64_values(
        sparse.rows,
        sparse.cols,
        sparse.col_ptrs.clone(),
        sparse.row_indices.clone(),
        sparse.materialize_f64(),
        dtype,
        builtin,
    )
}

fn dense_from_f64_values(
    values: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let tensor = match dtype {
        NumericDType::F32 => Tensor::from_f32(
            values.into_iter().map(|value| value as f32).collect(),
            shape,
        ),
        NumericDType::F64 => Tensor::new(values, shape),
        _ => unreachable!("floating sparse arithmetic requested an integer output dtype"),
    }
    .map_err(|err| map_internal_error(builtin, err))?;
    Ok(Value::Tensor(tensor))
}

fn sparse_scalar_value(sparse: &SparseTensor) -> f64 {
    sparse.get(0, 0).unwrap_or(0.0)
}

fn scalar_real(value: &Value) -> Option<RealScalar> {
    match value {
        Value::Num(n) => Some(RealScalar {
            value: *n,
            dtype: NumericDType::F64,
        }),
        Value::Int(i) => Some(RealScalar {
            value: i.to_f64(),
            dtype: NumericDType::F64,
        }),
        Value::Bool(b) => Some(RealScalar {
            value: if *b { 1.0 } else { 0.0 },
            dtype: NumericDType::F64,
        }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Some(RealScalar {
            value: tensor::tensor_value_f64(t, 0),
            dtype: floating_dtype(t.numeric_dtype()),
        }),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(RealScalar {
            value: if l.data[0] != 0 { 1.0 } else { 0.0 },
            dtype: NumericDType::F64,
        }),
        Value::CharArray(ca) if ca.rows.checked_mul(ca.cols) == Some(1) => Some(RealScalar {
            value: ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0),
            dtype: NumericDType::F64,
        }),
        _ => None,
    }
}

fn dense_tensor(value: &Value, builtin: &'static str) -> BuiltinResult<Option<Tensor>> {
    match value {
        Value::Tensor(t) => Ok(Some(t.clone())),
        Value::Num(n) => Tensor::new(vec![*n], vec![1, 1])
            .map(Some)
            .map_err(|err| map_internal_error(builtin, err)),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1])
            .map(Some)
            .map_err(|err| map_internal_error(builtin, err)),
        Value::Bool(b) => Tensor::new(vec![if *b { 1.0 } else { 0.0 }], vec![1, 1])
            .map(Some)
            .map_err(|err| map_internal_error(builtin, err)),
        Value::LogicalArray(logical) => logical_to_tensor(logical, builtin).map(Some),
        Value::CharArray(chars) => char_to_tensor(chars, builtin).map(Some),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(None),
        Value::GpuTensor(_) => Err(unsupported_error(
            builtin,
            "sparse arithmetic with gpuArray operands is not yet supported",
        )),
        _ => Ok(None),
    }
}

fn complex_tensor(value: &Value, builtin: &'static str) -> BuiltinResult<Option<ComplexTensor>> {
    match value {
        Value::Complex(re, im) => ComplexTensor::new(vec![(*re, *im)], vec![1, 1])
            .map(Some)
            .map_err(|err| map_internal_error(builtin, err)),
        Value::ComplexTensor(tensor) => Ok(Some(tensor.clone())),
        _ => Ok(None),
    }
}

fn logical_to_tensor(logical: &LogicalArray, builtin: &'static str) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = logical
        .data
        .iter()
        .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, logical.shape.clone()).map_err(|err| map_internal_error(builtin, err))
}

fn char_to_tensor(chars: &CharArray, builtin: &'static str) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|err| map_internal_error(builtin, err))
}

fn sparse_sparse(
    lhs: &SparseTensor,
    rhs: &SparseTensor,
    op: SparseBinaryOp,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let dtype = combined_floating_dtype(
        sparse_floating_dtype(lhs.numeric_dtype()),
        sparse_floating_dtype(rhs.numeric_dtype()),
    );
    if lhs.rows == rhs.rows && lhs.cols == rhs.cols {
        return match op {
            SparseBinaryOp::Add => sparse_sparse_union(lhs, rhs, |a, b| a + b, dtype, builtin),
            SparseBinaryOp::Sub => sparse_sparse_union(lhs, rhs, |a, b| a - b, dtype, builtin),
            SparseBinaryOp::Mul
                if sparse_stored_values_are_finite(lhs) && sparse_stored_values_are_finite(rhs) =>
            {
                sparse_sparse_intersection(lhs, rhs, |a, b| a * b, dtype, builtin)
            }
            SparseBinaryOp::Mul => sparse_sparse_union(lhs, rhs, |a, b| a * b, dtype, builtin),
        };
    }
    if is_sparse_scalar(lhs) {
        return sparse_scalar_sparse(sparse_scalar_value(lhs), rhs, op, dtype, builtin);
    }
    if is_sparse_scalar(rhs) {
        return sparse_sparse_scalar(lhs, sparse_scalar_value(rhs), op, dtype, builtin);
    }
    sparse_sparse_broadcast(lhs, rhs, op, dtype, builtin)
}

fn sparse_scalar_sparse(
    scalar: f64,
    sparse: &SparseTensor,
    op: SparseBinaryOp,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match op {
        SparseBinaryOp::Add => {
            if scalar == 0.0 {
                preserve_sparse_with_dtype(sparse, dtype, builtin)
            } else {
                sparse_sparse_scalar_full(
                    sparse,
                    scalar,
                    |sparse_value, scalar| scalar + sparse_value,
                    dtype,
                    builtin,
                )
            }
        }
        SparseBinaryOp::Sub => {
            if scalar == 0.0 {
                scale_sparse(sparse, -1.0, dtype, builtin)
            } else {
                sparse_sparse_scalar_full(
                    sparse,
                    scalar,
                    |sparse_value, scalar| scalar - sparse_value,
                    dtype,
                    builtin,
                )
            }
        }
        SparseBinaryOp::Mul if scalar.is_finite() || !sparse_has_implicit_zeros(sparse) => {
            scale_sparse(sparse, scalar, dtype, builtin)
        }
        SparseBinaryOp::Mul => sparse_sparse_scalar_full(
            sparse,
            scalar,
            |sparse_value, scalar| scalar * sparse_value,
            dtype,
            builtin,
        ),
    }
}

fn sparse_sparse_scalar(
    sparse: &SparseTensor,
    scalar: f64,
    op: SparseBinaryOp,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match op {
        SparseBinaryOp::Add => {
            if scalar == 0.0 {
                preserve_sparse_with_dtype(sparse, dtype, builtin)
            } else {
                sparse_sparse_scalar_full(
                    sparse,
                    scalar,
                    |sparse_value, scalar| sparse_value + scalar,
                    dtype,
                    builtin,
                )
            }
        }
        SparseBinaryOp::Sub => {
            if scalar == 0.0 {
                preserve_sparse_with_dtype(sparse, dtype, builtin)
            } else {
                sparse_sparse_scalar_full(
                    sparse,
                    scalar,
                    |sparse_value, scalar| sparse_value - scalar,
                    dtype,
                    builtin,
                )
            }
        }
        SparseBinaryOp::Mul if scalar.is_finite() || !sparse_has_implicit_zeros(sparse) => {
            scale_sparse(sparse, scalar, dtype, builtin)
        }
        SparseBinaryOp::Mul => sparse_sparse_scalar_full(
            sparse,
            scalar,
            |sparse_value, scalar| sparse_value * scalar,
            dtype,
            builtin,
        ),
    }
}

fn sparse_sparse_scalar_full(
    sparse: &SparseTensor,
    scalar: f64,
    combine: impl Fn(f64, f64) -> f64,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    checked_len(&sparse.shape(), builtin)?;
    let mut col_ptrs = Vec::with_capacity(sparse.cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..sparse.cols {
        for row in 0..sparse.rows {
            let value = combine(sparse.get(row, col).unwrap_or(0.0), scalar);
            if value != 0.0 {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    sparse_from_f64_values(
        sparse.rows,
        sparse.cols,
        col_ptrs,
        row_indices,
        values,
        dtype,
        builtin,
    )
}

fn sparse_sparse_union(
    lhs: &SparseTensor,
    rhs: &SparseTensor,
    combine: impl Fn(f64, f64) -> f64,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let lhs_values = lhs.materialize_f64();
    let rhs_values = rhs.materialize_f64();
    let mut col_ptrs = Vec::with_capacity(lhs.cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..lhs.cols {
        let mut l = lhs.col_ptrs[col];
        let l_end = lhs.col_ptrs[col + 1];
        let mut r = rhs.col_ptrs[col];
        let r_end = rhs.col_ptrs[col + 1];
        while l < l_end || r < r_end {
            let (row, value) =
                if r >= r_end || (l < l_end && lhs.row_indices[l] < rhs.row_indices[r]) {
                    let row = lhs.row_indices[l];
                    let value = combine(lhs_values[l], 0.0);
                    l += 1;
                    (row, value)
                } else if l >= l_end || rhs.row_indices[r] < lhs.row_indices[l] {
                    let row = rhs.row_indices[r];
                    let value = combine(0.0, rhs_values[r]);
                    r += 1;
                    (row, value)
                } else {
                    let row = lhs.row_indices[l];
                    let value = combine(lhs_values[l], rhs_values[r]);
                    l += 1;
                    r += 1;
                    (row, value)
                };
            if value != 0.0 {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    sparse_from_f64_values(
        lhs.rows,
        lhs.cols,
        col_ptrs,
        row_indices,
        values,
        dtype,
        builtin,
    )
}

fn sparse_sparse_intersection(
    lhs: &SparseTensor,
    rhs: &SparseTensor,
    combine: impl Fn(f64, f64) -> f64,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let lhs_values = lhs.materialize_f64();
    let rhs_values = rhs.materialize_f64();
    let mut col_ptrs = Vec::with_capacity(lhs.cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..lhs.cols {
        let mut l = lhs.col_ptrs[col];
        let l_end = lhs.col_ptrs[col + 1];
        let mut r = rhs.col_ptrs[col];
        let r_end = rhs.col_ptrs[col + 1];
        while l < l_end && r < r_end {
            if lhs.row_indices[l] < rhs.row_indices[r] {
                l += 1;
            } else if rhs.row_indices[r] < lhs.row_indices[l] {
                r += 1;
            } else {
                let value = combine(lhs_values[l], rhs_values[r]);
                if value != 0.0 {
                    row_indices.push(lhs.row_indices[l]);
                    values.push(value);
                }
                l += 1;
                r += 1;
            }
        }
        col_ptrs.push(values.len());
    }
    sparse_from_f64_values(
        lhs.rows,
        lhs.cols,
        col_ptrs,
        row_indices,
        values,
        dtype,
        builtin,
    )
}

fn sparse_sparse_broadcast(
    lhs: &SparseTensor,
    rhs: &SparseTensor,
    op: SparseBinaryOp,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let left_shape = sparse_shape(lhs);
    let right_shape = sparse_shape(rhs);
    let output_shape =
        checked_broadcast_shape(&left_shape, &right_shape, builtin, "sparse operands")?;
    checked_len(&output_shape, builtin)?;
    if output_shape.len() != 2 {
        return Err(unsupported_error(
            builtin,
            "sparse arithmetic currently supports 2-D sparse operands",
        ));
    }
    let plan = BroadcastPlan::new(&left_shape, &right_shape).map_err(|err| {
        shape_error(
            builtin,
            format!("sparse operands are not compatible: {err}"),
        )
    })?;
    let rows = output_shape[0];
    let cols = output_shape[1];
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for (out_idx, lhs_idx, rhs_idx) in plan.iter() {
        let row = out_idx % rows;
        let col = out_idx / rows;
        while col_ptrs.len() <= col {
            col_ptrs.push(values.len());
        }
        let a = sparse_value_by_linear(lhs, lhs_idx);
        let b = sparse_value_by_linear(rhs, rhs_idx);
        let value = apply_real_op(a, b, op);
        if value != 0.0 {
            row_indices.push(row);
            values.push(value);
        }
    }
    while col_ptrs.len() <= cols {
        col_ptrs.push(values.len());
    }
    sparse_from_f64_values(rows, cols, col_ptrs, row_indices, values, dtype, builtin)
}

fn sparse_other(
    sparse: &SparseTensor,
    other: &Value,
    op: SparseBinaryOp,
    sparse_is_lhs: bool,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    if let Some(complex) = complex_tensor(other, builtin)? {
        return sparse_complex(sparse, &complex, op, sparse_is_lhs, builtin);
    }
    if let Some(scalar) = scalar_real(other) {
        let dtype =
            combined_floating_dtype(sparse_floating_dtype(sparse.numeric_dtype()), scalar.dtype);
        return sparse_scalar(sparse, scalar.value, op, sparse_is_lhs, dtype, builtin);
    }
    if let Some(dense) = dense_tensor(other, builtin)? {
        let dtype = combined_floating_dtype(
            sparse_floating_dtype(sparse.numeric_dtype()),
            floating_dtype(dense.numeric_dtype()),
        );
        return sparse_dense(sparse, &dense, op, sparse_is_lhs, dtype, builtin);
    }
    Err(unsupported_error(
        builtin,
        format!("unsupported sparse operand type {other:?}"),
    ))
}

fn sparse_scalar(
    sparse: &SparseTensor,
    scalar: f64,
    op: SparseBinaryOp,
    sparse_is_lhs: bool,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match op {
        SparseBinaryOp::Add => {
            if scalar == 0.0 {
                return preserve_sparse_with_dtype(sparse, dtype, builtin);
            }
            sparse_dense_scalar_result(sparse, scalar, |a, b| a + b, sparse_is_lhs, dtype, builtin)
        }
        SparseBinaryOp::Sub => {
            if scalar == 0.0 && sparse_is_lhs {
                return preserve_sparse_with_dtype(sparse, dtype, builtin);
            }
            if scalar == 0.0 {
                return scale_sparse(sparse, -1.0, dtype, builtin);
            }
            sparse_dense_scalar_result(sparse, scalar, |a, b| a - b, sparse_is_lhs, dtype, builtin)
        }
        SparseBinaryOp::Mul if scalar.is_finite() || !sparse_has_implicit_zeros(sparse) => {
            scale_sparse(sparse, scalar, dtype, builtin)
        }
        SparseBinaryOp::Mul => {
            sparse_dense_scalar_result(sparse, scalar, |a, b| a * b, sparse_is_lhs, dtype, builtin)
        }
    }
}

fn sparse_dense_scalar_result(
    sparse: &SparseTensor,
    scalar: f64,
    combine: impl Fn(f64, f64) -> f64,
    sparse_is_lhs: bool,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    checked_len(&sparse.shape(), builtin)?;
    let dense = sparse_numeric_dense_tensor(sparse, builtin)?;
    let mut out = dense.materialize_f64();
    for value in &mut out {
        *value = if sparse_is_lhs {
            combine(*value, scalar)
        } else {
            combine(scalar, *value)
        };
    }
    dense_from_f64_values(out, dense.shape, dtype, builtin)
}

fn scale_sparse(
    sparse: &SparseTensor,
    scalar: f64,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    if scalar == 0.0 || sparse.nnz() == 0 {
        return Ok(Value::SparseTensor(sparse_zeros_with_dtype(
            sparse.rows,
            sparse.cols,
            dtype,
        )));
    }
    let stored_values = sparse.materialize_f64();
    let mut col_ptrs = Vec::with_capacity(sparse.cols.saturating_add(1));
    let mut row_indices = Vec::with_capacity(sparse.row_indices.len());
    let mut values = Vec::with_capacity(stored_values.len());
    col_ptrs.push(0);
    for col in 0..sparse.cols {
        for entry in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let value = stored_values[entry] * scalar;
            if value != 0.0 {
                row_indices.push(sparse.row_indices[entry]);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    sparse_from_f64_values(
        sparse.rows,
        sparse.cols,
        col_ptrs,
        row_indices,
        values,
        dtype,
        builtin,
    )
}

fn sparse_dense(
    sparse: &SparseTensor,
    dense: &Tensor,
    op: SparseBinaryOp,
    sparse_is_lhs: bool,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let sparse_shape = sparse_shape(sparse);
    let output_shape = checked_broadcast_shape(
        &sparse_shape,
        &dense.shape,
        builtin,
        "sparse and dense sizes",
    )?;
    match op {
        SparseBinaryOp::Add => sparse_dense_full(
            sparse,
            dense,
            &output_shape,
            |a, b| a + b,
            sparse_is_lhs,
            dtype,
            builtin,
        ),
        SparseBinaryOp::Sub => sparse_dense_full(
            sparse,
            dense,
            &output_shape,
            |a, b| a - b,
            sparse_is_lhs,
            dtype,
            builtin,
        ),
        SparseBinaryOp::Mul => {
            if output_shape == sparse_shape {
                if dense_has_nonfinite_at_sparse_implicit_zero(sparse, dense) {
                    sparse_dense_full(
                        sparse,
                        dense,
                        &output_shape,
                        |a, b| a * b,
                        sparse_is_lhs,
                        dtype,
                        builtin,
                    )
                } else {
                    sparse_dense_times_preserve_sparse(sparse, dense, sparse_is_lhs, dtype, builtin)
                }
            } else {
                sparse_dense_full(
                    sparse,
                    dense,
                    &output_shape,
                    |a, b| a * b,
                    sparse_is_lhs,
                    dtype,
                    builtin,
                )
            }
        }
    }
}

fn sparse_dense_full(
    sparse: &SparseTensor,
    dense: &Tensor,
    output_shape: &[usize],
    combine: impl Fn(f64, f64) -> f64,
    sparse_is_lhs: bool,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    checked_len(output_shape, builtin)?;
    let sparse_shape = sparse_shape(sparse);
    let plan = BroadcastPlan::new(&sparse_shape, &dense.shape).map_err(|err| {
        shape_error(
            builtin,
            format!("sparse and dense sizes are not compatible: {err}"),
        )
    })?;
    let dense_values = tensor::tensor_values_f64_cow(dense);
    let mut out = vec![0.0; plan.len()];
    for (out_idx, sparse_idx, dense_idx) in plan.iter() {
        let sparse_value = sparse_value_by_linear(sparse, sparse_idx);
        let dense_value = dense_values[dense_idx];
        out[out_idx] = if sparse_is_lhs {
            combine(sparse_value, dense_value)
        } else {
            combine(dense_value, sparse_value)
        };
    }
    dense_from_f64_values(out, output_shape.to_vec(), dtype, builtin)
}

fn sparse_has_implicit_zeros(sparse: &SparseTensor) -> bool {
    match sparse.rows.checked_mul(sparse.cols) {
        Some(len) => sparse.nnz() < len,
        None => true,
    }
}

fn sparse_stored_values_are_finite(sparse: &SparseTensor) -> bool {
    sparse.integer_storage().is_some()
        || sparse
            .materialize_f64()
            .iter()
            .all(|value| value.is_finite())
}

fn dense_has_nonfinite_at_sparse_implicit_zero(sparse: &SparseTensor, dense: &Tensor) -> bool {
    let dense_values = tensor::tensor_values_f64_cow(dense);
    if dense_values.iter().all(|value| value.is_finite()) {
        return false;
    }
    if tensor::is_scalar_tensor(dense) {
        return !dense_values[0].is_finite() && sparse_has_implicit_zeros(sparse);
    }

    for col in 0..sparse.cols {
        for row in 0..sparse.rows {
            let dense_idx = dense_index_for_sparse_position(dense, row, col);
            if !dense_values[dense_idx].is_finite() && sparse.get(row, col).is_none() {
                return true;
            }
        }
    }
    false
}

fn sparse_dense_times_preserve_sparse(
    sparse: &SparseTensor,
    dense: &Tensor,
    _sparse_is_lhs: bool,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let dense_values = tensor::tensor_values_f64_cow(dense);
    let sparse_values = sparse.materialize_f64();
    let mut col_ptrs = Vec::with_capacity(sparse.cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..sparse.cols {
        for entry in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let row = sparse.row_indices[entry];
            let dense_idx = dense_index_for_sparse_position(dense, row, col);
            let value = sparse_values[entry] * dense_values[dense_idx];
            if value != 0.0 {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    sparse_from_f64_values(
        sparse.rows,
        sparse.cols,
        col_ptrs,
        row_indices,
        values,
        dtype,
        builtin,
    )
}

fn dense_index_for_sparse_position(dense: &Tensor, row: usize, col: usize) -> usize {
    let dense_row = if dense.rows <= 1 { 0 } else { row };
    let dense_col = if dense.cols <= 1 { 0 } else { col };
    dense_row + dense_col * dense.rows
}

fn sparse_complex(
    sparse: &SparseTensor,
    complex: &ComplexTensor,
    op: SparseBinaryOp,
    sparse_is_lhs: bool,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let dtype = combined_floating_dtype(
        sparse_floating_dtype(sparse.numeric_dtype()),
        floating_dtype(complex.numeric_dtype()),
    );
    let sparse_shape = sparse_shape(sparse);
    let output_shape = checked_broadcast_shape(
        &sparse_shape,
        &complex.shape,
        builtin,
        "sparse and complex sizes",
    )?;
    checked_len(&output_shape, builtin)?;
    let plan = BroadcastPlan::new(&sparse_shape, &complex.shape).map_err(|err| {
        shape_error(
            builtin,
            format!("sparse and complex sizes are not compatible: {err}"),
        )
    })?;
    let mut out = vec![(0.0, 0.0); plan.len()];
    for (out_idx, sparse_idx, complex_idx) in plan.iter() {
        let sparse_value = sparse_value_by_linear(sparse, sparse_idx);
        let (cr, ci) = complex.materialize_f64()[complex_idx];
        out[out_idx] = if sparse_is_lhs {
            apply_real_complex_op(sparse_value, (cr, ci), op)
        } else {
            apply_complex_real_op((cr, ci), sparse_value, op)
        };
    }
    let tensor = ComplexTensor::from_f64_values_with_dtype(out, output_shape, dtype)
        .map_err(|err| map_internal_error(builtin, err))?;
    Ok(complex_tensor_into_value(tensor))
}

fn sparse_value_by_linear(sparse: &SparseTensor, linear: usize) -> f64 {
    if sparse.rows == 0 || sparse.cols == 0 {
        return 0.0;
    }
    let row = linear % sparse.rows;
    let col = linear / sparse.rows;
    sparse.get(row, col).unwrap_or(0.0)
}

fn apply_real_op(lhs: f64, rhs: f64, op: SparseBinaryOp) -> f64 {
    match op {
        SparseBinaryOp::Add => lhs + rhs,
        SparseBinaryOp::Sub => lhs - rhs,
        SparseBinaryOp::Mul => lhs * rhs,
    }
}

fn apply_real_complex_op(lhs: f64, rhs: (f64, f64), op: SparseBinaryOp) -> (f64, f64) {
    let (rr, ri) = rhs;
    match op {
        SparseBinaryOp::Add => (lhs + rr, ri),
        SparseBinaryOp::Sub => (lhs - rr, -ri),
        SparseBinaryOp::Mul => (lhs * rr, lhs * ri),
    }
}

fn apply_complex_real_op(lhs: (f64, f64), rhs: f64, op: SparseBinaryOp) -> (f64, f64) {
    let (lr, li) = lhs;
    match op {
        SparseBinaryOp::Add => (lr + rhs, li),
        SparseBinaryOp::Sub => (lr - rhs, li),
        SparseBinaryOp::Mul => (lr * rhs, li * rhs),
    }
}

/// Apply a real scalar operation to a sparse matrix while preserving sparse
/// storage exactly when the implicit zero remains zero. Operations such as a
/// bit complement, whose zero maps to a nonzero value, intentionally return a
/// full tensor and are bounded by the standard sparse materialization limit.
pub(crate) fn map_sparse_real_values(
    sparse: &SparseTensor,
    builtin: &'static str,
    map: impl Fn(f64) -> BuiltinResult<f64>,
) -> BuiltinResult<Value> {
    let implicit_zero = map(0.0)?;
    if implicit_zero != 0.0 {
        checked_len(&sparse.shape(), builtin)?;
        let dense = sparse_numeric_dense_tensor(sparse, builtin)?;
        let mut out = dense.materialize_f64();
        for value in &mut out {
            *value = map(*value)?;
        }
        return dense_from_f64_values(
            out,
            dense.shape,
            sparse_floating_dtype(sparse.numeric_dtype()),
            builtin,
        );
    }

    let stored_values = sparse.materialize_f64();
    let mut col_ptrs = Vec::with_capacity(sparse.cols.saturating_add(1));
    let mut row_indices = Vec::with_capacity(sparse.row_indices.len());
    let mut values = Vec::with_capacity(stored_values.len());
    col_ptrs.push(0);
    for col in 0..sparse.cols {
        for entry in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let value = map(stored_values[entry])?;
            if value != 0.0 {
                row_indices.push(sparse.row_indices[entry]);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    sparse_from_f64_values(
        sparse.rows,
        sparse.cols,
        col_ptrs,
        row_indices,
        values,
        sparse_floating_dtype(sparse.numeric_dtype()),
        builtin,
    )
}

fn sparse_numeric_dense_tensor(
    sparse: &SparseTensor,
    builtin: &'static str,
) -> BuiltinResult<Tensor> {
    if sparse.is_logical() {
        let logical = sparse
            .to_dense_logical()
            .map_err(|err| map_internal_error(builtin, err))?;
        return Tensor::new(
            logical.data.into_iter().map(f64::from).collect(),
            logical.shape,
        )
        .map_err(|err| map_internal_error(builtin, err));
    }
    sparse
        .to_dense()
        .map_err(|err| map_internal_error(builtin, err))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse_a() -> SparseTensor {
        SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![10.0, 30.0, 20.0])
            .expect("sparse a")
    }

    fn sparse_b() -> SparseTensor {
        SparseTensor::new(3, 2, vec![0, 1, 3], vec![2, 0, 1], vec![5.0, 7.0, -20.0])
            .expect("sparse b")
    }

    fn dense_3x2() -> Tensor {
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).expect("dense")
    }

    fn expect_sparse(value: Value) -> SparseTensor {
        match value {
            Value::SparseTensor(sparse) => sparse,
            other => panic!("expected sparse result, got {other:?}"),
        }
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected dense tensor result, got {other:?}"),
        }
    }

    fn expect_complex(value: Value) -> ComplexTensor {
        match value {
            Value::ComplexTensor(tensor) => tensor,
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[test]
    fn native_single_sparse_arithmetic_preserves_single_dominance() {
        let single = SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0 / 3.0, 2.0])
            .expect("single sparse");
        let double =
            SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![0.5, 4.0]).expect("double");

        let sum = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(single.clone()),
                &Value::SparseTensor(double),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("mixed sparse sum"),
        );
        assert_eq!(sum.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(
            sum.as_f32_slice(),
            Some(&[((1.0f32 / 3.0) as f64 + 0.5) as f32, 6.0][..])
        );

        let dense_single = Tensor::from_f32(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]).unwrap();
        let product = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(
                    SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]).unwrap(),
                ),
                &Value::Tensor(dense_single),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("sparse dense product"),
        );
        assert_eq!(product.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(product.as_f32_slice(), Some(&[6.0, 18.0][..]));

        let single_zero = Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap();
        let promoted_identity = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(
                    SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]).unwrap(),
                ),
                &Value::Tensor(single_zero),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("sparse single-zero sum"),
        );
        assert_eq!(promoted_identity.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(promoted_identity.as_f32_slice(), Some(&[2.0, 3.0][..]));

        let dense_sum = expect_tensor(
            sparse_binary(
                &Value::SparseTensor(single.clone()),
                &Value::Num(1.0),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("single sparse scalar sum"),
        );
        assert_eq!(dense_sum.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            dense_sum.as_f32_slice(),
            Some(&[1.0f32 / 3.0 + 1.0, 1.0, 1.0, 3.0][..])
        );

        let zero = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(single),
                &Value::Num(0.0),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("single sparse zero product"),
        );
        assert_eq!(zero.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(zero.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn sparse_complex_arithmetic_preserves_single_dominance() {
        let sparse =
            SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![2.0]).expect("double sparse");
        let complex =
            ComplexTensor::from_f32(vec![(1.0, 0.5), (2.0, -1.0)], vec![2, 1]).expect("single");
        let output = expect_complex(
            sparse_binary(
                &Value::SparseTensor(sparse),
                &Value::ComplexTensor(complex),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("sparse complex sum"),
        );
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(output.as_f32_slice(), Some(&[(3.0, 0.5), (2.0, -1.0)][..]));
    }

    fn double_values(tensor: &Tensor) -> &[f64] {
        tensor
            .as_f64_slice()
            .expect("sparse floating arithmetic returns a double tensor")
    }

    #[test]
    fn sparse_sparse_addition_preserves_sparse_union() {
        let result = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::SparseTensor(sparse_b()),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("sparse plus"),
        );
        assert_eq!(result.shape(), vec![3, 2]);
        assert_eq!(result.get(0, 0), Some(10.0));
        assert_eq!(result.get(2, 0), Some(35.0));
        assert_eq!(result.get(0, 1), Some(7.0));
        assert_eq!(result.get(1, 1).unwrap_or(0.0), 0.0);
    }

    #[test]
    fn sparse_dense_addition_returns_dense_result() {
        let result = expect_tensor(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::Tensor(dense_3x2()),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("sparse plus dense"),
        );
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(double_values(&result), [11.0, 2.0, 33.0, 4.0, 25.0, 6.0]);
    }

    #[test]
    fn dense_sparse_subtraction_returns_dense_result() {
        let result = expect_tensor(
            sparse_binary(
                &Value::Tensor(dense_3x2()),
                &Value::SparseTensor(sparse_a()),
                SparseBinaryOp::Sub,
                "minus",
            )
            .expect("dense minus sparse"),
        );
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(double_values(&result), [-9.0, 2.0, -27.0, 4.0, -15.0, 6.0]);
    }

    #[test]
    fn sparse_scalar_multiply_preserves_sparse_pattern() {
        let result = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::Num(2.0),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("sparse scale"),
        );
        assert_eq!(result.shape(), vec![3, 2]);
        assert_eq!(result.materialize_f64(), vec![20.0, 60.0, 40.0]);
    }

    #[test]
    fn sparse_dense_times_preserves_sparse_pattern() {
        let result = expect_sparse(
            sparse_binary(
                &Value::Tensor(dense_3x2()),
                &Value::SparseTensor(sparse_a()),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("dense times sparse"),
        );
        assert_eq!(result.shape(), vec![3, 2]);
        assert_eq!(result.get(0, 0), Some(10.0));
        assert_eq!(result.get(2, 0), Some(90.0));
        assert_eq!(result.get(1, 1), Some(100.0));
    }

    #[test]
    fn sparse_dense_times_materializes_nonfinite_implicit_zero_results() {
        let dense =
            Tensor::new(vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).expect("dense");
        let result = expect_tensor(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::Tensor(dense),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("sparse times dense with nan"),
        );
        assert_eq!(result.shape, vec![3, 2]);
        let values = double_values(&result);
        assert_eq!(values[0], 10.0);
        assert!(values[1].is_nan());
        assert_eq!(values[2], 90.0);
        assert_eq!(values[4], 100.0);
    }

    #[test]
    fn map_sparse_real_values_preserves_or_materializes_from_zero_semantics() {
        let sparse = sparse_a();
        let scaled = expect_sparse(
            map_sparse_real_values(&sparse, "test", |value| Ok(value * 2.0))
                .expect("zero-preserving map"),
        );
        assert_eq!(scaled.shape(), vec![3, 2]);
        assert_eq!(scaled.materialize_f64(), vec![20.0, 60.0, 40.0]);

        let complemented = expect_tensor(
            map_sparse_real_values(&sparse, "test", |value| Ok(1.0 - value))
                .expect("zero-changing map"),
        );
        assert_eq!(complemented.shape, vec![3, 2]);
        assert_eq!(
            double_values(&complemented),
            [-9.0, 1.0, -29.0, 1.0, -19.0, 1.0]
        );
    }

    #[test]
    fn sparse_sparse_times_keeps_nan_from_stored_value_times_implicit_zero() {
        let lhs = SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap();
        let rhs = SparseTensor::new(2, 1, vec![0, 1], vec![1], vec![5.0]).unwrap();
        let result = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(lhs),
                &Value::SparseTensor(rhs),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("sparse nan times implicit zero"),
        );
        assert_eq!(result.shape(), vec![2, 1]);
        assert!(result.get(0, 0).expect("stored nan").is_nan());
        assert_eq!(result.get(1, 0).unwrap_or(0.0), 0.0);
    }

    #[test]
    fn sparse_scalar_times_nonfinite_materializes_implicit_zero_results() {
        let result = expect_tensor(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::Num(f64::INFINITY),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("sparse times inf"),
        );
        assert_eq!(result.shape, vec![3, 2]);
        let values = double_values(&result);
        assert!(values[1].is_nan());
        assert!(values[3].is_nan());
        assert!(values[5].is_nan());
        assert!(values[0].is_infinite());
    }

    #[test]
    fn sparse_sparse_one_by_one_broadcasts_as_sparse() {
        let scalar = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![2.0]).unwrap();
        let result = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(scalar),
                &Value::SparseTensor(sparse_a()),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("sparse scalar plus sparse"),
        );
        assert_eq!(result.shape(), vec![3, 2]);
        assert_eq!(result.nnz(), 6);
        assert_eq!(result.get(0, 0), Some(12.0));
        assert_eq!(result.get(1, 0), Some(2.0));
        assert_eq!(result.get(2, 0), Some(32.0));
    }

    #[test]
    fn sparse_sparse_zero_scalar_broadcast_preserves_large_sparse_without_full_scan() {
        let large = SparseTensor::new(
            MAX_SPARSE_FULL_RESULT_ELEMENTS + 1,
            1,
            vec![0, 1],
            vec![0],
            vec![5.0],
        )
        .unwrap();
        let zero = SparseTensor::zeros(1, 1);
        let result = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(zero),
                &Value::SparseTensor(large.clone()),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("zero sparse scalar plus large sparse"),
        );
        assert_eq!(result, large);
    }

    #[test]
    fn sparse_sparse_scalar_times_large_sparse_scales_stored_entries_only() {
        let large = SparseTensor::new(
            MAX_SPARSE_FULL_RESULT_ELEMENTS + 1,
            1,
            vec![0, 1],
            vec![0],
            vec![5.0],
        )
        .unwrap();
        let scalar = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![3.0]).unwrap();
        let result = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(scalar),
                &Value::SparseTensor(large),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("sparse scalar times large sparse"),
        );
        assert_eq!(result.shape(), vec![MAX_SPARSE_FULL_RESULT_ELEMENTS + 1, 1]);
        assert_eq!(result.nnz(), 1);
        assert_eq!(result.get(0, 0), Some(15.0));
    }

    #[test]
    fn sparse_plus_char_array_promotes_chars_to_numeric_dense() {
        let chars = CharArray::new(vec!['A', 'B', 'C', 'D', 'E', 'F'], 3, 2).unwrap();
        let result = expect_tensor(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::CharArray(chars),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("sparse plus char"),
        );
        let values = double_values(&result);
        assert_eq!(values[0], 75.0);
        assert_eq!(values[1], 66.0);
        assert_eq!(values[2], 97.0);
    }

    #[test]
    fn sparse_plus_complex_returns_full_complex_result() {
        let complex = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let result = expect_complex(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::ComplexTensor(complex),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("sparse plus complex"),
        );
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(result.materialize_f64()[0], (11.0, 2.0));
        assert_eq!(result.materialize_f64()[1], (1.0, 2.0));
    }

    #[test]
    fn sparse_times_complex_returns_full_complex_result() {
        let complex = ComplexTensor::new(vec![(1.0, -1.0)], vec![1, 1]).unwrap();
        let result = expect_complex(
            sparse_binary(
                &Value::ComplexTensor(complex),
                &Value::SparseTensor(sparse_a()),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("complex times sparse"),
        );
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(result.materialize_f64()[0], (10.0, -10.0));
        assert_eq!(result.materialize_f64()[1], (0.0, -0.0));
    }

    #[test]
    fn sparse_times_logical_preserves_sparse_pattern() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0, 1, 1], vec![3, 2]).unwrap();
        let result = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::LogicalArray(logical),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("sparse times logical"),
        );
        assert_eq!(result.shape(), vec![3, 2]);
        assert_eq!(result.get(0, 0), Some(10.0));
        assert_eq!(result.get(2, 0), Some(30.0));
        assert_eq!(result.get(1, 1), Some(20.0));
    }

    #[test]
    fn sparse_zero_scalar_orders_preserve_sparse_storage() {
        let plus_zero = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::Num(0.0),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("sparse plus zero"),
        );
        assert_eq!(plus_zero, sparse_a());

        let zero_minus = expect_sparse(
            sparse_binary(
                &Value::Num(0.0),
                &Value::SparseTensor(sparse_a()),
                SparseBinaryOp::Sub,
                "minus",
            )
            .expect("zero minus sparse"),
        );
        assert_eq!(zero_minus.materialize_f64(), vec![-10.0, -30.0, -20.0]);

        let times_zero = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::Num(0.0),
                SparseBinaryOp::Mul,
                "times",
            )
            .expect("sparse times zero"),
        );
        assert_eq!(times_zero.shape(), vec![3, 2]);
        assert_eq!(times_zero.nnz(), 0);
    }

    #[test]
    fn sparse_broadcast_overflow_returns_stable_limit_error_before_plan_len() {
        let output_shape = checked_broadcast_shape(
            &[usize::MAX, 1],
            &[1, usize::MAX],
            "plus",
            "sparse operands",
        )
        .expect("broadcast shape");
        let err = checked_sparse_result_len(&output_shape, "plus")
            .expect_err("overflowing broadcast should fail before planning");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:plus:SparseDensifyTooLarge")
        );
    }

    #[test]
    fn sparse_sparse_subtraction_drops_cancellations() {
        let result = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(sparse_a()),
                &Value::SparseTensor(sparse_a()),
                SparseBinaryOp::Sub,
                "minus",
            )
            .expect("sparse minus itself"),
        );
        assert_eq!(result.shape(), vec![3, 2]);
        assert_eq!(result.nnz(), 0);
    }

    #[test]
    fn sparse_empty_arithmetic_preserves_empty_shape() {
        let empty = SparseTensor::zeros(0, 3);
        let result = expect_sparse(
            sparse_binary(
                &Value::SparseTensor(empty),
                &Value::SparseTensor(SparseTensor::zeros(0, 3)),
                SparseBinaryOp::Add,
                "plus",
            )
            .expect("empty sparse plus"),
        );
        assert_eq!(result.shape(), vec![0, 3]);
        assert_eq!(result.col_ptrs, vec![0, 0, 0, 0]);
    }

    #[test]
    fn sparse_shape_mismatch_uses_stable_sparse_identifier() {
        let err = sparse_binary(
            &Value::SparseTensor(sparse_a()),
            &Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            SparseBinaryOp::Add,
            "plus",
        )
        .expect_err("shape mismatch should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:plus:SparseSizeMismatch")
        );
    }

    #[test]
    fn sparse_densifying_scalar_result_has_stable_limit_error() {
        let sparse = SparseTensor::zeros(MAX_SPARSE_FULL_RESULT_ELEMENTS + 1, 1);
        let err = sparse_binary(
            &Value::SparseTensor(sparse),
            &Value::Num(1.0),
            SparseBinaryOp::Add,
            "plus",
        )
        .expect_err("large densification should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:plus:SparseDensifyTooLarge")
        );
    }
}
