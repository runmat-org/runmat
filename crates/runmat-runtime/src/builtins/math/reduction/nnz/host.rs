//! Borrowed native zero comparison and dimension count geometry.
use super::*;
pub(super) fn nnz_host_value(value: Value, dim: Option<usize>) -> BuiltinResult<Value> {
    match dim {
        None => {
            let count = count_nonzero_value(&value)?;
            Ok(Value::Num(count as f64))
        }
        Some(dim) => {
            if let Value::SparseTensor(sparse) = &value {
                return reduce_sparse_dim(sparse, dim).map(tensor::tensor_into_value);
            }
            let mask = mask_from_value(&value)?;
            let tensor = reduce_mask_dim(&mask, dim)?;
            Ok(tensor::tensor_into_value(tensor))
        }
    }
}

pub(super) fn count_nonzero_value(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) => Ok(count_nonzero_tensor(tensor)),
        Value::SparseTensor(sparse) => Ok(count_nonzero_sparse(sparse)),
        Value::ComplexTensor(ct) => Ok(count_nonzero_complex_tensor(ct)),
        Value::LogicalArray(logical) => Ok(count_nonzero_logical(logical)),
        Value::CharArray(chars) => Ok(count_nonzero_char(chars)),
        Value::Num(n) => Ok(if is_nonzero_scalar(*n) { 1 } else { 0 }),
        Value::Int(i) => Ok(if !i.is_zero() { 1 } else { 0 }),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Complex(re, im) => Ok(if is_nonzero_complex(*re, *im) { 1 } else { 0 }),
        Value::GpuTensor(_) => Err(nnz_descriptor_error_with_detail(
            &NNZ_ERROR_INTERNAL,
            "GPU inputs are handled before host evaluation",
        )),
        other => Err(nnz_descriptor_error_with_detail(
            &NNZ_ERROR_INVALID_INPUT,
            format!(
                "expected numeric, logical, complex, or char array, got {}",
                describe_value_kind(other)
            ),
        )),
    }
}

pub(super) fn count_nonzero_tensor(tensor: &Tensor) -> usize {
    (0..tensor.len())
        .filter(|&index| {
            tensor
                .numeric_value_at(index)
                .is_some_and(|value| !value.is_zero())
        })
        .count()
}

pub(super) fn count_nonzero_sparse(sparse: &SparseTensor) -> usize {
    (0..sparse.nnz())
        .filter(|&index| {
            sparse
                .numeric_value_at(index)
                .is_some_and(|value| !value.is_zero())
        })
        .count()
}

pub(super) fn count_nonzero_complex_tensor(tensor: &ComplexTensor) -> usize {
    (0..tensor.len())
        .filter(|&index| {
            tensor
                .numeric_value_at(index)
                .is_some_and(|(real, imaginary)| !real.is_zero() || !imaginary.is_zero())
        })
        .count()
}

pub(super) fn count_nonzero_logical(logical: &LogicalArray) -> usize {
    logical.data.iter().filter(|&&b| b != 0).count()
}

pub(super) fn count_nonzero_char(chars: &CharArray) -> usize {
    chars.data().iter().filter(|&&ch| ch.raw() != 0).count()
}

#[inline]
pub(super) fn is_nonzero_scalar(value: f64) -> bool {
    value.is_nan() || value != 0.0
}

#[inline]
pub(super) fn is_nonzero_complex(re: f64, im: f64) -> bool {
    re.is_nan() || im.is_nan() || re != 0.0 || im != 0.0
}

pub(super) struct Mask {
    bits: Vec<u8>,
    shape: Vec<usize>,
}

pub(super) fn mask_from_value(value: &Value) -> BuiltinResult<Mask> {
    match value {
        Value::Tensor(tensor) => {
            let shape = canonical_shape(tensor.shape(), tensor.len());
            let bits = if let Some(storage) = tensor.integer_storage() {
                (0..storage.len())
                    .map(|index| {
                        u8::from(
                            !storage
                                .value_at(index)
                                .expect("validated integer storage")
                                .is_zero(),
                        )
                    })
                    .collect()
            } else {
                tensor::tensor_values_f64_cow(tensor)
                    .iter()
                    .map(|&value| u8::from(is_nonzero_scalar(value)))
                    .collect()
            };
            Ok(Mask { bits, shape })
        }
        Value::ComplexTensor(tensor) => {
            let shape = canonical_shape(tensor.shape(), tensor.len());
            let bits = match tensor.complex_storage() {
                runmat_value::ComplexStorage::Integer(storage) => (0..storage.len())
                    .map(|index| {
                        u8::from(
                            storage
                                .is_nonzero_at(index)
                                .expect("typed complex integer storage is structurally valid"),
                        )
                    })
                    .collect(),
                runmat_value::ComplexStorage::F64(values) => values
                    .iter()
                    .map(|value| u8::from(is_nonzero_complex(value.0, value.1)))
                    .collect(),
                runmat_value::ComplexStorage::F32(values) => values
                    .iter()
                    .map(|value| {
                        u8::from(
                            value.0.is_nan()
                                || value.1.is_nan()
                                || value.0 != 0.0
                                || value.1 != 0.0,
                        )
                    })
                    .collect(),
            };
            Ok(Mask { bits, shape })
        }
        Value::SparseTensor(sparse) => mask_from_sparse(sparse),
        Value::LogicalArray(logical) => Ok(Mask {
            bits: logical
                .data
                .iter()
                .map(|&b| if b != 0 { 1 } else { 0 })
                .collect(),
            shape: canonical_shape(&logical.shape, logical.data.len()),
        }),
        Value::CharArray(chars) => {
            let bits = chars
                .data()
                .iter()
                .map(|&ch| if ch.raw() != 0 { 1u8 } else { 0u8 })
                .collect();
            Ok(Mask {
                bits,
                shape: chars.shape().to_vec(),
            })
        }
        Value::Num(n) => Ok(Mask {
            bits: vec![if is_nonzero_scalar(*n) { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::Int(i) => Ok(Mask {
            bits: vec![if !i.is_zero() { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::Bool(b) => Ok(Mask {
            bits: vec![if *b { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::Complex(re, im) => Ok(Mask {
            bits: vec![if is_nonzero_complex(*re, *im) { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::GpuTensor(_) => Err(nnz_descriptor_error_with_detail(
            &NNZ_ERROR_INTERNAL,
            "GPU inputs are handled before host evaluation",
        )),
        other => Err(nnz_descriptor_error_with_detail(
            &NNZ_ERROR_INVALID_INPUT,
            format!(
                "expected numeric, logical, complex, or char array, got {}",
                describe_value_kind(other)
            ),
        )),
    }
}

pub(super) fn reduce_mask_dim(mask: &Mask, dim: usize) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(nnz_descriptor_error_with_detail(
            &NNZ_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        ));
    }
    if mask.bits.is_empty() {
        let mut out_shape = canonical_shape(&mask.shape, 0);
        if dim <= out_shape.len() && !is_scalar_shape(&out_shape) {
            out_shape[dim - 1] = 1;
        }
        let out_len = out_shape.iter().copied().product::<usize>();
        let zeros = vec![0.0; out_len];
        return Tensor::new(zeros, out_shape)
            .map_err(|e| nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, &e));
    }
    if is_scalar_shape(&mask.shape) {
        let data = vec![mask.bits[0] as f64];
        return Tensor::new(data, canonical_scalar_shape())
            .map_err(|e| nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, &e));
    }
    if dim > mask.shape.len() {
        return mask_to_tensor(mask);
    }
    let dim_index = dim - 1;
    let reduce_len = mask.shape[dim_index];
    let stride_before = if dim_index == 0 {
        1
    } else {
        mask.shape[..dim_index].iter().copied().product::<usize>()
    };
    let stride_after = if dim_index + 1 >= mask.shape.len() {
        1
    } else {
        mask.shape[dim_index + 1..]
            .iter()
            .copied()
            .product::<usize>()
    };
    let out_len = stride_before.checked_mul(stride_after).ok_or_else(|| {
        nnz_descriptor_error_with_detail(&NNZ_ERROR_INVALID_ARGUMENT, "dimension too large")
    })?;
    let mut out_shape = mask.shape.clone();
    out_shape[dim_index] = 1;
    let mut output = vec![0f64; out_len];
    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut count = 0usize;
            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                if idx < mask.bits.len() && mask.bits[idx] != 0 {
                    count += 1;
                }
            }
            let out_idx = after * stride_before + before;
            if out_idx < output.len() {
                output[out_idx] = count as f64;
            }
        }
    }
    Tensor::new(output, out_shape)
        .map_err(|e| nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, &e))
}

pub(super) fn reduce_sparse_dim(sparse: &SparseTensor, dim: usize) -> BuiltinResult<Tensor> {
    match dim {
        0 => Err(nnz_descriptor_error_with_detail(
            &NNZ_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        )),
        1 => reduce_sparse_columns(sparse),
        2 => reduce_sparse_rows(sparse),
        _ => {
            let mask = mask_from_sparse(sparse)?;
            reduce_mask_dim(&mask, dim)
        }
    }
}

pub(super) fn zero_counts(len: usize, context: &str) -> BuiltinResult<Vec<f64>> {
    let mut counts = Vec::new();
    counts.try_reserve_exact(len).map_err(|err| {
        nnz_descriptor_error_with_detail(
            &NNZ_ERROR_INTERNAL,
            format!("{context} allocation failed: {err}"),
        )
    })?;
    counts.resize(len, 0.0);
    Ok(counts)
}

pub(super) fn reduce_sparse_columns(sparse: &SparseTensor) -> BuiltinResult<Tensor> {
    let mut output = zero_counts(sparse.cols, "sparse column count")?;
    for (col, count) in output.iter_mut().enumerate() {
        let start = sparse.col_ptrs.get(col).copied().ok_or_else(|| {
            nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, "sparse col_ptr missing")
        })?;
        let end = sparse.col_ptrs.get(col + 1).copied().ok_or_else(|| {
            nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, "sparse col_ptr missing")
        })?;
        if start > end || end > sparse.nnz() {
            return Err(nnz_descriptor_error_with_detail(
                &NNZ_ERROR_INTERNAL,
                "sparse col_ptr out of bounds",
            ));
        }
        *count = (start..end)
            .filter(|&index| {
                sparse
                    .numeric_value_at(index)
                    .is_some_and(|value| !value.is_zero())
            })
            .count() as f64;
    }
    Tensor::new(output, vec![1, sparse.cols])
        .map_err(|e| nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, &e))
}

pub(super) fn reduce_sparse_rows(sparse: &SparseTensor) -> BuiltinResult<Tensor> {
    let mut output = zero_counts(sparse.rows, "sparse row count")?;
    for col in 0..sparse.cols {
        let start = sparse.col_ptrs.get(col).copied().ok_or_else(|| {
            nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, "sparse col_ptr missing")
        })?;
        let end = sparse.col_ptrs.get(col + 1).copied().ok_or_else(|| {
            nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, "sparse col_ptr missing")
        })?;
        if start > end || end > sparse.nnz() || end > sparse.row_indices.len() {
            return Err(nnz_descriptor_error_with_detail(
                &NNZ_ERROR_INTERNAL,
                "sparse col_ptr out of bounds",
            ));
        }
        for idx in start..end {
            if sparse
                .numeric_value_at(idx)
                .is_none_or(|value| value.is_zero())
            {
                continue;
            }
            let row = sparse.row_indices[idx];
            let Some(count) = output.get_mut(row) else {
                return Err(nnz_descriptor_error_with_detail(
                    &NNZ_ERROR_INTERNAL,
                    "sparse row index out of bounds",
                ));
            };
            *count += 1.0;
        }
    }
    Tensor::new(output, vec![sparse.rows, 1])
        .map_err(|e| nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, &e))
}

pub(super) fn mask_from_sparse(sparse: &SparseTensor) -> BuiltinResult<Mask> {
    let total = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
        nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, "sparse dimensions overflow usize")
    })?;
    let mut bits = Vec::new();
    bits.try_reserve_exact(total).map_err(|err| {
        nnz_descriptor_error_with_detail(
            &NNZ_ERROR_INTERNAL,
            format!("sparse mask allocation failed: {err}"),
        )
    })?;
    bits.resize(total, 0);
    for col in 0..sparse.cols {
        let col_off = col.checked_mul(sparse.rows).ok_or_else(|| {
            nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, "sparse column offset overflow")
        })?;
        for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let row = sparse.row_indices[idx];
            if row < sparse.rows {
                let bit_idx = col_off.checked_add(row).ok_or_else(|| {
                    nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, "sparse index overflow")
                })?;
                if sparse
                    .numeric_value_at(idx)
                    .is_some_and(|value| !value.is_zero())
                {
                    bits[bit_idx] = 1;
                }
            }
        }
    }
    Ok(Mask {
        bits,
        shape: vec![sparse.rows, sparse.cols],
    })
}

pub(super) fn mask_to_tensor(mask: &Mask) -> BuiltinResult<Tensor> {
    let data = mask.bits.iter().map(|&b| b as f64).collect::<Vec<_>>();
    Tensor::new(data, canonical_shape(&mask.shape, mask.bits.len()))
        .map_err(|e| nnz_descriptor_error_with_detail(&NNZ_ERROR_INTERNAL, &e))
}

pub(super) fn canonical_shape(shape: &[usize], len: usize) -> Vec<usize> {
    if is_scalar_shape(shape) {
        if len <= 1 {
            return canonical_scalar_shape();
        }
        return vec![len, 1];
    }
    if tensor::element_count(shape) == len {
        return normalize_scalar_shape(shape);
    }
    shape.to_vec()
}
