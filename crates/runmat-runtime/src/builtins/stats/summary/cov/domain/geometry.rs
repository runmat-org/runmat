use super::super::*;

#[derive(Debug, Clone)]
pub(in super::super) struct Matrix {
    pub(super) data: Vec<f64>,
    pub(in super::super) rows: usize,
    pub(super) cols: usize,
}

// Shared host/resident result geometry: vectors and zero-column inputs are
// normalized to one variable; ordinary matrices retain their column count.
pub(in super::super) fn covariance_column_count(rows: usize, columns: usize) -> usize {
    if rows == 1 || columns <= 1 {
        1
    } else {
        columns
    }
}

pub(in super::super) fn covariance_element_count(shape: &[usize]) -> BuiltinResult<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| cov_error_with_detail(&COV_ERROR_INVALID_ARGUMENT, "input shape overflow"))
}

pub(in super::super) fn covariance_input_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(&array.shape),
        Value::GpuTensor(handle) => Some(handle.shape()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(&[1, 1]),
        _ => None,
    }
}

pub(in super::super) fn validate_covariance_pair_shapes(
    left: &[usize],
    right: &[usize],
) -> BuiltinResult<()> {
    if left.len() > 2 || right.len() > 2 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "inputs must be 2-D matrices or vectors",
        ));
    }
    let vector = |shape: &[usize]| {
        shape.first().copied().unwrap_or(1) == 1 || shape.get(1).copied().unwrap_or(1) == 1
    };
    let compatible = if vector(left) && vector(right) {
        covariance_element_count(left)? == covariance_element_count(right)?
    } else {
        left == right
    };
    if compatible {
        Ok(())
    } else {
        Err(cov_error(&COV_ERROR_ROWS_MISMATCH))
    }
}

impl Matrix {
    fn from_single_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        if tensor.shape().len() > 2 {
            return Err(cov_error_with_detail(
                &COV_ERROR_INVALID_ARGUMENT,
                "inputs must be 2-D matrices or vectors",
            ));
        }
        let original_rows = tensor.rows();
        let original_cols = tensor.cols();
        let is_vector = original_rows == 1 || original_cols == 1;
        let (rows, cols) = if is_vector {
            (
                tensor.len(),
                covariance_column_count(original_rows, original_cols),
            )
        } else if tensor.is_empty() && original_cols == 0 {
            (0, covariance_column_count(original_rows, original_cols))
        } else {
            (
                original_rows,
                covariance_column_count(original_rows, original_cols),
            )
        };
        Ok(Self {
            rows,
            cols,
            data: centered_tensor_values(&tensor, rows, cols),
        })
    }

    fn from_tensor_pair(left: Tensor, right: Tensor) -> BuiltinResult<Self> {
        validate_covariance_pair_shapes(left.shape(), right.shape())?;
        let rows = left.len();
        let mut data = centered_tensor_values(&left, rows, 1);
        data.extend(centered_tensor_values(&right, rows, 1));
        Ok(Self {
            data,
            rows,
            cols: 2,
        })
    }

    #[inline]
    pub(super) fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }

    #[inline]
    pub(super) fn column(&self, col: usize) -> &[f64] {
        let start = col * self.rows;
        let end = start + self.rows;
        &self.data[start..end]
    }
}

pub(in super::super) fn combine_tensors(
    left: Tensor,
    right: Option<Tensor>,
) -> BuiltinResult<Matrix> {
    if let Some(right) = right {
        Matrix::from_tensor_pair(left, right)
    } else {
        Matrix::from_single_tensor(left)
    }
}

fn centered_tensor_values(tensor: &Tensor, rows: usize, cols: usize) -> Vec<f64> {
    let Some(storage) = tensor.integer_storage() else {
        return tensor.materialize_f64();
    };
    let mut values = Vec::with_capacity(rows.saturating_mul(cols));
    for col in 0..cols {
        let start = col * rows;
        let anchor = storage
            .value_at(start)
            .map(|value| int_value_to_i128(&value))
            .unwrap_or(0);
        for index in start..start + rows {
            let value = storage
                .value_at(index)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(anchor);
            values.push((value - anchor) as f64);
        }
    }
    values
}

fn int_value_to_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn covariance_preserves_rank_zero_and_rank_one_observation_roles() {
        for shape in [Vec::new(), vec![1]] {
            let input = Value::Tensor(Tensor::new(vec![7.0], shape).unwrap());
            let Value::Tensor(output) = block_on(super::super::super::cov_builtin(
                input,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap() else {
                panic!("expected covariance tensor");
            };
            assert_eq!(output.shape(), &[1, 1]);
            assert_eq!(output.materialize_f64(), vec![0.0]);
        }
        let left = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap());
        let right = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap());
        let Value::Tensor(output) = block_on(super::super::super::cov_builtin(
            left,
            crate::builtins::common::test_support::builtin_arguments(vec![right]),
        ))
        .unwrap() else {
            panic!("expected paired covariance tensor");
        };
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.materialize_f64(), vec![1.0, 2.0, 2.0, 4.0]);
    }
}
