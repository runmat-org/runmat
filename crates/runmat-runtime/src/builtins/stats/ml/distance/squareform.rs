use super::*;
use runmat_builtins::{condensed_length, matrix_size_for_condensed_length, SquareformDirection};
use runmat_value::{LogicalArray, NumericDType, NumericScalar, NumericStorage};

pub(super) fn parse_squareform_force(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<SquareformDirection> {
    if args.is_empty() {
        return Ok(SquareformDirection::Auto);
    }
    if args.len() != 1 {
        return Err(distance_error(
            SQUAREFORM_NAME,
            "squareform: accepts at most one force option",
        ));
    }
    let text = distance_keyword(SQUAREFORM_NAME, &args[0])?.ok_or_else(|| {
        distance_error(
            SQUAREFORM_NAME,
            "squareform: force option must be 'tomatrix' or 'tovector'",
        )
    })?;
    SquareformDirection::from_keyword(&text).ok_or_else(|| {
        distance_error(
            SQUAREFORM_NAME,
            format!("squareform: unsupported force option '{text}'"),
        )
    })
}

pub(super) fn squareform_compute(
    tensor: Tensor,
    force: SquareformDirection,
) -> BuiltinResult<Tensor> {
    let (rows, cols) = observation_matrix_dimensions(&tensor);
    let is_vector = rows == 1 || cols == 1;
    match force {
        SquareformDirection::ToMatrix => vector_to_square(tensor),
        SquareformDirection::ToVector => square_to_vector(tensor),
        SquareformDirection::Auto if is_vector => vector_to_square(tensor),
        SquareformDirection::Auto => square_to_vector(tensor),
    }
}

pub(super) fn squareform_logical(
    logical: LogicalArray,
    force: SquareformDirection,
) -> BuiltinResult<Value> {
    if logical.shape.len() > 2 {
        return Err(distance_error(
            SQUAREFORM_NAME,
            "squareform: input must be a logical vector or 2-D matrix",
        ));
    }
    let rows = logical.shape.first().copied().unwrap_or(1);
    let cols = logical.shape.get(1).copied().unwrap_or(1);
    let is_vector = rows == 1 || cols == 1;
    let output = match force {
        SquareformDirection::ToMatrix => logical_vector_to_square(logical)?,
        SquareformDirection::ToVector => logical_square_to_vector(logical, rows, cols)?,
        SquareformDirection::Auto if is_vector => logical_vector_to_square(logical)?,
        SquareformDirection::Auto => logical_square_to_vector(logical, rows, cols)?,
    };
    Ok(Value::LogicalArray(output))
}

fn logical_vector_to_square(logical: LogicalArray) -> BuiltinResult<LogicalArray> {
    let rows = logical.shape.first().copied().unwrap_or(1);
    let cols = logical.shape.get(1).copied().unwrap_or(1);
    if rows != 1 && cols != 1 {
        return Err(distance_error(
            SQUAREFORM_NAME,
            "squareform: vector input is required for 'tomatrix'",
        ));
    }
    let size = matrix_size_for_condensed_length(logical.len()).ok_or_else(|| {
        distance_error(
            SQUAREFORM_NAME,
            "squareform: vector length must be n*(n-1)/2 for an integer n",
        )
    })?;
    let out_len = size
        .checked_mul(size)
        .ok_or_else(|| internal_error(SQUAREFORM_NAME, "squareform: output size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(out_len)
        .map_err(|_| internal_error(SQUAREFORM_NAME, "squareform: output allocation failed"))?;
    out.resize(out_len, 0);
    let mut source = 0;
    for col in 0..size {
        for row in (col + 1)..size {
            let value = logical.data[source];
            out[col * size + row] = value;
            out[row * size + col] = value;
            source += 1;
        }
    }
    LogicalArray::new(out, vec![size, size])
        .map_err(|error| internal_error(SQUAREFORM_NAME, format!("squareform: {error}")))
}

fn logical_square_to_vector(
    logical: LogicalArray,
    rows: usize,
    cols: usize,
) -> BuiltinResult<LogicalArray> {
    if rows != cols {
        return Err(distance_error(
            SQUAREFORM_NAME,
            "squareform: matrix input must be square",
        ));
    }
    let values = logical.data.as_slice();
    for index in 0..rows {
        if values[index * rows + index] != 0 {
            return Err(distance_error(
                SQUAREFORM_NAME,
                "squareform: matrix diagonal must be zero",
            ));
        }
    }
    let len = condensed_length(rows)
        .ok_or_else(|| internal_error(SQUAREFORM_NAME, "squareform: output size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| internal_error(SQUAREFORM_NAME, "squareform: output allocation failed"))?;
    for col in 0..rows {
        for row in (col + 1)..rows {
            let lower = values[col * rows + row];
            let upper = values[row * rows + col];
            if lower != upper {
                return Err(distance_error(
                    SQUAREFORM_NAME,
                    "squareform: matrix must be symmetric",
                ));
            }
            out.push(upper);
        }
    }
    LogicalArray::new(out, vec![1, len])
        .map_err(|error| internal_error(SQUAREFORM_NAME, format!("squareform: {error}")))
}

fn vector_to_square(tensor: Tensor) -> BuiltinResult<Tensor> {
    let (rows, cols) = observation_matrix_dimensions(&tensor);
    if !(rows == 1 || cols == 1) {
        return Err(distance_error(
            SQUAREFORM_NAME,
            "squareform: vector input is required for 'tomatrix'",
        ));
    }
    let len = tensor::tensor_element_len(&tensor);
    let size = matrix_size_for_condensed_length(len).ok_or_else(|| {
        distance_error(
            SQUAREFORM_NAME,
            "squareform: vector length must be n*(n-1)/2 for an integer n",
        )
    })?;
    let out_len = size
        .checked_mul(size)
        .ok_or_else(|| internal_error(SQUAREFORM_NAME, "squareform: output size overflow"))?;
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| internal_error(SQUAREFORM_NAME, format!("squareform: {err}")))?;
    let mut out = zero_storage(storage.numeric_dtype(), out_len)?;
    let mut idx = 0usize;
    for col in 0..size {
        for row in (col + 1)..size {
            let value = storage.value_at(idx).ok_or_else(|| {
                internal_error(SQUAREFORM_NAME, "squareform: input storage is inconsistent")
            })?;
            out.set_value(col * size + row, value)
                .map_err(|err| internal_error(SQUAREFORM_NAME, format!("squareform: {err}")))?;
            out.set_value(row * size + col, value)
                .map_err(|err| internal_error(SQUAREFORM_NAME, format!("squareform: {err}")))?;
            idx += 1;
        }
    }
    Tensor::from_numeric_storage(out, vec![size, size])
        .map_err(|err| internal_error(SQUAREFORM_NAME, format!("squareform: {err}")))
}

fn square_to_vector(tensor: Tensor) -> BuiltinResult<Tensor> {
    let (rows, cols) = observation_matrix_dimensions(&tensor);
    if rows != cols {
        return Err(distance_error(
            SQUAREFORM_NAME,
            "squareform: matrix input must be square",
        ));
    }
    let size = rows;
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| internal_error(SQUAREFORM_NAME, format!("squareform: {err}")))?;
    for idx in 0..size {
        let value = storage.value_at(idx * size + idx).ok_or_else(|| {
            internal_error(SQUAREFORM_NAME, "squareform: input storage is inconsistent")
        })?;
        if !squareform_scalar_is_zero(value) {
            return Err(distance_error(
                SQUAREFORM_NAME,
                "squareform: matrix diagonal must be zero",
            ));
        }
    }
    let len = condensed_length(size)
        .ok_or_else(|| internal_error(SQUAREFORM_NAME, "squareform: output size overflow"))?;
    let mut out = zero_storage(storage.numeric_dtype(), len)?;
    let mut out_idx = 0usize;
    for col in 0..size {
        for row in (col + 1)..size {
            let lower = storage.value_at(col * size + row).ok_or_else(|| {
                internal_error(SQUAREFORM_NAME, "squareform: input storage is inconsistent")
            })?;
            let upper = storage.value_at(row * size + col).ok_or_else(|| {
                internal_error(SQUAREFORM_NAME, "squareform: input storage is inconsistent")
            })?;
            if !squareform_scalars_equal(lower, upper) {
                return Err(distance_error(
                    SQUAREFORM_NAME,
                    "squareform: matrix must be symmetric",
                ));
            }
            out.set_value(out_idx, upper)
                .map_err(|err| internal_error(SQUAREFORM_NAME, format!("squareform: {err}")))?;
            out_idx += 1;
        }
    }
    Tensor::from_numeric_storage(out, vec![1, len])
        .map_err(|err| internal_error(SQUAREFORM_NAME, format!("squareform: {err}")))
}

fn squareform_scalar_is_zero(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value.abs() <= EPS,
        NumericScalar::F32(value) => f64::from(value).abs() <= EPS,
        NumericScalar::I8(value) => value == 0,
        NumericScalar::I16(value) => value == 0,
        NumericScalar::I32(value) => value == 0,
        NumericScalar::I64(value) => value == 0,
        NumericScalar::U8(value) => value == 0,
        NumericScalar::U16(value) => value == 0,
        NumericScalar::U32(value) => value == 0,
        NumericScalar::U64(value) => value == 0,
    }
}

fn squareform_scalars_equal(left: NumericScalar, right: NumericScalar) -> bool {
    match (left, right) {
        (NumericScalar::F64(left), NumericScalar::F64(right)) => (left - right).abs() <= EPS,
        (NumericScalar::F32(left), NumericScalar::F32(right)) => {
            f64::from(left - right).abs() <= EPS
        }
        _ => left == right,
    }
}

fn zero_storage(dtype: NumericDType, len: usize) -> BuiltinResult<NumericStorage> {
    fn allocated<T: Default + Clone>(len: usize) -> BuiltinResult<Vec<T>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| internal_error(SQUAREFORM_NAME, "squareform: output allocation failed"))?;
        values.resize(len, T::default());
        Ok(values)
    }
    Ok(match dtype {
        NumericDType::F64 => NumericStorage::F64(allocated(len)?),
        NumericDType::F32 => NumericStorage::F32(allocated(len)?),
        NumericDType::I8 => NumericStorage::I8(allocated(len)?),
        NumericDType::I16 => NumericStorage::I16(allocated(len)?),
        NumericDType::I32 => NumericStorage::I32(allocated(len)?),
        NumericDType::I64 => NumericStorage::I64(allocated(len)?),
        NumericDType::U8 => NumericStorage::U8(allocated(len)?),
        NumericDType::U16 => NumericStorage::U16(allocated(len)?),
        NumericDType::U32 => NumericStorage::U32(allocated(len)?),
        NumericDType::U64 => NumericStorage::U64(allocated(len)?),
    })
}

#[cfg(test)]
mod shape_privacy_tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn squareform_uses_dense_rank_zero_and_rank_one_matrix_geometry() {
        let scalar = Tensor::new(vec![0.0], vec![]).expect("valid scalar tensor");
        let vector = squareform_compute(scalar, SquareformDirection::Auto)
            .expect("rank-zero tensor follows square-to-vector path");
        assert_eq!(vector.shape(), &[1, 0]);
        assert!(vector.materialize_f64().is_empty());

        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("valid rank-one tensor");
        let matrix = squareform_compute(row, SquareformDirection::Auto)
            .expect("rank-one tensor follows vector-to-square path");
        assert_eq!(matrix.shape(), &[3, 3]);
        assert_eq!(
            matrix.materialize_f64(),
            vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0]
        );
    }
}
