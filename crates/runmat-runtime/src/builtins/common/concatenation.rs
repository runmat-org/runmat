//! Matrix and array concatenation operations
//!
//! This module provides language-compatible matrix concatenation operations.
//! Supports both horizontal concatenation [A, B] and vertical concatenation [A; B].

use crate::builtins::math::elementwise::integer_cast::{integer_values, IntegerTarget};
use runmat_builtins::{
    CharArray, IntValue, NumericDType, NumericScalar, NumericStorage, Tensor, Value,
};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn concat_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).build()
}

/// Converts an f64 code point to a 1x1 `CharArray`.
///
/// Validates that the value is a finite integer in the valid Unicode code point range,
/// then constructs a single-character array. The `error_prefix` is prepended to error
/// messages (e.g., "cat" or "char concat").
pub fn char_array_from_f64_with_prefix(value: f64, error_prefix: &str) -> BuiltinResult<CharArray> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(concat_error(format!(
            "{error_prefix}: expected integer code point"
        )));
    }
    if value < 0.0 || value > u32::MAX as f64 {
        return Err(concat_error(format!(
            "{error_prefix}: code point out of range"
        )));
    }
    let code = value as u32;
    let ch = char::from_u32(code)
        .ok_or_else(|| concat_error(format!("{error_prefix}: invalid code point")))?;
    CharArray::new(vec![ch], 1, 1).map_err(concat_error)
}

fn char_array_from_f64(value: f64) -> BuiltinResult<CharArray> {
    char_array_from_f64_with_prefix(value, "char concat")
}

fn char_array_from_int(value: &IntValue) -> BuiltinResult<CharArray> {
    let code = match value {
        IntValue::I8(value) => u32::try_from(*value),
        IntValue::I16(value) => u32::try_from(*value),
        IntValue::I32(value) => u32::try_from(*value),
        IntValue::I64(value) => u32::try_from(*value),
        IntValue::U8(value) => Ok(u32::from(*value)),
        IntValue::U16(value) => Ok(u32::from(*value)),
        IntValue::U32(value) => Ok(*value),
        IntValue::U64(value) => u32::try_from(*value),
    };
    let code = code.map_err(|_| concat_error("char concat: code point out of range"))?;
    let ch = char::from_u32(code).ok_or_else(|| concat_error("char concat: invalid code point"))?;
    CharArray::new(vec![ch], 1, 1).map_err(concat_error)
}

/// Horizontally concatenate two matrices [A, B]
/// In language: C = [A, B] creates a matrix with A and B side by side
pub fn hcat_matrices(a: &Tensor, b: &Tensor) -> BuiltinResult<Tensor> {
    // Language semantics: [] acts as a neutral element for concatenation
    if a.rows() == 0 && a.cols() == 0 {
        return Ok(b.clone());
    }
    if b.rows() == 0 && b.cols() == 0 {
        return Ok(a.clone());
    }
    if a.rows() != b.rows() {
        return Err(concat_error(format!(
            "Cannot horizontally concatenate matrices with different row counts: {} vs {}",
            a.rows, b.rows
        )));
    }

    if a.integer_storage().is_some() || b.integer_storage().is_some() {
        return hcat_integer_matrices(a, b);
    }

    let new_rows = a.rows();
    let new_cols = a.cols() + b.cols();
    let target = floating_concat_dtype(a, b)?;
    let mut indices = Vec::with_capacity(new_rows * new_cols);

    // Column-major layout: build column-by-column
    for col in 0..new_cols {
        if col < a.cols() {
            for row in 0..a.rows() {
                indices.push((a, row + col * a.rows()));
            }
        } else {
            let bcol = col - a.cols();
            for row in 0..b.rows() {
                indices.push((b, row + bcol * b.rows()));
            }
        }
    }

    floating_concat_tensor(indices, vec![new_rows, new_cols], target)
}

/// Vertically concatenate two matrices [A; B]
/// In language: C = [A; B] creates a matrix with A on top and B below
pub fn vcat_matrices(a: &Tensor, b: &Tensor) -> BuiltinResult<Tensor> {
    // Language semantics: [] acts as a neutral element for concatenation
    if a.rows() == 0 && a.cols() == 0 {
        return Ok(b.clone());
    }
    if b.rows() == 0 && b.cols() == 0 {
        return Ok(a.clone());
    }
    if a.cols() != b.cols() {
        return Err(concat_error(format!(
            "Cannot vertically concatenate matrices with different column counts: {} vs {}",
            a.cols, b.cols
        )));
    }

    if a.integer_storage().is_some() || b.integer_storage().is_some() {
        return vcat_integer_matrices(a, b);
    }

    let new_rows = a.rows() + b.rows();
    let new_cols = a.cols();
    let target = floating_concat_dtype(a, b)?;
    let mut indices = Vec::with_capacity(new_rows * new_cols);

    // Column-major: each output column contains A's rows followed by B's rows.
    for col in 0..a.cols() {
        for row in 0..a.rows() {
            indices.push((a, row + col * a.rows()));
        }
        for row in 0..b.rows() {
            indices.push((b, row + col * b.rows()));
        }
    }

    floating_concat_tensor(indices, vec![new_rows, new_cols], target)
}

fn floating_concat_dtype(a: &Tensor, b: &Tensor) -> BuiltinResult<NumericDType> {
    match (a.numeric_dtype(), b.numeric_dtype()) {
        (NumericDType::F64, NumericDType::F64) => Ok(NumericDType::F64),
        (NumericDType::F32, NumericDType::F32)
        | (NumericDType::F32, NumericDType::F64)
        | (NumericDType::F64, NumericDType::F32) => Ok(NumericDType::F32),
        (left, right) => Err(concat_error(format!(
            "floating concatenation received unexpected {} and {} storage",
            left.class_name(),
            right.class_name()
        ))),
    }
}

fn floating_concat_tensor(
    indices: Vec<(&Tensor, usize)>,
    shape: Vec<usize>,
    target: NumericDType,
) -> BuiltinResult<Tensor> {
    let storage = match target {
        NumericDType::F64 => NumericStorage::F64(
            indices
                .into_iter()
                .map(|(tensor, index)| floating_value_f64(tensor, index))
                .collect::<BuiltinResult<Vec<_>>>()?,
        ),
        NumericDType::F32 => NumericStorage::F32(
            indices
                .into_iter()
                .map(|(tensor, index)| floating_value_f64(tensor, index).map(|value| value as f32))
                .collect::<BuiltinResult<Vec<_>>>()?,
        ),
        NumericDType::I8
        | NumericDType::I16
        | NumericDType::I32
        | NumericDType::I64
        | NumericDType::U8
        | NumericDType::U16
        | NumericDType::U32
        | NumericDType::U64 => unreachable!("floating concat target is validated"),
    };
    Tensor::from_numeric_storage(storage, shape).map_err(concat_error)
}

fn floating_value_f64(tensor: &Tensor, index: usize) -> BuiltinResult<f64> {
    match tensor.numeric_value_at(index) {
        Some(NumericScalar::F64(value)) => Ok(value),
        Some(NumericScalar::F32(value)) => Ok(f64::from(value)),
        Some(value) => Err(concat_error(format!(
            "floating concatenation received unexpected {} sample",
            value.class_name()
        ))),
        None => Err(concat_error(format!(
            "floating concatenation could not read {} element {index}",
            tensor.numeric_dtype().class_name()
        ))),
    }
}

fn hcat_integer_matrices(a: &Tensor, b: &Tensor) -> BuiltinResult<Tensor> {
    let target = leftmost_tensor_integer_target(a, b)
        .expect("integer hcat path requires at least one integer tensor");
    let new_rows = a.rows();
    let new_cols = a.cols() + b.cols();
    let mut values = Vec::with_capacity(new_rows * new_cols);

    for col in 0..new_cols {
        if col < a.cols() {
            for row in 0..a.rows() {
                values.push(integer_value_at(target, a, row + col * a.rows()));
            }
        } else {
            let bcol = col - a.cols();
            for row in 0..b.rows() {
                values.push(integer_value_at(target, b, row + bcol * b.rows()));
            }
        }
    }

    Tensor::new_integer(target.storage(values), vec![new_rows, new_cols]).map_err(concat_error)
}

fn vcat_integer_matrices(a: &Tensor, b: &Tensor) -> BuiltinResult<Tensor> {
    let target = leftmost_tensor_integer_target(a, b)
        .expect("integer vcat path requires at least one integer tensor");
    let new_rows = a.rows() + b.rows();
    let new_cols = a.cols();
    let mut values = Vec::with_capacity(new_rows * new_cols);

    for col in 0..new_cols {
        for row in 0..a.rows() {
            values.push(integer_value_at(target, a, row + col * a.rows()));
        }
        for row in 0..b.rows() {
            values.push(integer_value_at(target, b, row + col * b.rows()));
        }
    }

    Tensor::new_integer(target.storage(values), vec![new_rows, new_cols]).map_err(concat_error)
}

fn leftmost_tensor_integer_target(a: &Tensor, b: &Tensor) -> Option<IntegerTarget> {
    a.integer_storage()
        .map(IntegerTarget::from_storage)
        .or_else(|| b.integer_storage().map(IntegerTarget::from_storage))
}

fn integer_value_at(target: IntegerTarget, tensor: &Tensor, index: usize) -> IntValue {
    match tensor.integer_storage() {
        Some(storage) => target.cast_int(
            &storage
                .value_at(index)
                .expect("integer tensor storage length matches tensor shape"),
        ),
        None => target.cast_scalar(
            floating_value_f64(tensor, index)
                .expect("noninteger concat operand has floating storage"),
        ),
    }
}

/// Concatenate values horizontally - handles mixed scalars and matrices
pub fn hcat_values(values: &[Value]) -> BuiltinResult<Value> {
    if values.is_empty() {
        return Ok(Value::Tensor(
            Tensor::new(vec![], vec![0, 0]).map_err(concat_error)?,
        ));
    }

    // If any operand is a string or string array, perform string-array concatenation
    let has_str = values
        .iter()
        .any(|v| matches!(v, Value::String(_) | Value::StringArray(_)));
    let has_char = values.iter().any(|v| matches!(v, Value::CharArray(_)));
    if has_str {
        // Normalize all to string-arrays, then horizontal concat by columns
        // Determine row count: if any is string array, its rows; if string scalar or numeric scalar, rows=1
        let mut rows: Option<usize> = None;
        let mut cols_total = 0usize;
        let mut blocks: Vec<runmat_builtins::StringArray> = Vec::new();
        for v in values {
            match v {
                Value::StringArray(sa) => {
                    if rows.is_none() {
                        rows = Some(sa.rows());
                    } else if rows != Some(sa.rows()) {
                        return Err(concat_error("string hcat: row mismatch"));
                    }
                    cols_total += sa.cols();
                    blocks.push(sa.clone());
                }
                Value::String(s) => {
                    let sa =
                        runmat_builtins::StringArray::new(vec![s.clone()], vec![1, 1]).unwrap();
                    if rows.is_none() {
                        rows = Some(1);
                    } else if rows != Some(1) {
                        return Err(concat_error("string hcat: row mismatch"));
                    }
                    cols_total += 1;
                    blocks.push(sa);
                }
                Value::CharArray(ca) => {
                    // Convert char array to string array by rows
                    if ca.rows == 0 {
                        continue;
                    }
                    if rows.is_none() {
                        rows = Some(ca.rows);
                    } else if rows != Some(ca.rows) {
                        return Err(concat_error("string hcat: row mismatch"));
                    }
                    let mut out: Vec<String> = Vec::with_capacity(ca.rows);
                    for r in 0..ca.rows {
                        let mut s = String::with_capacity(ca.cols);
                        for c in 0..ca.cols {
                            s.push(ca.data[r * ca.cols + c]);
                        }
                        out.push(s);
                    }
                    let sa = runmat_builtins::StringArray::new(out, vec![ca.rows, 1]).unwrap();
                    cols_total += 1;
                    blocks.push(sa);
                }
                Value::Num(n) => {
                    let sa =
                        runmat_builtins::StringArray::new(vec![n.to_string()], vec![1, 1]).unwrap();
                    if rows.is_none() {
                        rows = Some(1);
                    } else if rows != Some(1) {
                        return Err(concat_error("string hcat: row mismatch"));
                    }
                    cols_total += 1;
                    blocks.push(sa);
                }
                Value::Complex(re, im) => {
                    let sa = runmat_builtins::StringArray::new(
                        vec![runmat_builtins::Value::Complex(*re, *im).to_string()],
                        vec![1, 1],
                    )
                    .unwrap();
                    if rows.is_none() {
                        rows = Some(1);
                    } else if rows != Some(1) {
                        return Err(concat_error("string hcat: row mismatch"));
                    }
                    cols_total += 1;
                    blocks.push(sa);
                }
                Value::Int(i) => {
                    let sa =
                        runmat_builtins::StringArray::new(vec![i.decimal_string()], vec![1, 1])
                            .unwrap();
                    if rows.is_none() {
                        rows = Some(1);
                    } else if rows != Some(1) {
                        return Err(concat_error("string hcat: row mismatch"));
                    }
                    cols_total += 1;
                    blocks.push(sa);
                }
                Value::Tensor(_) | Value::Cell(_) => {
                    return Err(concat_error(format!(
                        "Cannot concatenate value of type {v:?} with string array"
                    )))
                }
                _ => {
                    return Err(concat_error(format!(
                        "Cannot concatenate value of type {v:?} with string array"
                    )))
                }
            }
        }
        let rows = rows.unwrap_or(0);
        let mut data: Vec<String> = Vec::with_capacity(rows * cols_total);
        for cacc in 0..cols_total {
            let _ = cacc;
        }
        // Stitch columns block-by-block in column-major
        for block in &blocks {
            for c in 0..block.cols() {
                for r in 0..rows {
                    let idx = r + c * rows;
                    data.push(block.data[idx].clone());
                }
            }
        }
        let sa = runmat_builtins::StringArray::new(data, vec![rows, cols_total])
            .map_err(|e| concat_error(format!("string hcat: {e}")))?;
        return Ok(Value::StringArray(sa));
    }

    if has_char {
        let mut rows: Option<usize> = None;
        let mut cols_total = 0usize;
        let mut blocks: Vec<CharArray> = Vec::new();
        for v in values {
            match v {
                Value::CharArray(ca) => {
                    if ca.rows == 0 && ca.cols == 0 {
                        continue;
                    }
                    if rows.is_none() {
                        rows = Some(ca.rows);
                    } else if rows != Some(ca.rows) {
                        return Err(concat_error("char hcat: row mismatch"));
                    }
                    cols_total += ca.cols;
                    blocks.push(ca.clone());
                }
                Value::Num(n) => {
                    let ca = char_array_from_f64(*n)?;
                    if rows.is_none() {
                        rows = Some(1);
                    } else if rows != Some(1) {
                        return Err(concat_error("char hcat: row mismatch"));
                    }
                    cols_total += 1;
                    blocks.push(ca);
                }
                Value::Int(i) => {
                    let ca = char_array_from_int(i)?;
                    if rows.is_none() {
                        rows = Some(1);
                    } else if rows != Some(1) {
                        return Err(concat_error("char hcat: row mismatch"));
                    }
                    cols_total += 1;
                    blocks.push(ca);
                }
                Value::Bool(flag) => {
                    let ca = char_array_from_f64(if *flag { 1.0 } else { 0.0 })?;
                    if rows.is_none() {
                        rows = Some(1);
                    } else if rows != Some(1) {
                        return Err(concat_error("char hcat: row mismatch"));
                    }
                    cols_total += 1;
                    blocks.push(ca);
                }
                _ => {
                    return Err(concat_error(format!(
                        "Cannot concatenate value of type {v:?} with char array"
                    )))
                }
            }
        }
        let rows = rows.unwrap_or(0);
        let mut data: Vec<char> = Vec::with_capacity(rows * cols_total);
        for r in 0..rows {
            for block in &blocks {
                for c in 0..block.cols {
                    data.push(block.data[r * block.cols + c]);
                }
            }
        }
        let ca = CharArray::new(data, rows, cols_total)
            .map_err(|e| concat_error(format!("char hcat: {e}")))?;
        return Ok(Value::CharArray(ca));
    }

    if let Some(target) = leftmost_value_integer_target(values) {
        return hcat_integer_values(target, values);
    }

    // Convert all scalars to 1x1 matrices for uniform processing
    let mut matrices = Vec::new();
    let mut _total_cols = 0;
    let mut rows = 0;

    for value in values {
        match value {
            Value::Num(n) => {
                let matrix = Tensor::new_2d(vec![*n], 1, 1).map_err(concat_error)?;
                if rows == 0 {
                    rows = 1;
                } else if rows != 1 {
                    return Err(concat_error(
                        "Cannot concatenate scalar with multi-row matrix",
                    ));
                }
                _total_cols += 1;
                matrices.push(matrix);
            }
            Value::Complex(re, _im) => {
                let matrix = Tensor::new_2d(vec![*re], 1, 1).map_err(concat_error)?; // real part in numeric hcat coercion
                if rows == 0 {
                    rows = 1;
                } else if rows != 1 {
                    return Err(concat_error(
                        "Cannot concatenate scalar with multi-row matrix",
                    ));
                }
                _total_cols += 1;
                matrices.push(matrix);
            }
            Value::Int(i) => {
                let matrix = Tensor::new_2d(vec![i.to_f64()], 1, 1).map_err(concat_error)?;
                if rows == 0 {
                    rows = 1;
                } else if rows != 1 {
                    return Err(concat_error(
                        "Cannot concatenate scalar with multi-row matrix",
                    ));
                }
                _total_cols += 1;
                matrices.push(matrix);
            }
            Value::Tensor(m) => {
                // Skip true empty 0x0 operands (neutral element)
                if m.rows() == 0 && m.cols() == 0 {
                    continue;
                }
                if rows == 0 {
                    rows = m.rows();
                } else if rows != m.rows() {
                    return Err(concat_error(format!(
                        "Cannot concatenate matrices with different row counts: {} vs {}",
                        rows,
                        m.rows()
                    )));
                }
                _total_cols += m.cols();
                matrices.push(m.clone());
            }
            _ => {
                return Err(concat_error(format!(
                    "Cannot concatenate value of type {value:?}"
                )))
            }
        }
    }

    // Now concatenate all matrices horizontally
    if matrices.is_empty() {
        return Ok(Value::Tensor(
            Tensor::new(Vec::new(), vec![0, 0]).map_err(concat_error)?,
        ));
    }
    let mut result = matrices[0].clone();
    for matrix in &matrices[1..] {
        result = hcat_matrices(&result, matrix)?;
    }

    Ok(Value::Tensor(result))
}

/// Concatenate values vertically - handles mixed scalars and matrices
pub fn vcat_values(values: &[Value]) -> BuiltinResult<Value> {
    if values.is_empty() {
        return Ok(Value::Tensor(
            Tensor::new(vec![], vec![0, 0]).map_err(concat_error)?,
        ));
    }

    // If any operand is a string or string array, perform string-array vertical concatenation by stacking rows
    let has_str = values
        .iter()
        .any(|v| matches!(v, Value::String(_) | Value::StringArray(_)));
    let has_char = values.iter().any(|v| matches!(v, Value::CharArray(_)));
    if has_str {
        // Normalize to string-arrays; for scalars, treat as 1x1
        let mut cols: Option<usize> = None;
        let mut rows_total = 0usize;
        let mut blocks: Vec<runmat_builtins::StringArray> = Vec::new();
        for v in values {
            match v {
                Value::StringArray(sa) => {
                    if cols.is_none() {
                        cols = Some(sa.cols());
                    } else if cols != Some(sa.cols()) {
                        return Err(concat_error("string vcat: column mismatch"));
                    }
                    rows_total += sa.rows();
                    blocks.push(sa.clone());
                }
                Value::String(s) => {
                    let sa =
                        runmat_builtins::StringArray::new(vec![s.clone()], vec![1, 1]).unwrap();
                    rows_total += 1;
                    if cols.is_none() {
                        cols = Some(1);
                    } else if cols != Some(1) {
                        return Err(concat_error("string vcat: column mismatch"));
                    }
                    blocks.push(sa);
                }
                Value::CharArray(ca) => {
                    if ca.cols == 0 {
                        continue;
                    }
                    let out: String = ca.data.iter().collect();
                    let sa = runmat_builtins::StringArray::new(vec![out], vec![1, 1]).unwrap();
                    rows_total += 1;
                    if cols.is_none() {
                        cols = Some(1);
                    } else if cols != Some(1) {
                        return Err(concat_error("string vcat: column mismatch"));
                    }
                    blocks.push(sa);
                }
                Value::Num(n) => {
                    let sa =
                        runmat_builtins::StringArray::new(vec![n.to_string()], vec![1, 1]).unwrap();
                    rows_total += 1;
                    if cols.is_none() {
                        cols = Some(1);
                    } else if cols != Some(1) {
                        return Err(concat_error("string vcat: column mismatch"));
                    }
                    blocks.push(sa);
                }
                Value::Complex(re, im) => {
                    let sa = runmat_builtins::StringArray::new(
                        vec![runmat_builtins::Value::Complex(*re, *im).to_string()],
                        vec![1, 1],
                    )
                    .unwrap();
                    rows_total += 1;
                    if cols.is_none() {
                        cols = Some(1);
                    } else if cols != Some(1) {
                        return Err(concat_error("string vcat: column mismatch"));
                    }
                    blocks.push(sa);
                }
                Value::Int(i) => {
                    let sa =
                        runmat_builtins::StringArray::new(vec![i.decimal_string()], vec![1, 1])
                            .unwrap();
                    rows_total += 1;
                    if cols.is_none() {
                        cols = Some(1);
                    } else if cols != Some(1) {
                        return Err(concat_error("string vcat: column mismatch"));
                    }
                    blocks.push(sa);
                }
                _ => {
                    return Err(concat_error(format!(
                        "Cannot concatenate value of type {v:?} with string array"
                    )))
                }
            }
        }
        let cols = cols.unwrap_or(0);
        let mut data: Vec<String> = Vec::with_capacity(rows_total * cols);
        // Stack rows: copy columns for each block into data
        for block in &blocks {
            for c in 0..cols {
                for r in 0..block.rows() {
                    let idx = r + c * block.rows();
                    data.push(block.data[idx].clone());
                }
            }
        }
        let sa = runmat_builtins::StringArray::new(data, vec![rows_total, cols])
            .map_err(|e| concat_error(format!("string vcat: {e}")))?;
        return Ok(Value::StringArray(sa));
    }

    if has_char {
        let mut cols: Option<usize> = None;
        let mut rows_total = 0usize;
        let mut blocks: Vec<CharArray> = Vec::new();
        for v in values {
            match v {
                Value::CharArray(ca) => {
                    if ca.rows == 0 && ca.cols == 0 {
                        continue;
                    }
                    if cols.is_none() {
                        cols = Some(ca.cols);
                    } else if cols != Some(ca.cols) {
                        return Err(concat_error("char vcat: column mismatch"));
                    }
                    rows_total += ca.rows;
                    blocks.push(ca.clone());
                }
                Value::Num(n) => {
                    let ca = char_array_from_f64(*n)?;
                    if cols.is_none() {
                        cols = Some(1);
                    } else if cols != Some(1) {
                        return Err(concat_error("char vcat: column mismatch"));
                    }
                    rows_total += 1;
                    blocks.push(ca);
                }
                Value::Int(i) => {
                    let ca = char_array_from_int(i)?;
                    if cols.is_none() {
                        cols = Some(1);
                    } else if cols != Some(1) {
                        return Err(concat_error("char vcat: column mismatch"));
                    }
                    rows_total += 1;
                    blocks.push(ca);
                }
                Value::Bool(flag) => {
                    let ca = char_array_from_f64(if *flag { 1.0 } else { 0.0 })?;
                    if cols.is_none() {
                        cols = Some(1);
                    } else if cols != Some(1) {
                        return Err(concat_error("char vcat: column mismatch"));
                    }
                    rows_total += 1;
                    blocks.push(ca);
                }
                _ => {
                    return Err(concat_error(format!(
                        "Cannot concatenate value of type {v:?} with char array"
                    )))
                }
            }
        }
        let cols = cols.unwrap_or(0);
        let mut data: Vec<char> = Vec::with_capacity(rows_total * cols);
        for block in &blocks {
            for r in 0..block.rows {
                for c in 0..cols {
                    data.push(block.data[r * block.cols + c]);
                }
            }
        }
        let ca = CharArray::new(data, rows_total, cols)
            .map_err(|e| concat_error(format!("char vcat: {e}")))?;
        return Ok(Value::CharArray(ca));
    }

    if let Some(target) = leftmost_value_integer_target(values) {
        return vcat_integer_values(target, values);
    }

    // Convert all scalars to 1x1 matrices for uniform processing
    let mut matrices = Vec::new();
    let mut _total_rows = 0;
    let mut cols = 0;

    for value in values {
        match value {
            Value::Num(n) => {
                let matrix = Tensor::new_2d(vec![*n], 1, 1).map_err(concat_error)?;
                if cols == 0 {
                    cols = 1;
                } else if cols != 1 {
                    return Err(concat_error(
                        "Cannot concatenate scalar with multi-column matrix",
                    ));
                }
                _total_rows += 1;
                matrices.push(matrix);
            }
            Value::Complex(re, _im) => {
                let matrix = Tensor::new_2d(vec![*re], 1, 1).map_err(concat_error)?;
                if cols == 0 {
                    cols = 1;
                } else if cols != 1 {
                    return Err(concat_error(
                        "Cannot concatenate scalar with multi-column matrix",
                    ));
                }
                _total_rows += 1;
                matrices.push(matrix);
            }
            Value::Int(i) => {
                let matrix = Tensor::new_2d(vec![i.to_f64()], 1, 1).map_err(concat_error)?;
                if cols == 0 {
                    cols = 1;
                } else if cols != 1 {
                    return Err(concat_error(
                        "Cannot concatenate scalar with multi-column matrix",
                    ));
                }
                _total_rows += 1;
                matrices.push(matrix);
            }
            Value::Tensor(m) => {
                // Skip true empty 0x0 operands (neutral element)
                if m.rows() == 0 && m.cols() == 0 {
                    continue;
                }
                if cols == 0 {
                    cols = m.cols();
                } else if cols != m.cols() {
                    return Err(concat_error(format!(
                        "Cannot concatenate matrices with different column counts: {} vs {}",
                        cols,
                        m.cols()
                    )));
                }
                _total_rows += m.rows();
                matrices.push(m.clone());
            }
            _ => {
                return Err(concat_error(format!(
                    "Cannot concatenate value of type {value:?}"
                )))
            }
        }
    }

    // Now concatenate all matrices vertically
    if matrices.is_empty() {
        return Ok(Value::Tensor(
            Tensor::new(Vec::new(), vec![0, 0]).map_err(concat_error)?,
        ));
    }
    let mut result = matrices[0].clone();
    for matrix in &matrices[1..] {
        result = vcat_matrices(&result, matrix)?;
    }

    Ok(Value::Tensor(result))
}

fn leftmost_value_integer_target(values: &[Value]) -> Option<IntegerTarget> {
    let mut empty_target = None;
    for value in values {
        match value {
            Value::Int(value) => return Some(IntegerTarget::from_int_value(value)),
            Value::Tensor(tensor) => {
                if let Some(storage) = tensor.integer_storage() {
                    let target = IntegerTarget::from_storage(storage);
                    if !(tensor.rows() == 0 && tensor.cols() == 0) {
                        return Some(target);
                    }
                    empty_target.get_or_insert(target);
                }
            }
            _ => {}
        }
    }
    empty_target
}

fn hcat_integer_values(target: IntegerTarget, values: &[Value]) -> BuiltinResult<Value> {
    let mut matrices = Vec::new();
    let mut rows = 0;

    for value in values {
        let matrix = integer_matrix_from_value(target, value)?;
        if matrix.rows() == 0 && matrix.cols() == 0 {
            continue;
        }
        if rows == 0 {
            rows = matrix.rows();
        } else if rows != matrix.rows() {
            return Err(concat_error(format!(
                "Cannot concatenate matrices with different row counts: {} vs {}",
                rows,
                matrix.rows()
            )));
        }
        matrices.push(matrix);
    }

    if matrices.is_empty() {
        return Ok(Value::Tensor(
            Tensor::new_integer(target.storage(Vec::new()), vec![0, 0]).map_err(concat_error)?,
        ));
    }

    let mut result = matrices[0].clone();
    for matrix in &matrices[1..] {
        result = hcat_matrices(&result, matrix)?;
    }
    Ok(Value::Tensor(result))
}

fn vcat_integer_values(target: IntegerTarget, values: &[Value]) -> BuiltinResult<Value> {
    let mut matrices = Vec::new();
    let mut cols = 0;

    for value in values {
        let matrix = integer_matrix_from_value(target, value)?;
        if matrix.rows() == 0 && matrix.cols() == 0 {
            continue;
        }
        if cols == 0 {
            cols = matrix.cols();
        } else if cols != matrix.cols() {
            return Err(concat_error(format!(
                "Cannot concatenate matrices with different column counts: {} vs {}",
                cols,
                matrix.cols()
            )));
        }
        matrices.push(matrix);
    }

    if matrices.is_empty() {
        return Ok(Value::Tensor(
            Tensor::new_integer(target.storage(Vec::new()), vec![0, 0]).map_err(concat_error)?,
        ));
    }

    let mut result = matrices[0].clone();
    for matrix in &matrices[1..] {
        result = vcat_matrices(&result, matrix)?;
    }
    Ok(Value::Tensor(result))
}

fn integer_matrix_from_value(target: IntegerTarget, value: &Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Int(value) => {
            Tensor::new_integer(target.storage(vec![target.cast_int(value)]), vec![1, 1])
                .map_err(concat_error)
        }
        Value::Num(value) => {
            Tensor::new_integer(target.storage(vec![target.cast_scalar(*value)]), vec![1, 1])
                .map_err(concat_error)
        }
        Value::Tensor(tensor) => {
            let values = match tensor.integer_storage() {
                Some(storage) => integer_values(storage.clone())
                    .iter()
                    .map(|value| target.cast_int(value))
                    .collect(),
                None => (0..tensor.len())
                    .map(|index| {
                        floating_value_f64(tensor, index).map(|value| target.cast_scalar(value))
                    })
                    .collect::<BuiltinResult<Vec<_>>>()?,
            };
            Tensor::new_integer(target.storage(values), tensor.shape.clone()).map_err(concat_error)
        }
        other => Err(concat_error(format!(
            "Cannot concatenate integer value with type {other:?}"
        ))),
    }
}

/// Create a matrix from a 2D array of Values with proper concatenation semantics
/// This handles the case where matrix elements can be variables, not just literals
pub async fn create_matrix_from_values(rows: &[Vec<Value>]) -> BuiltinResult<Value> {
    if rows.is_empty() {
        return Ok(Value::Tensor(
            Tensor::new(vec![], vec![0, 0]).map_err(concat_error)?,
        ));
    }

    // Build each row using horzcat builtin to preserve canonical semantics
    let mut row_matrices: Vec<Value> = Vec::with_capacity(rows.len());
    for row in rows {
        let row_value = if row.is_empty() {
            Value::Tensor(Tensor::new(vec![], vec![0, 0]).map_err(concat_error)?)
        } else {
            crate::call_builtin_async("horzcat", row).await?
        };
        row_matrices.push(row_value);
    }

    // Stack rows using vertcat builtin
    if row_matrices.is_empty() {
        Ok(Value::Tensor(
            Tensor::new(vec![], vec![0, 0]).map_err(concat_error)?,
        ))
    } else if row_matrices.len() == 1 {
        Ok(row_matrices.into_iter().next().unwrap())
    } else {
        Ok(crate::call_builtin_async("vertcat", &row_matrices).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_matrices() {
        let a = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = Tensor::new_2d(vec![5.0, 6.0], 2, 1).unwrap();

        let result = hcat_matrices(&a, &b).unwrap();
        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 3);
        // Column-major result: [ [1 3 5]; [2 4 6] ] data
        assert_eq!(
            result.as_f64_slice().expect("double result"),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_vcat_matrices() {
        let a = Tensor::new_2d(vec![1.0, 2.0], 1, 2).unwrap();
        let b = Tensor::new_2d(vec![3.0, 4.0], 1, 2).unwrap();

        let result = vcat_matrices(&a, &b).unwrap();
        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
        // [1 2; 3 4] in column-major storage.
        assert_eq!(
            result.as_f64_slice().expect("double result"),
            &[1.0, 3.0, 2.0, 4.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_scalars() {
        let values = vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)];
        let result = hcat_values(&values).unwrap();

        if let Value::Tensor(m) = result {
            assert_eq!(m.rows(), 1);
            assert_eq!(m.cols(), 3);
            // Column-major: 1x3 row vector still row-major visually, data order follows cols
            assert_eq!(m.as_f64_slice().expect("double result"), &[1.0, 2.0, 3.0]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_vcat_values_scalars() {
        let values = vec![Value::Num(1.0), Value::Num(2.0)];
        let result = vcat_values(&values).unwrap();

        if let Value::Tensor(m) = result {
            assert_eq!(m.rows(), 2);
            assert_eq!(m.cols(), 1);
            assert_eq!(m.as_f64_slice().expect("double result"), &[1.0, 2.0]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[test]
    fn floating_concatenation_preserves_native_single_and_layout() {
        let single_row = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let double_row = Tensor::new(vec![3.25, 4.5], vec![1, 2]).unwrap();
        let vertical = vcat_matrices(&single_row, &double_row).unwrap();
        assert_eq!(vertical.shape, vec![2, 2]);
        assert_eq!(
            vertical.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 3.25, 2.0, 4.5])
        );

        let single_column = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let double_column = Tensor::new(vec![3.25, 4.5], vec![2, 1]).unwrap();
        let horizontal = hcat_matrices(&single_column, &double_column).unwrap();
        assert_eq!(horizontal.shape, vec![2, 2]);
        assert_eq!(
            horizontal.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 2.0, 3.25, 4.5])
        );
    }

    #[test]
    fn all_neutral_double_operands_return_empty_matrix() {
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        let Value::Tensor(horizontal) = hcat_values(&[empty.clone(), empty.clone()]).unwrap()
        else {
            panic!("expected horizontal empty tensor");
        };
        let Value::Tensor(vertical) = vcat_values(&[empty.clone(), empty]).unwrap() else {
            panic!("expected vertical empty tensor");
        };
        assert_eq!(horizontal.shape, vec![0, 0]);
        assert_eq!(vertical.shape, vec![0, 0]);
        assert!(horizontal.is_empty());
        assert!(vertical.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn direct_matrix_concatenation_preserves_leftmost_integer_storage() {
        let left = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]),
            vec![2, 1],
        )
        .expect("left integer tensor");
        let right = Tensor::new(vec![3.5, 4.5], vec![2, 1]).expect("right double tensor");

        let horizontal = hcat_matrices(&left, &right).expect("integer hcat");
        assert_eq!(horizontal.shape, vec![2, 2]);
        assert_eq!(
            horizontal.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::U64(vec![
                u64::MAX,
                9_223_372_036_854_775_808,
                4,
                5,
            ]))
        );

        let top = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I8(vec![12, -8]),
            vec![1, 2],
        )
        .expect("top integer tensor");
        let bottom = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX, 2]),
            vec![1, 2],
        )
        .expect("bottom integer tensor");
        let vertical = vcat_matrices(&top, &bottom).expect("integer vcat");
        assert_eq!(vertical.shape, vec![2, 2]);
        assert_eq!(
            vertical.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::I8(vec![
                12,
                i8::MAX,
                -8,
                2,
            ]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn direct_value_concatenation_preserves_exact_integer_scalars() {
        let row = hcat_values(&[
            Value::Int(runmat_builtins::IntValue::U64(9_223_372_036_854_775_808)),
            Value::Num(2.5),
            Value::Int(runmat_builtins::IntValue::U64(u64::MAX)),
        ])
        .expect("integer hcat values");
        let Value::Tensor(row) = row else {
            panic!("expected integer tensor row");
        };
        assert_eq!(row.shape, vec![1, 3]);
        assert_eq!(
            row.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::U64(vec![
                9_223_372_036_854_775_808,
                3,
                u64::MAX,
            ]))
        );

        let column = vcat_values(&[
            Value::Int(runmat_builtins::IntValue::I16(-7)),
            Value::Num(40000.0),
            Value::Int(runmat_builtins::IntValue::U64(u64::MAX)),
        ])
        .expect("integer vcat values");
        let Value::Tensor(column) = column else {
            panic!("expected integer tensor column");
        };
        assert_eq!(column.shape, vec![3, 1]);
        assert_eq!(
            column.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::I16(vec![
                -7,
                i16::MAX,
                i16::MAX,
            ]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn direct_value_concatenation_preserves_empty_integer_class() {
        let empty =
            Tensor::new_integer(runmat_builtins::IntegerStorage::U32(Vec::new()), vec![0, 0])
                .expect("empty integer tensor");

        let Value::Tensor(row) = hcat_values(&[Value::Tensor(empty.clone())]).expect("hcat empty")
        else {
            panic!("expected tensor");
        };
        assert_eq!(row.shape, vec![0, 0]);
        assert_eq!(
            row.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::U32(Vec::new()))
        );

        let Value::Tensor(column) = vcat_values(&[Value::Tensor(empty)]).expect("vcat empty")
        else {
            panic!("expected tensor");
        };
        assert_eq!(column.shape, vec![0, 0]);
        assert_eq!(
            column.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::U32(Vec::new()))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_concatenation_preserves_exact_uint64_scalar_text() {
        let maximum = Value::Int(runmat_builtins::IntValue::U64(u64::MAX));

        let Value::StringArray(horizontal) =
            hcat_values(&[Value::String("id".to_string()), maximum.clone()]).expect("hcat")
        else {
            panic!("expected string array");
        };
        assert_eq!(horizontal.shape, vec![1, 2]);
        assert_eq!(horizontal.data, vec!["id", "18446744073709551615"]);

        let Value::StringArray(vertical) =
            vcat_values(&[Value::String("id".to_string()), maximum]).expect("vcat")
        else {
            panic!("expected string array");
        };
        assert_eq!(vertical.shape, vec![2, 1]);
        assert_eq!(vertical.data, vec!["id", "18446744073709551615"]);
    }
}
