//! Matrix and array concatenation operations
//!
//! This module provides language-compatible matrix concatenation operations.
//! Supports both horizontal concatenation [A, B] and vertical concatenation [A; B].

use runmat_builtins::{CharArray, SymbolicArray, SymbolicExpr, Tensor, Value};

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

fn has_symbolic_operand(values: &[Value]) -> bool {
    values
        .iter()
        .any(|value| matches!(value, Value::Symbolic(_) | Value::SymbolicArray(_)))
}

fn numeric_scalar_to_symbolic(value: &Value) -> Option<SymbolicExpr> {
    match value {
        Value::Symbolic(expr) => Some(expr.clone()),
        Value::Num(n) => Some(SymbolicExpr::constant(*n)),
        Value::Int(i) => Some(SymbolicExpr::constant(i.to_f64())),
        Value::Bool(flag) => Some(SymbolicExpr::constant(if *flag { 1.0 } else { 0.0 })),
        _ => None,
    }
}

fn normalize_symbolic_concat_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() == 1 && shape[0] != 1 {
        vec![1, shape[0]]
    } else if crate::builtins::common::shape::is_scalar_shape(shape) {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

#[derive(Clone)]
struct SymbolicConcatBlock {
    data: Vec<SymbolicExpr>,
    shape: Vec<usize>,
}

impl SymbolicConcatBlock {
    fn new(data: Vec<SymbolicExpr>, shape: Vec<usize>) -> BuiltinResult<Self> {
        let expected_len = checked_shape_len(&shape)?;
        if data.len() != expected_len {
            return Err(concat_error(format!(
                "Symbolic array data length {} does not match shape {:?} (expected {} elements)",
                data.len(),
                shape,
                expected_len
            )));
        }
        Ok(Self { data, shape })
    }

    fn rows(&self) -> usize {
        self.shape[0]
    }

    fn cols(&self) -> usize {
        self.shape[1]
    }
}

fn checked_shape_len(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| concat_error("Symbolic array dimensions overflow"))
    })
}

fn dim_or_one(shape: &[usize], dim: usize) -> usize {
    shape.get(dim).copied().unwrap_or(1)
}

fn symbolic_block_from_value(value: &Value) -> BuiltinResult<Option<SymbolicConcatBlock>> {
    match value {
        Value::SymbolicArray(array) => {
            let shape = normalize_symbolic_concat_shape(&array.shape);
            if shape[0] == 0 && shape[1] == 0 {
                return Ok(None);
            }
            SymbolicConcatBlock::new(array.data.clone(), shape).map(Some)
        }
        Value::Tensor(tensor) => {
            let shape = normalize_symbolic_concat_shape(&tensor.shape);
            if shape[0] == 0 && shape[1] == 0 {
                return Ok(None);
            }
            let data = tensor
                .data
                .iter()
                .map(|value| SymbolicExpr::constant(*value))
                .collect();
            SymbolicConcatBlock::new(data, shape).map(Some)
        }
        Value::LogicalArray(array) => {
            let shape = normalize_symbolic_concat_shape(&array.shape);
            if shape[0] == 0 && shape[1] == 0 {
                return Ok(None);
            }
            let data = array
                .data
                .iter()
                .map(|value| SymbolicExpr::constant(if *value == 0 { 0.0 } else { 1.0 }))
                .collect();
            SymbolicConcatBlock::new(data, shape).map(Some)
        }
        _ => {
            if let Some(expr) = numeric_scalar_to_symbolic(value) {
                SymbolicConcatBlock::new(vec![expr], vec![1, 1]).map(Some)
            } else {
                Err(concat_error(format!(
                    "Cannot concatenate value of type {value:?} with symbolic array"
                )))
            }
        }
    }
}

fn hcat_symbolic_values(values: &[Value]) -> BuiltinResult<Value> {
    let mut blocks = Vec::new();

    for value in values {
        let Some(block) = symbolic_block_from_value(value)? else {
            continue;
        };
        blocks.push(block);
    }

    if blocks.is_empty() {
        return SymbolicArray::new(Vec::new(), vec![0, 0])
            .map(Value::SymbolicArray)
            .map_err(concat_error);
    }

    let rank = blocks
        .iter()
        .map(|block| block.shape.len())
        .max()
        .unwrap_or(2)
        .max(2);
    let rows = dim_or_one(&blocks[0].shape, 0);
    let mut cols_total = 0usize;
    let mut output_shape = vec![1; rank];
    output_shape[0] = rows;
    for dim in 2..rank {
        output_shape[dim] = dim_or_one(&blocks[0].shape, dim);
    }

    for block in &blocks {
        if dim_or_one(&block.shape, 0) != rows {
            return Err(concat_error(format!(
                "Cannot horizontally concatenate symbolic arrays with different row counts: {} vs {}",
                rows,
                block.rows()
            )));
        }
        for (dim, expected) in output_shape.iter().enumerate().skip(2) {
            let actual = dim_or_one(&block.shape, dim);
            if actual != *expected {
                return Err(concat_error(format!(
                    "Cannot horizontally concatenate symbolic arrays with different dimension {} sizes: {} vs {}",
                    dim + 1,
                    expected,
                    actual
                )));
            }
        }
        cols_total = cols_total
            .checked_add(block.cols())
            .ok_or_else(|| concat_error("Symbolic array dimensions overflow"))?;
    }
    output_shape[1] = cols_total;

    let output_len = checked_shape_len(&output_shape)?;
    let tail_len = output_shape.iter().skip(2).try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| concat_error("Symbolic array dimensions overflow"))
    })?;
    let mut data = Vec::with_capacity(output_len);
    for tail in 0..tail_len {
        for block in &blocks {
            let block_plane_len = block.rows() * block.cols();
            let block_tail_offset = tail * block_plane_len;
            for c in 0..block.cols() {
                for r in 0..rows {
                    data.push(block.data[block_tail_offset + r + c * block.rows()].clone());
                }
            }
        }
    }
    SymbolicArray::new(data, output_shape)
        .map(Value::SymbolicArray)
        .map_err(concat_error)
}

fn vcat_symbolic_values(values: &[Value]) -> BuiltinResult<Value> {
    let mut blocks = Vec::new();

    for value in values {
        let Some(block) = symbolic_block_from_value(value)? else {
            continue;
        };
        blocks.push(block);
    }

    if blocks.is_empty() {
        return SymbolicArray::new(Vec::new(), vec![0, 0])
            .map(Value::SymbolicArray)
            .map_err(concat_error);
    }

    let rank = blocks
        .iter()
        .map(|block| block.shape.len())
        .max()
        .unwrap_or(2)
        .max(2);
    let cols = dim_or_one(&blocks[0].shape, 1);
    let mut rows_total = 0usize;
    let mut output_shape = vec![1; rank];
    output_shape[1] = cols;
    for dim in 2..rank {
        output_shape[dim] = dim_or_one(&blocks[0].shape, dim);
    }

    for block in &blocks {
        if dim_or_one(&block.shape, 1) != cols {
            return Err(concat_error(format!(
                "Cannot vertically concatenate symbolic arrays with different column counts: {} vs {}",
                cols,
                block.cols()
            )));
        }
        for (dim, expected) in output_shape.iter().enumerate().skip(2) {
            let actual = dim_or_one(&block.shape, dim);
            if actual != *expected {
                return Err(concat_error(format!(
                    "Cannot vertically concatenate symbolic arrays with different dimension {} sizes: {} vs {}",
                    dim + 1,
                    expected,
                    actual
                )));
            }
        }
        rows_total = rows_total
            .checked_add(block.rows())
            .ok_or_else(|| concat_error("Symbolic array dimensions overflow"))?;
    }
    output_shape[0] = rows_total;

    let output_len = checked_shape_len(&output_shape)?;
    let tail_len = output_shape.iter().skip(2).try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| concat_error("Symbolic array dimensions overflow"))
    })?;
    let mut data = Vec::with_capacity(output_len);
    for tail in 0..tail_len {
        for c in 0..cols {
            for block in &blocks {
                let block_plane_len = block.rows() * block.cols();
                let block_tail_offset = tail * block_plane_len;
                for r in 0..block.rows() {
                    data.push(block.data[block_tail_offset + r + c * block.rows()].clone());
                }
            }
        }
    }
    SymbolicArray::new(data, output_shape)
        .map(Value::SymbolicArray)
        .map_err(concat_error)
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

    let new_rows = a.rows();
    let new_cols = a.cols() + b.cols();
    let mut new_data = Vec::with_capacity(new_rows * new_cols);

    // Column-major layout: build column-by-column
    for col in 0..new_cols {
        if col < a.cols() {
            for row in 0..a.rows() {
                new_data.push(a.data[row + col * a.rows()]);
            }
        } else {
            let bcol = col - a.cols();
            for row in 0..b.rows() {
                new_data.push(b.data[row + bcol * b.rows()]);
            }
        }
    }

    Tensor::new_2d(new_data, new_rows, new_cols).map_err(concat_error)
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

    let new_rows = a.rows() + b.rows();
    let new_cols = a.cols();
    let mut new_data = Vec::with_capacity(new_rows * new_cols);

    // Column-major: copy columns of A then columns of B
    for col in 0..a.cols() {
        for row in 0..a.rows() {
            new_data.push(a.data[row + col * a.rows()]);
        }
    }
    for col in 0..b.cols() {
        for row in 0..b.rows() {
            new_data.push(b.data[row + col * b.rows()]);
        }
    }

    Tensor::new_2d(new_data, new_rows, new_cols).map_err(concat_error)
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
                        runmat_builtins::StringArray::new(vec![i.to_i64().to_string()], vec![1, 1])
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
                    let ca = char_array_from_f64(i.to_f64())?;
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

    if has_symbolic_operand(values) {
        return hcat_symbolic_values(values);
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
                        runmat_builtins::StringArray::new(vec![i.to_i64().to_string()], vec![1, 1])
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
                    let ca = char_array_from_f64(i.to_f64())?;
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

    if has_symbolic_operand(values) {
        return vcat_symbolic_values(values);
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
    let mut result = matrices[0].clone();
    for matrix in &matrices[1..] {
        result = vcat_matrices(&result, matrix)?;
    }

    Ok(Value::Tensor(result))
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
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_vcat_matrices() {
        let a = Tensor::new_2d(vec![1.0, 2.0], 1, 2).unwrap();
        let b = Tensor::new_2d(vec![3.0, 4.0], 1, 2).unwrap();

        let result = vcat_matrices(&a, &b).unwrap();
        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
        // Column-major: columns preserved
        // With our current vcat implementation, data appends column-wise preserving row order within each input
        // For 1x2 stacked over 1x2, result data is [1,2,3,4]
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
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
            assert_eq!(m.data, vec![1.0, 2.0, 3.0]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_mixed_symbolic_numeric() {
        let values = vec![
            Value::Symbolic(SymbolicExpr::variable("dA")),
            Value::Num(95.0),
            Value::Num(0.0),
        ];
        let result = hcat_values(&values).unwrap();

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![1, 3]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["dA", "95", "0"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_promotes_numeric_and_logical_arrays() {
        let numeric = Tensor::new_2d(vec![1.0, 2.0], 2, 1).unwrap();
        let logical = runmat_builtins::LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        let values = vec![
            Value::SymbolicArray(
                SymbolicArray::new_2d(
                    vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
                    2,
                    1,
                )
                .unwrap(),
            ),
            Value::Tensor(numeric),
            Value::LogicalArray(logical),
        ];
        let result = hcat_values(&values).unwrap();

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![2, 3]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x", "y", "1", "2", "1", "0"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_promotes_one_dimensional_logical_array() {
        let logical = runmat_builtins::LogicalArray::new(vec![1], vec![1]).unwrap();
        let values = vec![
            Value::Symbolic(SymbolicExpr::variable("x")),
            Value::LogicalArray(logical),
        ];
        let result = hcat_values(&values).unwrap();

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![1, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x", "1"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_promotes_non_scalar_one_dimensional_logical_array() {
        let logical = runmat_builtins::LogicalArray::new(vec![1, 0], vec![2]).unwrap();
        let values = vec![
            Value::Symbolic(SymbolicExpr::variable("x")),
            Value::LogicalArray(logical),
        ];
        let result = hcat_values(&values).unwrap();

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![1, 3]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x", "1", "0"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_promotes_one_dimensional_empty_numeric_tensor_as_empty_row() {
        let empty = Tensor::new(vec![], vec![0]).unwrap();
        let values = vec![
            Value::Symbolic(SymbolicExpr::variable("x")),
            Value::Tensor(empty),
        ];
        let result = hcat_values(&values).unwrap();

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![1, 1]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_normalizes_one_dimensional_symbolic_array_as_row() {
        let array = SymbolicArray::new(
            vec![SymbolicExpr::variable("y"), SymbolicExpr::variable("z")],
            vec![2],
        )
        .unwrap();
        let values = vec![
            Value::Symbolic(SymbolicExpr::variable("x")),
            Value::SymbolicArray(array),
        ];
        let result = hcat_values(&values).unwrap();

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![1, 3]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x", "y", "z"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_normalizes_one_dimensional_empty_symbolic_array_as_empty_row() {
        let empty = SymbolicArray::new(vec![], vec![0]).unwrap();
        let values = vec![
            Value::Symbolic(SymbolicExpr::variable("x")),
            Value::SymbolicArray(empty),
        ];
        let result = hcat_values(&values).unwrap();

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![1, 1]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_preserves_trailing_symbolic_dimensions() {
        let left = SymbolicArray::new(
            vec![SymbolicExpr::variable("a1"), SymbolicExpr::variable("a2")],
            vec![1, 1, 2],
        )
        .unwrap();
        let right = SymbolicArray::new(
            vec![SymbolicExpr::variable("b1"), SymbolicExpr::variable("b2")],
            vec![1, 1, 2],
        )
        .unwrap();
        let result = hcat_values(&[Value::SymbolicArray(left), Value::SymbolicArray(right)])
            .expect("symbolic hcat");

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![1, 2, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["a1", "b1", "a2", "b2"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_preserves_nd_column_major_order_with_multiple_rows() {
        let left = SymbolicArray::new(
            vec![
                SymbolicExpr::variable("a1"),
                SymbolicExpr::variable("a2"),
                SymbolicExpr::variable("a3"),
                SymbolicExpr::variable("a4"),
            ],
            vec![2, 1, 2],
        )
        .unwrap();
        let right = SymbolicArray::new(
            vec![
                SymbolicExpr::variable("b1"),
                SymbolicExpr::variable("b2"),
                SymbolicExpr::variable("b3"),
                SymbolicExpr::variable("b4"),
            ],
            vec![2, 1, 2],
        )
        .unwrap();
        let result = hcat_values(&[Value::SymbolicArray(left), Value::SymbolicArray(right)])
            .expect("symbolic hcat");

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![2, 2, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["a1", "a2", "b1", "b2", "a3", "a4", "b3", "b4"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_hcat_values_rejects_trailing_symbolic_dimension_mismatch() {
        let left = SymbolicArray::new(
            vec![SymbolicExpr::variable("a1"), SymbolicExpr::variable("a2")],
            vec![1, 1, 2],
        )
        .unwrap();
        let right = SymbolicArray::new(
            vec![
                SymbolicExpr::variable("b1"),
                SymbolicExpr::variable("b2"),
                SymbolicExpr::variable("b3"),
            ],
            vec![1, 1, 3],
        )
        .unwrap();

        assert!(hcat_values(&[Value::SymbolicArray(left), Value::SymbolicArray(right)]).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_vcat_values_scalars() {
        let values = vec![Value::Num(1.0), Value::Num(2.0)];
        let result = vcat_values(&values).unwrap();

        if let Value::Tensor(m) = result {
            assert_eq!(m.rows(), 2);
            assert_eq!(m.cols(), 1);
            assert_eq!(m.data, vec![1.0, 2.0]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_vcat_values_mixed_symbolic_numeric() {
        let values = vec![
            Value::Symbolic(SymbolicExpr::variable("dA")),
            Value::Num(95.0),
            Value::Num(0.0),
        ];
        let result = vcat_values(&values).unwrap();

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![3, 1]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["dA", "95", "0"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_vcat_values_preserves_trailing_symbolic_dimensions() {
        let top = SymbolicArray::new(
            vec![SymbolicExpr::variable("a1"), SymbolicExpr::variable("a2")],
            vec![1, 1, 2],
        )
        .unwrap();
        let bottom = SymbolicArray::new(
            vec![SymbolicExpr::variable("b1"), SymbolicExpr::variable("b2")],
            vec![1, 1, 2],
        )
        .unwrap();
        let result = vcat_values(&[Value::SymbolicArray(top), Value::SymbolicArray(bottom)])
            .expect("symbolic vcat");

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![2, 1, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["a1", "b1", "a2", "b2"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_vcat_values_preserves_nd_column_major_order_with_multiple_columns() {
        let top = SymbolicArray::new(
            vec![
                SymbolicExpr::variable("a1"),
                SymbolicExpr::variable("a2"),
                SymbolicExpr::variable("a3"),
                SymbolicExpr::variable("a4"),
            ],
            vec![1, 2, 2],
        )
        .unwrap();
        let bottom = SymbolicArray::new(
            vec![
                SymbolicExpr::variable("b1"),
                SymbolicExpr::variable("b2"),
                SymbolicExpr::variable("b3"),
                SymbolicExpr::variable("b4"),
            ],
            vec![1, 2, 2],
        )
        .unwrap();
        let result = vcat_values(&[Value::SymbolicArray(top), Value::SymbolicArray(bottom)])
            .expect("symbolic vcat");

        if let Value::SymbolicArray(array) = result {
            assert_eq!(array.shape, vec![2, 2, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4"]
            );
        } else {
            panic!("Expected symbolic array result");
        }
    }
}
