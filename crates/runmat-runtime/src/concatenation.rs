//! Matrix and array concatenation operations
//!
//! This module provides language-compatible matrix concatenation operations.
//! Supports both horizontal concatenation [A, B] and vertical concatenation [A; B].

use runmat_builtins::{Tensor, Value};

/// Horizontally concatenate two matrices [A, B]
/// In language: C = [A, B] creates a matrix with A and B side by side
pub fn hcat_matrices(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    // Language semantics: [] acts as a neutral element for concatenation
    if a.rows() == 0 && a.cols() == 0 {
        return Ok(b.clone());
    }
    if b.rows() == 0 && b.cols() == 0 {
        return Ok(a.clone());
    }
    if a.rows() != b.rows() {
        return Err(format!(
            "Cannot horizontally concatenate matrices with different row counts: {} vs {}",
            a.rows, b.rows
        ));
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

    Tensor::new_2d(new_data, new_rows, new_cols)
}

/// Vertically concatenate two matrices [A; B]
/// In language: C = [A; B] creates a matrix with A on top and B below
pub fn vcat_matrices(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    // Language semantics: [] acts as a neutral element for concatenation
    if a.rows() == 0 && a.cols() == 0 {
        return Ok(b.clone());
    }
    if b.rows() == 0 && b.cols() == 0 {
        return Ok(a.clone());
    }
    if a.cols() != b.cols() {
        return Err(format!(
            "Cannot vertically concatenate matrices with different column counts: {} vs {}",
            a.cols, b.cols
        ));
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

    Tensor::new_2d(new_data, new_rows, new_cols)
}

/// Concatenate values horizontally - handles mixed scalars and matrices
pub fn hcat_values(values: &[Value]) -> Result<Value, String> {
    if values.is_empty() {
        return Ok(Value::Tensor(Tensor::new(vec![], vec![0, 0])?));
    }

    // If any operand is a string or string array, perform string-array concatenation
    let has_str = values.iter().any(|v| {
        matches!(
            v,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        )
    });
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
                        return Err("string hcat: row mismatch".to_string());
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
                        return Err("string hcat: row mismatch".to_string());
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
                        return Err("string hcat: row mismatch".to_string());
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
                        return Err("string hcat: row mismatch".to_string());
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
                        return Err("string hcat: row mismatch".to_string());
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
                        return Err("string hcat: row mismatch".to_string());
                    }
                    cols_total += 1;
                    blocks.push(sa);
                }
                Value::Tensor(_) | Value::Cell(_) => {
                    return Err(format!(
                        "Cannot concatenate value of type {v:?} with string array"
                    ))
                }
                _ => {
                    return Err(format!(
                        "Cannot concatenate value of type {v:?} with string array"
                    ))
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
            .map_err(|e| format!("string hcat: {e}"))?;
        return Ok(Value::StringArray(sa));
    }

    // Convert all scalars to 1x1 matrices for uniform processing
    let mut matrices = Vec::new();
    let mut _total_cols = 0;
    let mut rows = 0;

    for value in values {
        match value {
            Value::Num(n) => {
                let matrix = Tensor::new_2d(vec![*n], 1, 1)?;
                if rows == 0 {
                    rows = 1;
                } else if rows != 1 {
                    return Err("Cannot concatenate scalar with multi-row matrix".to_string());
                }
                _total_cols += 1;
                matrices.push(matrix);
            }
            Value::Complex(re, _im) => {
                let matrix = Tensor::new_2d(vec![*re], 1, 1)?; // real part in numeric hcat coercion
                if rows == 0 {
                    rows = 1;
                } else if rows != 1 {
                    return Err("Cannot concatenate scalar with multi-row matrix".to_string());
                }
                _total_cols += 1;
                matrices.push(matrix);
            }
            Value::Int(i) => {
                let matrix = Tensor::new_2d(vec![i.to_f64()], 1, 1)?;
                if rows == 0 {
                    rows = 1;
                } else if rows != 1 {
                    return Err("Cannot concatenate scalar with multi-row matrix".to_string());
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
                    return Err(format!(
                        "Cannot concatenate matrices with different row counts: {} vs {}",
                        rows,
                        m.rows()
                    ));
                }
                _total_cols += m.cols();
                matrices.push(m.clone());
            }
            _ => return Err(format!("Cannot concatenate value of type {value:?}")),
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
pub fn vcat_values(values: &[Value]) -> Result<Value, String> {
    if values.is_empty() {
        return Ok(Value::Tensor(Tensor::new(vec![], vec![0, 0])?));
    }

    // If any operand is a string or string array, perform string-array vertical concatenation by stacking rows
    let has_str = values.iter().any(|v| {
        matches!(
            v,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        )
    });
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
                        return Err("string vcat: column mismatch".to_string());
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
                        return Err("string vcat: column mismatch".to_string());
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
                        return Err("string vcat: column mismatch".to_string());
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
                        return Err("string vcat: column mismatch".to_string());
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
                        return Err("string vcat: column mismatch".to_string());
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
                        return Err("string vcat: column mismatch".to_string());
                    }
                    blocks.push(sa);
                }
                _ => {
                    return Err(format!(
                        "Cannot concatenate value of type {v:?} with string array"
                    ))
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
            .map_err(|e| format!("string vcat: {e}"))?;
        return Ok(Value::StringArray(sa));
    }

    // Convert all scalars to 1x1 matrices for uniform processing
    let mut matrices = Vec::new();
    let mut _total_rows = 0;
    let mut cols = 0;

    for value in values {
        match value {
            Value::Num(n) => {
                let matrix = Tensor::new_2d(vec![*n], 1, 1)?;
                if cols == 0 {
                    cols = 1;
                } else if cols != 1 {
                    return Err("Cannot concatenate scalar with multi-column matrix".to_string());
                }
                _total_rows += 1;
                matrices.push(matrix);
            }
            Value::Complex(re, _im) => {
                let matrix = Tensor::new_2d(vec![*re], 1, 1)?;
                if cols == 0 {
                    cols = 1;
                } else if cols != 1 {
                    return Err("Cannot concatenate scalar with multi-column matrix".to_string());
                }
                _total_rows += 1;
                matrices.push(matrix);
            }
            Value::Int(i) => {
                let matrix = Tensor::new_2d(vec![i.to_f64()], 1, 1)?;
                if cols == 0 {
                    cols = 1;
                } else if cols != 1 {
                    return Err("Cannot concatenate scalar with multi-column matrix".to_string());
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
                    return Err(format!(
                        "Cannot concatenate matrices with different column counts: {} vs {}",
                        cols,
                        m.cols()
                    ));
                }
                _total_rows += m.rows();
                matrices.push(m.clone());
            }
            _ => return Err(format!("Cannot concatenate value of type {value:?}")),
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
pub fn create_matrix_from_values(rows: &[Vec<Value>]) -> Result<Value, String> {
    if rows.is_empty() {
        return Ok(Value::Tensor(Tensor::new(vec![], vec![0, 0])?));
    }

    // Build each row using horzcat builtin to preserve canonical semantics
    let mut row_matrices: Vec<Value> = Vec::with_capacity(rows.len());
    for row in rows {
        let row_value = if row.is_empty() {
            Value::Tensor(Tensor::new(vec![], vec![0, 0])?)
        } else {
            crate::call_builtin("horzcat", row)?
        };
        row_matrices.push(row_value);
    }

    // Stack rows using vertcat builtin
    if row_matrices.is_empty() {
        Ok(Value::Tensor(Tensor::new(vec![], vec![0, 0])?))
    } else if row_matrices.len() == 1 {
        Ok(row_matrices.into_iter().next().unwrap())
    } else {
        Ok(crate::call_builtin("vertcat", &row_matrices)?)
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
}
