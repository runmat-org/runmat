use anyhow::{anyhow, Result};

const EPS: f64 = 1.0e-12;

#[derive(Debug, Clone)]
pub struct LuHostFactors {
    pub combined: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub perm_matrix: Vec<f64>,
    pub pivot_vector: Vec<f64>,
    pub combined_shape: Vec<usize>,
    pub lower_shape: Vec<usize>,
    pub upper_shape: Vec<usize>,
    pub perm_shape: Vec<usize>,
    pub pivot_shape: Vec<usize>,
}

pub fn lu_factor_host(data: &[f64], shape: &[usize]) -> Result<LuHostFactors> {
    let (rows, cols) = matrix_dims(shape)?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("lu: dimension product overflows"))?;
    if expected != data.len() {
        return Err(anyhow!(
            "lu: data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected
        ));
    }

    let mut matrix = column_major_to_row_major(data, rows, cols);
    let mut perm: Vec<usize> = (0..rows).collect();
    let min_dim = rows.min(cols);

    for k in 0..min_dim {
        let mut pivot_row = k;
        let mut pivot_abs = 0.0;
        for r in k..rows {
            let value = matrix[r * cols + k];
            let abs = value.abs();
            if abs > pivot_abs {
                pivot_abs = abs;
                pivot_row = r;
            }
        }

        if pivot_row != k {
            swap_rows(&mut matrix, rows, cols, k, pivot_row);
            perm.swap(k, pivot_row);
        }

        if pivot_abs <= EPS {
            for r in (k + 1)..rows {
                matrix[r * cols + k] = 0.0;
            }
            continue;
        }

        let pivot = matrix[k * cols + k];
        for r in (k + 1)..rows {
            let idx = r * cols + k;
            let factor = matrix[idx] / pivot;
            matrix[idx] = factor;
            for c in (k + 1)..cols {
                let target = r * cols + c;
                matrix[target] -= factor * matrix[k * cols + c];
            }
        }
    }

    let combined = row_major_to_column_major(&matrix, rows, cols);

    let mut lower_row_major = vec![0.0; rows.saturating_mul(rows)];
    let limit = rows.min(cols);
    for i in 0..rows {
        for j in 0..rows {
            if i == j {
                lower_row_major[i * rows + j] = 1.0;
            } else if i > j && j < limit {
                lower_row_major[i * rows + j] = matrix[i * cols + j];
            }
        }
    }
    let lower = row_major_to_column_major(&lower_row_major, rows, rows);

    let mut upper_row_major = vec![0.0; rows.saturating_mul(cols)];
    for i in 0..rows {
        for j in 0..cols {
            if i <= j {
                upper_row_major[i * cols + j] = matrix[i * cols + j];
            }
        }
    }
    let upper = row_major_to_column_major(&upper_row_major, rows, cols);

    let mut perm_matrix = vec![0.0; rows.saturating_mul(rows)];
    for (row_index, &col_index) in perm.iter().enumerate() {
        if col_index < rows {
            perm_matrix[row_index + col_index * rows] = 1.0;
        }
    }

    let pivot_vector: Vec<f64> = perm.iter().map(|idx| (*idx + 1) as f64).collect();

    Ok(LuHostFactors {
        combined,
        lower,
        upper,
        perm_matrix,
        pivot_vector,
        combined_shape: vec![rows, cols],
        lower_shape: vec![rows, rows],
        upper_shape: vec![rows, cols],
        perm_shape: vec![rows, rows],
        pivot_shape: vec![rows, 1],
    })
}

fn matrix_dims(shape: &[usize]) -> Result<(usize, usize)> {
    match shape.len() {
        0 => Ok((1, 1)),
        1 => Ok((shape[0], 1)),
        2 => Ok((shape[0], shape[1])),
        _ => Err(anyhow!("lu: input must be 2-D")),
    }
}

fn column_major_to_row_major(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows.saturating_mul(cols)];
    if rows == 0 || cols == 0 {
        return out;
    }
    for col in 0..cols {
        for row in 0..rows {
            out[row * cols + col] = data[row + col * rows];
        }
    }
    out
}

fn row_major_to_column_major(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows.saturating_mul(cols)];
    if rows == 0 || cols == 0 {
        return out;
    }
    for col in 0..cols {
        for row in 0..rows {
            out[row + col * rows] = data[row * cols + col];
        }
    }
    out
}

fn swap_rows(matrix: &mut [f64], rows: usize, cols: usize, r1: usize, r2: usize) {
    if r1 == r2 || rows == 0 || cols == 0 {
        return;
    }
    for col in 0..cols {
        matrix.swap(r1 * cols + col, r2 * cols + col);
    }
}
