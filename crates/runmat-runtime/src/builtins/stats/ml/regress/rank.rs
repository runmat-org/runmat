use super::*;

pub(super) fn independent_columns(design: &DMatrix<f64>) -> BuiltinResult<Vec<usize>> {
    let rows = design.nrows();
    let cols = design.ncols();
    let mut largest_norm = 0.0_f64;
    for col in 0..cols {
        let norm = column_norm(design, col);
        largest_norm = largest_norm.max(norm);
    }
    let tolerance = (rows.max(cols) as f64) * f64::EPSILON * largest_norm.max(1.0);
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut active = Vec::new();
    let mut selected = vec![false; cols];

    loop {
        let mut best: Option<(usize, Vec<f64>, f64)> = None;
        for (col, is_selected) in selected.iter().copied().enumerate() {
            if is_selected {
                continue;
            }
            let residual = column_residual(design, col, &basis);
            let norm = residual
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if best
                .as_ref()
                .is_none_or(|(_, _, best_norm)| norm > *best_norm)
            {
                best = Some((col, residual, norm));
            }
        }
        let Some((col, mut residual, norm)) = best else {
            break;
        };
        if norm <= tolerance {
            break;
        }
        for value in &mut residual {
            *value /= norm;
        }
        selected[col] = true;
        basis.push(residual);
        active.push(col);
    }
    Ok(active)
}

pub(super) fn column_residual(design: &DMatrix<f64>, col: usize, basis: &[Vec<f64>]) -> Vec<f64> {
    let rows = design.nrows();
    let mut residual = (0..rows).map(|row| design[(row, col)]).collect::<Vec<_>>();
    for vector in basis {
        let projection = residual
            .iter()
            .zip(vector)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        for row in 0..rows {
            residual[row] -= projection * vector[row];
        }
    }
    residual
}

pub(super) fn column_norm(design: &DMatrix<f64>, col: usize) -> f64 {
    (0..design.nrows())
        .map(|row| {
            let value = design[(row, col)];
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

pub(super) fn active_design(design: &DMatrix<f64>, active_columns: &[usize]) -> DMatrix<f64> {
    let rows = design.nrows();
    let cols = active_columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in active_columns {
            data.push(design[(row, *col)]);
        }
    }
    DMatrix::from_row_slice(rows, cols, &data)
}
