//! Polynomial least-squares basis construction and QR solve.

use num_complex::Complex64;
use runmat_accelerate_api::{ProviderPolyfitBasis, ProviderPolyfitOutputs};

use super::{
    column_scale, normalization, response_scale::ResponseScale, PolyfitKernelError,
    PolyfitKernelResult, PolyfitSolution, SolverGeometry,
};

pub(super) fn validate_weights(weights: &[f64]) -> PolyfitKernelResult<()> {
    for (idx, w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(PolyfitKernelError::argument(format!(
                "polyfit: weight at position {} must be finite",
                idx + 1
            )));
        }
        if *w < 0.0 {
            return Err(PolyfitKernelError::argument(
                "polyfit: weights must be non-negative",
            ));
        }
    }
    Ok(())
}

pub(super) fn solve_polyfit(
    x_data: &[f64],
    y_data: &[Complex64],
    degree: usize,
    weights: Option<&[f64]>,
    outputs: ProviderPolyfitOutputs,
) -> PolyfitKernelResult<PolyfitSolution> {
    if x_data.len() != y_data.len() {
        return Err(PolyfitKernelError::input(
            "polyfit: X and Y vectors must be the same length",
        ));
    }
    if x_data.is_empty() {
        return Err(PolyfitKernelError::input(
            "polyfit: X and Y must contain at least one sample",
        ));
    }
    if let Some(w) = weights {
        if w.len() != x_data.len() {
            return Err(PolyfitKernelError::argument(
                "polyfit: weight vector must match the size of X",
            ));
        }
    }

    // Structural geometry is checked before numerical payload validation,
    // matching resident preflight and the provider's real-data adapter.
    let geometry = SolverGeometry::for_outputs(x_data.len(), degree, outputs)?;
    if let Some(w) = weights {
        validate_weights(w)?;
    }
    let (mean, scale) = match outputs.basis() {
        ProviderPolyfitBasis::Original => (0.0, 1.0),
        ProviderPolyfitBasis::CenteredScaled => {
            let mean = normalization::sample_mean(x_data)
                .ok_or_else(|| PolyfitKernelError::input("polyfit: mean of X must be finite"))?;
            (mean, compute_scale(x_data, mean)?)
        }
    };
    for (idx, value) in y_data.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(PolyfitKernelError::input(format!(
                "polyfit: Y must contain finite values (encountered NaN/Inf at position {})",
                idx + 1
            )));
        }
    }
    let scaled = scaled_coordinates(x_data, mean, scale)?;
    // QR works in a dimensionless RHS basis too. A residual may exceed f64
    // even when the requested coefficients are finite (for example opposite
    // MAX-valued observations fitted by a constant).
    // The real Vandermonde basis acts independently on both components. A
    // shared response scale would erase a tiny representable component next
    // to a large component, even in an exact one-sample constant fit.
    let rhs_scale = ResponseScale::from_values(y_data);
    let mut rhs = y_data
        .iter()
        .map(|value| rhs_scale.normalize(value))
        .collect::<Vec<_>>();
    if let Some(w) = weights {
        apply_weights_rhs(&mut rhs, w)?;
    }

    let rows = geometry.rows;
    let cols = geometry.cols;
    let mut vandermonde = build_vandermonde(&scaled, &geometry);
    if let Some(w) = weights {
        apply_weights_matrix(&mut vandermonde, rows, cols, w)?;
    }

    let column_scales = column_scale::equilibrate(&mut vandermonde, rows, cols)?;

    let mut transformed_rhs = rhs.clone();
    let column_norms: Vec<f64> = (0..cols)
        .map(|col| stable_l2((0..rows).map(|row| vandermonde[row + col * rows])))
        .collect();
    householder_qr(&mut vandermonde, rows, cols, &mut transformed_rhs)?;
    let scaled_coeffs = solve_upper(&vandermonde, rows, cols, &transformed_rhs, &column_norms)?;
    let coeffs = column_scale::restore_coefficients(scaled_coeffs, &column_scales, rhs_scale)?;

    let normr = outputs
        .wants_statistics()
        .then(|| {
            let residual_start = cols.min(rows);
            rhs_scale.residual_norm(&transformed_rhs[residual_start..rows])
        })
        .transpose()?;
    let df = if rows > cols {
        (rows - cols) as f64
    } else {
        0.0
    };
    let r_matrix = if outputs.wants_statistics() {
        let mut r = extract_upper(&vandermonde, &geometry);
        column_scale::restore_upper(&mut r, cols, &column_scales)?;
        Some(r)
    } else {
        None
    };
    let is_complex = coeffs.iter().any(|c| c.im != 0.0);

    Ok(PolyfitSolution {
        coeffs,
        r_matrix,
        mu_mean: mean,
        mu_scale: scale,
        normr,
        df,
        cols,
        is_complex,
    })
}

pub(super) fn compute_scale(data: &[f64], mean: f64) -> PolyfitKernelResult<f64> {
    normalization::sample_scale(data, mean).ok_or_else(|| {
        PolyfitKernelError::input("polyfit: failed to compute a finite, nonzero scaling factor")
    })
}

pub(super) fn scaled_coordinates(
    data: &[f64],
    mean: f64,
    scale: f64,
) -> PolyfitKernelResult<Vec<f64>> {
    data.iter()
        .map(|&value| {
            normalization::standardize(value, mean, scale).ok_or_else(finite_solver_range_error)
        })
        .collect()
}

pub(super) fn stable_l2(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0, f64::hypot)
}

pub(super) fn finite_solver_range_error() -> PolyfitKernelError {
    PolyfitKernelError::input("polyfit: polynomial basis exceeds finite solver range")
}

fn build_vandermonde(u: &[f64], geometry: &SolverGeometry) -> Vec<f64> {
    let rows = geometry.rows;
    let cols = geometry.cols;
    let mut matrix = vec![0.0; geometry.rectangular_len];
    if cols == 0 {
        return matrix;
    }
    for (row_idx, &value) in u.iter().enumerate() {
        let mut powers = vec![0.0; cols];
        powers[cols - 1] = 1.0;
        for idx in (0..cols - 1).rev() {
            powers[idx] = powers[idx + 1] * value;
        }
        for col_idx in 0..cols {
            matrix[row_idx + col_idx * rows] = powers[col_idx];
        }
    }
    matrix
}

fn apply_weights_matrix(
    matrix: &mut [f64],
    rows: usize,
    cols: usize,
    weights: &[f64],
) -> PolyfitKernelResult<()> {
    for (row, weight) in weights.iter().enumerate().take(rows) {
        let sqrt_w = weight.sqrt();
        if !sqrt_w.is_finite() {
            return Err(PolyfitKernelError::input(format!(
                "polyfit: weight at position {} must be finite",
                row + 1
            )));
        }
        for col in 0..cols {
            let idx = row + col * rows;
            matrix[idx] *= sqrt_w;
        }
    }
    Ok(())
}

fn apply_weights_rhs(rhs: &mut [Complex64], weights: &[f64]) -> PolyfitKernelResult<()> {
    for (idx, (value, weight)) in rhs.iter_mut().zip(weights.iter()).enumerate() {
        let sqrt_w = weight.sqrt();
        if !sqrt_w.is_finite() {
            return Err(PolyfitKernelError::input(format!(
                "polyfit: weight at position {} must be finite",
                idx + 1
            )));
        }
        *value *= sqrt_w;
    }
    Ok(())
}

pub(super) fn householder_qr(
    matrix: &mut [f64],
    rows: usize,
    cols: usize,
    rhs: &mut [Complex64],
) -> PolyfitKernelResult<()> {
    if matrix.iter().any(|value| !value.is_finite())
        || rhs
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(finite_solver_range_error());
    }
    let min_dim = rows.min(cols);
    for k in 0..min_dim {
        let norm = stable_l2((k..rows).map(|row| matrix[row + k * rows]));
        if !norm.is_finite() {
            return Err(finite_solver_range_error());
        }
        if norm == 0.0 {
            continue;
        }
        let x0 = matrix[k + k * rows];
        let alpha = if x0 >= 0.0 { -norm } else { norm };
        let mut v = vec![0.0; rows - k];
        // Dividing the column before forming the reflector avoids x0-alpha
        // overflowing when both x0 and the column norm are finite.
        v[0] = x0 / norm + if x0 >= 0.0 { 1.0 } else { -1.0 };
        for row in (k + 1)..rows {
            v[row - k] = matrix[row + k * rows] / norm;
        }
        let v_norm = stable_l2(v.iter().copied());
        if v_norm == 0.0 {
            continue;
        }
        for component in &mut v {
            *component /= v_norm;
        }
        matrix[k + k * rows] = alpha;
        for row in (k + 1)..rows {
            matrix[row + k * rows] = 0.0;
        }
        for col in (k + 1)..cols {
            let mut dot = 0.0;
            for (idx, &vi) in v.iter().enumerate() {
                let row_idx = k + idx;
                dot += vi * matrix[row_idx + col * rows];
            }
            let factor = 2.0 * dot;
            if !factor.is_finite() {
                return Err(finite_solver_range_error());
            }
            for (idx, &vi) in v.iter().enumerate() {
                let row_idx = k + idx;
                matrix[row_idx + col * rows] -= factor * vi;
                if !matrix[row_idx + col * rows].is_finite() {
                    return Err(finite_solver_range_error());
                }
            }
        }
        let mut dot = Complex64::new(0.0, 0.0);
        for (idx, &vi) in v.iter().enumerate() {
            let row_idx = k + idx;
            dot += rhs[row_idx] * vi;
        }
        let factor = Complex64::new(2.0, 0.0) * dot;
        if !factor.re.is_finite() || !factor.im.is_finite() {
            return Err(finite_solver_range_error());
        }
        for (idx, &vi) in v.iter().enumerate() {
            let row_idx = k + idx;
            rhs[row_idx] -= factor * vi;
            if !rhs[row_idx].re.is_finite() || !rhs[row_idx].im.is_finite() {
                return Err(finite_solver_range_error());
            }
        }
    }
    Ok(())
}

pub(super) fn solve_upper(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    rhs: &[Complex64],
    column_norms: &[f64],
) -> PolyfitKernelResult<Vec<Complex64>> {
    if rhs.len() < rows {
        return Err(PolyfitKernelError::input(
            "polyfit internal error: RHS dimension mismatch",
        ));
    }
    let mut coeffs = vec![Complex64::new(0.0, 0.0); cols];
    for col in (0..cols).rev() {
        let diag = if col < rows {
            matrix[col + col * rows]
        } else {
            0.0
        };
        if diag.abs() <= f64::EPSILON * column_norms[col] {
            coeffs[col] = Complex64::new(0.0, 0.0);
            continue;
        }
        let mut acc = if col < rows {
            rhs[col]
        } else {
            Complex64::new(0.0, 0.0)
        };
        for next in (col + 1)..cols {
            let idx = if col < rows {
                matrix[col + next * rows]
            } else {
                0.0
            };
            acc -= Complex64::new(idx, 0.0) * coeffs[next];
        }
        let coefficient = Complex64::new(acc.re / diag, acc.im / diag);
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(finite_solver_range_error());
        }
        coeffs[col] = coefficient;
    }
    Ok(coeffs)
}

fn extract_upper(matrix: &[f64], geometry: &SolverGeometry) -> Vec<f64> {
    let rows = geometry.rows;
    let cols = geometry.cols;
    let mut output = vec![0.0; geometry.square_len];
    for col in 0..cols {
        for row in 0..=col {
            if row < rows {
                output[row + col * cols] = matrix[row + col * rows];
            }
        }
    }
    output
}
