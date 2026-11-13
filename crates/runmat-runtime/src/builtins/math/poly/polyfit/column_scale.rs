//! Column equilibration for the polynomial least-squares QR basis.

use super::{
    response_scale::ResponseScale, solver::finite_solver_range_error, Complex64,
    PolyfitKernelError, PolyfitKernelResult,
};

/// Factor each original weighted Vandermonde column into a finite scale and
/// dimensionless working column. The published coefficients and, when asked
/// for, the R matrix are restored to the original basis after solving.
pub(super) fn equilibrate(
    matrix: &mut [f64],
    rows: usize,
    cols: usize,
) -> PolyfitKernelResult<Vec<f64>> {
    let mut scales = Vec::with_capacity(cols);
    for col in 0..cols {
        let column = &mut matrix[col * rows..(col + 1) * rows];
        if column.iter().any(|value| !value.is_finite()) {
            return Err(finite_solver_range_error());
        }
        let magnitude = column.iter().map(|value| value.abs()).fold(0.0, f64::max);
        let scale = if magnitude == 0.0 { 1.0 } else { magnitude };
        for value in column {
            *value /= scale;
        }
        scales.push(scale);
    }
    Ok(scales)
}

pub(super) fn restore_coefficients(
    scaled: Vec<Complex64>,
    scales: &[f64],
    rhs_scale: ResponseScale,
) -> PolyfitKernelResult<Vec<Complex64>> {
    scaled
        .into_iter()
        .zip(scales)
        .map(|(value, &scale)| {
            let restored = Complex64::new(
                restore_component(value.re, scale, rhs_scale.real),
                restore_component(value.im, scale, rhs_scale.imaginary),
            );
            if restored.re.is_finite() && restored.im.is_finite() {
                Ok(restored)
            } else {
                Err(finite_solver_range_error())
            }
        })
        .collect()
}

fn restore_component(value: f64, column_scale: f64, rhs_scale: f64) -> f64 {
    let scale_ratio = rhs_scale / column_scale;
    if scale_ratio.is_finite() && scale_ratio != 0.0 {
        let result = value * scale_ratio;
        if result.is_finite() {
            return result;
        }
    }
    value / column_scale * rhs_scale
}

pub(super) fn restore_upper(
    upper: &mut [f64],
    cols: usize,
    scales: &[f64],
) -> PolyfitKernelResult<()> {
    for (col, &scale) in scales.iter().enumerate() {
        for row in 0..=col {
            let value = &mut upper[row + col * cols];
            *value *= scale;
            if !value.is_finite() {
                return Err(PolyfitKernelError::input(
                    "polyfit: requested QR statistics exceed finite output-basis range",
                ));
            }
        }
    }
    Ok(())
}
