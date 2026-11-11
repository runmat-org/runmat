//! Per-mode algebraic accuracy of a modal basis: both published measures of each reduced
//! eigenpair.
use super::ModalBasisMode;

impl ModalBasisMode {
    /// Normwise backward error of the reduced eigenpair,
    /// `‖K x − λ M x‖_∞ / ((‖K‖_∞ + λ ‖M‖_∞) ‖x‖_∞)`. `residual_norm()` is its relative
    /// residual, `‖K x − λ M x‖₂ / (‖K x‖₂ + λ ‖M x‖₂)`.
    pub const fn backward_error(&self) -> f64 {
        self.backward_error
    }
}

/// The frequency, relative residual and backward error of each mode, in mode order.
pub(super) fn mode_scalar_columns(modes: &[ModalBasisMode]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let frequencies = modes.iter().map(|mode| mode.frequency_hz).collect();
    let (residuals, backward_errors) = modes
        .iter()
        .map(|mode| (mode.residual_norm, mode.backward_error))
        .unzip();
    (frequencies, residuals, backward_errors)
}

/// Whether every per-mode input holds exactly `count` entries.
pub(super) fn columns_match(
    count: usize,
    dof_shapes: &[Vec<f64>],
    residual_norms: &[f64],
    backward_errors: &[f64],
) -> bool {
    dof_shapes.len() == count && residual_norms.len() == count && backward_errors.len() == count
}
