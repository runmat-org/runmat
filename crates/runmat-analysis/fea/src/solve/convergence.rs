//! Preserve the relative residual of the original solved equation space.
// A balanced zero equation has zero error; a nonzero defect without a scale is unproven.
pub(crate) fn relative_residual(residual: f64, reference: f64) -> Option<f64> {
    if !residual.is_finite() || residual < 0.0 || !reference.is_finite() || reference < 0.0 {
        return None;
    }
    if reference == 0.0 {
        return (residual == 0.0).then_some(0.0);
    }
    let ratio = residual / reference;
    ratio.is_finite().then_some(ratio)
}
