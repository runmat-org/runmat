use super::{EnergyError, Interval};

/// Weyl's quadratic-form bound: a verified local lower bound survives only after
/// subtracting an outward upper bound for every coefficient of the perturbation.
pub(super) fn perturbation_lower(
    original: &[f64],
    lower: f64,
    current: &[f64],
    width: usize,
) -> Option<f64> {
    let mut rows = vec![0.0_f64; width - 1];
    let mut columns = vec![0.0_f64; width - 1];
    for i in 1..width {
        for j in 1..width {
            let delta = Interval::exact(original[i * width + j])
                .sub(Interval::exact(current[i * width + j]))
                .absolute_upper();
            rows[i - 1] = (rows[i - 1] + delta).next_up();
            columns[j - 1] = (columns[j - 1] + delta).next_up();
        }
    }
    let norm = ((rows.into_iter().fold(0.0, f64::max) * columns.into_iter().fold(0.0, f64::max))
        .next_up()
        .sqrt())
    .next_up();
    let bound = (lower - norm).next_down();
    (bound.is_finite() && bound > 0.0).then_some(bound)
}
/// Verify a positive LDL candidate with an outward residual. Its triangular inverse
/// majorant bounds both norms without assuming floating point solves are exact. `None` when the
/// block cannot be certified positive (a nonpositive pivot, or roundoff that swamps the bound):
/// that is no disproof of the assembled operator, so the proof becomes indeterminate.
pub(super) fn lower_bound(block: &[f64], width: usize) -> Result<Option<f64>, EnergyError> {
    if block.iter().any(|v| !v.is_finite()) {
        return Err(EnergyError::InvalidBlock);
    }
    let n = width - 1;
    let mut l = vec![0.0; n * n];
    let mut d = vec![0.0; n];
    for i in 0..n {
        l[i * n + i] = 1.0;
        for j in 0..i {
            let mut v = block[(i + 1) * width + j + 1];
            for k in 0..j {
                v -= l[i * n + k] * d[k] * l[j * n + k];
            }
            l[i * n + j] = v / d[j];
        }
        let mut v = block[(i + 1) * width + i + 1];
        for k in 0..i {
            v -= l[i * n + k] * l[i * n + k] * d[k];
        }
        if !v.is_finite() || v <= 0.0 {
            return Ok(None);
        }
        d[i] = v;
    }
    if l.iter().any(|v| !v.is_finite()) {
        return Err(EnergyError::Arithmetic);
    }
    let mut inverse = vec![0.0_f64; n * n];
    for i in 0..n {
        inverse[i * n + i] = 1.0;
        for j in 0..i {
            let mut upper = 0.0;
            for k in j..i {
                upper = (upper + (l[i * n + k].abs() * inverse[k * n + j]).next_up()).next_up();
            }
            inverse[i * n + j] = upper;
        }
    }
    let mut rows = vec![0.0_f64; n];
    let mut columns = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            rows[i] = (rows[i] + inverse[i * n + j]).next_up();
            columns[j] = (columns[j] + inverse[i * n + j]).next_up();
        }
    }
    let norm_product =
        (rows.into_iter().fold(0.0, f64::max) * columns.into_iter().fold(0.0, f64::max)).next_up();
    let positive_lower =
        (d.iter().copied().fold(f64::INFINITY, f64::min) / norm_product).next_down();
    let mut error_rows = vec![0.0_f64; n];
    let mut error_columns = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            let mut product = Interval::exact(0.0);
            for k in 0..=i.min(j) {
                product = product.add(
                    Interval::exact(l[i * n + k])
                        .mul(Interval::exact(d[k]))
                        .mul(Interval::exact(l[j * n + k])),
                );
            }
            let error = product
                .sub(Interval::exact(block[(i + 1) * width + j + 1]))
                .absolute_upper();
            error_rows[i] = (error_rows[i] + error).next_up();
            error_columns[j] = (error_columns[j] + error).next_up();
        }
    }
    let error = ((error_rows.into_iter().fold(0.0, f64::max)
        * error_columns.into_iter().fold(0.0, f64::max))
    .next_up()
    .sqrt())
    .next_up();
    let lower = (positive_lower - error).next_down();
    if !lower.is_finite() || lower <= 0.0 {
        return Ok(None);
    }
    Ok(Some(lower))
}
