//! Homogeneous generalized eigenvectors from retained QZ factors.
//!
//! This is a triangular solve and unitary back-transform, not a second QZ
//! factorization. Defective pencils may yield dependent columns, but each
//! requested column must pass its homogeneous residual check.

use faer::{c64, Mat};

use super::algorithm::ComplexQz;

pub(super) fn conj(z: c64) -> c64 {
    c64::new(z.re, -z.im)
}

pub(super) fn magnitude(z: c64) -> f64 {
    z.re.hypot(z.im)
}

pub(super) fn component_scale(z: c64) -> f64 {
    z.re.abs().max(z.im.abs())
}

pub(super) fn scaled(z: c64, by: f64) -> c64 {
    c64::new(z.re / by, z.im / by)
}

pub(super) fn divide(numerator: c64, denominator: c64) -> c64 {
    if denominator.re.abs() >= denominator.im.abs() {
        let r = denominator.im / denominator.re;
        let scale = denominator.re + denominator.im * r;
        c64::new(
            (numerator.re + numerator.im * r) / scale,
            (numerator.im - numerator.re * r) / scale,
        )
    } else {
        let r = denominator.re / denominator.im;
        let scale = denominator.im + denominator.re * r;
        c64::new(
            (numerator.re * r + numerator.im) / scale,
            (numerator.im * r - numerator.re) / scale,
        )
    }
}

pub(super) fn normalize(vector: &mut [c64]) -> Result<(), &'static str> {
    let scale = vector
        .iter()
        .copied()
        .map(component_scale)
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return Err("generalized eigenvector basis is singular or nonfinite");
    }
    let squared = vector
        .iter()
        .map(|&value| {
            let value = scaled(value, scale);
            value.re * value.re + value.im * value.im
        })
        .sum::<f64>();
    let denominator = scale * squared.sqrt();
    if !denominator.is_finite() || denominator == 0.0 {
        // Divide in two stages when the physical norm overflows.
        let relative = squared.sqrt();
        for value in vector {
            *value = scaled(scaled(*value, scale), relative);
        }
    } else {
        for value in vector {
            *value = scaled(*value, denominator);
        }
    }
    Ok(())
}

/// Compare the homogeneous residual against both sides of the pencil, before
/// their subtraction. Scaling by |(beta A - alpha B) v| instead would reject
/// a rounded one-component eigenpair regardless of its backward error.
pub(super) fn homogeneous_row_residual(
    terms: impl IntoIterator<Item = (c64, c64)>,
    tolerance: f64,
) -> bool {
    let mut residual = c64::new(0.0, 0.0);
    let mut scale = 0.0;
    for (a, b) in terms {
        let a_size = magnitude(a);
        let b_size = magnitude(b);
        if !a_size.is_finite() || !b_size.is_finite() {
            return false;
        }
        residual += a - b;
        scale += a_size + b_size;
    }
    if !scale.is_finite() {
        return false;
    }
    if scale == 0.0 {
        magnitude(residual) == 0.0
    } else {
        magnitude(residual) / scale <= tolerance
    }
}

fn residual_terms(
    form: &ComplexQz,
    position: (usize, usize),
    eigenvalue: (c64, c64),
    scale: f64,
    left: bool,
    component: c64,
) -> (c64, c64) {
    let (row, col) = position;
    let (alpha, beta) = eigenvalue;
    let (a, b) = if left {
        (form.aa[(col, row)], form.bb[(col, row)])
    } else {
        (form.aa[(row, col)], form.bb[(row, col)])
    };
    let a = beta * scaled(a, scale);
    let b = alpha * scaled(b, scale);
    if left {
        (conj(a) * component, conj(b) * component)
    } else {
        (a * component, b * component)
    }
}

fn coefficient(form: &ComplexQz, row: usize, col: usize, alpha: c64, beta: c64, scale: f64) -> c64 {
    beta * scaled(form.aa[(row, col)], scale) - alpha * scaled(form.bb[(row, col)], scale)
}

fn solve_column(
    form: &ComplexQz,
    selected: usize,
    left: bool,
    scale: f64,
) -> Result<Vec<c64>, &'static str> {
    let n = form.aa.nrows();
    let a = form.alpha[selected];
    let b = form.beta[selected];
    let eig_scale = component_scale(a).max(component_scale(b));
    if eig_scale == 0.0 {
        return Err("indeterminate generalized eigenvalue has no unique eigenvector");
    }
    let alpha = scaled(a, eig_scale);
    let beta = scaled(b, eig_scale);
    let mut vector = vec![c64::new(0.0, 0.0); n];
    vector[selected] = c64::new(1.0, 0.0);
    let threshold = 128.0 * f64::EPSILON * (n as f64).max(1.0);
    if left {
        for row in (selected + 1)..n {
            let mut rhs = c64::new(0.0, 0.0);
            for col in selected..row {
                rhs -= conj(coefficient(form, col, row, alpha, beta, scale)) * vector[col];
            }
            let diagonal = conj(coefficient(form, row, row, alpha, beta, scale));
            let local_scale = magnitude(diagonal).max(magnitude(rhs));
            if magnitude(diagonal) <= threshold * local_scale {
                if magnitude(rhs) > threshold * local_scale {
                    return Err("left generalized eigenvector solve is ill-conditioned");
                }
                vector[row] = c64::new(0.0, 0.0);
            } else {
                vector[row] = divide(rhs, diagonal);
            }
            if !magnitude(vector[row]).is_finite() {
                return Err("left generalized eigenvector solve overflowed");
            }
        }
    } else {
        for row in (0..selected).rev() {
            let mut rhs = c64::new(0.0, 0.0);
            for col in (row + 1)..=selected {
                rhs -= coefficient(form, row, col, alpha, beta, scale) * vector[col];
            }
            let diagonal = coefficient(form, row, row, alpha, beta, scale);
            let local_scale = magnitude(diagonal).max(magnitude(rhs));
            if magnitude(diagonal) <= threshold * local_scale {
                if magnitude(rhs) > threshold * local_scale {
                    return Err("right generalized eigenvector solve is ill-conditioned");
                }
                vector[row] = c64::new(0.0, 0.0);
            } else {
                vector[row] = divide(rhs, diagonal);
            }
            if !magnitude(vector[row]).is_finite() {
                return Err("right generalized eigenvector solve overflowed");
            }
        }
    }
    normalize(&mut vector)?;
    // A residual in the homogeneous equation is meaningful for finite,
    // infinite, and subnormal beta alike; alpha/beta is never formed here.
    for row in 0..n {
        let valid = homogeneous_row_residual(
            (0..n).map(|col| {
                residual_terms(form, (row, col), (alpha, beta), scale, left, vector[col])
            }),
            threshold * 16.0,
        );
        if !valid {
            return Err("generalized eigenvector residual is too large");
        }
    }
    Ok(vector)
}

fn backtransform(unitary: &Mat<c64>, basis: &[c64]) -> Result<Vec<c64>, &'static str> {
    let n = unitary.nrows();
    let mut result = vec![c64::new(0.0, 0.0); n];
    for row in 0..n {
        for col in 0..n {
            result[row] += unitary[(row, col)] * basis[col];
        }
    }
    normalize(&mut result)?;
    Ok(result)
}

pub(super) fn same_homogeneous_eigenvalue(a: c64, b: c64, other_a: c64, other_b: c64) -> bool {
    let first_scale = component_scale(a).max(component_scale(b));
    let other_scale = component_scale(other_a).max(component_scale(other_b));
    if first_scale == 0.0 || other_scale == 0.0 {
        return false;
    }
    let (a, b) = (scaled(a, first_scale), scaled(b, first_scale));
    let (other_a, other_b) = (scaled(other_a, other_scale), scaled(other_b, other_scale));
    magnitude(a * other_b - other_a * b) <= 128.0 * f64::EPSILON
}

fn solve_equivalent_column(
    form: &ComplexQz,
    selected: usize,
    left: bool,
    scale: f64,
) -> Result<Vec<c64>, &'static str> {
    let alpha = form.alpha[selected];
    let beta = form.beta[selected];
    if component_scale(alpha).max(component_scale(beta)) == 0.0 {
        return Err("indeterminate generalized eigenvalue has no unique eigenvector");
    }
    let mut failure = "generalized eigenvector solve is ill-conditioned";
    let eig_scale = component_scale(alpha).max(component_scale(beta));
    let desired_alpha = scaled(alpha, eig_scale);
    let desired_beta = scaled(beta, eig_scale);
    let tolerance = 128.0 * f64::EPSILON * (form.aa.nrows() as f64).max(1.0) * 16.0;
    for candidate in
        std::iter::once(selected).chain((0..form.aa.nrows()).filter(|&i| i != selected))
    {
        if !same_homogeneous_eigenvalue(alpha, beta, form.alpha[candidate], form.beta[candidate]) {
            continue;
        }
        match solve_column(form, candidate, left, scale) {
            Ok(vector) => {
                let satisfies_selected = (0..form.aa.nrows()).all(|row| {
                    homogeneous_row_residual(
                        (0..form.aa.nrows()).map(|col| {
                            residual_terms(
                                form,
                                (row, col),
                                (desired_alpha, desired_beta),
                                scale,
                                left,
                                vector[col],
                            )
                        }),
                        tolerance,
                    )
                });
                if satisfies_selected {
                    return Ok(vector);
                }
                failure = "generalized eigenvector residual is too large";
            }
            Err(error) => failure = error,
        }
    }
    Err(failure)
}

pub(super) struct Eigenvectors {
    pub right: Mat<c64>,
    pub left: Option<Mat<c64>>,
}

pub(super) fn complex_eigenvectors(
    form: &ComplexQz,
    need_left: bool,
) -> Result<Eigenvectors, &'static str> {
    let n = form.aa.nrows();
    let scale = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .fold(0.0_f64, |m, (i, j)| {
            m.max(component_scale(form.aa[(i, j)]))
                .max(component_scale(form.bb[(i, j)]))
        });
    let scale = if scale == 0.0 { 1.0 } else { scale };
    let mut right = Mat::<c64>::zeros(n, n);
    let mut left = need_left.then(|| Mat::<c64>::zeros(n, n));
    for selected in 0..n {
        let basis = solve_equivalent_column(form, selected, false, scale)?;
        let vector = backtransform(&form.right, &basis)?;
        for row in 0..n {
            right[(row, selected)] = vector[row];
        }
        if let Some(left_matrix) = left.as_mut() {
            let basis = solve_equivalent_column(form, selected, true, scale)?;
            let vector = backtransform(&form.left, &basis)?;
            for row in 0..n {
                left_matrix[(row, selected)] = vector[row];
            }
        }
    }
    Ok(Eigenvectors { right, left })
}
