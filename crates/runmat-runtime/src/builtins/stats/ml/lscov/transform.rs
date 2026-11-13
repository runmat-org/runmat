use super::*;

#[derive(Debug)]
pub(super) struct TransformedReal {
    pub(super) a: DMatrix<f64>,
    pub(super) b: DMatrix<f64>,
}

#[derive(Debug)]
pub(super) struct TransformedComplex {
    pub(super) a: DMatrix<Complex64>,
    pub(super) b: DMatrix<Complex64>,
}

pub(super) fn transform_real_problem(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    weighting: &Weighting,
    algorithm: Algorithm,
) -> BuiltinResult<TransformedReal> {
    match weighting {
        Weighting::Identity => Ok(TransformedReal {
            a: a.clone(),
            b: b.clone(),
        }),
        Weighting::Weights(weights) => {
            let rows = a.nrows();
            let cols = a.ncols();
            let rhs_cols = b.ncols();
            let mut transformed_a = DMatrix::<f64>::zeros(rows, cols);
            let mut transformed_b = DMatrix::<f64>::zeros(rows, rhs_cols);
            for row in 0..rows {
                let scale = weights[row].sqrt();
                for col in 0..cols {
                    transformed_a[(row, col)] = a[(row, col)] * scale;
                }
                for rhs_col in 0..rhs_cols {
                    transformed_b[(row, rhs_col)] = b[(row, rhs_col)] * scale;
                }
            }
            Ok(TransformedReal {
                a: transformed_a,
                b: transformed_b,
            })
        }
        Weighting::Covariance(covariance) => {
            let transform = covariance_transform(covariance, algorithm)?;
            Ok(TransformedReal {
                a: &transform * a,
                b: &transform * b,
            })
        }
    }
}

pub(super) fn transform_complex_problem(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    weighting: &Weighting,
    algorithm: Algorithm,
) -> BuiltinResult<TransformedComplex> {
    match weighting {
        Weighting::Identity => Ok(TransformedComplex {
            a: a.clone(),
            b: b.clone(),
        }),
        Weighting::Weights(weights) => {
            let rows = a.nrows();
            let cols = a.ncols();
            let rhs_cols = b.ncols();
            let mut transformed_a = DMatrix::<Complex64>::zeros(rows, cols);
            let mut transformed_b = DMatrix::<Complex64>::zeros(rows, rhs_cols);
            for row in 0..rows {
                let scale = weights[row].sqrt();
                for col in 0..cols {
                    transformed_a[(row, col)] = a[(row, col)] * scale;
                }
                for rhs_col in 0..rhs_cols {
                    transformed_b[(row, rhs_col)] = b[(row, rhs_col)] * scale;
                }
            }
            Ok(TransformedComplex {
                a: transformed_a,
                b: transformed_b,
            })
        }
        Weighting::Covariance(covariance) => {
            let transform = covariance_transform(covariance, algorithm)?;
            let transform_complex = transform.map(|value| Complex64::new(value, 0.0));
            Ok(TransformedComplex {
                a: &transform_complex * a,
                b: &transform_complex * b,
            })
        }
    }
}

pub(super) fn covariance_transform(
    covariance: &DMatrix<f64>,
    algorithm: Algorithm,
) -> BuiltinResult<DMatrix<f64>> {
    if matches!(algorithm, Algorithm::Chol) {
        if let Some(chol) = covariance.clone().cholesky() {
            let lower = chol.l();
            let identity = DMatrix::<f64>::identity(covariance.nrows(), covariance.ncols());
            if let Some(transform) = lower.lu().solve(&identity) {
                return Ok(transform);
            }
        }
    }
    covariance_orth_transform(covariance)
}

pub(super) fn covariance_orth_transform(covariance: &DMatrix<f64>) -> BuiltinResult<DMatrix<f64>> {
    let eigen = SymmetricEigen::new(covariance.clone());
    let tolerance = scaled_psd_tolerance(eigen.eigenvalues.as_slice());
    let positive = eigen
        .eigenvalues
        .iter()
        .copied()
        .filter(|value| *value > tolerance)
        .count();
    if positive == 0 {
        return Err(numerical(
            "lscov: V covariance matrix must have at least one positive direction",
        ));
    }
    let mut transform = DMatrix::<f64>::zeros(positive, covariance.ncols());
    let mut out_row = 0usize;
    for eig_idx in 0..eigen.eigenvalues.len() {
        let lambda = eigen.eigenvalues[eig_idx];
        if lambda < -tolerance {
            return Err(invalid(
                "lscov: V covariance matrix must be positive semidefinite",
            ));
        }
        if lambda <= tolerance {
            continue;
        }
        let scale = 1.0 / lambda.sqrt();
        for col in 0..covariance.ncols() {
            transform[(out_row, col)] = eigen.eigenvectors[(col, eig_idx)] * scale;
        }
        out_row += 1;
    }
    Ok(transform)
}
