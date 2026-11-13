use super::*;
use nalgebra::DMatrix;

pub(super) fn prepare_model(
    training: &FloatingMatrix,
    labels: LabelVector,
    discriminant: DiscriminantType,
    prior_value: prior::PriorSpec,
) -> BuiltinResult<PreparedModel> {
    let unique = unique_labels(&labels.labels, labels.kind);
    if unique.len() < 2 {
        return Err(invalid_argument(
            "classify: group must contain at least two nonmissing classes",
        ));
    }
    let mut rows_by_class = vec![Vec::<usize>::new(); unique.len()];
    for (row, label) in labels.labels.iter().enumerate() {
        if is_missing_label(label) {
            continue;
        }
        let Some(class_index) = unique.iter().position(|class| same_label(class, label)) else {
            continue;
        };
        rows_by_class[class_index].push(row);
    }
    if rows_by_class.iter().any(Vec::is_empty) {
        return Err(invalid_argument(
            "classify: every class must contain at least one training row",
        ));
    }
    let priors = prior_value.resolve(&unique, labels.kind, &rows_by_class)?;
    let cols = training.cols;
    let mut classes = Vec::with_capacity(unique.len());
    for (class_index, rows) in rows_by_class.iter().enumerate() {
        let mean = class_mean(training, rows);
        let mut covariance = class_covariance(training, rows, &mean)?;
        if discriminant.uses_diagonal_covariance() {
            covariance = diagonalized(&covariance);
        }
        let (inverse, log_det) = invert_covariance(&covariance, "class covariance")?;
        classes.push(ClassStats {
            label: unique[class_index].clone(),
            rows: rows.clone(),
            mean,
            covariance,
            inverse,
            log_det,
        });
    }
    let (shared_inverse, shared_log_det) = if discriminant.uses_shared_covariance() {
        let mut pooled = pooled_covariance(&classes, cols)?;
        if discriminant.uses_diagonal_covariance() {
            pooled = diagonalized(&pooled);
        }
        let (inverse, log_det) = invert_covariance(&pooled, "pooled covariance")?;
        (Some(inverse), Some(log_det))
    } else {
        (None, None)
    };
    Ok(PreparedModel {
        kind: labels.kind,
        discriminant,
        classes,
        priors,
        shared_inverse,
        shared_log_det,
        cols,
    })
}

pub(super) fn class_mean(training: &FloatingMatrix, rows: &[usize]) -> Vec<f64> {
    let mut mean = vec![0.0; training.cols];
    for row in rows {
        for col in 0..training.cols {
            mean[col] += training.values[row + col * training.rows];
        }
    }
    for value in &mut mean {
        *value /= rows.len() as f64;
    }
    mean
}

pub(super) fn class_covariance(
    training: &FloatingMatrix,
    rows: &[usize],
    mean: &[f64],
) -> BuiltinResult<DMatrix<f64>> {
    let cols = training.cols;
    let mut cov = DMatrix::<f64>::zeros(cols, cols);
    if rows.len() < 2 {
        for idx in 0..cols {
            cov[(idx, idx)] = REGULARIZATION;
        }
        return Ok(cov);
    }
    for row in rows {
        for a in 0..cols {
            let da = training.values[row + a * training.rows] - mean[a];
            for b in 0..cols {
                let db = training.values[row + b * training.rows] - mean[b];
                cov[(a, b)] += da * db;
            }
        }
    }
    let denom = (rows.len() - 1) as f64;
    for value in cov.iter_mut() {
        *value /= denom;
    }
    Ok(cov)
}

pub(super) fn pooled_covariance(
    classes: &[ClassStats],
    cols: usize,
) -> BuiltinResult<DMatrix<f64>> {
    let total_rows = classes.iter().map(|class| class.rows.len()).sum::<usize>();
    if total_rows <= classes.len() {
        let mut identity = DMatrix::<f64>::zeros(cols, cols);
        for idx in 0..cols {
            identity[(idx, idx)] = REGULARIZATION;
        }
        return Ok(identity);
    }
    let mut pooled = DMatrix::<f64>::zeros(cols, cols);
    for class in classes {
        let weight = class.rows.len().saturating_sub(1) as f64;
        pooled += class.covariance.clone() * weight;
    }
    pooled /= (total_rows - classes.len()) as f64;
    Ok(pooled)
}

pub(super) fn diagonalized(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let mut out = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());
    for idx in 0..matrix.nrows().min(matrix.ncols()) {
        out[(idx, idx)] = matrix[(idx, idx)];
    }
    out
}

pub(super) fn invert_covariance(
    matrix: &DMatrix<f64>,
    label: &'static str,
) -> BuiltinResult<(DMatrix<f64>, f64)> {
    let mut adjusted = matrix.clone();
    for idx in 0..adjusted.nrows().min(adjusted.ncols()) {
        adjusted[(idx, idx)] += REGULARIZATION;
    }
    let lu = adjusted.lu();
    let det = lu.determinant();
    if !det.is_finite() || det.abs() <= EPS {
        return Err(numerical_error(format!("classify: singular {label}")));
    }
    let Some(inverse) = lu.try_inverse() else {
        return Err(numerical_error(format!("classify: singular {label}")));
    };
    Ok((inverse, det.abs().ln()))
}
