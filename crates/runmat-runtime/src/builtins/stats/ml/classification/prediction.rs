use super::*;
use nalgebra::DVector;
use std::cmp::Ordering;

pub(super) fn predict_rows(
    model: &PreparedModel,
    data: &FloatingMatrix,
    need_posterior: bool,
    need_logp: bool,
) -> BuiltinResult<PredictionResult> {
    let rows = data.rows;
    let class_count = model.classes.len();
    let mut label_indices = Vec::new();
    label_indices
        .try_reserve_exact(rows)
        .map_err(|err| internal_error(format!("classify: cannot allocate labels: {err}")))?;
    let mut posterior_data = if need_posterior {
        let count = rows
            .checked_mul(class_count)
            .ok_or_else(|| internal_error("classify: posterior shape overflows"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|err| internal_error(format!("classify: cannot allocate posterior: {err}")))?;
        values.resize(count, 0.0);
        Some(values)
    } else {
        None
    };
    let mut logp_data = if need_logp {
        let mut values = Vec::new();
        values.try_reserve_exact(rows).map_err(|err| {
            internal_error(format!("classify: cannot allocate log density: {err}"))
        })?;
        values.resize(rows, 0.0);
        Some(values)
    } else {
        None
    };
    for row in 0..rows {
        let x = row_vector(data, row);
        let mut scores = Vec::new();
        scores.try_reserve_exact(class_count).map_err(|err| {
            internal_error(format!("classify: cannot allocate class scores: {err}"))
        })?;
        for class_index in 0..class_count {
            scores.push(discriminant_score(model, class_index, &x)?);
        }
        let best = scores
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| internal_error("classify: no classes"))?;
        label_indices.push(best);
        if model.discriminant == DiscriminantType::Mahalanobis {
            if let Some(posterior) = &mut posterior_data {
                for class_index in 0..class_count {
                    posterior[row + class_index * rows] = f64::NAN;
                }
            }
            if let Some(logp) = &mut logp_data {
                logp[row] = f64::NAN;
            }
        } else if need_posterior || need_logp {
            let logp = log_sum_exp(&scores);
            if let Some(values) = &mut logp_data {
                values[row] = logp;
            }
            if let Some(posterior) = &mut posterior_data {
                for class_index in 0..class_count {
                    posterior[row + class_index * rows] = (scores[class_index] - logp).exp();
                }
            }
        }
    }
    Ok(PredictionResult {
        labels: label_indices,
        posterior: posterior_data
            .map(|values| {
                Tensor::new(values, vec![rows, class_count])
                    .map_err(|err| internal_error(format!("classify: {err}")))
            })
            .transpose()?,
        logp: logp_data
            .map(|values| {
                Tensor::new(values, vec![rows, 1])
                    .map_err(|err| internal_error(format!("classify: {err}")))
            })
            .transpose()?,
    })
}

pub(super) fn row_vector(data: &FloatingMatrix, row: usize) -> DVector<f64> {
    DVector::from_iterator(
        data.cols,
        (0..data.cols).map(|col| data.values[row + col * data.rows]),
    )
}

pub(super) fn mean_vector(mean: &[f64]) -> DVector<f64> {
    DVector::from_iterator(mean.len(), mean.iter().copied())
}

pub(super) fn discriminant_score(
    model: &PreparedModel,
    class_index: usize,
    x: &DVector<f64>,
) -> BuiltinResult<f64> {
    let class = &model.classes[class_index];
    let mean = mean_vector(&class.mean);
    let diff = x - &mean;
    let (inverse, log_det) = if let Some(inverse) = &model.shared_inverse {
        (inverse, model.shared_log_det.unwrap_or(0.0))
    } else {
        (&class.inverse, class.log_det)
    };
    let distance = (diff.transpose() * inverse * diff)[(0, 0)];
    if model.discriminant == DiscriminantType::Mahalanobis {
        Ok(-distance)
    } else {
        let gaussian_norm = -0.5 * (model.cols as f64) * (2.0 * std::f64::consts::PI).ln();
        Ok(model.priors[class_index].ln() + gaussian_norm - 0.5 * log_det - 0.5 * distance)
    }
}

pub(super) fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(value));
    max + values
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f64>()
        .ln()
}

pub(super) fn apparent_error(model: &PreparedModel, predicted: &[usize]) -> f64 {
    let mut total = 0.0;
    for (class_index, class) in model.classes.iter().enumerate() {
        let mut wrong = 0usize;
        for row in &class.rows {
            if predicted.get(*row).copied() != Some(class_index) {
                wrong += 1;
            }
        }
        let class_error = if class.rows.is_empty() {
            0.0
        } else {
            wrong as f64 / class.rows.len() as f64
        };
        total += model.priors[class_index] * class_error;
    }
    total
}
