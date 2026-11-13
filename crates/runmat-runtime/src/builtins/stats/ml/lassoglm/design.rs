use super::*;

pub(super) fn prepare_data(
    x: &Tensor,
    response: (Vec<f64>, Vec<f64>),
    options: &Options,
) -> BuiltinResult<PreparedData> {
    let rows = x.rows();
    let cols = x.cols();
    let mut weights = options.weights.clone().unwrap_or_else(|| vec![1.0; rows]);
    let trials = if response.1.is_empty() {
        vec![1.0; rows]
    } else {
        response.1
    };
    let offset = options.offset.clone().unwrap_or_else(|| vec![0.0; rows]);
    if offset.iter().any(|value| !value.is_finite()) {
        return Err(invalid("lassoglm: Offset must contain finite values"));
    }
    for (weight, trial) in weights.iter_mut().zip(trials.iter()) {
        *weight *= *trial;
    }
    let weight_sum: f64 = weights.iter().sum();
    if !(weight_sum > 0.0 && weight_sum.is_finite()) {
        return Err(invalid("lassoglm: Weights must have a positive finite sum"));
    }
    for weight in &mut weights {
        *weight /= weight_sum;
    }

    let mut x_means = vec![0.0; cols];
    let mut x_scales = vec![1.0; cols];
    let x_values = tensor::tensor_values_f64_cow(x);
    let mut x_work = vec![0.0; x_values.len()];
    for col in 0..cols {
        let mean = if options.intercept {
            weighted_column_mean(x_values.as_ref(), rows, &weights, col)
        } else {
            0.0
        };
        let mut scale = 1.0;
        if options.intercept && options.standardize {
            let variance = (0..rows)
                .map(|row| {
                    let centered = matrix_value(x_values.as_ref(), rows, row, col) - mean;
                    weights[row] * centered * centered
                })
                .sum::<f64>();
            scale = variance.sqrt();
            if scale <= EPS || !scale.is_finite() {
                scale = 1.0;
            }
        }
        x_means[col] = mean;
        x_scales[col] = scale;
        for row in 0..rows {
            let idx = row + col * rows;
            x_work[idx] = (x_values[idx] - mean) / scale;
        }
    }

    Ok(PreparedData {
        x_original: x_values.into_owned(),
        x_work,
        y: response.0,
        trials,
        weights,
        offset,
        x_means,
        x_scales,
        rows,
        cols,
    })
}

pub(super) fn weighted_column_mean(x: &[f64], rows: usize, weights: &[f64], col: usize) -> f64 {
    (0..rows)
        .map(|row| matrix_value(x, rows, row, col) * weights[row])
        .sum()
}

pub(super) fn matrix_value(values: &[f64], rows: usize, row: usize, col: usize) -> f64 {
    values[row + col * rows]
}
