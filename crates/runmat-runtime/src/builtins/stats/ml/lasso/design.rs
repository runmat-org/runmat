use super::*;

pub(super) fn prepare_data(
    x: &Tensor,
    y: &[f64],
    options: &LassoOptions,
) -> BuiltinResult<PreparedData> {
    let rows = x.rows();
    let cols = x.cols();
    let x_values = tensor::tensor_values_f64_cow(x);
    let mut weights = options.weights.clone().unwrap_or_else(|| vec![1.0; rows]);
    let weight_sum: f64 = weights.iter().sum();
    if !(weight_sum > 0.0 && weight_sum.is_finite()) {
        return Err(invalid_argument(
            "lasso: Weights must have a positive finite sum",
        ));
    }
    for weight in &mut weights {
        *weight /= weight_sum;
    }

    let y_mean = if options.intercept {
        weighted_mean(y, &weights)
    } else {
        0.0
    };
    let y_work = y.iter().map(|value| *value - y_mean).collect::<Vec<_>>();

    let mut x_means = vec![0.0; cols];
    let mut x_scales = vec![1.0; cols];
    let mut x_work = vec![0.0; x_values.len()];
    for col in 0..cols {
        let mean = if options.intercept {
            weighted_column_mean(&x_values, rows, &weights, col)
        } else {
            0.0
        };
        let mut scale = 1.0;
        if options.intercept && options.standardize {
            let variance = (0..rows)
                .map(|row| {
                    let centered = x_values[row + col * rows] - mean;
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
        x_original: x_values.to_vec(),
        x_work,
        y_original: y.to_vec(),
        y_work,
        weights,
        x_means,
        x_scales,
        y_mean,
        rows,
        cols,
    })
}

pub(super) fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| value * weight)
        .sum()
}

pub(super) fn weighted_column_mean(
    x_values: &[f64],
    rows: usize,
    weights: &[f64],
    col: usize,
) -> f64 {
    (0..rows)
        .map(|row| x_values[row + col * rows] * weights[row])
        .sum()
}
