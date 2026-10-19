use super::*;

pub(super) fn fit_ols(spec: &FitSpec) -> BuiltinResult<OlsFit> {
    let clean = clean_data(spec)?;
    let observations = clean.y.len();
    if observations == 0 {
        return Err(fitlm_invalid(
            "fitlm: at least one complete observation is required",
        ));
    }
    let svd = clean.weighted_design.clone().svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| fitlm_numerical("fitlm: SVD did not return left singular vectors"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| fitlm_numerical("fitlm: SVD did not return right singular vectors"))?;
    let singular_values = svd.singular_values.iter().copied().collect::<Vec<_>>();
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = (observations.max(spec.terms.len()) as f64) * f64::EPSILON * largest.max(1.0);
    let rank = singular_values
        .iter()
        .filter(|value| value.abs() > tolerance)
        .count();
    let mut beta = DVector::zeros(spec.terms.len());
    let mut inv_xtx = DMatrix::<f64>::zeros(spec.terms.len(), spec.terms.len());
    for (index, singular_value) in singular_values.iter().copied().enumerate() {
        if singular_value.abs() <= tolerance {
            continue;
        }
        let uty = u.column(index).dot(&clean.weighted_y);
        let contribution = uty / singular_value;
        for row in 0..spec.terms.len() {
            beta[row] += v_t[(index, row)] * contribution;
        }
        let inv_s2 = 1.0 / (singular_value * singular_value);
        for row in 0..spec.terms.len() {
            for col in 0..spec.terms.len() {
                inv_xtx[(row, col)] += v_t[(index, row)] * v_t[(index, col)] * inv_s2;
            }
        }
    }
    let fitted_vec = &clean.design * &beta;
    let residual_vec = &clean.y - &fitted_vec;
    let sse = residual_vec
        .iter()
        .zip(clean.weights.iter())
        .map(|(residual, weight)| residual * residual * weight)
        .sum::<f64>();
    let weight_sum = clean.weights.iter().sum::<f64>();
    let y_mean = if weight_sum > 0.0 {
        clean
            .y
            .iter()
            .zip(clean.weights.iter())
            .map(|(y, weight)| y * weight)
            .sum::<f64>()
            / weight_sum
    } else {
        f64::NAN
    };
    let sst = clean
        .y
        .iter()
        .zip(clean.weights.iter())
        .map(|(y, weight)| {
            let diff = y - y_mean;
            diff * diff * weight
        })
        .sum::<f64>();
    let dfe = observations as f64 - rank as f64;
    let mse = if dfe > 0.0 { sse / dfe } else { f64::NAN };
    let rmse = mse.sqrt();
    let r_squared = if sst > EPS {
        1.0 - sse / sst
    } else if sse <= EPS {
        1.0
    } else {
        f64::NAN
    };
    let adjusted_r_squared = if dfe > 0.0 && observations > 1 && r_squared.is_finite() {
        1.0 - (1.0 - r_squared) * ((observations - 1) as f64) / dfe
    } else {
        f64::NAN
    };
    let covariance = inv_xtx.iter().map(|value| value * mse).collect::<Vec<_>>();
    let mut se = Vec::with_capacity(spec.terms.len());
    let mut tstat = Vec::with_capacity(spec.terms.len());
    let mut pvalue = Vec::with_capacity(spec.terms.len());
    for i in 0..spec.terms.len() {
        let variance = covariance[i + i * spec.terms.len()];
        let std_error = if variance >= 0.0 {
            variance.sqrt()
        } else {
            f64::NAN
        };
        let t = beta[i] / std_error;
        let p = if dfe > 0.0 {
            (2.0 * student_t_cdf_upper(t.abs(), dfe)).min(1.0)
        } else {
            f64::NAN
        };
        se.push(std_error);
        tstat.push(t);
        pvalue.push(p);
    }
    Ok(OlsFit {
        beta: beta.iter().copied().collect(),
        fitted: fitted_vec.iter().copied().collect(),
        residuals: residual_vec.iter().copied().collect(),
        covariance,
        se,
        tstat,
        pvalue,
        sse,
        mse,
        rmse,
        r_squared,
        adjusted_r_squared,
        dfe,
        rank,
        observations,
    })
}

fn clean_data(spec: &FitSpec) -> BuiltinResult<CleanData> {
    let (rows, cols) = fitlm_matrix_dimensions(&spec.x);
    let exclude_mask = exclude_mask(spec.exclude.as_ref(), rows)?;
    let weights = spec.weights.as_ref();
    if let Some(weights) = weights {
        if weights.len() != rows {
            return Err(fitlm_invalid(
                "fitlm: Weights length must match the number of observations",
            ));
        }
    }
    let mut design_rows = Vec::new();
    let mut y_values = Vec::new();
    let mut clean_weights = Vec::new();
    for row in 0..rows {
        if exclude_mask.as_ref().is_some_and(|mask| mask[row]) {
            continue;
        }
        let y = spec.y[row];
        let weight = weights.map(|w| w[row]).unwrap_or(1.0);
        if weight < 0.0 || !weight.is_finite() {
            return Err(fitlm_invalid(
                "fitlm: weights must be nonnegative finite values",
            ));
        }
        let mut raw = Vec::with_capacity(cols);
        let mut has_nan = y.is_nan();
        let mut has_inf = y.is_infinite();
        for col in 0..cols {
            let value = x_value(&spec.x, row, col);
            has_nan |= value.is_nan();
            has_inf |= value.is_infinite();
            raw.push(value);
        }
        if has_inf {
            return Err(fitlm_invalid("fitlm: X and y must not contain Inf values"));
        }
        if has_nan || weight == 0.0 {
            continue;
        }
        for term in &spec.terms {
            design_rows.push(evaluate_term(&raw, term));
        }
        y_values.push(y);
        clean_weights.push(weight);
    }
    let observations = y_values.len();
    let predictors = spec.terms.len();
    let design = DMatrix::from_row_slice(observations, predictors, &design_rows);
    let mut weighted_rows = design_rows;
    let mut weighted_y = y_values.clone();
    for row in 0..observations {
        let scale = clean_weights[row].sqrt();
        weighted_y[row] *= scale;
        for col in 0..predictors {
            weighted_rows[row * predictors + col] *= scale;
        }
    }
    Ok(CleanData {
        design,
        weighted_design: DMatrix::from_row_slice(observations, predictors, &weighted_rows),
        y: DVector::from_column_slice(&y_values),
        weighted_y: DVector::from_column_slice(&weighted_y),
        weights: clean_weights,
    })
}

pub(super) fn evaluate_term(row: &[f64], term: &[u32]) -> f64 {
    let mut value = 1.0;
    for (x, power) in row.iter().zip(term) {
        match *power {
            0 => {}
            1 => value *= *x,
            2 => value *= x * x,
            n => value *= x.powi(n as i32),
        }
    }
    value
}
