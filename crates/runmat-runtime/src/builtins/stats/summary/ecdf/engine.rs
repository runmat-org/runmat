//! Host empirical distribution, censoring, frequency, and confidence arithmetic.
use super::*;
use std::cmp::Ordering;

use crate::builtins::common::tensor;
pub(super) fn evaluate_ecdf(value: Value, options: Options) -> BuiltinResult<EcdfEvaluation> {
    let tensor = tensor_from_value(value, ECDF_OWNER)?;
    if tensor.rows() > 1 && tensor.cols() > 1 {
        if tensor.cols() == 2 {
            return Err(unsupported(
                "ecdf: interval-censored two-column input is not supported yet",
            ));
        }
        return Err(invalid_argument(
            ECDF_OWNER,
            format!(
                "ecdf: expected a vector or two-column interval matrix, got {}x{} matrix",
                tensor.rows(),
                tensor.cols()
            ),
        ));
    }
    let data = tensor::tensor_values_f64(&tensor);
    if data.is_empty() {
        return Err(invalid_argument(
            ECDF_OWNER,
            "ecdf: sample data must be nonempty",
        ));
    }
    let n = data.len();
    let censoring = options.censoring.unwrap_or_else(|| vec![0.0; n]);
    let frequency = options.frequency.unwrap_or_else(|| vec![1.0; n]);
    if censoring.len() != n {
        return Err(invalid_argument(
            ECDF_OWNER,
            "ecdf: Censoring vector must have the same number of elements as y",
        ));
    }
    if frequency.len() != n {
        return Err(invalid_argument(
            ECDF_OWNER,
            "ecdf: Frequency vector must have the same number of elements as y",
        ));
    }

    let mut observations = Vec::new();
    let mut has_left = false;
    let mut has_right = false;
    for ((time, censor), freq) in data
        .into_iter()
        .zip(censoring.into_iter())
        .zip(frequency.into_iter())
    {
        if time.is_nan() || censor.is_nan() || freq.is_nan() || freq == 0.0 {
            continue;
        }
        if !time.is_finite() {
            return Err(invalid_argument(
                ECDF_OWNER,
                "ecdf: sample data must be finite after removing NaN values",
            ));
        }
        if !(freq.is_finite() && freq >= 0.0 && freq.fract() == 0.0) {
            return Err(invalid_argument(
                ECDF_OWNER,
                "ecdf: Frequency values must be finite nonnegative integer counts",
            ));
        }
        let censor = if censor == 0.0 {
            0.0
        } else if censor == 1.0 {
            has_right = true;
            1.0
        } else if censor == -1.0 {
            has_left = true;
            -1.0
        } else {
            return Err(invalid_argument(
                ECDF_OWNER,
                "ecdf: Censoring values must be 0, 1, or -1",
            ));
        };
        observations.push(Observation {
            time,
            censor,
            frequency: freq,
        });
    }
    if observations.is_empty() {
        return Err(invalid_argument(
            ECDF_OWNER,
            "ecdf: no finite positive-frequency observations remain",
        ));
    }
    if has_left && has_right {
        return Err(unsupported(
            "ecdf: double-censored data with both left and right censoring is not supported yet",
        ));
    }
    if has_left {
        return evaluate_left_censored(&observations, options.alpha)
            .map(|eval| transform_eval(eval, options.function, options.bounds_on));
    }

    observations.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(Ordering::Equal));
    let eval = if has_right {
        evaluate_kaplan_meier(&observations, options.alpha)
    } else {
        evaluate_fully_observed(&observations, options.alpha)
    }?;
    Ok(transform_eval(eval, options.function, options.bounds_on))
}

fn evaluate_fully_observed(
    observations: &[Observation],
    alpha: f64,
) -> BuiltinResult<EcdfEvaluation> {
    let total: f64 = observations.iter().map(|obs| obs.frequency).sum();
    if total <= 0.0 {
        return Err(invalid_argument(
            ECDF_OWNER,
            "ecdf: total frequency must be positive",
        ));
    }
    let z = standard_normal_inv(1.0 - alpha / 2.0);
    let mut x = Vec::new();
    let mut f = Vec::new();
    let mut flo = Vec::new();
    let mut fup = Vec::new();
    let mut hazard = Vec::new();
    let mut hazard_flo = Vec::new();
    let mut hazard_fup = Vec::new();
    let first = observations[0].time;
    x.push(first);
    f.push(0.0);
    flo.push(0.0);
    fup.push(0.0);
    hazard.push(0.0);
    hazard_flo.push(0.0);
    hazard_fup.push(0.0);

    let mut cumulative = 0.0;
    let mut cumulative_hazard = 0.0;
    let mut hazard_variance = 0.0;
    let mut idx = 0;
    while idx < observations.len() {
        let time = observations[idx].time;
        let mut count = 0.0;
        while idx < observations.len() && observations[idx].time == time {
            count += observations[idx].frequency;
            idx += 1;
        }
        let risk = total - cumulative;
        cumulative += count;
        let p = (cumulative / total).clamp(0.0, 1.0);
        let se = (p * (1.0 - p) / total).sqrt();
        if risk > 0.0 {
            cumulative_hazard += count / risk;
            hazard_variance += count / (risk * risk);
        }
        let hazard_se = hazard_variance.sqrt();
        x.push(time);
        f.push(p);
        flo.push((p - z * se).clamp(0.0, 1.0));
        fup.push((p + z * se).clamp(0.0, 1.0));
        hazard.push(cumulative_hazard);
        hazard_flo.push((cumulative_hazard - z * hazard_se).max(0.0));
        hazard_fup.push(cumulative_hazard + z * hazard_se);
    }

    Ok(EcdfEvaluation {
        f,
        x,
        flo,
        fup,
        hazard,
        hazard_flo,
        hazard_fup,
        function: FunctionKind::Cdf,
        bounds_on: false,
    })
}

fn evaluate_kaplan_meier(
    observations: &[Observation],
    alpha: f64,
) -> BuiltinResult<EcdfEvaluation> {
    let z = standard_normal_inv(1.0 - alpha / 2.0);
    let total: f64 = observations.iter().map(|obs| obs.frequency).sum();
    if total <= 0.0 {
        return Err(invalid_argument(
            ECDF_OWNER,
            "ecdf: total frequency must be positive",
        ));
    }
    let mut x = Vec::new();
    let mut f = Vec::new();
    let mut flo = Vec::new();
    let mut fup = Vec::new();
    let mut hazard = Vec::new();
    let mut hazard_flo = Vec::new();
    let mut hazard_fup = Vec::new();
    let mut risk = total;
    let mut survivor = 1.0;
    let mut greenwood = 0.0;
    let mut cumulative_hazard = 0.0;
    let mut hazard_variance = 0.0;
    let mut idx = 0;
    while idx < observations.len() {
        let time = observations[idx].time;
        let mut events = 0.0;
        let mut censored = 0.0;
        while idx < observations.len() && observations[idx].time == time {
            if observations[idx].censor == 0.0 {
                events += observations[idx].frequency;
            } else {
                censored += observations[idx].frequency;
            }
            idx += 1;
        }
        if events > 0.0 && risk > 0.0 {
            if x.is_empty() {
                x.push(time);
                f.push(0.0);
                flo.push(0.0);
                fup.push(0.0);
                hazard.push(0.0);
                hazard_flo.push(0.0);
                hazard_fup.push(0.0);
            }
            let decrement = (1.0_f64 - events / risk).max(0.0_f64);
            survivor *= decrement;
            if risk > events {
                greenwood += events / (risk * (risk - events));
            }
            cumulative_hazard += events / risk;
            hazard_variance += events / (risk * risk);
            let cdf = (1.0 - survivor).clamp(0.0, 1.0);
            let se = (survivor * survivor * greenwood).sqrt();
            let hazard_se = hazard_variance.sqrt();
            x.push(time);
            f.push(cdf);
            flo.push((cdf - z * se).clamp(0.0, 1.0));
            fup.push((cdf + z * se).clamp(0.0, 1.0));
            hazard.push(cumulative_hazard);
            hazard_flo.push((cumulative_hazard - z * hazard_se).max(0.0));
            hazard_fup.push(cumulative_hazard + z * hazard_se);
        }
        risk -= events + censored;
    }

    Ok(EcdfEvaluation {
        f,
        x,
        flo,
        fup,
        hazard,
        hazard_flo,
        hazard_fup,
        function: FunctionKind::Cdf,
        bounds_on: false,
    })
}

fn evaluate_left_censored(
    observations: &[Observation],
    alpha: f64,
) -> BuiltinResult<EcdfEvaluation> {
    let z = standard_normal_inv(1.0 - alpha / 2.0);
    let total: f64 = observations.iter().map(|obs| obs.frequency).sum();
    if total <= 0.0 {
        return Err(invalid_argument(
            ECDF_OWNER,
            "ecdf: total frequency must be positive",
        ));
    }

    let mut descending = observations.to_vec();
    descending.sort_by(|a, b| b.time.partial_cmp(&a.time).unwrap_or(Ordering::Equal));
    let mut risk = total;
    let mut reverse_survivor: f64 = 1.0;
    let mut greenwood: f64 = 0.0;
    let mut points = Vec::new();
    let mut idx = 0;
    while idx < descending.len() {
        let time = descending[idx].time;
        let mut events = 0.0;
        let mut censored = 0.0;
        while idx < descending.len() && descending[idx].time == time {
            if descending[idx].censor == 0.0 {
                events += descending[idx].frequency;
            } else {
                censored += descending[idx].frequency;
            }
            idx += 1;
        }
        if events > 0.0 && risk > 0.0 {
            let cdf = reverse_survivor.clamp(0.0, 1.0);
            let se = (reverse_survivor * reverse_survivor * greenwood).sqrt();
            points.push((
                time,
                cdf,
                (cdf - z * se).clamp(0.0, 1.0),
                (cdf + z * se).clamp(0.0, 1.0),
            ));
            let decrement = (1.0_f64 - events / risk).max(0.0_f64);
            if risk > events {
                greenwood += events / (risk * (risk - events));
            }
            reverse_survivor *= decrement;
        }
        risk -= events + censored;
    }
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut x = Vec::new();
    let mut f = Vec::new();
    let mut flo = Vec::new();
    let mut fup = Vec::new();
    if let Some((first_time, _, _, _)) = points.first().copied() {
        x.push(first_time);
        f.push(0.0);
        flo.push(0.0);
        fup.push(0.0);
    }
    for (time, cdf, lower, upper) in points {
        x.push(time);
        f.push(cdf);
        flo.push(lower);
        fup.push(upper);
    }
    let (hazard, hazard_flo, hazard_fup) = hazard_from_cdf_bounds(&f, &flo, &fup);

    Ok(EcdfEvaluation {
        f,
        x,
        flo,
        fup,
        hazard,
        hazard_flo,
        hazard_fup,
        function: FunctionKind::Cdf,
        bounds_on: false,
    })
}

fn transform_eval(
    mut eval: EcdfEvaluation,
    function: FunctionKind,
    bounds_on: bool,
) -> EcdfEvaluation {
    if function == FunctionKind::Cdf {
        eval.bounds_on = bounds_on;
        return eval;
    }
    match function {
        FunctionKind::Cdf => unreachable!(),
        FunctionKind::Survivor => {
            let f = std::mem::take(&mut eval.f);
            let flo = std::mem::take(&mut eval.flo);
            let fup = std::mem::take(&mut eval.fup);
            eval.f = f.iter().map(|value| 1.0 - value).collect();
            eval.flo = fup.iter().map(|value| 1.0 - value).collect();
            eval.fup = flo.iter().map(|value| 1.0 - value).collect();
        }
        FunctionKind::CumulativeHazard => {
            eval.f = eval.hazard.clone();
            eval.flo = eval.hazard_flo.clone();
            eval.fup = eval.hazard_fup.clone();
        }
    }
    eval.function = function;
    eval.bounds_on = bounds_on;
    eval
}

fn hazard_from_survivor(survivor: f64) -> f64 {
    if survivor <= 0.0 {
        f64::INFINITY
    } else {
        -survivor.ln()
    }
}

fn hazard_from_cdf_bounds(f: &[f64], flo: &[f64], fup: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let hazard = f
        .iter()
        .map(|value| hazard_from_survivor(1.0 - value))
        .collect();
    let hazard_flo = flo
        .iter()
        .map(|value| hazard_from_survivor(1.0 - value))
        .collect();
    let hazard_fup = fup
        .iter()
        .map(|value| hazard_from_survivor(1.0 - value))
        .collect();
    (hazard, hazard_flo, hazard_fup)
}
