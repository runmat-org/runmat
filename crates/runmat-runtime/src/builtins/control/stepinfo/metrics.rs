//! Pure sampled-response metrics and scalar-structure assembly.

use super::*;

pub(super) struct StepMetrics {
    rise_time: f64,
    transient_time: f64,
    settling_time: f64,
    settling_min: f64,
    settling_max: f64,
    overshoot: f64,
    undershoot: f64,
    peak: f64,
    peak_time: f64,
    steady_state_value: f64,
}

pub(super) fn validate_inputs(y: &[f64], t: &[f64]) -> BuiltinResult<()> {
    if y.is_empty() || t.is_empty() {
        return Err(error("response and time vectors cannot be empty"));
    }
    if y.len() != t.len() {
        return Err(error("y and t must have the same number of samples"));
    }
    if y.iter().chain(t).any(|value| !value.is_finite()) {
        return Err(error("response and time values must be finite"));
    }
    if t.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(error("t must be nondecreasing"));
    }
    Ok(())
}

pub(super) fn compute(
    y: &[f64],
    t: &[f64],
    y_final: f64,
    threshold: f64,
) -> BuiltinResult<StepMetrics> {
    validate_inputs(y, t)?;
    if !y_final.is_finite() {
        return Err(error("yfinal must be finite"));
    }
    let y0 = y[0];
    let amplitude = y_final - y0;
    let amplitude_abs = amplitude.abs();
    let scale = amplitude_abs.max(y_final.abs()).max(1.0);
    let direction = if amplitude >= 0.0 { 1.0 } else { -1.0 };

    let (rise_time, rise_idx) = if amplitude_abs <= 1.0e-12 {
        (f64::NAN, 0)
    } else {
        let t10 = crossing(y, t, y0 + 0.1 * amplitude, direction);
        let (t90, idx90) = crossing_with_index(y, t, y0 + 0.9 * amplitude, direction);
        match (t10, t90) {
            (Some(start), Some(end)) => (end - start, idx90.unwrap_or(0)),
            _ => (f64::NAN, 0),
        }
    };

    let band = threshold * scale;
    let last_outside = y.iter().rposition(|value| (*value - y_final).abs() > band);
    let settling_time = match last_outside {
        None => t[0],
        Some(index) if index + 1 < t.len() => t[index + 1],
        Some(_) => f64::NAN,
    };
    let window = &y[rise_idx.min(y.len() - 1)..];
    let settling_min = window.iter().copied().fold(f64::INFINITY, f64::min);
    let settling_max = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_y = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_y = y.iter().copied().fold(f64::INFINITY, f64::min);
    let overshoot = if amplitude_abs <= 1.0e-12 {
        0.0
    } else if amplitude >= 0.0 {
        ((max_y - y_final) / amplitude_abs * 100.0).max(0.0)
    } else {
        ((y_final - min_y) / amplitude_abs * 100.0).max(0.0)
    };
    let undershoot = if amplitude_abs <= 1.0e-12 {
        0.0
    } else if amplitude >= 0.0 {
        ((y0 - min_y) / amplitude_abs * 100.0).max(0.0)
    } else {
        ((max_y - y0) / amplitude_abs * 100.0).max(0.0)
    };
    let (peak_index, peak) = y
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value.abs()))
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((0, y[0].abs()));
    Ok(StepMetrics {
        rise_time,
        transient_time: settling_time,
        settling_time,
        settling_min,
        settling_max,
        overshoot,
        undershoot,
        peak,
        peak_time: t[peak_index],
        steady_state_value: y_final,
    })
}

fn crossing(y: &[f64], t: &[f64], threshold: f64, direction: f64) -> Option<f64> {
    crossing_with_index(y, t, threshold, direction).0
}

fn crossing_with_index(
    y: &[f64],
    t: &[f64],
    threshold: f64,
    direction: f64,
) -> (Option<f64>, Option<usize>) {
    for index in 0..y.len() {
        let current = direction * (y[index] - threshold);
        if current >= -1.0e-12 {
            if index == 0 {
                return (Some(t[0]), Some(0));
            }
            let previous = direction * (y[index - 1] - threshold);
            let denominator = current - previous;
            if denominator.abs() <= 1.0e-12 {
                return (Some(t[index]), Some(index));
            }
            let alpha = (-previous / denominator).clamp(0.0, 1.0);
            return (
                Some(t[index - 1] + alpha * (t[index] - t[index - 1])),
                Some(index),
            );
        }
    }
    (None, None)
}

pub(super) fn to_value(metrics: StepMetrics) -> BuiltinResult<Value> {
    let mut value = runmat_value::UnboundedStructValueBuilder::new();
    value.insert("RiseTime", Value::Num(metrics.rise_time));
    value.insert("TransientTime", Value::Num(metrics.transient_time));
    value.insert("SettlingTime", Value::Num(metrics.settling_time));
    value.insert("SettlingMin", Value::Num(metrics.settling_min));
    value.insert("SettlingMax", Value::Num(metrics.settling_max));
    value.insert("Overshoot", Value::Num(metrics.overshoot));
    value.insert("Undershoot", Value::Num(metrics.undershoot));
    value.insert("Peak", Value::Num(metrics.peak));
    value.insert("PeakTime", Value::Num(metrics.peak_time));
    value.insert("SteadyStateValue", Value::Num(metrics.steady_state_value));
    Ok(Value::Struct(value.finish()))
}

fn error(message: impl Into<String>) -> crate::RuntimeError {
    control_error(
        BUILTIN_NAME,
        "RunMat:stepinfo:InvalidData",
        format!("stepinfo: {}", message.into()),
    )
}
