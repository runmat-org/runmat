//! Real SISO unit-step simulation and response materialization.
use super::*;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[derive(Clone, Debug)]
pub(crate) struct StepResponse {
    pub y: Vec<f64>,
    pub t: Vec<f64>,
}

pub(crate) fn evaluate_step(
    model: &RealTfModel,
    time: TimeSpec,
    caller: StepSimulationCaller,
) -> BuiltinResult<StepResponse> {
    if model.sample_time() > 0.0 {
        evaluate_discrete_step(model, time, caller)
    } else {
        evaluate_continuous_step(model, time, caller)
    }
}

fn evaluate_continuous_step(
    model: &RealTfModel,
    time: TimeSpec,
    caller: StepSimulationCaller,
) -> BuiltinResult<StepResponse> {
    let t = continuous_time_vector(model, time, caller)?;
    let (num, den) = model.normalized(caller.name())?;
    let response =
        continuous_response(&num, &den, &t).map_err(|error| error.into_runtime(caller))?;
    Ok(StepResponse { y: response, t })
}

fn continuous_time_vector(
    model: &RealTfModel,
    time: TimeSpec,
    caller: StepSimulationCaller,
) -> BuiltinResult<Vec<f64>> {
    match time {
        TimeSpec::Auto => Ok(linspace(
            0.0,
            automatic_final_time(model, caller)?,
            DEFAULT_SAMPLES,
        )),
        TimeSpec::FinalTime(final_time) => Ok(linspace(0.0, final_time, DEFAULT_SAMPLES)),
        TimeSpec::Vector(values) => Ok(values),
    }
}

fn continuous_response(num: &[f64], den: &[f64], t: &[f64]) -> SimulationResult<Vec<f64>> {
    let order = den.len() - 1;
    if order == 0 {
        let gain = num.last().copied().unwrap_or(0.0) / den[0];
        return Ok(vec![gain; t.len()]);
    }

    let mut padded_num = vec![0.0; order + 1 - num.len()];
    padded_num.extend_from_slice(num);
    let direct = padded_num[0];
    let a = &den[1..];
    let mut c = Vec::with_capacity(order);
    for state_idx in 0..order {
        let coeff_idx = order - state_idx;
        c.push(padded_num[coeff_idx] - direct * den[coeff_idx]);
    }

    let mut state = vec![0.0; order];
    let mut current_t = 0.0;
    let mut response = Vec::with_capacity(t.len());
    for &target_t in t {
        if target_t > current_t {
            integrate_to(&mut state, a, current_t, target_t);
            current_t = target_t;
        }
        response.push(dot(&c, &state) + direct);
    }
    Ok(response)
}

fn integrate_to(state: &mut [f64], a: &[f64], start: f64, end: f64) {
    let duration = end - start;
    if duration <= 0.0 {
        return;
    }
    let steps = ((duration / 0.01).ceil() as usize).clamp(1, 10_000);
    let h = duration / steps as f64;
    for _ in 0..steps {
        rk4_step(state, a, h);
    }
}

fn rk4_step(state: &mut [f64], a: &[f64], h: f64) {
    let k1 = derivative(state, a);
    let s2 = add_scaled(state, &k1, h * 0.5);
    let k2 = derivative(&s2, a);
    let s3 = add_scaled(state, &k2, h * 0.5);
    let k3 = derivative(&s3, a);
    let s4 = add_scaled(state, &k3, h);
    let k4 = derivative(&s4, a);
    for idx in 0..state.len() {
        state[idx] += h * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]) / 6.0;
    }
}

fn derivative(state: &[f64], a: &[f64]) -> Vec<f64> {
    let order = state.len();
    let mut dx = vec![0.0; order];
    if order > 1 {
        dx[..(order - 1)].copy_from_slice(&state[1..order]);
    }
    let mut last = 1.0;
    for state_idx in 0..order {
        let coeff = a[order - 1 - state_idx];
        last -= coeff * state[state_idx];
    }
    dx[order - 1] = last;
    dx
}

fn add_scaled(state: &[f64], delta: &[f64], scale: f64) -> Vec<f64> {
    state
        .iter()
        .zip(delta)
        .map(|(value, delta)| value + scale * delta)
        .collect()
}

fn evaluate_discrete_step(
    model: &RealTfModel,
    time: TimeSpec,
    caller: StepSimulationCaller,
) -> BuiltinResult<StepResponse> {
    let t = discrete_time_vector(model.sample_time(), time)
        .map_err(|error| error.into_runtime(caller))?;
    if t.len() > MAX_DISCRETE_SAMPLES {
        return Err(simulation_error(
            SimulationErrorKind::DiscreteLimit,
            format!("discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"),
        )
        .into_runtime(caller));
    }
    let sample_indices = t
        .iter()
        .map(|&value| checked_discrete_sample_index(model.sample_time(), value))
        .collect::<SimulationResult<Vec<_>>>()
        .map_err(|error| error.into_runtime(caller))?;
    let max_k = sample_indices.iter().copied().max().unwrap_or(0);
    let count = max_k
        .checked_add(1)
        .ok_or_else(|| {
            simulation_error(
                SimulationErrorKind::DiscreteLimit,
                "discrete sample index exceeds platform limits",
            )
        })
        .map_err(|error| error.into_runtime(caller))?;
    let (num, den) = model.normalized(caller.name())?;
    let all_y = discrete_response(&num, &den, count).map_err(|error| error.into_runtime(caller))?;
    let y = sample_indices
        .into_iter()
        .map(|idx| {
            all_y.get(idx).copied().ok_or_else(|| {
                simulation_error(
                    SimulationErrorKind::DiscreteLimit,
                    "discrete sample index exceeds response length",
                )
            })
        })
        .collect::<SimulationResult<Vec<_>>>()
        .map_err(|error| error.into_runtime(caller))?;
    Ok(StepResponse { y, t })
}

fn discrete_time_vector(sample_time: f64, time: TimeSpec) -> SimulationResult<Vec<f64>> {
    match time {
        TimeSpec::Auto => Ok((0..DEFAULT_SAMPLES)
            .map(|idx| idx as f64 * sample_time)
            .collect()),
        TimeSpec::FinalTime(final_time) => {
            let steps = checked_discrete_sample_steps(sample_time, final_time)?;
            Ok((0..=steps).map(|idx| idx as f64 * sample_time).collect())
        }
        TimeSpec::Vector(values) => Ok(values),
    }
}

fn checked_discrete_sample_steps(sample_time: f64, final_time: f64) -> SimulationResult<usize> {
    let steps = (final_time / sample_time).floor();
    if !steps.is_finite() || steps < 0.0 || steps > usize::MAX as f64 {
        return Err(simulation_error(
            SimulationErrorKind::DiscreteLimit,
            "discrete sample count exceeds platform limits",
        ));
    }
    if steps >= MAX_DISCRETE_SAMPLES as f64 {
        return Err(simulation_error(
            SimulationErrorKind::DiscreteLimit,
            format!("discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"),
        ));
    }
    Ok(steps as usize)
}

fn checked_discrete_sample_index(sample_time: f64, time: f64) -> SimulationResult<usize> {
    let index = (time / sample_time).round();
    if !index.is_finite() || index < 0.0 || index > usize::MAX as f64 {
        return Err(simulation_error(
            SimulationErrorKind::DiscreteLimit,
            "discrete sample index exceeds platform limits",
        ));
    }
    if index >= MAX_DISCRETE_SAMPLES as f64 {
        return Err(simulation_error(
            SimulationErrorKind::DiscreteLimit,
            format!("discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"),
        ));
    }
    Ok(index as usize)
}

fn discrete_response(num: &[f64], den: &[f64], count: usize) -> SimulationResult<Vec<f64>> {
    let order = den.len() - 1;
    let mut padded_num = vec![0.0; order + 1 - num.len()];
    padded_num.extend_from_slice(num);
    let mut y = vec![0.0; count];
    for k in 0..count {
        let mut value = 0.0;
        for (idx, &coeff) in padded_num.iter().enumerate() {
            if k >= idx {
                value += coeff;
            }
        }
        for idx in 1..den.len() {
            if k >= idx {
                value -= den[idx] * y[k - idx];
            }
        }
        y[k] = value;
    }
    Ok(y)
}

fn automatic_final_time(model: &RealTfModel, caller: StepSimulationCaller) -> BuiltinResult<f64> {
    let (_, den) = model.normalized(caller.name())?;
    let poles = polynomial_roots(&den).map_err(|error| error.into_runtime(caller))?;
    let slowest_decay = poles
        .iter()
        .filter_map(|pole| if pole.re < -EPS { Some(-pole.re) } else { None })
        .fold(f64::INFINITY, f64::min);
    if slowest_decay.is_finite() && slowest_decay > EPS {
        Ok((5.0 / slowest_decay).clamp(1.0, 100.0))
    } else {
        Ok(10.0)
    }
}

fn polynomial_roots(coeffs: &[f64]) -> SimulationResult<Vec<Complex64>> {
    let trimmed = trim_leading_zeros(coeffs.to_vec());
    if trimmed.len() <= 1 {
        return Ok(Vec::new());
    }
    if trimmed.len() == 2 {
        return Ok(vec![Complex64::new(-trimmed[1] / trimmed[0], 0.0)]);
    }
    let degree = trimmed.len() - 1;
    let leading = trimmed[0];
    let mut companion = DMatrix::<Complex64>::zeros(degree, degree);
    for row in 1..degree {
        companion[(row, row - 1)] = Complex64::new(1.0, 0.0);
    }
    for (idx, coeff) in trimmed.iter().enumerate().skip(1) {
        companion[(0, idx - 1)] = Complex64::new(-coeff / leading, 0.0);
    }
    let eigenvalues = companion.eigenvalues().ok_or_else(|| {
        simulation_error(
            SimulationErrorKind::Internal,
            "failed to compute transfer-function poles",
        )
    })?;
    Ok(eigenvalues.iter().copied().collect())
}

fn trim_leading_zeros(coeffs: Vec<f64>) -> Vec<f64> {
    let first_nonzero = coeffs
        .iter()
        .position(|value| *value != 0.0)
        .unwrap_or(coeffs.len());
    coeffs[first_nonzero..].to_vec()
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![end];
    }
    let step = (end - start) / (count - 1) as f64;
    (0..count).map(|idx| start + idx as f64 * step).collect()
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}
