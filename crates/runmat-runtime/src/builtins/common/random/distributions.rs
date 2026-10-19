//! Distribution samplers over the admitted session host stream.

use super::*;

pub(crate) fn generate_exponential(
    turn: &mut RandomTurn,
    mu: f64,
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state_mut(turn, label, len != 0, |state| {
        (0..len)
            .map(|_| {
                let u = next_uniform_state(&mut state.state).max(MIN_UNIFORM);
                -mu * u.ln()
            })
            .collect()
    })
}

/// Fill admitted native-class output while retaining the ordered turn. The
/// same one-uniform exponential transform is used for scalar and array means.
pub(crate) async fn fill_exponential_array(
    turn: &mut RandomTurn,
    mu: &[f64],
    len: usize,
    label: &str,
    mut write: impl FnMut(usize, f64) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    debug_assert!(mu.len() == 1 || mu.len() == len);
    super::reservation::checked_length(len, label)?;
    let owner = turn.owner().cloned();
    if let Some(runtime) = owner.as_ref() {
        let mut budget = crate::execution::CpuCooperationBudget::new(
            runtime.execution().clone(),
            runtime.cancellation(),
        );
        budget
            .checkpoint(0)
            .await
            .map_err(|source| super::reservation::service_error(label, Some(runtime), source))?;
        for index in 0..len {
            turn.with_state(label, |state| {
                state.state.prepare_mutation(owner.as_ref())?;
                let u = next_uniform_state(&mut state.state).max(MIN_UNIFORM);
                let mean = if mu.len() == 1 { mu[0] } else { mu[index] };
                write(index, -mean * u.ln())
            })??;
            budget.checkpoint(1).await.map_err(|source| {
                super::reservation::service_error(label, Some(runtime), source)
            })?;
        }
    } else {
        turn.with_state(label, |state| -> BuiltinResult<()> {
            if len != 0 {
                state.state.prepare_mutation(None)?;
            }
            for index in 0..len {
                let u = next_uniform_state(&mut state.state).max(MIN_UNIFORM);
                let mean = if mu.len() == 1 { mu[0] } else { mu[index] };
                write(index, -mean * u.ln())?;
            }
            Ok(())
        })??;
    }
    Ok(())
}

pub(crate) fn generate_normal_scaled(
    turn: &mut RandomTurn,
    mu: f64,
    sigma: f64,
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state_mut(turn, label, len != 0, |state| {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            if matches!(&state.state, RandomState::Mt(_)) {
                out.push(mu + sigma * next_normal(&mut state.state));
                continue;
            }
            let (z0, z1) = next_normal_pair(&mut state.state);
            out.push(mu + sigma * z0);
            if out.len() < len {
                out.push(mu + sigma * z1);
            }
        }
        out
    })
}

pub(crate) fn generate_student_t(
    turn: &mut RandomTurn,
    nu: &[f64],
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state_mut(turn, label, len != 0, |state| {
        let mut out = Vec::with_capacity(len);
        let mut spare_normal = None;
        for idx in 0..len {
            let df = if nu.len() == 1 { nu[0] } else { nu[idx] };
            let z = if matches!(&state.state, RandomState::Mt(_)) {
                next_normal(&mut state.state)
            } else if let Some(value) = spare_normal.take() {
                value
            } else {
                let (z0, z1) = next_normal_pair(&mut state.state);
                spare_normal = Some(z1);
                z0
            };
            if df == f64::INFINITY {
                out.push(z);
            } else {
                let chi_square = sample_gamma_shape_scale(&mut state.state, df / 2.0, 2.0);
                out.push(z / (chi_square / df).sqrt());
            }
        }
        out
    })
}

pub(crate) fn generate_gamma_shape_scale(
    turn: &mut RandomTurn,
    shape: &[f64],
    scale: &[f64],
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state_mut(turn, label, len != 0, |state| {
        let mut out = Vec::with_capacity(len);
        for idx in 0..len {
            let a = if shape.len() == 1 {
                shape[0]
            } else {
                shape[idx]
            };
            let b = if scale.len() == 1 {
                scale[0]
            } else {
                scale[idx]
            };
            out.push(if a == 0.0 {
                0.0
            } else {
                sample_gamma_shape_scale(&mut state.state, a, b)
            });
        }
        out
    })
}

pub(crate) fn generate_binomial(
    turn: &mut RandomTurn,
    trials: &[f64],
    probability: &[f64],
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state_mut(turn, label, len != 0, |state| {
        let mut out = Vec::with_capacity(len);
        for idx in 0..len {
            let n = if trials.len() == 1 {
                trials[0]
            } else {
                trials[idx]
            };
            let p = if probability.len() == 1 {
                probability[0]
            } else {
                probability[idx]
            };
            out.push(sample_binomial(&mut state.state, n, p));
        }
        out
    })
}

pub(crate) fn generate_weibull(
    turn: &mut RandomTurn,
    scale: &[f64],
    shape: &[f64],
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state_mut(turn, label, len != 0, |state| {
        let mut out = Vec::with_capacity(len);
        for idx in 0..len {
            let a = if scale.len() == 1 {
                scale[0]
            } else {
                scale[idx]
            };
            let b = if shape.len() == 1 {
                shape[0]
            } else {
                shape[idx]
            };
            let u = next_uniform_state(&mut state.state).max(MIN_UNIFORM);
            out.push(a * (-u.ln()).powf(1.0 / b));
        }
        out
    })
}

fn sample_gamma_shape_scale(state: &mut RandomState, shape: f64, scale: f64) -> f64 {
    if shape < 1.0 {
        let u = next_uniform_state(state).max(MIN_UNIFORM);
        return sample_gamma_shape_scale(state, shape + 1.0, scale) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let x = next_normal(state);
        let v_base = 1.0 + c * x;
        if v_base <= 0.0 {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = next_uniform_state(state);
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 || u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return scale * d * v;
        }
    }
}

fn sample_binomial(state: &mut RandomState, trials: f64, probability: f64) -> f64 {
    if trials == 0.0 || probability == 0.0 {
        return 0.0;
    }
    if probability == 1.0 {
        return trials;
    }

    let effective_p = probability.min(1.0 - probability);
    let mean = trials * effective_p;
    let sampled = if trials <= 10_000.0 {
        sample_binomial_direct(state, trials as u64, effective_p)
    } else if mean < 30.0 {
        sample_poisson_knuth(state, mean).min(trials)
    } else {
        sample_binomial_normal_approx(state, trials, effective_p)
    };

    if probability <= 0.5 {
        sampled
    } else {
        trials - sampled
    }
}

fn sample_binomial_direct(state: &mut RandomState, trials: u64, probability: f64) -> f64 {
    let mut successes = 0_u64;
    for _ in 0..trials {
        if next_uniform_state(state) < probability {
            successes += 1;
        }
    }
    successes as f64
}

fn sample_poisson_knuth(state: &mut RandomState, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    let threshold = (-lambda).exp();
    let mut product = 1.0;
    let mut count = 0_u64;
    loop {
        count += 1;
        product *= next_uniform_state(state).max(MIN_UNIFORM);
        if product <= threshold {
            return (count - 1) as f64;
        }
    }
}

fn sample_binomial_normal_approx(state: &mut RandomState, trials: f64, probability: f64) -> f64 {
    let mean = trials * probability;
    let variance = mean * (1.0 - probability);
    if variance <= 0.0 {
        return mean.round().clamp(0.0, trials);
    }
    let z = next_normal(state);
    (mean + variance.sqrt() * z).round().clamp(0.0, trials)
}
