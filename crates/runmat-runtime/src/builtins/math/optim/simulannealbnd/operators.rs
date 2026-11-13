//! Distinct proposal, acceptance and cooling mechanisms over original data.
use super::{
    evaluation,
    inputs::{self, Domain},
    invalid, observation,
    options::{Options, Source},
    random,
    state::State,
    ScalarCooperation,
};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_value::Value;
pub(super) async fn proposal(
    state: &State,
    domain: &Domain,
    problem: &RootedValue,
    options: &Options,
    started: f64,
    count: usize,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    match &options.annealing {
        Source::Predefined(kind) => {
            let Domain::Double { lower, upper, .. } = domain else {
                return Err(invalid("numeric proposal requires double domain"));
            };
            let previous = inputs::doubles(state.current.value(), "current x")?;
            let direction = random::direction(previous.len(), c).await?;
            let mut values = Vec::with_capacity(previous.len());
            for (i, &value) in previous.iter().enumerate() {
                let scale = match kind {
                    runmat_builtins::OptimAnnealing::Fast => state.temperature[i],
                    runmat_builtins::OptimAnnealing::Boltzmann => state.temperature[i].sqrt(),
                };
                values.push(value + scale * direction[i]);
                c.completed(1).await?;
            }
            let values = honor_bounds(values, previous, lower, upper, c).await?;
            // This candidate buffer is moved, not copied into another point.
            domain.validate_trial(domain.value(values)?, c).await
        }
        Source::Custom(target) => {
            let metadata = observation::metadata(state, count, started, c).await?;
            let returned = evaluation::invoke(
                target.value(),
                &[metadata.value().clone(), problem.value().clone()],
                1,
                c,
            )
            .await?;
            let proposed = {
                let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
                returned.try_into_single()?
            };
            let Domain::Double { lower, upper, .. } = domain else {
                return Ok(proposed);
            };
            let source = inputs::doubles(proposed.value(), "proposed x")?;
            if source.len() != lower.len() {
                return Err(invalid("proposal extent differs from x0"));
            }
            let mut outside = false;
            for (i, &value) in source.iter().enumerate() {
                if value.is_nan() {
                    return Err(invalid("proposal contains NaN"));
                }
                outside |= value < lower[i] || value > upper[i];
                c.completed(1).await?;
            }
            if !outside {
                return domain.validate_trial(proposed.value().clone(), c).await;
            }
            // Keep the returned root through repair and final publication.
            // Only a bound-changing custom result needs a replacement buffer.
            let mut values = Vec::with_capacity(source.len());
            for &value in source {
                values.push(value);
                c.completed(1).await?;
            }
            let previous = inputs::doubles(state.current.value(), "current x")?;
            let values = honor_bounds(values, previous, lower, upper, c).await?;
            domain.validate_trial(domain.value(values)?, c).await
        }
    }
}
pub(super) async fn honor_bounds(
    mut values: Vec<f64>,
    previous: &[f64],
    lower: &[f64],
    upper: &[f64],
    c: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    if values.len() != previous.len() || values.len() != lower.len() || values.len() != upper.len()
    {
        return Err(invalid("bound repair extent mismatch"));
    }
    // Draw only for actual violations, never before a custom proposal callback.
    for i in 0..values.len() {
        let v = values[i];
        if v.is_nan() {
            return Err(invalid("proposal contains NaN"));
        }
        let (bound, violated) = if v < lower[i] {
            (lower[i], true)
        } else if v > upper[i] {
            (upper[i], true)
        } else {
            (0.0, false)
        };
        let v = if violated {
            let u = random::uniform(1, c).await?[0];
            // A convex combination avoids overflowing (bound-previous).
            (1.0 - u) * previous[i] + u * bound
        } else {
            v
        };
        if !v.is_finite() {
            return Err(invalid(
                "proposal step is not representable in the unbounded coordinate",
            ));
        }
        values[i] = v.max(lower[i]).min(upper[i]);
        c.completed(1).await?;
    }
    Ok(values)
}
pub(super) fn probability(old: f64, new: f64, temperature: f64) -> f64 {
    if new < old {
        return 1.0;
    }
    if temperature <= 0.0 {
        return if new == old { 0.5 } else { 0.0 };
    }
    // Divide before subtracting if the finite objective difference overflows.
    let delta = new - old;
    let scaled = if delta.is_finite() {
        delta / temperature
    } else {
        new / temperature - old / temperature
    };
    1.0 / (1.0 + scaled.exp())
}
pub(super) async fn accept(
    state: &State,
    trial: &RootedValue,
    fitness: f64,
    options: &Options,
    started: f64,
    count: usize,
    c: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    match &options.acceptance {
        Source::Predefined(_) => {
            if fitness < state.current_f {
                return Ok(true);
            }
            let temperature = state.temperature.iter().copied().fold(0.0, f64::max);
            Ok(
                random::uniform(1, c).await?[0]
                    < probability(state.current_f, fitness, temperature),
            )
        }
        Source::Custom(target) => {
            let metadata = observation::metadata(state, count, started, c).await?;
            let returned = evaluation::invoke(
                target.value(),
                &[
                    metadata.value().clone(),
                    trial.value().clone(),
                    Value::Num(fitness),
                ],
                1,
                c,
            )
            .await?;
            let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
            observation::logical(returned.try_into_single()?.value(), "AcceptanceFcn")
        }
    }
}
pub(super) fn temperature(
    kind: runmat_builtins::OptimTemperature,
    t0: f64,
    k: f64,
) -> BuiltinResult<f64> {
    let value = match kind {
        runmat_builtins::OptimTemperature::Exponential => t0 * 0.95_f64.powf(k),
        runmat_builtins::OptimTemperature::Fast => t0 / k,
        runmat_builtins::OptimTemperature::Boltzmann => t0 / k.ln().max(1.0),
    };
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(invalid(
            "temperature schedule has a nonfinite or negative value",
        ))
    }
}
pub(super) async fn cool(
    state: &mut State,
    options: &Options,
    started: f64,
    count: usize,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    for k in &mut state.k {
        *k += 1.0;
        c.completed(1).await?;
    }
    let values = match &options.schedule {
        Source::Predefined(kind) => {
            let mut values = Vec::with_capacity(state.k.len());
            for (i, &k) in state.k.iter().enumerate() {
                values.push(temperature(*kind, options.initial_temperature[i], k)?);
                c.completed(1).await?;
            }
            values
        }
        Source::Custom(target) => {
            let metadata = observation::metadata(state, count, started, c).await?;
            let returned = evaluation::invoke(
                target.value(),
                &[metadata.value().clone(), options.public.value().clone()],
                1,
                c,
            )
            .await?;
            let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
            let value = returned.try_into_single()?;
            let values = inputs::doubles(value.value(), "TemperatureFcn result")?;
            if values.len() != state.k.len() {
                return Err(invalid("TemperatureFcn must return numel(x0) temperatures"));
            }
            values.to_vec()
        }
    };
    for &value in &values {
        if !value.is_finite() || value < 0.0 {
            return Err(invalid(
                "TemperatureFcn must return finite nonnegative temperatures",
            ));
        }
        c.completed(1).await?;
    }
    state.temperature = values;
    Ok(())
}
