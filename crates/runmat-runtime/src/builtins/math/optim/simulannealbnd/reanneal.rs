//! Accepted-point sensitivity refresh; every extra probe is a real evaluation.
use super::{
    evaluation::Evaluator,
    inputs::{self, Domain},
    invalid,
    options::Options,
    state::State,
    ScalarCooperation,
};
use crate::BuiltinResult;
pub(super) async fn refresh(
    state: &mut State,
    domain: &Domain,
    options: &Options,
    evaluator: &mut Evaluator,
    c: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let n = domain.variables();
    let mut sensitivities = Vec::with_capacity(n);
    for _ in 0..n {
        sensitivities.push(0.0);
        c.completed(1).await?;
    }
    if let Domain::Double { lower, upper, .. } = domain {
        let x = inputs::doubles(state.current.value(), "reannealing x")?;
        let mut values = Vec::with_capacity(n);
        for &value in x {
            values.push(value);
            c.completed(1).await?;
        }
        let mut probe = domain.value(values)?;
        let mut previous = None;
        for i in 0..n {
            c.completed(1).await?;
            if lower[i] == upper[i] {
                sensitivities[i] = f64::NEG_INFINITY;
                continue;
            }
            if evaluator.count >= options.evaluations {
                return Ok(false);
            }
            let nominal = f64::EPSILON.sqrt() * x[i].abs().max(1.0);
            let mut point = x[i] + nominal;
            if !point.is_finite() || point > upper[i] {
                point = x[i] - nominal;
            }
            point = point.max(lower[i]).min(upper[i]);
            if point == x[i] {
                point = if x[i] < upper[i] {
                    x[i].next_up()
                } else {
                    x[i].next_down()
                };
                point = point.max(lower[i]).min(upper[i]);
            }
            let offset = point - x[i];
            if !point.is_finite() || offset == 0.0 || !offset.is_finite() {
                return Err(invalid(
                    "reannealing has no usable actual coordinate offset",
                ));
            }
            set_probe(&mut probe, previous, i, point, x, c).await?;
            previous = Some(i);
            let value = evaluator.evaluate(&probe, c).await?;
            // Ratios of sensitivities are evaluated in logarithmic form.
            // Opposite-sign finite objectives/bounds can have an overflowing
            // raw difference even when their scaled sensitivity ratio is usable.
            let log_gradient = log_difference(value, state.current_f) - offset.abs().ln();
            let log_width = if lower[i].is_finite() && upper[i].is_finite() {
                log_difference(upper[i], lower[i])
            } else {
                0.0
            };
            sensitivities[i] = log_gradient + log_width;
        }
    }
    // Custom proposal coordinates have no public differentiable metric. Use
    // the topology-neutral equal-sensitivity safeguard; do not alter opaque
    // data or fabricate numerical coordinate probes. Reheating still occurs
    // at the actual accepted-point interval with the original temperature.
    let mut maximum = f64::NEG_INFINITY;
    for &s in &sensitivities {
        maximum = maximum.max(s);
        c.completed(1).await?;
    }
    for (i, k) in state.k.iter_mut().enumerate() {
        let s = sensitivities[i];
        let t0 = options.initial_temperature[i];
        let t = state.temperature[i];
        let log_ratio = if maximum.is_finite() && s.is_finite() {
            maximum - s
        } else {
            0.0
        };
        let refreshed = if t0 > 0.0 && t > 0.0 {
            (t0.ln() - t.ln() + log_ratio).max(1.0)
        } else {
            1.0
        };
        // Reannealing reheats: it never moves the parameter beyond its current
        // value. Guard unusable sensitivities without manufacturing a gradient.
        *k = if refreshed.is_finite() {
            refreshed.min(*k).max(1.0)
        } else {
            1.0
        };
        c.completed(1).await?;
    }
    state.accepted = 0;
    Ok(true)
}
fn log_difference(a: f64, b: f64) -> f64 {
    if a == b {
        return f64::NEG_INFINITY;
    }
    let difference = (a - b).abs();
    if difference.is_finite() {
        difference.ln()
    } else {
        let scale = a.abs().max(b.abs());
        scale.ln() + (a / scale - b / scale).abs().ln()
    }
}
async fn set_probe(
    probe: &mut runmat_value::Value,
    previous: Option<usize>,
    index: usize,
    point: f64,
    original: &[f64],
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let runmat_value::Value::Tensor(tensor) = probe else {
        unreachable!("double probe")
    };
    if tensor.try_as_f64_slice_mut_unique().is_none() {
        // Only a callback-retained predecessor forces a new physical buffer.
        // Ordinary coordinate probes reuse this one authoritative workspace.
        let mut values = Vec::with_capacity(original.len());
        for &value in tensor.as_f64_slice().expect("double probe") {
            values.push(value);
            c.completed(1).await?;
        }
        *tensor = runmat_value::Tensor::new(values, tensor.shape().to_vec()).map_err(invalid)?;
    }
    let values = tensor
        .try_as_f64_slice_mut_unique()
        .expect("unique probe after COW");
    if let Some(previous) = previous {
        values[previous] = original[previous];
    }
    values[index] = point;
    Ok(())
}
