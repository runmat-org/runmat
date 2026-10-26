//! Piecewise reflected rays with analytical folding, not one clipped bounce.
//! Backtracking uses actual original residuals at each represented trial.
use super::*;
pub(super) struct Trial {
    pub point: Point,
    pub ratio: f64,
    pub boundary: bool,
}
pub(super) use trust_quadratic::bounds::folded;
pub(super) async fn search(
    model: &Model,
    point: &Point,
    direction: &[f64],
    metric: &trust_quadratic::bounds::Metric,
    radius: f64,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Trial>> {
    let mut descent = 0f64;
    for (&g, &p) in point.gradient.iter().zip(direction) {
        descent = g.mul_add(-p, descent);
        c.completed(1).await?;
    }
    if !(descent > 0. && descent.is_finite()) {
        return Ok(None);
    }
    let mut alpha = 1f64;
    while alpha > 0. {
        let mut x = runmat_value::exact_capacity_vec(point.x.len()).map_err(allocation)?;
        let mut length = 0f64;
        let mut changed = false;
        let mut valid = true;
        for (i, (&old, &p)) in point.x.iter().zip(direction).enumerate() {
            let Some(mut value) = folded(old, alpha * p, model.lower[i], model.upper[i]) else {
                valid = false;
                break;
            };
            if value <= model.lower[i] {
                value = model.lower[i].next_up();
            }
            if value >= model.upper[i] {
                value = model.upper[i].next_down();
            }
            if !value.is_finite() || value <= model.lower[i] || value >= model.upper[i] {
                valid = false;
                break;
            }
            changed |= value != old;
            length = length.hypot((value - old) / metric.scale[i]);
            x.push(value);
            c.completed(1).await?;
        }
        if valid && !changed {
            return Ok(None);
        }
        if valid
            && length.is_finite()
            && length <= radius * (1. + 128. * f64::EPSILON * point.x.len().max(1) as f64)
        {
            if let Some(trial) = model.evaluate(x, c).await? {
                let actual = 0.5 * model.units * (point.value - trial.value);
                let mut predicted = 0f64;
                for ((&g, &next), (&x, &old)) in point
                    .gradient
                    .iter()
                    .zip(&trial.gradient)
                    .zip(trial.x.iter().zip(&point.x))
                {
                    predicted = (-(0.5 * g + 0.5 * next)).mul_add(x - old, predicted);
                    c.completed(1).await?;
                }
                if actual > 0.
                    && actual.is_finite()
                    && predicted > 0.
                    && predicted.is_finite()
                    && actual >= 1e-4 * alpha * descent
                {
                    let ratio = actual / predicted;
                    if ratio.is_finite() {
                        return Ok(Some(Trial {
                            point: trial,
                            ratio,
                            boundary: length / radius >= 0.9,
                        }));
                    }
                }
            }
        }
        alpha *= 0.5;
        c.completed(1).await?;
    }
    Ok(None)
}
