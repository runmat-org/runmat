//! Shared Coleman--Li distance metric on borrowed original bounds.
use super::*;
#[derive(Clone, Copy)]
pub(crate) struct Bounds<'a> {
    pub lower: &'a [f64],
    pub upper: &'a [f64],
}
pub(crate) struct Metric {
    pub scale: Vec<f64>,
    pub diagonal: Vec<f64>,
    pub gradient: Vec<f64>,
    policy: Policy,
}
impl Metric {
    pub(crate) async fn new(
        x: &[f64],
        gradient: &[f64],
        bounds: Bounds<'_>,
        typical: impl Fn(usize) -> f64,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        if gradient.len() != x.len()
            || bounds.lower.len() != x.len()
            || bounds.upper.len() != x.len()
        {
            return Err(policy.invalid("reflective distance metric dimensions disagree"));
        }
        let mut scale = policy.buffer(x.len())?;
        let mut diagonal = policy.buffer(x.len())?;
        let mut scaled = policy.buffer(x.len())?;
        for (i, (&x, &g)) in x.iter().zip(gradient).enumerate() {
            let upper = g < 0. && bounds.upper[i].is_finite();
            let lower = g >= 0. && bounds.lower[i].is_finite();
            let d = if upper || lower {
                let distance = if upper {
                    bounds.upper[i] - x
                } else {
                    x - bounds.lower[i]
                };
                if !(distance > 0. && distance.is_finite()) {
                    return Ok(None);
                }
                distance.sqrt()
            } else {
                let value = typical(i);
                if !(value > 0.0 && value.is_finite()) {
                    return Ok(None);
                }
                value.max(1.)
            };
            let correction = if upper || lower { g.abs() } else { 0. };
            let value = d * g;
            if !(d > 0. && d.is_finite() && value.is_finite()) {
                return Ok(None);
            }
            scale.push(d);
            diagonal.push(correction);
            scaled.push(value);
            c.completed(1).await?;
        }
        Ok(Some(Self {
            scale,
            diagonal,
            gradient: scaled,
            policy,
        }))
    }
    pub(crate) async fn physical(
        &self,
        z: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        if z.len() != self.scale.len() {
            return Err(self
                .policy
                .invalid("reflective physical step dimensions disagree"));
        }
        let mut values = self.policy.buffer(z.len())?;
        for (&z, &scale) in z.iter().zip(&self.scale) {
            let value = z * scale;
            if !value.is_finite() {
                return Ok(None);
            }
            values.push(value);
            c.completed(1).await?;
        }
        Ok(Some(values))
    }
}

pub(crate) fn folded(x: f64, step: f64, lower: f64, upper: f64) -> Option<f64> {
    if !step.is_finite() {
        return None;
    }
    let (edge, other, distance, magnitude, positive) = if step >= 0. {
        (upper, lower, upper - x, step, true)
    } else {
        (lower, upper, x - lower, -step, false)
    };
    if !edge.is_finite() || magnitude <= distance {
        let value = x + step;
        return value.is_finite().then_some(value);
    }
    if !(distance >= 0. && distance.is_finite()) {
        return None;
    }
    let remaining = magnitude - distance;
    let width = if positive { edge - other } else { other - edge };
    let period = 2. * width;
    let offset = if other.is_finite() && period.is_finite() {
        if period <= 0. {
            return None;
        }
        remaining.rem_euclid(period)
    } else {
        remaining
    };
    let value = if offset <= width {
        if positive {
            edge - offset
        } else {
            edge + offset
        }
    } else if positive {
        other + (offset - width)
    } else {
        other - (offset - width)
    };
    value.is_finite().then_some(value)
}
