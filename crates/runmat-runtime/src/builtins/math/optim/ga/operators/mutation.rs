//! Distinct Gaussian and uniform mutations; feasibility is a separate owner.
use super::super::{invalid, ScalarCooperation, POLICY};
use crate::BuiltinResult;

/// RunMat's numerical power-mutation policy is S=U^2. It is explicit
/// algorithm policy, not a claim about proprietary MATLAB RNG trajectories.
const POWER_EXPONENT: i32 = 2;

pub(in super::super) async fn bit_flip(
    parent: &[f64],
    rate: f64,
    draws: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    if parent.len() != draws.len() || !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
        return Err(invalid("invalid bit mutation extent or rate"));
    }
    let mut child = POLICY.capacity(parent.len(), cooperation.runtime())?;
    for (&value, &draw) in parent.iter().zip(draws) {
        if value != 0. && value != 1. || !draw.is_finite() || !(0.0..1.0).contains(&draw) {
            return Err(invalid(
                "bit mutation requires exact zero/one parents and uniform variates",
            ));
        }
        child.push(if draw < rate { 1. - value } else { value });
        cooperation.completed(1).await?;
    }
    Ok(child)
}

pub(in super::super) async fn power(
    parent: &[f64],
    intervals: &[(f64, f64)],
    draws: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let n = parent.len();
    if intervals.len() != n
        || n.checked_mul(2) != Some(draws.len())
        || draws
            .iter()
            .any(|value| !value.is_finite() || !(0.0..1.0).contains(value))
    {
        return Err(invalid("invalid power mutation random or interval extent"));
    }
    let mut values = POLICY.capacity(n, cooperation.runtime())?;
    for column in 0..n {
        let value = parent[column];
        let (lower, upper) = intervals[column];
        if !value.is_finite()
            || !lower.is_finite()
            || !upper.is_finite()
            || value < lower
            || value > upper
        {
            return Err(invalid(
                "power mutation requires a finite interval containing its parent",
            ));
        }
        let next = if lower == upper {
            lower
        } else {
            let width = upper - lower;
            let position = if width.is_finite() {
                (value - lower) / width
            } else {
                let scale = lower.abs().max(upper.abs()).max(value.abs());
                (value / scale - lower / scale) / (upper / scale - lower / scale)
            };
            let target = if position < draws[column] {
                lower
            } else {
                upper
            };
            let fraction = draws[n + column].powi(POWER_EXPONENT);
            super::crossover::interpolate(value, target, fraction)
                .ok_or_else(|| invalid("power mutation result cannot be represented"))?
        };
        values.push(next);
        cooperation.completed(1).await?;
    }
    Ok(values)
}

pub(in super::super) struct Gaussian {
    sigma: Vec<f64>,
    shrink: f64,
}
impl Gaussian {
    pub async fn new(
        ranges: &[(f64, f64)],
        scale: f64,
        shrink: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if !scale.is_finite() || scale < 0.0 || !shrink.is_finite() {
            return Err(invalid(
                "Gaussian scale must be finite nonnegative and shrink must be finite",
            ));
        }
        let mut sigma = POLICY.capacity(ranges.len(), cooperation.runtime())?;
        for &(lower, upper) in ranges {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(invalid(
                    "Gaussian initial ranges must be finite ordered intervals",
                ));
            }
            let width = upper - lower;
            let deviation = if scale == 0.0 {
                0.0
            } else if width.is_finite() {
                scale * width
            } else {
                scale * upper - scale * lower
            };
            if !deviation.is_finite() || deviation < 0.0 {
                return Err(invalid(
                    "Gaussian initial standard deviation cannot be represented",
                ));
            }
            sigma.push(deviation);
            cooperation.completed(1).await?;
        }
        Ok(Self { sigma, shrink })
    }

    /// The specified recursive shrink schedule, not a replacement linear
    /// schedule applied afresh to the original range at every generation.
    pub async fn advance(
        &mut self,
        generation: usize,
        generations: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if generations == 0 || generation > generations {
            return Err(invalid("invalid Gaussian generation extent"));
        }
        let factor = 1.0 - self.shrink * (generation as f64 / generations as f64);
        if !factor.is_finite() || self.sigma.iter().any(|value| !(value * factor).is_finite()) {
            return Ok(false);
        }
        for value in &mut self.sigma {
            // A negative Gaussian multiplier has the same distribution as its
            // magnitude. This also handles finite user shrink values above 1.
            *value = (*value * factor).abs();
            cooperation.completed(1).await?;
        }
        Ok(true)
    }

    pub async fn child(
        &self,
        parent: &[f64],
        standard_normal: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        if parent.len() != self.sigma.len()
            || standard_normal.len() != parent.len()
            || parent
                .iter()
                .chain(standard_normal)
                .any(|value| !value.is_finite())
        {
            return Err(invalid("invalid Gaussian parent or normal variate extent"));
        }
        let mut child = POLICY.capacity(parent.len(), cooperation.runtime())?;
        for ((&value, &sigma), &normal) in parent.iter().zip(&self.sigma).zip(standard_normal) {
            let next = value + sigma * normal;
            if !next.is_finite() {
                return Ok(None);
            }
            child.push(next);
            cooperation.completed(1).await?;
        }
        Ok(Some(child))
    }
}

pub(in super::super) async fn uniform(
    parent: &[f64],
    ranges: &[(f64, f64)],
    rate: f64,
    draws: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let n = parent.len();
    if !rate.is_finite()
        || !(0.0..=1.0).contains(&rate)
        || ranges.len() != n
        || n.checked_mul(2) != Some(draws.len())
        || draws
            .iter()
            .any(|value| !value.is_finite() || !(0.0..1.0).contains(value))
    {
        return Err(invalid("invalid uniform mutation rate or random extent"));
    }
    let mut child = POLICY.capacity(n, cooperation.runtime())?;
    for column in 0..n {
        let (lower, upper) = ranges[column];
        if !lower.is_finite() || !upper.is_finite() || lower > upper || !parent[column].is_finite()
        {
            return Err(invalid(
                "uniform mutation requires finite parents and ordered ranges",
            ));
        }
        let value = if draws[column] < rate {
            let fraction = draws[n + column];
            if (upper - lower).is_finite() {
                lower + fraction * (upper - lower)
            } else {
                (1.0 - fraction) * lower + fraction * upper
            }
        } else {
            parent[column]
        };
        if !value.is_finite() {
            return Err(invalid("uniform mutation result cannot be represented"));
        }
        child.push(value);
        cooperation.completed(1).await?;
    }
    Ok(child)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gaussian_uses_recursive_schedule_and_uniform_changes_only_selected_genes() {
        futures::executor::block_on(async {
            let mut cooperation =
                ScalarCooperation::for_builtin(super::super::super::IDENTITY, None);
            assert_eq!(
                bit_flip(&[0., 1., 0.], 0.5, &[0.1, 0.9, 0.3], &mut cooperation)
                    .await
                    .unwrap(),
                [1., 1., 1.]
            );
            let mut gaussian = Gaussian::new(&[(0.0, 10.0)], 1.0, 1.0, &mut cooperation)
                .await
                .unwrap();
            assert!(gaussian.advance(1, 4, &mut cooperation).await.unwrap());
            assert!(gaussian.advance(2, 4, &mut cooperation).await.unwrap());
            assert_eq!(
                gaussian
                    .child(&[2.0], &[1.0], &mut cooperation)
                    .await
                    .unwrap()
                    .unwrap(),
                [5.75]
            );
            let mutated = uniform(
                &[2.0, 3.0],
                &[(0.0, 10.0), (0.0, 20.0)],
                0.5,
                &[0.1, 0.9, 0.7, 0.4],
                &mut cooperation,
            )
            .await
            .unwrap();
            assert_eq!(mutated, [7.0, 3.0]);
            let power = power(
                &[5.0, 4.0],
                &[(0.0, 10.0), (4.0, 4.0)],
                &[0.9, 0.2, 0.5, 0.7],
                &mut cooperation,
            )
            .await
            .unwrap();
            assert_eq!(power, [3.75, 4.0]);
        });
    }
}
