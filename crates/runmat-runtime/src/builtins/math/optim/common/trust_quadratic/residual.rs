//! Trust-ball secular selection through a borrowed residual factorization.
use super::*;
mod stationarity;
pub(crate) use stationarity::StationarityScale;
pub(crate) trait ShiftedResidual {
    async fn solve(
        &mut self,
        shift: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>>;
}
pub(crate) async fn step(
    augmented: &mut impl ShiftedResidual,
    gradient: &[f64],
    radius: f64,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let Some(initial) = augmented.solve(0., c).await? else {
        return Ok(None);
    };
    let (displacement, multiplier) = if length(&initial, c).await? <= radius {
        (initial, 0.)
    } else {
        let mut lower = 0f64;
        // H is positive semidefinite, so ||g||/radius is an upper shift bracket.
        let mut upper = (length(gradient, c).await? / radius).max(f64::from_bits(1));
        if !upper.is_finite() {
            return Ok(None);
        }
        let Some(mut candidate) = augmented.solve(upper, c).await? else {
            return Ok(None);
        };
        if length(&candidate, c).await? > radius {
            return Ok(None);
        }
        loop {
            let shift = 0.5 * lower + 0.5 * upper;
            if shift == lower || shift == upper {
                break;
            }
            let Some(trial) = augmented.solve(shift, c).await? else {
                return Ok(None);
            };
            let trial_length = length(&trial, c).await?;
            if trial_length > radius {
                lower = shift;
            } else {
                upper = shift;
                candidate = trial;
                if (1. - trial_length / radius).abs() <= tolerance(gradient.len()) {
                    break;
                }
            }
            c.completed(1).await?;
        }
        (candidate, upper)
    };
    if !qualify(
        length(&displacement, c).await?,
        multiplier,
        radius,
        gradient.len(),
    ) {
        return Ok(None);
    }
    Ok(Some(displacement))
}
pub(crate) fn tolerance(n: usize) -> f64 {
    128. * f64::EPSILON * n.max(1) as f64
}
fn qualify(length: f64, multiplier: f64, radius: f64, n: usize) -> bool {
    let tolerance = tolerance(n);
    length.is_finite()
        && length <= radius * (1. + tolerance)
        && (multiplier == 0. || (length / radius - 1.).abs() <= tolerance)
}

async fn length(values: &[f64], c: &mut ScalarCooperation) -> BuiltinResult<f64> {
    let mut length = 0f64;
    for chunk in values.chunks(1024) {
        for &value in chunk {
            length = length.hypot(value);
        }
        c.completed(chunk.len() as u64).await?;
    }
    Ok(length)
}
