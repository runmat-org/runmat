//! Numerical chart reductions do not authorize solver convergence or fitness.
use super::super::{invalid, ScalarCooperation, POLICY};
use crate::BuiltinResult;
#[cfg(test)]
#[path = "plot_metrics/tests.rs"]
mod tests;

pub(super) struct Summary {
    pub minimum: f64,
    pub maximum: f64,
    pub mean: f64,
    pub best: Option<usize>,
    pub count: usize,
}
pub(super) async fn summarize(
    values: &[Option<f64>],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Summary> {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut best = None;
    let mut scale = 0_f64;
    let mut count = 0usize;
    for (index, value) in values.iter().enumerate() {
        if let Some(value) = value {
            if *value < minimum {
                minimum = *value;
                best = Some(index);
            }
            maximum = maximum.max(*value);
            scale = scale.max(value.abs());
            count += 1;
        }
        cooperation.completed(1).await?;
    }
    let mut sum = 0.;
    if scale != 0. {
        for value in values.iter().flatten() {
            sum += *value / scale;
            cooperation.completed(1).await?;
        }
    }
    Ok(Summary {
        minimum: if count == 0 { f64::NAN } else { minimum },
        maximum: if count == 0 { f64::NAN } else { maximum },
        mean: if count == 0 {
            f64::NAN
        } else {
            (sum / count as f64) * scale
        },
        best,
        count,
    })
}
pub(super) async fn distance(
    shape: &[usize],
    values: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<f64> {
    if shape.len() != 2 || shape[0].checked_mul(shape[1]) != Some(values.len()) {
        return Err(invalid("distance plot requires a population matrix"));
    }
    let rows = shape[0];
    let mut scale = 0_f64;
    for value in values {
        if !value.is_finite() {
            return Ok(f64::NAN);
        }
        scale = scale.max(value.abs());
        cooperation.completed(1).await?;
    }
    if rows < 2 || scale == 0. {
        return Ok(0.);
    }
    let pairs = (rows as u128)
        .checked_mul((rows - 1) as u128)
        .map(|n| n / 2)
        .ok_or_else(|| invalid("population distance pair count overflow"))?;
    let mut sum = 0.;
    for first in 0..rows {
        for second in first + 1..rows {
            let mut norm = 0_f64;
            for column in 0..shape[1] {
                norm = norm.hypot(
                    values[first + rows * column] / scale - values[second + rows * column] / scale,
                );
                cooperation.completed(1).await?;
            }
            // Divide before accumulating; the plotted average does not overflow
            // merely because a sum of otherwise representable distances does.
            sum += norm / pairs as f64;
        }
    }
    Ok(sum * scale)
}
pub(super) async fn histogram(
    values: &[Option<f64>],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(Vec<String>, Vec<f64>)> {
    let summary = summarize(values, cooperation).await?;
    let count = summary.count;
    if count == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    // RunMat's histogram policy: ceil(sqrt(valid rows)) equal-width bins,
    // with one bin for a constant sample and an inclusive right endpoint.
    let bins = if summary.minimum == summary.maximum {
        1
    } else {
        (count as f64).sqrt().ceil() as usize
    };
    let mut frequencies = POLICY.capacity(bins, cooperation.runtime())?;
    frequencies.resize(bins, 0.);
    let mut labels = POLICY.capacity(bins, cooperation.runtime())?;
    let scale = summary.minimum.abs().max(summary.maximum.abs());
    let low = if scale == 0. {
        0.
    } else {
        summary.minimum / scale
    };
    let high = if scale == 0. {
        0.
    } else {
        summary.maximum / scale
    };
    for value in values.iter().flatten() {
        let bin = if bins == 1 {
            0
        } else {
            ((((value / scale - low) / (high - low)) * bins as f64) as usize).min(bins - 1)
        };
        frequencies[bin] += 1.;
        cooperation.completed(1).await?;
    }
    for bin in 0..bins {
        let left = super::super::common::affine_coordinate::interpolate(
            summary.minimum,
            summary.maximum,
            bin as f64 / bins as f64,
        )
        .ok_or_else(|| invalid("histogram interval is not representable"))?;
        let right = super::super::common::affine_coordinate::interpolate(
            summary.minimum,
            summary.maximum,
            (bin + 1) as f64 / bins as f64,
        )
        .ok_or_else(|| invalid("histogram interval is not representable"))?;
        labels.push(format!(
            "[{left:.5}, {right:.5}{}",
            if bin + 1 == bins { "]" } else { ")" }
        ));
        cooperation.completed(1).await?;
    }
    Ok((labels, frequencies))
}
