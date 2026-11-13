//! Robust normal-reference bandwidth, without an arbitrary constant scale.
use super::*;
use crate::builtins::common::cooperative_sort::{SortProgress, StableSortWork};
pub(super) async fn normal_reference(
    data: &mut Vec<f64>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<f64> {
    sort(data, cooperation).await?;
    let center = median(data);
    let mut deviations = buffer(data.len(), runtime)?;
    for chunk in data.chunks(1024) {
        for &value in chunk {
            let difference = value - center;
            let logarithm = if difference.is_finite() {
                difference.abs().ln()
            } else {
                let scale = value.abs().max(center.abs());
                scale.ln() + (value / scale - center / scale).abs().ln()
            };
            deviations.push(logarithm);
        }
        cooperation.completed(chunk.len()).await?;
    }
    sort(&mut deviations, cooperation).await?;
    let log_mad = if deviations.len() % 2 == 1 {
        deviations[deviations.len() / 2]
    } else {
        let left = deviations[deviations.len() / 2 - 1];
        let right = deviations[deviations.len() / 2];
        if left == f64::NEG_INFINITY {
            right - 2.0f64.ln()
        } else {
            right + ((left - right).exp()).ln_1p() - 2.0f64.ln()
        }
    };
    let log_reference = (4.0f64.ln() - 3.0f64.ln() - (data.len() as f64).ln()) / 5.0;
    let log_width = if log_mad.is_finite() {
        log_mad - super::super::distribution_math::standard_normal_inv(0.75).ln() + log_reference
    } else if data.len() >= 2 {
        // MAD may vanish for tied but nonconstant observations. Use their
        // actual sample spread, never a fabricated unit-width distribution.
        let moments = inferential::moments(data, cooperation).await?;
        moments.scale.ln()
            + 0.5 * (moments.normalized_squares.ln() - ((data.len() - 1) as f64).ln())
            + log_reference
    } else {
        f64::NEG_INFINITY
    };
    let width = log_width.exp();
    if width.is_finite() && width > 0.0 {
        Ok(width)
    } else {
        Err(invalid(NAME,"automatic bandwidth has no finite positive identifiable scale; specify a finite positive Bandwidth (required for constant or singleton samples)"))
    }
}
fn median(values: &[f64]) -> f64 {
    let upper = values[values.len() / 2];
    if values.len() % 2 == 1 {
        upper
    } else {
        let lower = values[values.len() / 2 - 1];
        if lower.abs() <= f64::MAX / 2.0 && upper.abs() <= f64::MAX / 2.0 {
            (lower + upper) * 0.5
        } else {
            lower * 0.5 + upper * 0.5
        }
    }
}
async fn sort(data: &mut Vec<f64>, cooperation: &mut Cooperation<'_>) -> BuiltinResult<()> {
    let mut work = StableSortWork::new(data, |a: &f64, b: &f64| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    loop {
        let progress = work.advance().map_err(|source| {
            crate::build_runtime_error("ksdensity: sort allocation failed")
                .with_builtin(NAME)
                .with_identifier("RunMat:ksdensity:AllocationFailed")
                .with_source(source.0)
                .build()
        })?;
        let units = match progress {
            SortProgress::Pending(units) | SortProgress::Complete(units) => units,
        };
        cooperation.completed(units as usize).await?;
        if matches!(progress, SortProgress::Complete(_)) {
            return Ok(());
        }
    }
}
