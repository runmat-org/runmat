//! Cooperative stable selection over already admitted index storage.
use super::{super::Sigma, work::Work};
use crate::BuiltinResult;
use num_complex::Complex64;
fn magnitude(value: Complex64) -> f64 {
    let scale = value.re.abs().max(value.im.abs());
    if scale == 0.0 {
        return f64::NEG_INFINITY;
    }
    if !scale.is_finite() {
        return f64::INFINITY;
    }
    scale.ln() + (value / scale).norm().ln()
}
fn distance(value: Complex64, target: Complex64) -> f64 {
    let scale = value
        .re
        .abs()
        .max(value.im.abs())
        .max(target.re.abs())
        .max(target.im.abs());
    if scale == 0.0 {
        return f64::NEG_INFINITY;
    }
    if !scale.is_finite() {
        return f64::INFINITY;
    }
    scale.ln() + ((value / scale) - (target / scale)).norm().ln()
}

pub(super) async fn select(
    values: &[Complex64],
    indices: &mut [usize],
    selected: &mut [usize],
    sigma: Sigma,
    work: &mut Work<'_>,
) -> BuiltinResult<()> {
    for (i, slot) in indices.iter_mut().enumerate() {
        *slot = i;
        work.tick().await?;
    }
    let metric = |value: Complex64| match sigma {
        Sigma::LargestAbs => -magnitude(value),
        Sigma::SmallestAbs => magnitude(value),
        Sigma::LargestReal => -value.re,
        Sigma::SmallestReal | Sigma::BothEndsReal => value.re,
        Sigma::LargestImag => -value.im,
        Sigma::SmallestImag | Sigma::BothEndsImag => value.im,
        Sigma::Near(target) => distance(value, target),
    };
    // No sort-owned merge scratch or noncooperative comparison loop.
    for i in 1..indices.len() {
        let index = indices[i];
        let mut j = i;
        while j > 0 {
            work.tick().await?;
            let previous = indices[j - 1];
            if metric(values[previous])
                .total_cmp(&metric(values[index]))
                .then_with(|| previous.cmp(&index))
                .is_le()
            {
                break;
            }
            indices[j] = previous;
            j -= 1;
        }
        indices[j] = index;
    }
    if matches!(sigma, Sigma::BothEndsReal | Sigma::BothEndsImag) {
        let low = selected.len() / 2;
        let high = selected.len() - low;
        for (destination, &source) in selected[..low].iter_mut().zip(&indices[..low]) {
            *destination = source;
            work.tick().await?;
        }
        for (destination, &source) in selected[low..]
            .iter_mut()
            .zip(&indices[indices.len() - high..])
        {
            *destination = source;
            work.tick().await?;
        }
    } else {
        for (destination, &source) in selected.iter_mut().zip(indices.iter()) {
            *destination = source;
            work.tick().await?;
        }
    }
    if matches!(sigma, Sigma::BothEndsImag) {
        for index in 1..selected.len() {
            let value = selected[index];
            let mut cursor = index;
            while cursor > 0 {
                work.tick().await?;
                let previous = selected[cursor - 1];
                let order = values[value]
                    .im
                    .abs()
                    .total_cmp(&values[previous].im.abs())
                    .then_with(|| values[value].im.total_cmp(&values[previous].im));
                if !order.is_gt() {
                    break;
                }
                selected[cursor] = previous;
                cursor -= 1;
            }
            selected[cursor] = value;
        }
    }
    work.flush().await
}
