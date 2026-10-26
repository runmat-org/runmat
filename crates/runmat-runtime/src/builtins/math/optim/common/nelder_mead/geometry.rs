//! Conserved simplex geometry, ordering, affine steps and convergence.
use super::*;
pub(in crate::builtins::math::optim) fn geometry(policy: Policy, n: usize) -> BuiltinResult<usize> {
    let count = n
        .checked_add(1)
        .ok_or_else(|| policy.invalid("simplex geometry overflow"))?;
    n.checked_mul(count)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| policy.invalid("simplex backing geometry is not representable"))?;
    Ok(count)
}

pub(super) fn less(left: f64, right: f64) -> bool {
    if left.is_nan() {
        false
    } else if right.is_nan() {
        true
    } else {
        left < right
    }
}
pub(super) async fn sort(
    policy: Policy,
    order: &mut Vec<usize>,
    vertices: &[Point],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    use crate::builtins::common::cooperative_sort::{SortProgress, StableSortWork};
    let mut work = StableSortWork::new(order, |&left: &usize, &right: &usize| {
        let left = vertices[left].f;
        let right = vertices[right].f;
        if less(left, right) {
            std::cmp::Ordering::Less
        } else if less(right, left) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    loop {
        let progress = work
            .advance()
            .map_err(|source| policy.sort_failure(source.0))?;
        let (done, units) = match progress {
            SortProgress::Pending(units) => (false, units),
            SortProgress::Complete(units) => (true, units),
        };
        cooperation.completed(units).await?;
        if done {
            return Ok(());
        }
    }
}
pub(super) async fn converged(
    vertices: &[Point],
    order: &[usize],
    options: &Controls,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let best = &vertices[order[0]];
    if !best.f.is_finite() {
        return Ok(false);
    }
    for &slot in order.iter().skip(1) {
        let point = &vertices[slot];
        if !point.f.is_finite() || (point.f - best.f).abs() > options.tol_fun {
            return Ok(false);
        }
        for first in (0..best.x.len()).step_by(1024) {
            let end = first.saturating_add(1024).min(best.x.len());
            if (first..end).any(|index| (point.x[index] - best.x[index]).abs() > options.tol_x) {
                return Ok(false);
            }
            cooperation.completed((end - first) as u64).await?;
        }
    }
    Ok(true)
}
pub(super) async fn center(
    policy: Policy,
    vertices: &[Point],
    order: &[usize],
    center: &mut [f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let n = center.len();
    for (coordinate, out) in center.iter_mut().enumerate() {
        let mut scale = 0.0f64;
        for chunk in order[..n].chunks(1024) {
            for &slot in chunk {
                scale = scale.max(vertices[slot].x[coordinate].abs());
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        let mut sum = 0.0;
        let mut correction = 0.0;
        for chunk in order[..n].chunks(1024) {
            for &slot in chunk {
                let value = if scale == 0.0 {
                    0.0
                } else {
                    (vertices[slot].x[coordinate] / scale) / n as f64
                };
                let next = value - correction;
                let total = sum + next;
                correction = (total - sum) - next;
                sum = total;
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        *out = sum * scale;
        if !out.is_finite() {
            return Err(policy.invalid("simplex centroid is not finite"));
        }
    }
    Ok(())
}
pub(super) async fn affine(
    policy: Policy,
    left: &[f64],
    right: &[f64],
    weight: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut out = policy.buffer(left.len())?;
    for first in (0..left.len()).step_by(1024) {
        let end = first.saturating_add(1024).min(left.len());
        for index in first..end {
            let difference = right[index] - left[index];
            let value = if difference.is_finite() {
                left[index] + weight * difference
            } else {
                let scale = left[index].abs().max(right[index].abs());
                (left[index] / scale + weight * (right[index] / scale - left[index] / scale))
                    * scale
            };
            if !value.is_finite() {
                return Err(policy.invalid("simplex trial coordinate is not finite"));
            }
            out.push(value);
        }
        cooperation.completed((end - first) as u64).await?;
    }
    Ok(out)
}
pub(super) async fn copy(
    policy: Policy,
    values: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut copy = policy.buffer(values.len())?;
    for chunk in values.chunks(1024) {
        copy.extend_from_slice(chunk);
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(copy)
}
