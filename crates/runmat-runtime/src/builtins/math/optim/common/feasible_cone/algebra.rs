//! Sparse exact vector operations shared by cone pivots and facet cuts.
use super::Policy;
use super::Ray;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
pub(super) async fn normalize(
    ray: &mut [(usize, BigRational)],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let mut scale = BigRational::zero();
    for (_, value) in ray.iter() {
        scale = scale.max(value.abs());
        cooperation.completed(1).await?;
    }
    if scale.is_zero() {
        return Ok(false);
    }
    if !scale.is_one() {
        for (_, value) in ray {
            *value /= &scale;
            cooperation.completed(1).await?;
        }
    }
    Ok(true)
}

pub(super) async fn product(
    left: &[(usize, BigRational)],
    right: &[(usize, BigRational)],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<BigRational> {
    if left.is_empty() || right.is_empty() {
        return Ok(BigRational::zero());
    }
    let (short, long) = if left.len() < right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let comparisons = (long.len().ilog2() + 1) as usize;
    let binary_work = short.len().saturating_mul(comparisons);
    let merge_work = left.len().saturating_add(right.len());
    if binary_work < merge_work {
        let mut result = BigRational::zero();
        for (column, value) in short {
            if let Ok(index) = long.binary_search_by_key(column, |entry| entry.0) {
                result += value * &long[index].1;
            }
            cooperation.completed(comparisons as u64).await?;
        }
        return Ok(result);
    }
    let (mut i, mut j) = (0, 0);
    let mut result = BigRational::zero();
    while i < left.len() && j < right.len() {
        match left[i].0.cmp(&right[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result += &left[i].1 * &right[j].1;
                i += 1;
                j += 1;
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(result)
}

/// left_scale*left - right_scale*right, retaining sorted sparse coordinates.
pub(super) async fn combine(
    left: &Ray,
    left_scale: &BigRational,
    right: &Ray,
    right_scale: &BigRational,
    cooperation: &mut ScalarCooperation,
    policy: Policy,
) -> BuiltinResult<Ray> {
    let capacity = left
        .len()
        .checked_add(right.len())
        .ok_or_else(|| policy.invalid("direction extent overflow"))?;
    let mut result = policy.buffer(capacity)?;
    let (mut i, mut j) = (0, 0);
    while i < left.len() || j < right.len() {
        let column = match (left.get(i), right.get(j)) {
            (Some(left), Some(right)) => left.0.min(right.0),
            (Some(left), None) => left.0,
            (None, Some(right)) => right.0,
            (None, None) => unreachable!("at least one retained coordinate"),
        };
        let mut value = BigRational::zero();
        if left.get(i).is_some_and(|entry| entry.0 == column) {
            value += left_scale * &left[i].1;
            i += 1;
        }
        if right.get(j).is_some_and(|entry| entry.0 == column) {
            value -= right_scale * &right[j].1;
            j += 1;
        }
        if !value.is_zero() {
            result.push((column, value));
        }
        cooperation.completed(1).await?;
    }
    Ok(result)
}
