//! Bounded agglomerative merge engine over admitted condensed distances.

use std::cmp::Ordering;
use std::collections::HashMap;

use runmat_builtins::LinkageMethod;
use runmat_value::Tensor;

use super::{internal, invalid, validate_observation_count, EPS};
use crate::BuiltinResult;

pub(super) fn compute_linkage(
    observations: usize,
    condensed: Vec<f64>,
    method: LinkageMethod,
) -> BuiltinResult<Tensor> {
    validate_observation_count(observations)?;
    let expected = observations
        .checked_mul(observations.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| internal("linkage: output size overflow"))?;
    if condensed.len() != expected {
        return Err(invalid(
            "linkage: condensed distance vector has the wrong length",
        ));
    }

    let mut distances = HashMap::new();
    distances
        .try_reserve(expected.saturating_add(observations))
        .map_err(|_| internal("linkage: failed to allocate cluster distance workspace"))?;
    let mut offset = 0usize;
    for col in 0..observations {
        for row in (col + 1)..observations {
            distances.insert(cluster_key(row, col), condensed[offset]);
            offset += 1;
        }
    }

    let mut active = (0..observations).collect::<Vec<_>>();
    let mut sizes = vec![1usize; observations];
    let rows = observations - 1;
    let mut output = vec![0.0; rows * 3];

    for step in 0..rows {
        let (left, right, merge_distance) = find_next_merge(&active, &distances)?;
        let left_size = sizes[left];
        let right_size = sizes[right];
        output[step] = (left.min(right) + 1) as f64;
        output[rows + step] = (left.max(right) + 1) as f64;
        output[2 * rows + step] = merge_distance;

        let new_id = sizes.len();
        let new_size = left_size
            .checked_add(right_size)
            .ok_or_else(|| internal("linkage: cluster size overflow"))?;
        let mut new_distances = Vec::with_capacity(active.len().saturating_sub(2));
        for &other in &active {
            if other == left || other == right {
                continue;
            }
            let d_left = distance_between(&distances, left, other)?;
            let d_right = distance_between(&distances, right, other)?;
            let updated = update_distance(
                method,
                d_left,
                d_right,
                merge_distance,
                left_size,
                right_size,
                sizes[other],
            )?;
            new_distances.push((other, updated));
        }

        sizes.push(new_size);
        for &other in &active {
            if other != left {
                distances.remove(&cluster_key(left, other));
            }
            if other != right {
                distances.remove(&cluster_key(right, other));
            }
        }
        distances.remove(&cluster_key(left, right));
        active.retain(|id| *id != left && *id != right);
        for (other, distance) in new_distances {
            distances.insert(cluster_key(new_id, other), distance);
        }
        active.push(new_id);
        active.sort_unstable();
    }

    Tensor::new(output, vec![rows, 3]).map_err(|err| internal(format!("linkage: {err}")))
}

fn find_next_merge(
    active: &[usize],
    distances: &HashMap<(usize, usize), f64>,
) -> BuiltinResult<(usize, usize, f64)> {
    let mut best: Option<(usize, usize, f64)> = None;
    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            let left = active[i];
            let right = active[j];
            let distance = distance_between(distances, left, right)?;
            let candidate = (left.min(right), left.max(right), distance);
            if let Some(current) = best {
                if compare_merge(candidate, current) == Ordering::Less {
                    best = Some(candidate);
                }
            } else {
                best = Some(candidate);
            }
        }
    }
    best.ok_or_else(|| internal("linkage: no active cluster pair found"))
}

fn compare_merge(a: (usize, usize, f64), b: (usize, usize, f64)) -> Ordering {
    compare_distance(a.2, b.2)
        .then_with(|| a.0.cmp(&b.0))
        .then_with(|| a.1.cmp(&b.1))
}

fn compare_distance(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

fn update_distance(
    method: LinkageMethod,
    d_left: f64,
    d_right: f64,
    d_join: f64,
    left_size: usize,
    right_size: usize,
    other_size: usize,
) -> BuiltinResult<f64> {
    let nl = left_size as f64;
    let nr = right_size as f64;
    let nk = other_size as f64;
    match method {
        LinkageMethod::Single => {
            if d_left.is_nan() || d_right.is_nan() {
                Ok(f64::NAN)
            } else {
                Ok(d_left.min(d_right))
            }
        }
        LinkageMethod::Complete => {
            if d_left.is_nan() || d_right.is_nan() {
                Ok(f64::NAN)
            } else {
                Ok(d_left.max(d_right))
            }
        }
        LinkageMethod::Average => Ok((nl * d_left + nr * d_right) / (nl + nr)),
        LinkageMethod::Weighted => Ok(0.5 * (d_left + d_right)),
        LinkageMethod::Centroid => nonnegative_sqrt(
            (nl * d_left.powi(2) + nr * d_right.powi(2)) / (nl + nr)
                - (nl * nr * d_join.powi(2)) / (nl + nr).powi(2),
        ),
        LinkageMethod::Median => {
            nonnegative_sqrt(0.5 * d_left.powi(2) + 0.5 * d_right.powi(2) - 0.25 * d_join.powi(2))
        }
        LinkageMethod::Ward => nonnegative_sqrt(
            ((nl + nk) * d_left.powi(2) + (nr + nk) * d_right.powi(2) - nk * d_join.powi(2))
                / (nl + nr + nk),
        ),
    }
}

fn nonnegative_sqrt(value: f64) -> BuiltinResult<f64> {
    if value.is_nan() {
        Ok(f64::NAN)
    } else if value < 0.0 && value.abs() <= EPS {
        Ok(0.0)
    } else if value < 0.0 {
        Err(invalid(
            "linkage: Euclidean linkage update produced an invalid negative distance",
        ))
    } else {
        Ok(value.sqrt())
    }
}

fn distance_between(
    distances: &HashMap<(usize, usize), f64>,
    a: usize,
    b: usize,
) -> BuiltinResult<f64> {
    distances
        .get(&cluster_key(a, b))
        .copied()
        .ok_or_else(|| internal("linkage: missing inter-cluster distance"))
}

fn cluster_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}
