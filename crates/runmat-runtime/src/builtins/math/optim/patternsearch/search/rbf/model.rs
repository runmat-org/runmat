//! Callback-free proposal minimization of one authenticated cubic model.
//! This bounded local proposal search never certifies original optimality.
use super::super::super::{
    buffer, invalid, linear::Linear, mesh::Basis, scaling::Scaling, timing::Deadline,
};
use super::{bounds::Window, history::Training};
use crate::builtins::math::optim::common::{
    cubic_rbf::{self, Training as _},
    ScalarCooperation,
};
use crate::BuiltinResult;

pub(super) async fn propose(
    training: &Training<'_>,
    model: &cubic_rbf::Model,
    window: &Window,
    geometry: &Linear,
    tolerance: f64,
    deadline: &Deadline,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let n = training.dimensions();
    // This is a model-quality work bound, not a callback limit or a public
    // convergence test. No original evaluation occurs within this loop.
    let limit = n
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| invalid("RBF proposal work count overflow"))?;
    let mut best = None;
    for row in 0..training.rows().len() {
        if deadline.reached() {
            return Ok(None);
        }
        let mut point = buffer(n)?;
        for axis in 0..n {
            point.push(training.coordinate(row, axis));
            cooperation.completed(1).await?;
        }
        if window.contains(&point) && geometry.feasible(&point, tolerance, cooperation).await? {
            let score = training.measurement(row, 0);
            if best.as_ref().is_none_or(|(_, previous)| score < *previous) {
                best = Some((point, score));
            }
        }
    }
    let Some((mut point, mut value)) = best else {
        return Ok(None);
    };
    let mut radius = 0.0f64;
    for (&l, &u) in window.lower.iter().zip(&window.upper) {
        // Half-span is finite even when the full endpoint subtraction is not.
        radius = radius.max((0.5 * u - 0.5 * l).abs());
        cooperation.completed(1).await?;
    }
    if radius == 0.0 || !radius.is_finite() {
        return Ok(None);
    }
    let scale = Scaling::unit(n);
    let mut prediction = [0.0];
    let mut used = 0usize;
    let mut changed = false;
    while used < limit && radius > 0.0 && !deadline.reached() {
        let Some(cone) = geometry
            .cone(&point, radius, tolerance, &scale, cooperation)
            .await?
        else {
            return Ok(None);
        };
        let count = cone.len(Basis::Maximal)?;
        let mut improved = None;
        for index in 0..count {
            if used >= limit || deadline.reached() {
                break;
            }
            used += 1;
            let Some(mut direction) = cone
                .oriented_direction(index, Basis::Maximal, false, cooperation)
                .await?
            else {
                continue;
            };
            let maximum = direction.iter().fold(0.0f64, |s, &v| s.max(v.abs()));
            if maximum == 0.0 || !maximum.is_finite() {
                continue;
            }
            let mut norm = 0.0f64;
            for value in &mut direction {
                *value /= maximum;
                norm = norm.hypot(*value);
                cooperation.completed(1).await?;
            }
            if !norm.is_finite() || norm == 0.0 {
                continue;
            }
            for value in &mut direction {
                *value /= norm;
                cooperation.completed(1).await?;
            }
            let Some(length) = window
                .length(&point, &direction, radius, cooperation)
                .await?
            else {
                continue;
            };
            let Ok(candidate) = geometry
                .candidate_with_step(&point, &direction, length, tolerance, cooperation)
                .await?
            else {
                continue;
            };
            if !window.contains(&candidate.point) {
                continue;
            }
            if model
                .predict_into(training, &candidate.point, &mut prediction, cooperation)
                .await?
                == cubic_rbf::Prediction::Finite
                && prediction[0] < value
            {
                improved = Some((candidate.point, prediction[0]));
                break;
            }
        }
        if let Some((next, score)) = improved {
            point = next;
            value = score;
            changed = true;
        } else {
            let next = radius * 0.5;
            if next == radius {
                break;
            }
            radius = next;
            // Without a representable proposal no further numerical progress
            // is possible; used remains a model-direction count, not a fake call.
            if count == 0 {
                break;
            }
        }
    }
    Ok(changed.then_some(point))
}
