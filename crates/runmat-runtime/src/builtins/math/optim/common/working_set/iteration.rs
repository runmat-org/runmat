//! Feasible primal working-set iterations with deterministic blocking/release.
use super::{rows, workspace, Change, EqualityTarget, Limits, Outcome, Rows, Step, Stop, Target};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use runmat_builtins::BuiltinCatalogIdentity;

pub(crate) async fn minimize<T: Target>(
    objective: &mut T,
    model: Rows<'_>,
    mut x: Vec<f64>,
    limits: Limits,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    let tolerance = limits.feasibility;
    let Some(activity) = rows::product(model.a, &x, owner, cooperation).await? else {
        return Ok(Outcome::Numerical { point: x });
    };
    let mut active = match objective
        .initial_active(model, &x, &activity, cooperation)
        .await?
    {
        Some(active) if active.len() == model.b.len() => active,
        Some(_) => {
            return Err(crate::build_runtime_error(
                "warm working-set geometry disagrees with current rows",
            )
            .with_builtin(owner.name)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build())
        }
        None => {
            let mut active = workspace::capacity(model.b.len(), owner)?;
            for (row, (&value, &rhs)) in activity.iter().zip(model.b).enumerate() {
                // Feasibility tolerance is not equality membership: two distinct
                // sides of a narrow feasible interval can both be within tolerance.
                // Only an actual represented boundary starts as an active inequality.
                active.push(row < model.equalities || value == rhs);
                cooperation.completed(1).await?;
            }
            active
        }
    };
    let pivot_limit = limits.pivots;
    let mut stationary_pivots = 0usize;
    for _ in 0..pivot_limit {
        let (a, b, original) = rows::select(model, &active, owner, cooperation).await?;
        objective.begin()?;
        let Some(target) = objective.equality(&a, &b, &x, cooperation).await? else {
            return Ok(Outcome::Numerical { point: x });
        };
        let target = match target {
            EqualityTarget::Point(target) => target,
            EqualityTarget::Recession(direction) => {
                match super::recession::advance(
                    objective,
                    model,
                    super::recession::State {
                        x: &mut x,
                        active: &mut active,
                        stationary: &mut stationary_pivots,
                    },
                    direction,
                    tolerance,
                    owner,
                    cooperation,
                )
                .await?
                {
                    Some(outcome) => return Ok(outcome),
                    None => continue,
                }
            }
        };
        if target.direction.len() != x.len() || target.multipliers.len() != b.len() {
            return Err(crate::build_runtime_error(
                "working-set target returned inconsistent original coordinates",
            )
            .with_builtin(owner.name)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        let Some(current_activity) = rows::product(model.a, &x, owner, cooperation).await? else {
            return Ok(Outcome::Numerical { point: x });
        };
        let Some(target_activity) =
            rows::product(model.a, &target.direction, owner, cooperation).await?
        else {
            return Ok(Outcome::Numerical { point: x });
        };
        let mut alpha = 1.0_f64;
        let mut blocking = None;
        for row in model.equalities..model.b.len() {
            if !active[row] {
                let delta = target_activity[row] - current_activity[row];
                // A target already feasible at the original requested
                // tolerance needs no extra equality. In particular, numerical
                // residuals of a redundant row must not make an incompatible
                // working set out of an otherwise checked candidate.
                if delta > 0.0 && target_activity[row] - model.b[row] > tolerance {
                    let ratio = ((model.b[row] - current_activity[row]) / delta).max(0.0);
                    if !ratio.is_finite() {
                        return Ok(Outcome::Numerical { point: x });
                    }
                    if ratio < alpha {
                        alpha = ratio;
                        blocking = Some(row);
                    }
                }
            }
            cooperation.completed(1).await?;
        }
        if let Some(blocking) = blocking {
            let mut moved = false;
            for (x, &target) in x.iter_mut().zip(&target.direction) {
                let next = *x + alpha * (target - *x);
                moved |= next != *x;
                *x = next;
                cooperation.completed(1).await?;
            }
            stationary_pivots = if moved {
                0
            } else {
                stationary_pivots.saturating_add(1)
            };
            if stationary_pivots > model.b.len() {
                return Ok(Outcome::Numerical { point: x });
            }
            if !rows::feasible(model, &x, tolerance, owner, cooperation).await? {
                return Ok(Outcome::Numerical { point: x });
            }
            active[blocking] = true;
            if let Some(reason) = objective.advanced(&x, cooperation).await? {
                objective.retain_active(active);
                return Ok(Outcome::Stopped { point: x, reason });
            }
            continue;
        }
        let mut moved = false;
        let mut change = Change {
            length: 0.,
            reference: 0.,
        };
        for (&previous, &next) in x.iter().zip(&target.direction) {
            moved |= previous != next;
            change.length = change.length.hypot(next - previous);
            change.reference = change.reference.hypot(previous);
            cooperation.completed(1).await?;
        }
        stationary_pivots = if moved {
            0
        } else {
            stationary_pivots.saturating_add(1)
        };
        change.reference = change.reference.max(1.);
        x = target.direction;
        if let Some(reason) = objective.advanced(&x, cooperation).await? {
            objective.retain_active(active);
            return Ok(Outcome::Stopped { point: x, reason });
        }
        let mut multipliers = workspace::capacity(model.b.len(), owner)?;
        for _ in model.b {
            multipliers.push(0.0);
            cooperation.completed(1).await?;
        }
        let mut release = None;
        for (&row, &dual) in original.iter().zip(&target.multipliers) {
            multipliers[row] = dual;
            // Bland ordering: smallest original eligible row, not a varying
            // most-negative tie. Blocking ties use the same ascending IDs.
            if row >= model.equalities && dual < -limits.dual && release.is_none() {
                release = Some(row);
            }
            cooperation.completed(1).await?;
        }
        if let Some(release) = release {
            // Redundant/dependent rows can have nonunique duals. They may
            // leave normally, but stationary pivot cycling is not success.
            if stationary_pivots > model.b.len() {
                return Ok(Outcome::Numerical { point: x });
            }
            active[release] = false;
            continue;
        }
        if objective
            .checked(model, &x, &multipliers, cooperation)
            .await?
        {
            objective.retain_active(active);
            return Ok(Outcome::Optimal(Step {
                direction: x,
                multipliers,
            }));
        }
        if objective.small_step(change) {
            objective.retain_active(active);
            return Ok(Outcome::Stopped {
                point: x,
                reason: Stop::StepStagnation,
            });
        }
        return Ok(Outcome::Numerical { point: x });
    }
    objective.retain_active(active);
    Ok(Outcome::Stopped {
        point: x,
        reason: Stop::IterationLimit,
    })
}
