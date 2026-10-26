//! Selected feasible-start waves use the existing callback worker owner.
//! The parent's selected Clock is a soft boundary between genuine worker waves.
use super::super::{common, inputs::Problem, invalid, options, random, ScalarCooperation, POLICY};
use super::nonlinear::Counts;
use crate::{execution::CallbackBatch, BuiltinResult};
use common::solver_timing::SolverTiming;
use runmat_value::{Tensor, Value};

pub(super) async fn create(
    draft: &mut super::Draft,
    problem: &Problem,
    options: &options::Options,
    prepared: options::NonlinearCreation,
    batch: CallbackBatch,
    timing: &SolverTiming,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Counts> {
    let missing = draft.missing();
    let starts = prepared.starts.unwrap_or(missing.len());
    let constraints = problem
        .nonlinear
        .as_ref()
        .ok_or_else(|| invalid("nonlinear feasible creation requires nonlcon"))?;
    let geometry = super::geometry_wire::from_problem(&problem.public)?;
    let solver = prepared.solver.value();
    let mut best = POLICY.capacity(missing.len(), cooperation.runtime())?;
    for _ in missing.clone() {
        best.push(f64::INFINITY);
        cooperation.completed(1).await?;
    }
    let mut expected_extents = None;
    let mut counts = Counts::default();
    let mut start = 0;
    while start < starts {
        if options.maximum.reached(timing.elapsed()) {
            counts.timed_out = true;
            break;
        }
        let end = start.saturating_add(batch.width()).min(starts);
        if end == start {
            return Err(invalid(
                "selected feasible-start worker wave has zero width",
            ));
        }
        let mut arguments = POLICY.capacity(end - start, cooperation.runtime())?;
        for index in start..end {
            let point = if index < missing.len() {
                draft.row(missing.start + index, cooperation).await?
            } else {
                let mut point = random::uniform(draft.variables(), cooperation).await?;
                for (column, value) in point.iter_mut().enumerate() {
                    let (lower, upper) = options.ranges[column];
                    *value = super::super::operators::crossover::interpolate(lower, upper, *value)
                        .ok_or_else(|| invalid("parallel nonlinear start cannot be represented"))?;
                    cooperation.completed(1).await?;
                }
                point
            };
            let n = point.len();
            let mut args = POLICY.capacity(4, cooperation.runtime())?;
            args.push(Value::Tensor(
                Tensor::new(point, vec![1, n]).map_err(invalid)?,
            ));
            args.push(geometry.clone());
            args.push(constraints.clone());
            args.push(solver.clone());
            arguments.push(args);
            cooperation.completed(1).await?;
        }
        let returned = {
            let (_, budget) = cooperation.runtime_budget();
            let budget = budget.ok_or_else(|| {
                let mut error = crate::context::RuntimeCapabilityError::new(
                    crate::context::RuntimeCapability::Builtin,
                    "parallel feasible starts require their selected invocation budget",
                )
                .into_runtime_error();
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error
            })?;
            batch.execute(arguments, 1, budget)?.await?
        };
        if returned.len() != end - start {
            return Err(invalid("feasible-start worker response count changed"));
        }
        for (offset, returned) in returned.into_iter().enumerate() {
            let result = super::worker::decode(returned, draft.variables(), cooperation).await?;
            counts.calls = counts
                .calls
                .checked_add(result.calls)
                .ok_or_else(|| invalid("creation constraint callback count overflow"))?;
            counts.points = counts
                .points
                .checked_add(result.points)
                .ok_or_else(|| invalid("creation constraint point count overflow"))?;
            counts.timed_out |= result.timed_out;
            if let Some(extents) = result.extents {
                if expected_extents.is_some_and(|expected| expected != extents) {
                    return Err(invalid(
                        "nonlcon changed its inequality or equality count between starts",
                    ));
                }
                expected_extents = Some(extents);
            }
            let slot = (start + offset) % missing.len();
            if let Some(violation) = result.violation {
                if violation < best[slot] {
                    draft
                        .set(missing.start + slot, &result.x, false, cooperation)
                        .await?;
                    best[slot] = violation;
                }
            }
        }
        // Every completed wave's real counts and candidates survive the soft
        // deadline. There is no serial replay of a submitted start.
        counts.timed_out |= options.maximum.reached(timing.elapsed());
        if counts.timed_out {
            break;
        }
        start = end;
    }
    counts.extents = expected_extents;
    Ok(counts)
}
