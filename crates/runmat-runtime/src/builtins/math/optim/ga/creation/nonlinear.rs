//! Serial nonlinear multistart retains one constrained geometry and one clock.
//! The parallel dispatcher uses a separate selected-callable worker boundary;
//! this owner is the serial kernel, not a serial interpretation of On.
use super::super::super::fmincon::feasibility;
use super::super::{
    common, inputs::Problem, invalid, linear, numeric::INPUT_POLICY, options, random,
    ScalarCooperation, POLICY,
};
use crate::BuiltinResult;
use common::solver_timing::SolverTiming;

#[derive(Default)]
pub(in super::super) struct Counts {
    pub calls: usize,
    pub points: usize,
    pub timed_out: bool,
    pub extents: Option<(usize, usize)>,
}
mod launch;
pub(in super::super) fn prepare<'a>(
    draft: &'a mut super::Draft,
    problem: &'a mut Problem,
    options: &'a mut options::Options,
    timing: &'a SolverTiming,
    source: Option<&'a crate::call::descriptor::RepeatableCallable>,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<
    Output = BuiltinResult<crate::execution::RuntimeContinuationFuture<'a, Counts>>,
> + 'a {
    launch::prepare(draft, problem, options, timing, source, cooperation)
}
impl Counts {
    fn add(&mut self, candidate: &feasibility::Candidate) -> BuiltinResult<()> {
        if let Some(extents) = candidate.extents {
            if self.extents.is_some_and(|expected| expected != extents) {
                return Err(invalid(
                    "nonlcon changed its extents between creation starts",
                ));
            }
            self.extents = Some(extents);
        }
        self.calls = self
            .calls
            .checked_add(candidate.constraint_calls)
            .ok_or_else(|| invalid("creation constraint callback count overflow"))?;
        self.points = self
            .points
            .checked_add(candidate.points)
            .ok_or_else(|| invalid("creation constraint point count overflow"))?;
        self.timed_out |= candidate.timed_out;
        Ok(())
    }
}

pub(in super::super) async fn serial(
    draft: &mut super::Draft,
    problem: &mut Problem,
    options: &options::Options,
    prepared: options::NonlinearCreation,
    timing: &SolverTiming,
    source: Option<&crate::call::descriptor::RepeatableCallable>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Counts> {
    let missing = draft.missing();
    let starts = prepared.starts.unwrap_or(missing.len());
    if starts == 0 || missing.is_empty() {
        return Ok(Counts::default());
    }
    if starts > draft.members() {
        return Err(invalid("NumStartPts cannot exceed PopulationSize"));
    }
    let constraints = problem
        .nonlinear
        .as_ref()
        .ok_or_else(|| invalid("nonlinear feasible creation requires nonlcon"))?;
    let selected = if source.is_none() {
        Some(super::super::callbacks::select(constraints, cooperation)?)
    } else {
        None
    };
    let source = source
        .or(selected.as_ref())
        .expect("feasible starts retain their selected constraint source");
    let data = problem
        .linear
        .take()
        .ok_or_else(|| invalid("creation geometry was already consumed"))?;
    let (input, integers) = data.into_nonlinear_creation();
    let mut domain = feasibility::Domain::new(
        input,
        prepared.solver,
        feasibility::Limit {
            timing: timing.clone(),
            maximum: options.maximum,
        },
    )?;
    let mut best = POLICY.capacity(missing.len(), cooperation.runtime())?;
    for _ in missing.clone() {
        best.push(f64::INFINITY);
        cooperation.completed(1).await?;
    }
    let mut counts = Counts::default();
    for start in 0..starts {
        if options.maximum.reached(timing.elapsed()) {
            counts.timed_out = true;
            break;
        }
        let slot = start % missing.len();
        let row = missing.start + slot;
        let point = if start < missing.len() {
            draft.row(row, cooperation).await?
        } else {
            let mut point = random::uniform(draft.variables(), cooperation).await?;
            for (column, value) in point.iter_mut().enumerate() {
                let (lower, upper) = options.ranges[column];
                *value = super::super::operators::crossover::interpolate(lower, upper, *value)
                    .ok_or_else(|| invalid("nonlinear creation start cannot be represented"))?;
                cooperation.completed(1).await?;
            }
            point
        };
        let candidate = domain
            .run(point, constraints, source, INPUT_POLICY, cooperation)
            .await?;
        counts.add(&candidate)?;
        if let Some(violation) = candidate.violation.filter(|value| value.is_finite()) {
            if violation < best[slot] {
                draft.set(row, &candidate.x, false, cooperation).await?;
                best[slot] = violation;
            }
        }
        if counts.timed_out {
            break;
        }
    }
    // Even a normal time stop returns exactly the transferred original buffers.
    // No partial candidate is installed when its physical constraints are unknown.
    problem.linear = Some(linear::Data::after_nonlinear_creation(
        domain.into_input(),
        integers,
    ));
    Ok(counts)
}
