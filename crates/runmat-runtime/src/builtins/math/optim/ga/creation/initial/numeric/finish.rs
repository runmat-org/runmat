//! Final geometry acquisition starts only after nonlinear creation retires.
use super::*;
pub(super) async fn finish<F: creation::factory::Factory, I: Invocation>(
    seed: ga::control::Retained<seed::Seed>,
    problem: &mut Problem,
    options: &options::Options,
    timing: &SolverTiming,
    factory: F,
    counts: creation::nonlinear::Counts,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, Initial<F::Geometry>>> {
    let seed::Seed {
        mut draft,
        mut changed,
        bitstring,
    } = seed.into_inner();
    let mut admission = if counts.timed_out {
        creation::Admission::TimeLimit
    } else {
        creation::Admission::Complete
    };
    let domain = if bitstring {
        None
    } else {
        let runtime = ga::controller::continuation::original(cooperation)?;
        let original_draft = &mut draft;
        let original_problem = &mut *problem;
        let original_options = options;
        let original_admission = &mut admission;
        let original_changed = &mut changed;
        let original = &mut *cooperation;
        let domain = ga::controller::continuation::run(&runtime, move || {
            creation::domain::prepare(
                factory,
                original_draft,
                original_problem,
                original_options,
                timing,
                original_admission,
                original_changed,
                original,
            )
        })?
        .await?;
        Some(domain)
    };
    let supplied = draft.missing().start;
    Ok(ControlFlow::Continue(Initial {
        generation: Generation::Numeric(draft.finish()),
        domain,
        admission,
        constraint_calls: counts.calls,
        constraint_points: counts.points,
        extents: counts.extents,
        supplied,
        invalidated_scores: changed.rows,
    }))
}
