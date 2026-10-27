//! The active borrowed GA phase begins after options preparation retires.
use super::*;

// The invocation capability retires before selected source/control state on
// admission failure or an unpolled child drop. No evaluator reference is stored.
pub(super) struct Prepared<'a> {
    pub scope: Scope,
    pub controls: auxiliary::Controls,
    pub timing: crate::builtins::math::optim::patternsearch::timing::AuxiliaryTiming<'a>,
    pub initial: f64,
}

pub(super) async fn run(
    base: &Point,
    options: &Options,
    linear: &Linear,
    augmented: Option<&Augmented>,
    deadline: &Deadline,
    bounds: &super::super::super::Bounds,
    evaluator: &mut Evaluator<'_>,
    prepared: Prepared<'_>,
) -> BuiltinResult<Outcome> {
    let Prepared {
        scope,
        controls,
        timing,
        initial,
    } = prepared;
    let maximum = controls.maximum();
    let geometry = Geometry::new(
        linear,
        bounds,
        &timing,
        maximum,
        options.controls.constraint_tolerance,
        controls.tolerance(),
    )?;
    let started = CountsAtEntry {
        points: evaluator.state.objective_calls,
        constraints: evaluator.state.constraint_calls,
        callbacks: evaluator.state.callbacks,
    };
    let mut callbacks = Callbacks {
        scope,
        oracle: Oracle {
            context: evaluator.context,
            state: &mut *evaluator.state,
            execution: None,
            base,
            initial,
            geometry: linear,
            tolerance: options.controls.constraint_tolerance,
            augmented,
            timing: &timing,
            maximum,
            best: None,
            started,
        },
    };
    let result = auxiliary::run(
        controls,
        geometry,
        timing.local().clone(),
        &mut callbacks,
        evaluator.cooperation,
    )
    .await;
    let completed = match result {
        Ok(completed) => completed,
        Err(error) => {
            drop(callbacks);
            return Err(error);
        }
    };
    let stopped = completed.stopped().copied();
    let Callbacks { scope, mut oracle } = callbacks;
    let best = oracle.best.take();
    // Revoke the local merit before dropping any retained GA source/user
    // state. The registration never held the borrowed evaluator itself.
    drop(scope);
    drop(completed);
    drop(oracle);
    let completion = if deadline.reached() {
        Completion::Time
    } else if evaluator.remaining() == 0 {
        Completion::Budget
    } else {
        match stopped {
            Some(Stop::ParentTime) => Completion::Time,
            Some(Stop::Budget) => Completion::Budget,
            Some(Stop::LocalTime) | None => Completion::Finished,
        }
    };
    Ok(Outcome {
        improved: best
            .filter(|best| best.score < initial)
            .map(|best| Improvement {
                point: best.point,
                direction: None,
                merit: MeritChange {
                    before: initial,
                    after: best.score,
                },
            }),
        completion,
        rotatable: false,
    })
}
