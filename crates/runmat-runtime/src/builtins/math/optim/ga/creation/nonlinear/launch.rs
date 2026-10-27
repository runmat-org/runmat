//! Default preparation returns the already admitted selected continuation.
//! Its own state retires before any serial/parallel child callback is polled.
use super::*;
use crate::execution::{CallbackBatch, RuntimeContinuationFuture};
pub(super) async fn prepare<'a>(
    draft: &'a mut super::super::Draft,
    problem: &'a mut Problem,
    options: &'a mut options::Options,
    timing: &'a SolverTiming,
    source: Option<&'a crate::call::descriptor::RepeatableCallable>,
    cooperation: &'a mut ScalarCooperation,
) -> BuiltinResult<RuntimeContinuationFuture<'a, Counts>> {
    let creation = match options.prepared.nonlinear_creation.take() {
        Some(prepared) => prepared,
        None => options::default_nonlinear_creation(cooperation).await?,
    };
    if creation
        .starts
        .is_some_and(|starts| starts > draft.members())
    {
        return Err(invalid("NumStartPts cannot exceed PopulationSize"));
    }
    let target = runmat_value::Value::FunctionHandle(super::super::worker::NAME.into());
    let selected = CallbackBatch::select(cooperation.runtime(), &target, creation.parallel)?;
    let runtime = super::super::super::controller::continuation::original(cooperation)?;
    match selected {
        Some(selected) => super::super::super::controller::continuation::run(&runtime, move || {
            super::super::parallel::create(
                draft,
                problem,
                options,
                creation,
                selected,
                timing,
                cooperation,
            )
        }),
        None => super::super::super::controller::continuation::run(&runtime, move || {
            super::serial(
                draft,
                problem,
                options,
                creation,
                timing,
                source,
                cooperation,
            )
        }),
    }
}
