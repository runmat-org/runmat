//! Private modeled scalar consumer of the existing fminunc evaluator/controller.
pub(in crate::builtins::math::optim) use super::reporting::TerminationReason as StopReason;
use super::*;
use crate::builtins::math::optim::common::{self, ScalarCooperation};
pub(in crate::builtins::math::optim) use crate::builtins::math::optim::fmincon::compiled::Oracle;
mod options;
mod run;
pub(in crate::builtins::math::optim) use options::prepare;
pub(in crate::builtins::math::optim) struct PreparedOptions {
    controls: FminuncOptions,
    values: Option<common::PreparedSolverOptions>,
}
pub(in crate::builtins::math::optim) struct Outcome(algorithm::MinimizeOutcome);
impl Outcome {
    pub fn candidate(&self) -> &[f64] {
        self.0.candidate()
    }
    pub fn objective(&self) -> f64 {
        self.0.value()
    }
    pub fn reason(&self) -> StopReason {
        self.0.reason()
    }
    pub fn information(&self) -> StructValue {
        self.0.information()
    }
}
pub(in crate::builtins::math::optim) fn solve<'a>(
    initial: Vec<f64>,
    options: PreparedOptions,
    oracle: &'a mut dyn Oracle,
    cooperation: &'a mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RuntimeContinuationFuture<'a, Outcome>> {
    let runtime = cooperation.runtime().cloned();
    crate::execution::admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
        run::solve(initial, options, oracle, cooperation)
    })
}
