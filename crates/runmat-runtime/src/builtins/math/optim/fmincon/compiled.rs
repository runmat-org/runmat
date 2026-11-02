//! Private modeled-objective consumer of the actual constrained controllers.
//! The original model owns the oracle; no function handle or source identity
//! is invented, and no public builtin is re-entered.
use super::{
    evaluation::{Budget, Evaluator, Objective},
    model::{Geometry, LinearConstraints},
    options::Controls,
    outcome::{Outcome as NumericalOutcome, Parameters},
};
use crate::builtins::math::optim::common::{ScalarCooperation, ScalarOptionFields};
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
use runmat_builtins::{OptimAlgorithm, OptimOptionField as Field};
use runmat_value::Value;

pub(in crate::builtins::math::optim) use super::evaluation::internal::Oracle;
pub(in crate::builtins::math::optim) use super::evaluation::{EvaluationStop, Probe, ProbeRole};
pub(in crate::builtins::math::optim) use super::validation::Multipliers;

pub(in crate::builtins::math::optim) struct Problem {
    pub x: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub inequalities: usize,
    pub equalities: usize,
}
pub(in crate::builtins::math::optim) struct Outcome {
    numerical: NumericalOutcome,
    variables: usize,
}
impl Outcome {
    pub fn candidate(&self) -> &[f64] {
        &self.numerical.x
    }
    pub fn objective(&self) -> Option<f64> {
        self.numerical.point.as_ref().map(|point| point.f)
    }
    pub fn exitflag(&self) -> i32 {
        self.numerical.termination.flag()
    }
}

impl Outcome {
    pub fn information(
        &self,
        timing: Option<&crate::builtins::math::optim::common::solver_timing::SolverTiming>,
    ) -> BuiltinResult<Value> {
        super::publication::information::value(&self.numerical, timing, None)
    }
    pub async fn multipliers(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Multipliers>> {
        let Some(constraints) = self
            .numerical
            .constraints
            .as_ref()
            .filter(|_| self.numerical.multipliers_available)
        else {
            return Ok(None);
        };
        if constraints.roles.len() != self.numerical.multipliers.len() {
            return Err(super::invalid("multiplier role count is inconsistent"));
        }
        super::validation::split(
            constraints,
            &self.numerical.multipliers,
            self.variables,
            0,
            0,
            self.numerical
                .point
                .as_ref()
                .map_or(0, |point| point.inequality.len()),
            self.numerical
                .point
                .as_ref()
                .map_or(0, |point| point.equality.len()),
            super::IDENTITY,
            cooperation,
        )
        .await
        .map(Some)
    }
}

mod run;
mod algorithm;
mod options;
pub(in crate::builtins::math::optim) use options::prepare as prepare_options;
pub(in crate::builtins::math::optim) struct PreparedOptions {
    controls: Controls,
}

// Forwarders retain the admitted owner, not the full inactive algorithm future.
pub(in crate::builtins::math::optim) fn solve<'a>(
    problem: Problem,
    prepared: PreparedOptions,
    oracle: &'a mut dyn Oracle,
    policy: super::numeric::Policy,
    cooperation: &'a mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RuntimeContinuationFuture<'a, Outcome>> {
    let runtime = cooperation.runtime().cloned();
    crate::execution::admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
        run::solve(problem, prepared, oracle, policy, cooperation)
    })
}
