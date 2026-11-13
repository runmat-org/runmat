//! Local scalar stops remain distinct from residual and algebraic evidence.
use super::*;
use crate::builtins::math::optim::fminunc::compiled as unconstrained;
use futures::future::Either;
use runmat_builtins::OptimSolver;
use runmat_builtins::OptimizationExitflag as Flag;
type Scalar = Either<compiled::Outcome, unconstrained::Outcome>;
pub(super) struct ScalarOutcome(Scalar);
pub(super) struct Outcome(Either<ScalarOutcome, residual::Outcome>);
impl Outcome {
    pub(super) fn constrained(value: compiled::Outcome) -> Self {
        Self(Either::Left(ScalarOutcome(Either::Left(value))))
    }
    pub(super) fn unconstrained(value: unconstrained::Outcome) -> Self {
        Self(Either::Left(ScalarOutcome(Either::Right(value))))
    }
    pub(super) fn residual(value: residual::Outcome) -> Self {
        Self(Either::Right(value))
    }
    pub(super) fn into_parts(self) -> Either<ScalarOutcome, residual::Outcome> {
        self.0
    }
}
impl ScalarOutcome {
    pub(super) fn solver(&self) -> OptimSolver {
        match &self.0 {
            Either::Left(_) => OptimSolver::Fmincon,
            Either::Right(_) => OptimSolver::Fminunc,
        }
    }
    pub(super) fn candidate(&self) -> &[f64] {
        match &self.0 {
            Either::Left(value) => value.candidate(),
            Either::Right(value) => value.candidate(),
        }
    }
    pub(super) fn objective(&self) -> Option<f64> {
        match &self.0 {
            Either::Left(value) => value.objective(),
            Either::Right(value) => Some(value.objective()),
        }
    }
    pub(super) fn flag(&self, policy: Policy) -> BuiltinResult<Flag> {
        match &self.0 {
            Either::Left(value) => match value.exitflag() {
                1 => Ok(Flag::OptimalSolution),
                2 => Ok(Flag::StepSizeBelowTolerance),
                3 => Ok(Flag::FunctionChangeBelowTolerance),
                0 => Ok(Flag::SolverLimitExceeded),
                -1 => Ok(Flag::OutputFcnStop),
                -2 => Ok(Flag::NoFeasiblePointFound),
                -4 => Err(run::unverified(policy,
                    "The nonlinear solver could not obtain a reliable local step; no global infeasibility or minimum is established.")),
                _ => Err(policy.invalid("unsupported modeled constrained solver termination")),
            },
            Either::Right(value) => match value.reason() {
                unconstrained::StopReason::Optimality => Ok(Flag::OptimalSolution),
                unconstrained::StopReason::Step => Ok(Flag::StepSizeBelowTolerance),
                unconstrained::StopReason::Limit => Ok(Flag::SolverLimitExceeded),
                unconstrained::StopReason::OutputFunction => Ok(Flag::OutputFcnStop),
                // Public solve names this threshold stop Unbounded(-3). Its
                // internal reason and actual point remain ObjectiveLimit;
                // no original unboundedness certificate is created or used.
                unconstrained::StopReason::ObjectiveLimit => Ok(Flag::Unbounded),
                unconstrained::StopReason::NumericalFailure => Err(run::unverified(policy,
                    "The unconstrained line search did not produce a reliable local step; no minimum or unboundedness certificate is established.")),
            },
        }
    }
    pub(super) fn information(
        &self,
        timing: Option<&crate::builtins::math::optim::common::solver_timing::SolverTiming>,
    ) -> BuiltinResult<Value> {
        match &self.0 {
            Either::Left(value) => value.information(timing),
            Either::Right(value) => Ok(Value::Struct(value.information())),
        }
    }
    pub(super) async fn multipliers(
        &self,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<compiled::Multipliers>> {
        match &self.0 {
            Either::Left(value) => value.multipliers(c).await,
            Either::Right(_) => Ok(None),
        }
    }
}
