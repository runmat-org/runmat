//! Typed minimax/goal-attainment domain. The original vector objective is kept
//! separate from the smooth attainment coordinate consumed by constrained SQP.
pub(crate) mod controls;
pub(crate) mod formulation;
pub(crate) mod inputs;
pub(crate) mod jacobian;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kind {
    Minimax,
    GoalAttain,
}
impl Kind {
    pub(crate) const fn solver(self) -> runmat_builtins::OptimSolver {
        match self {
            Self::Minimax => runmat_builtins::OptimSolver::Fminimax,
            Self::GoalAttain => runmat_builtins::OptimSolver::Fgoalattain,
        }
    }
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Minimax => "fminimax",
            Self::GoalAttain => "fgoalattain",
        }
    }
}
pub(crate) fn invalid(kind: Kind, detail: impl AsRef<str>) -> crate::RuntimeError {
    crate::build_runtime_error(format!("{}: {}", kind.name(), detail.as_ref()))
        .with_builtin(kind.name())
        .with_identifier(format!("RunMat:{}:InvalidArgument", kind.name()))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
