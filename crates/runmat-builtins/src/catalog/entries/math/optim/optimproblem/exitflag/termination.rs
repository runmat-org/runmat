//! Problem-based solver termination members. Numeric meanings are semantic names,
//! not a cast of arbitrary backend status integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationExitflag {
    OptimalWithPoorFeasibility,
    OptimalSolution,
    SolverLimitExceeded,
    NoFeasiblePointFound,
    Unbounded,
    FoundNaN,
    PrimalDualInfeasible,
    DirectionTooSmall,
    FeasibilityLost,
    Nonconvex,
    FunctionChangeBelowTolerance,
    StepSizeBelowTolerance,
    OutputFcnStop,
    IntegerFeasible,
    ObjectiveLimitReached,
    IllConditioned,
    NoDescentDirectionFound,
    EquationSolved,
}
impl OptimizationExitflag {
    pub const ALL: [Self; 18] = [
        Self::OptimalWithPoorFeasibility,
        Self::OptimalSolution,
        Self::SolverLimitExceeded,
        Self::NoFeasiblePointFound,
        Self::Unbounded,
        Self::FoundNaN,
        Self::PrimalDualInfeasible,
        Self::DirectionTooSmall,
        Self::FeasibilityLost,
        Self::Nonconvex,
        Self::FunctionChangeBelowTolerance,
        Self::StepSizeBelowTolerance,
        Self::OutputFcnStop,
        Self::IntegerFeasible,
        Self::ObjectiveLimitReached,
        Self::IllConditioned,
        Self::NoDescentDirectionFound,
        Self::EquationSolved,
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::OptimalWithPoorFeasibility => "OptimalWithPoorFeasibility",
            Self::OptimalSolution => "OptimalSolution",
            Self::SolverLimitExceeded => "SolverLimitExceeded",
            Self::NoFeasiblePointFound => "NoFeasiblePointFound",
            Self::Unbounded => "Unbounded",
            Self::FoundNaN => "FoundNaN",
            Self::PrimalDualInfeasible => "PrimalDualInfeasible",
            Self::DirectionTooSmall => "DirectionTooSmall",
            Self::FeasibilityLost => "FeasibilityLost",
            Self::Nonconvex => "Nonconvex",
            Self::FunctionChangeBelowTolerance => "FunctionChangeBelowTolerance",
            Self::StepSizeBelowTolerance => "StepSizeBelowTolerance",
            Self::OutputFcnStop => "OutputFcnStop",
            Self::IntegerFeasible => "IntegerFeasible",
            Self::ObjectiveLimitReached => "ObjectiveLimitReached",
            Self::IllConditioned => "IllConditioned",
            Self::NoDescentDirectionFound => "NoDescentDirectionFound",
            Self::EquationSolved => "EquationSolved",
        }
    }
    pub const fn numeric(self) -> i32 {
        match self {
            Self::OptimalWithPoorFeasibility => 3,
            Self::OptimalSolution => 1,
            Self::SolverLimitExceeded => 0,
            Self::NoFeasiblePointFound => -2,
            Self::Unbounded => -3,
            Self::FoundNaN => -4,
            Self::PrimalDualInfeasible => -5,
            Self::DirectionTooSmall => -7,
            Self::FeasibilityLost => -9,
            Self::Nonconvex => -6,
            Self::FunctionChangeBelowTolerance => 3,
            Self::StepSizeBelowTolerance => 2,
            Self::OutputFcnStop => -1,
            Self::IntegerFeasible => 2,
            Self::ObjectiveLimitReached => -3,
            Self::IllConditioned => -4,
            Self::NoDescentDirectionFound => -8,
            Self::EquationSolved => 1,
        }
    }
    pub fn parse_member(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|member| member.name() == name)
    }
}
