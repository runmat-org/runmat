//! Public stopping reasons are distinct from numerical/provider errors and
//! from any claim of a globally optimal genetic-search result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Termination {
    Function,
    Augmented,
    Unchanged,
    FitnessLimit,
    Generations,
    Observer,
    NoFeasiblePoint,
    StallTime,
    Time,
    Unverified,
}
impl Termination {
    pub fn flag(self) -> i32 {
        match self {
            Self::Function | Self::Augmented => 1,
            Self::Unchanged => 3,
            Self::FitnessLimit => 5,
            Self::Generations => 0,
            Self::Observer => -1,
            Self::NoFeasiblePoint => -2,
            Self::StallTime => -4,
            Self::Time => -5,
            Self::Unverified => -9,
        }
    }
    pub fn message(self) -> &'static str {
        match self {
            Self::Function=>"Completed-generation relative change meets FunctionTolerance and the returned point satisfies the requested constraints.",
            Self::Augmented=>"The fixed-parameter subproblem meets FunctionTolerance, complementarity meets sqrt(ConstraintTolerance), and the returned point satisfies the requested constraints.",
            Self::Unchanged=>"The comparison value was unchanged over MaxStallGenerations and the returned point satisfies the requested constraints.",
            Self::FitnessLimit=>"FitnessLimit was reached at an admitted feasible point.",
            Self::Generations=>"MaxGenerations reached.",
            Self::Observer=>"An output or plot function requested termination.",
            Self::NoFeasiblePoint=>"No feasible point with a finite real fitness value was found.",
            Self::StallTime=>"MaxStallTime reached without a best-value improvement.",
            Self::Time=>"MaxTime reached.",
            Self::Unverified=>"A numerical feasibility proposal could not be verified; no unverified point was accepted.",
        }
    }
    pub fn may_run_hybrid(self) -> bool {
        matches!(
            self,
            Self::Function
                | Self::Augmented
                | Self::Unchanged
                | Self::FitnessLimit
                | Self::Generations
        )
    }
}
