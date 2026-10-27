//! Genetic operator identities at the public option-value boundary.
//! Callables retain semantic callback authority; these selectors do not dispatch
//! arbitrary function names or grant callback effects.
macro_rules! selector {
    ($name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum $name { $($variant),+ }
        impl $name {
            pub fn parse(value: &str) -> Option<Self> {
                let value=value.trim();
                $(if value.eq_ignore_ascii_case($text) { return Some(Self::$variant); })+
                None
            }
            pub const fn name(self) -> &'static str {
                match self { $(Self::$variant => $text),+ }
            }
        }
    };
}

selector!(OptimGeneticPopulation {
    DoubleVector => "doubleVector",
    BitString => "bitstring",
    Custom => "custom",
});
selector!(OptimGeneticConstraint {
    AugmentedLagrangian => "auglag",
    Penalty => "penalty",
});
selector!(OptimGeneticCreation {
    Uniform => "gacreationuniform",
    LinearFeasible => "gacreationlinearfeasible",
    NonlinearFeasible => "gacreationnonlinearfeasible",
    UniformInteger => "gacreationuniformint",
    Sobol => "gacreationsobol",
});
selector!(OptimGeneticCrossover {
    Scattered => "crossoverscattered",
    SinglePoint => "crossoversinglepoint",
    TwoPoint => "crossovertwopoint",
    Intermediate => "crossoverintermediate",
    Laplace => "crossoverlaplace",
    Heuristic => "crossoverheuristic",
    Arithmetic => "crossoverarithmetic",
});
selector!(OptimGeneticMutation {
    Gaussian => "mutationgaussian",
    Uniform => "mutationuniform",
    AdaptiveFeasible => "mutationadaptfeasible",
    Power => "mutationpower",
    PositiveBasis => "mutationpositivebasis",
});
selector!(OptimGeneticSelection {
    StochasticUniform => "selectionstochunif",
    Remainder => "selectionremainder",
    Uniform => "selectionuniform",
    Roulette => "selectionroulette",
    Tournament => "selectiontournament",
});
selector!(OptimGeneticScaling {
    Rank => "fitscalingrank",
    Proportional => "fitscalingprop",
    Top => "fitscalingtop",
    ShiftLinear => "fitscalingshiftlinear",
});
selector!(OptimGeneticMigration {
    Forward => "forward",
    Both => "both",
});
selector!(OptimGeneticStall {
    AverageChange => "averageChange",
    GeometricWeighted => "geometricWeighted",
});
selector!(OptimGeneticAction {
    InfeasiblePoint => "Infeasible point",
    UpdateMultipliers => "Update multipliers",
    IncreasePenalty => "Increase penalty",
});

/// GA observers return state (and, for OutputFcn, options plus optchanged).
/// This is not the stop-boolean protocol of local optimization callbacks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimGeneticObserverState {
    Init,
    Iter,
    Interrupt,
    Done,
}
impl OptimGeneticObserverState {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "init" => Some(Self::Init),
            "iter" => Some(Self::Iter),
            "interrupt" => Some(Self::Interrupt),
            "done" => Some(Self::Done),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::Iter => "iter",
            Self::Interrupt => "interrupt",
            Self::Done => "done",
        }
    }
}

/// Public GA state keys. Presence depends on the selected algorithm/event;
/// these identities do not assert that user-returned state facts are valid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimGeneticStateField {
    Generation,
    StartTime,
    StopFlag,
    LastImprovement,
    LastImprovementTime,
    Best,
    How,
    GeneticOrigin,
    GeneticParents,
    FunEval,
    Expectation,
    Selection,
    EvalElites,
    HaveDuplicates,
    Population,
    Score,
    NonlinearInequality,
    NonlinearEquality,
    LinearConstraintType,
    MixedInteger,
    Fitness,
    InnerGeneration,
    TotalInnerGenerations,
}
impl OptimGeneticStateField {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Generation => "Generation",
            Self::StartTime => "StartTime",
            Self::StopFlag => "StopFlag",
            Self::LastImprovement => "LastImprovement",
            Self::LastImprovementTime => "LastImprovementTime",
            Self::Best => "Best",
            Self::How => "how",
            Self::GeneticOrigin => "GeneticOrigin",
            Self::GeneticParents => "GeneticParents",
            Self::FunEval => "FunEval",
            Self::Expectation => "Expectation",
            Self::Selection => "Selection",
            Self::EvalElites => "EvalElites",
            Self::HaveDuplicates => "HaveDuplicates",
            Self::Population => "Population",
            Self::Score => "Score",
            Self::NonlinearInequality => "NonlinIneq",
            Self::NonlinearEquality => "NonlinEq",
            Self::LinearConstraintType => "LinearConstrType",
            Self::MixedInteger => "IsMixedInteger",
            Self::Fitness => "Fitness",
            Self::InnerGeneration => "InnerGeneration",
            Self::TotalInnerGenerations => "TotalInnerGenerations",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn genetic_roles_do_not_alias_other_algorithms_or_observer_protocols() {
        assert_eq!(
            OptimGeneticConstraint::parse("AUGLAG"),
            Some(OptimGeneticConstraint::AugmentedLagrangian)
        );
        assert_ne!(
            OptimGeneticConstraint::parse("penalty"),
            OptimGeneticConstraint::parse("auglag")
        );
        assert!(OptimGeneticCrossover::parse("selectiontournament").is_none());
        assert_eq!(
            OptimGeneticObserverState::parse("interrupt"),
            Some(OptimGeneticObserverState::Interrupt)
        );
        assert!(OptimGeneticObserverState::parse("true").is_none());
    }
}
