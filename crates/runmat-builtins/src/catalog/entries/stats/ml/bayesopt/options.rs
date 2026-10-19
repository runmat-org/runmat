/// Name-value option identity; only this parser interprets public spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BayesoptOption {
    MaxObjectiveEvaluations,
    NumSeedPoints,
    AcquisitionFunctionName,
    ExplorationRatio,
    IsObjectiveDeterministic,
    Verbose,
    UseParallel,
    XConstraintFcn,
    PlotFcn,
    OutputFcn,
    ConditionalVariableFcn,
    InitialX,
    InitialObjective,
    InitialConstraintViolations,
}

impl BayesoptOption {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "maxobjectiveevaluations" => Self::MaxObjectiveEvaluations,
            "numseedpoints" => Self::NumSeedPoints,
            "acquisitionfunctionname" => Self::AcquisitionFunctionName,
            "explorationratio" => Self::ExplorationRatio,
            "isobjectivedeterministic" => Self::IsObjectiveDeterministic,
            "verbose" => Self::Verbose,
            "useparallel" => Self::UseParallel,
            "xconstraintfcn" => Self::XConstraintFcn,
            "plotfcn" => Self::PlotFcn,
            "outputfcn" => Self::OutputFcn,
            "conditionalvariablefcn" => Self::ConditionalVariableFcn,
            "initialx" => Self::InitialX,
            "initialobjective" => Self::InitialObjective,
            "initialconstraintviolations" => Self::InitialConstraintViolations,
            _ => return None,
        })
    }

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::MaxObjectiveEvaluations => "MaxObjectiveEvaluations",
            Self::NumSeedPoints => "NumSeedPoints",
            Self::AcquisitionFunctionName => "AcquisitionFunctionName",
            Self::ExplorationRatio => "ExplorationRatio",
            Self::IsObjectiveDeterministic => "IsObjectiveDeterministic",
            Self::Verbose => "Verbose",
            Self::UseParallel => "UseParallel",
            Self::XConstraintFcn => "XConstraintFcn",
            Self::PlotFcn => "PlotFcn",
            Self::OutputFcn => "OutputFcn",
            Self::ConditionalVariableFcn => "ConditionalVariableFcn",
            Self::InitialX => "InitialX",
            Self::InitialObjective => "InitialObjective",
            Self::InitialConstraintViolations => "InitialConstraintViolations",
        }
    }

    pub const fn always_unsupported(self) -> bool {
        matches!(
            self,
            Self::InitialX | Self::InitialObjective | Self::InitialConstraintViolations
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BayesoptAcquisition {
    ExpectedImprovement,
    ExpectedImprovementPlus,
    LowerConfidenceBound,
}

impl BayesoptAcquisition {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(
            match name.trim().to_ascii_lowercase().replace('_', "-").as_str() {
                "expected-improvement" | "expected-improvement-per-second" => {
                    Self::ExpectedImprovement
                }
                "expected-improvement-plus" | "expected-improvement-per-second-plus" => {
                    Self::ExpectedImprovementPlus
                }
                "lower-confidence-bound" => Self::LowerConfidenceBound,
                _ => return None,
            },
        )
    }

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::ExpectedImprovement => "expected-improvement",
            Self::ExpectedImprovementPlus => "expected-improvement-plus",
            Self::LowerConfidenceBound => "lower-confidence-bound",
        }
    }
}
