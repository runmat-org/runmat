//! Public option spellings and their semantic identities.
use super::OptimSolver;
mod roles;
mod spelling;
use spelling::OptionSpelling;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimOptionField {
    TolX,
    TolFun,
    TolCon,
    MaxIter,
    MaxFunEvals,
    Display,
    Algorithm,
    SpecifyObjectiveGradient,
    SpecifyConstraintGradient,
    FiniteDifferenceType,
    FiniteDifferenceStepSize,
    TypicalX,
    OutputFcn,
    OptimalityTolerance,
    FunctionTolerance,
    HessianFcn,
    HessianMultiplyFcn,
    HessPattern,
    MaxPCGIter,
    TolPCG,
    PrecondBandWidth,
    SubproblemAlgorithm,
    AbsoluteGapTolerance,
    RelativeGapTolerance,
    MaxFeasiblePoints,
    MaxNodes,
    MaxTime,
    ObjectiveCutOff,
    PlotFcn,
    HessianApproximation,
    HessUpdate,
    ObjectiveLimit,
    UseParallel,
    AbsoluteMaxObjectiveCount,
    EqualityGoalCount,
    MeritFunction,
    Diagnostics,
    DiffMaxChange,
    DiffMinChange,
    FunValCheck,
    MaxSQPIter,
    RelLineSrchBnd,
    RelLineSrchBndDuration,
    TolConSQP,
    CreationFcn,
    HybridFcn,
    DisplayInterval,
    InertiaRange,
    InitialPoints,
    InitialSwarmSpan,
    MaxStallIterations,
    MaxStallTime,
    MinNeighborsFraction,
    SelfAdjustmentWeight,
    SocialAdjustmentWeight,
    SwarmSize,
    UseVectorized,
    MeshTolerance,
    InitialMeshSize,
    MaxMeshSize,
    MeshContractionFactor,
    MeshExpansionFactor,
    MeshRotate,
    AccelerateMesh,
    ScaleMesh,
    PollMethod,
    PollOrderAlgorithm,
    UseCompletePoll,
    SearchFcn,
    UseCompleteSearch,
    TolBind,
    InitialPenalty,
    PenaltyFactor,
    Cache,
    CacheSize,
    CacheTol,
    PlotInterval,
    AcceptanceFcn,
    AnnealingFcn,
    DataType,
    HybridInterval,
    InitialTemperature,
    ReannealInterval,
    TemperatureFcn,
    PopulationType,
    PopulationSize,
    EliteCount,
    CrossoverFraction,
    InitialPopulationMatrix,
    InitialScoresMatrix,
    InitialPopulationRange,
    CrossoverFcn,
    MutationFcn,
    SelectionFcn,
    FitnessScalingFcn,
    NonlinearConstraintAlgorithm,
    MigrationDirection,
    MigrationFraction,
    MigrationInterval,
    StallTest,
    BatchUpdateInterval,
    CheckpointFile,
    MinSampleDistance,
    MinSurrogatePoints,
    JacobianMultiplyFcn,
}

impl OptimOptionField {
    /// Scalar range only; count conversion must additionally preserve the
    /// original integer class and fit the host's exact structural index type.
    pub fn intlinprog_scalar_range(self, value: f64) -> bool {
        match self {
            Self::AbsoluteGapTolerance => value.is_finite() && value >= 0.0,
            Self::RelativeGapTolerance => value.is_finite() && (0.0..=1.0).contains(&value),
            Self::TolCon => value.is_finite() && (1e-10..=1e-3).contains(&value),
            Self::MaxTime => !value.is_nan() && value >= 0.0,
            Self::ObjectiveCutOff => !value.is_nan() && value > f64::NEG_INFINITY,
            Self::MaxFeasiblePoints => {
                value == f64::INFINITY || value.is_finite() && value > 0.0 && value.fract() == 0.0
            }
            Self::MaxNodes => value.is_finite() && value > 0.0 && value.fract() == 0.0,
            _ => false,
        }
    }
    pub const ALL: [Self; 105] = [
        Self::TolX,
        Self::TolFun,
        Self::TolCon,
        Self::MaxIter,
        Self::MaxFunEvals,
        Self::Display,
        Self::Algorithm,
        Self::SpecifyObjectiveGradient,
        Self::SpecifyConstraintGradient,
        Self::FiniteDifferenceType,
        Self::FiniteDifferenceStepSize,
        Self::TypicalX,
        Self::OutputFcn,
        Self::OptimalityTolerance,
        Self::FunctionTolerance,
        Self::HessianFcn,
        Self::HessianMultiplyFcn,
        Self::HessPattern,
        Self::MaxPCGIter,
        Self::TolPCG,
        Self::PrecondBandWidth,
        Self::SubproblemAlgorithm,
        Self::AbsoluteGapTolerance,
        Self::RelativeGapTolerance,
        Self::MaxFeasiblePoints,
        Self::MaxNodes,
        Self::MaxTime,
        Self::ObjectiveCutOff,
        Self::PlotFcn,
        Self::HessianApproximation,
        Self::HessUpdate,
        Self::ObjectiveLimit,
        Self::UseParallel,
        Self::AbsoluteMaxObjectiveCount,
        Self::EqualityGoalCount,
        Self::MeritFunction,
        Self::Diagnostics,
        Self::DiffMaxChange,
        Self::DiffMinChange,
        Self::FunValCheck,
        Self::MaxSQPIter,
        Self::RelLineSrchBnd,
        Self::RelLineSrchBndDuration,
        Self::TolConSQP,
        Self::CreationFcn,
        Self::HybridFcn,
        Self::DisplayInterval,
        Self::InertiaRange,
        Self::InitialPoints,
        Self::InitialSwarmSpan,
        Self::MaxStallIterations,
        Self::MaxStallTime,
        Self::MinNeighborsFraction,
        Self::SelfAdjustmentWeight,
        Self::SocialAdjustmentWeight,
        Self::SwarmSize,
        Self::UseVectorized,
        Self::MeshTolerance,
        Self::InitialMeshSize,
        Self::MaxMeshSize,
        Self::MeshContractionFactor,
        Self::MeshExpansionFactor,
        Self::MeshRotate,
        Self::AccelerateMesh,
        Self::ScaleMesh,
        Self::PollMethod,
        Self::PollOrderAlgorithm,
        Self::UseCompletePoll,
        Self::SearchFcn,
        Self::UseCompleteSearch,
        Self::TolBind,
        Self::InitialPenalty,
        Self::PenaltyFactor,
        Self::Cache,
        Self::CacheSize,
        Self::CacheTol,
        Self::PlotInterval,
        Self::AcceptanceFcn,
        Self::AnnealingFcn,
        Self::DataType,
        Self::HybridInterval,
        Self::InitialTemperature,
        Self::ReannealInterval,
        Self::TemperatureFcn,
        Self::PopulationType,
        Self::PopulationSize,
        Self::EliteCount,
        Self::CrossoverFraction,
        Self::InitialPopulationMatrix,
        Self::InitialScoresMatrix,
        Self::InitialPopulationRange,
        Self::CrossoverFcn,
        Self::MutationFcn,
        Self::SelectionFcn,
        Self::FitnessScalingFcn,
        Self::NonlinearConstraintAlgorithm,
        Self::MigrationDirection,
        Self::MigrationFraction,
        Self::MigrationInterval,
        Self::StallTest,
        Self::BatchUpdateInterval,
        Self::CheckpointFile,
        Self::MinSampleDistance,
        Self::MinSurrogatePoints,
        Self::JacobianMultiplyFcn,
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::JacobianMultiplyFcn => "JacobianMultiplyFcn",
            Self::TolX => "TolX",
            Self::TolFun => "TolFun",
            Self::TolCon => "TolCon",
            Self::MaxIter => "MaxIter",
            Self::MaxFunEvals => "MaxFunEvals",
            Self::Display => "Display",
            Self::Algorithm => "Algorithm",
            Self::SpecifyObjectiveGradient => "SpecifyObjectiveGradient",
            Self::SpecifyConstraintGradient => "SpecifyConstraintGradient",
            Self::FiniteDifferenceType => "FiniteDifferenceType",
            Self::FiniteDifferenceStepSize => "FiniteDifferenceStepSize",
            Self::TypicalX => "TypicalX",
            Self::OutputFcn => "OutputFcn",
            Self::OptimalityTolerance => "OptimalityTolerance",
            Self::FunctionTolerance => "FunctionTolerance",
            Self::HessianFcn => "HessianFcn",
            Self::HessianMultiplyFcn => "HessianMultiplyFcn",
            Self::HessPattern => "HessPattern",
            Self::MaxPCGIter => "MaxPCGIter",
            Self::TolPCG => "TolPCG",
            Self::PrecondBandWidth => "PrecondBandWidth",
            Self::SubproblemAlgorithm => "SubproblemAlgorithm",
            Self::AbsoluteGapTolerance => "AbsoluteGapTolerance",
            Self::RelativeGapTolerance => "RelativeGapTolerance",
            Self::MaxFeasiblePoints => "MaxFeasiblePoints",
            Self::MaxNodes => "MaxNodes",
            Self::MaxTime => "MaxTime",
            Self::ObjectiveCutOff => "ObjectiveCutOff",
            Self::PlotFcn => "PlotFcn",
            Self::HessianApproximation => "HessianApproximation",
            Self::HessUpdate => "HessUpdate",
            Self::ObjectiveLimit => "ObjectiveLimit",
            Self::UseParallel => "UseParallel",
            Self::AbsoluteMaxObjectiveCount => "AbsoluteMaxObjectiveCount",
            Self::EqualityGoalCount => "EqualityGoalCount",
            Self::MeritFunction => "MeritFunction",
            Self::Diagnostics => "Diagnostics",
            Self::DiffMaxChange => "DiffMaxChange",
            Self::DiffMinChange => "DiffMinChange",
            Self::FunValCheck => "FunValCheck",
            Self::MaxSQPIter => "MaxSQPIter",
            Self::RelLineSrchBnd => "RelLineSrchBnd",
            Self::RelLineSrchBndDuration => "RelLineSrchBndDuration",
            Self::TolConSQP => "TolConSQP",
            Self::CreationFcn => "CreationFcn",
            Self::HybridFcn => "HybridFcn",
            Self::DisplayInterval => "DisplayInterval",
            Self::InertiaRange => "InertiaRange",
            Self::InitialPoints => "InitialPoints",
            Self::InitialSwarmSpan => "InitialSwarmSpan",
            Self::MaxStallIterations => "MaxStallIterations",
            Self::MaxStallTime => "MaxStallTime",
            Self::MinNeighborsFraction => "MinNeighborsFraction",
            Self::SelfAdjustmentWeight => "SelfAdjustmentWeight",
            Self::SocialAdjustmentWeight => "SocialAdjustmentWeight",
            Self::SwarmSize => "SwarmSize",
            Self::MeshTolerance => "MeshTolerance",
            Self::InitialMeshSize => "InitialMeshSize",
            Self::MaxMeshSize => "MaxMeshSize",
            Self::MeshContractionFactor => "MeshContractionFactor",
            Self::MeshExpansionFactor => "MeshExpansionFactor",
            Self::MeshRotate => "MeshRotate",
            Self::AccelerateMesh => "AccelerateMesh",
            Self::ScaleMesh => "ScaleMesh",
            Self::PollMethod => "PollMethod",
            Self::PollOrderAlgorithm => "PollOrderAlgorithm",
            Self::UseCompletePoll => "UseCompletePoll",
            Self::SearchFcn => "SearchFcn",
            Self::UseCompleteSearch => "UseCompleteSearch",
            Self::TolBind => "TolBind",
            Self::InitialPenalty => "InitialPenalty",
            Self::PenaltyFactor => "PenaltyFactor",
            Self::Cache => "Cache",
            Self::CacheSize => "CacheSize",
            Self::CacheTol => "CacheTol",
            Self::PlotInterval => "PlotInterval",
            Self::UseVectorized => "UseVectorized",
            Self::AcceptanceFcn => "AcceptanceFcn",
            Self::AnnealingFcn => "AnnealingFcn",
            Self::DataType => "DataType",
            Self::HybridInterval => "HybridInterval",
            Self::InitialTemperature => "InitialTemperature",
            Self::ReannealInterval => "ReannealInterval",
            Self::TemperatureFcn => "TemperatureFcn",
            Self::PopulationType => "PopulationType",
            Self::PopulationSize => "PopulationSize",
            Self::EliteCount => "EliteCount",
            Self::CrossoverFraction => "CrossoverFraction",
            Self::InitialPopulationMatrix => "InitialPopulationMatrix",
            Self::InitialScoresMatrix => "InitialScoresMatrix",
            Self::InitialPopulationRange => "InitialPopulationRange",
            Self::CrossoverFcn => "CrossoverFcn",
            Self::MutationFcn => "MutationFcn",
            Self::SelectionFcn => "SelectionFcn",
            Self::FitnessScalingFcn => "FitnessScalingFcn",
            Self::NonlinearConstraintAlgorithm => "NonlinearConstraintAlgorithm",
            Self::MigrationDirection => "MigrationDirection",
            Self::MigrationFraction => "MigrationFraction",
            Self::MigrationInterval => "MigrationInterval",
            Self::StallTest => "StallTest",
            Self::BatchUpdateInterval => "BatchUpdateInterval",
            Self::CheckpointFile => "CheckpointFile",
            Self::MinSampleDistance => "MinSampleDistance",
            Self::MinSurrogatePoints => "MinSurrogatePoints",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        OptionSpelling::parse(name).map(OptionSpelling::field)
    }

    /// One semantic field per active control. Modern function/optimality
    /// tolerances stay distinct for fminunc; older solver profiles keep their
    /// established canonical TolFun field and meaning.
    pub fn parse_for_solver(name: &str, solver: OptimSolver) -> Option<Self> {
        if solver == OptimSolver::Lsqnonlin && name.eq_ignore_ascii_case("Jacobian") {
            return Some(Self::SpecifyObjectiveGradient);
        }
        if solver == OptimSolver::Simulannealbnd {
            let alias = match name.to_ascii_lowercase().as_str() {
                "tolfun" => Some(Self::FunctionTolerance),
                "outputfcns" => Some(Self::OutputFcn),
                "stalliterlimit" => Some(Self::MaxStallIterations),
                "timelimit" => Some(Self::MaxTime),
                _ => None,
            };
            if alias.is_some() {
                return alias;
            }
        }
        if matches!(solver, OptimSolver::Ga) && name.eq_ignore_ascii_case("StallTimeLimit") {
            return Some(Self::MaxStallTime);
        }
        let field = Self::parse(name)?;
        if matches!(solver, OptimSolver::Ga) {
            let field = if matches!(field, Self::TolFun) {
                Self::FunctionTolerance
            } else {
                field
            };
            return solver.accepts(field).then_some(field);
        }
        if matches!(solver, OptimSolver::Patternsearch) {
            if matches!(field, Self::OptimalityTolerance) {
                return None;
            }
            let field = if matches!(field, Self::TolFun) {
                Self::FunctionTolerance
            } else {
                field
            };
            return solver.accepts(field).then_some(field);
        }
        if solver == OptimSolver::Lsqlin {
            return solver.accepts(field).then_some(field);
        }
        let field = match field {
            Self::FunctionTolerance if matches!(solver, OptimSolver::Quadprog) => return None,
            Self::FunctionTolerance | Self::OptimalityTolerance
                if !matches!(
                    solver,
                    OptimSolver::Fminunc
                        | OptimSolver::Fmincon
                        | OptimSolver::Particleswarm
                        | OptimSolver::Simulannealbnd
                        | OptimSolver::Fseminf
                        | OptimSolver::Fminimax
                        | OptimSolver::Fgoalattain
                        | OptimSolver::Generic
                        | OptimSolver::Lsqnonlin
                ) =>
            {
                Self::TolFun
            }
            field => field,
        };
        solver.accepts(field).then_some(field)
    }
    pub fn parse_for_constructor(name: &str, solver: OptimSolver) -> Option<Self> {
        Self::parse_for_solver(name, solver).filter(|field| *field != Self::HessUpdate)
    }
    pub const fn name_for_solver(self, solver: OptimSolver) -> &'static str {
        match (solver, self) {
            (OptimSolver::Surrogateopt, Self::MaxFunEvals) => "MaxFunctionEvaluations",
            (OptimSolver::Surrogateopt, Self::TolCon) => "ConstraintTolerance",
            (OptimSolver::Lsqnonlin, Self::TolX) => "StepTolerance",
            (OptimSolver::Lsqnonlin, Self::MaxIter) => "MaxIterations",
            (OptimSolver::Lsqnonlin, Self::MaxFunEvals) => "MaxFunctionEvaluations",
            (OptimSolver::Ga, Self::MaxIter) => "MaxGenerations",
            (OptimSolver::Ga, Self::MaxStallIterations) => "MaxStallGenerations",
            (OptimSolver::Ga, Self::ObjectiveLimit) => "FitnessLimit",
            (OptimSolver::Ga, Self::TolCon) => "ConstraintTolerance",
            (OptimSolver::Particleswarm | OptimSolver::Simulannealbnd, Self::MaxIter) => {
                "MaxIterations"
            }
            (OptimSolver::Simulannealbnd, Self::MaxFunEvals) => "MaxFunctionEvaluations",
            (OptimSolver::Patternsearch, Self::MaxIter) => "MaxIterations",
            (OptimSolver::Patternsearch, Self::MaxFunEvals) => "MaxFunctionEvaluations",
            (OptimSolver::Patternsearch, Self::TolX) => "StepTolerance",
            (OptimSolver::Patternsearch, Self::TolCon) => "ConstraintTolerance",
            _ => self.name(),
        }
    }
}
