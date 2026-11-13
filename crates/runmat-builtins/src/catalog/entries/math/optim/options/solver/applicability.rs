//! Solver applicability over the single canonical field set.
use super::{OptimOptionField, OptimSolver};
impl OptimSolver {
    pub const fn accepts(self, field: OptimOptionField) -> bool {
        use OptimOptionField as Field;
        if matches!(self, Self::Surrogateopt) {
            return matches!(
                field,
                Field::BatchUpdateInterval
                    | Field::CheckpointFile
                    | Field::TolCon
                    | Field::Display
                    | Field::InitialPoints
                    | Field::MaxFunEvals
                    | Field::MaxTime
                    | Field::MinSampleDistance
                    | Field::MinSurrogatePoints
                    | Field::ObjectiveLimit
                    | Field::OutputFcn
                    | Field::PlotFcn
                    | Field::UseParallel
                    | Field::UseVectorized
            );
        }
        if matches!(self, Self::Ga) {
            return super::super::genetic_profile::accepts(field);
        }
        if matches!(self, Self::Simulannealbnd) {
            return matches!(
                field,
                Field::Display
                    | Field::FunctionTolerance
                    | Field::MaxIter
                    | Field::MaxFunEvals
                    | Field::MaxTime
                    | Field::ObjectiveLimit
                    | Field::OutputFcn
                    | Field::PlotFcn
                    | Field::HybridFcn
                    | Field::DisplayInterval
                    | Field::MaxStallIterations
                    | Field::AcceptanceFcn
                    | Field::AnnealingFcn
                    | Field::DataType
                    | Field::HybridInterval
                    | Field::InitialTemperature
                    | Field::PlotInterval
                    | Field::ReannealInterval
                    | Field::TemperatureFcn
            );
        }
        if matches!(self, Self::Particleswarm) {
            return matches!(
                field,
                Field::Display
                    | Field::FunctionTolerance
                    | Field::MaxIter
                    | Field::FunValCheck
                    | Field::MaxTime
                    | Field::ObjectiveLimit
                    | Field::OutputFcn
                    | Field::PlotFcn
                    | Field::UseParallel
                    | Field::CreationFcn
                    | Field::HybridFcn
                    | Field::DisplayInterval
                    | Field::InertiaRange
                    | Field::InitialPoints
                    | Field::InitialSwarmSpan
                    | Field::MaxStallIterations
                    | Field::MaxStallTime
                    | Field::MinNeighborsFraction
                    | Field::SelfAdjustmentWeight
                    | Field::SocialAdjustmentWeight
                    | Field::SwarmSize
                    | Field::UseVectorized
            );
        }
        if matches!(self, Self::Patternsearch) {
            return matches!(
                field,
                Field::TolX
                    | Field::FunctionTolerance
                    | Field::TolCon
                    | Field::MaxIter
                    | Field::MaxFunEvals
                    | Field::Display
                    | Field::Algorithm
                    | Field::OutputFcn
                    | Field::PlotFcn
                    | Field::MaxTime
                    | Field::UseParallel
                    | Field::MeshTolerance
                    | Field::InitialMeshSize
                    | Field::MaxMeshSize
                    | Field::MeshContractionFactor
                    | Field::MeshExpansionFactor
                    | Field::MeshRotate
                    | Field::AccelerateMesh
                    | Field::ScaleMesh
                    | Field::PollMethod
                    | Field::PollOrderAlgorithm
                    | Field::UseCompletePoll
                    | Field::SearchFcn
                    | Field::UseCompleteSearch
                    | Field::TolBind
                    | Field::InitialPenalty
                    | Field::PenaltyFactor
                    | Field::Cache
                    | Field::CacheSize
                    | Field::CacheTol
                    | Field::PlotInterval
                    | Field::UseVectorized
            );
        }
        if matches!(self, Self::Fseminf) {
            return matches!(
                field,
                Field::TolX
                    | Field::TolFun
                    | Field::TolCon
                    | Field::OptimalityTolerance
                    | Field::FunctionTolerance
                    | Field::MaxIter
                    | Field::MaxFunEvals
                    | Field::Display
                    | Field::SpecifyObjectiveGradient
                    | Field::FiniteDifferenceType
                    | Field::FiniteDifferenceStepSize
                    | Field::TypicalX
                    | Field::OutputFcn
                    | Field::PlotFcn
                    | Field::Diagnostics
                    | Field::DiffMaxChange
                    | Field::DiffMinChange
                    | Field::FunValCheck
                    | Field::MaxSQPIter
                    | Field::RelLineSrchBnd
                    | Field::RelLineSrchBndDuration
                    | Field::TolConSQP
            );
        }
        if matches!(self, Self::Fminimax | Self::Fgoalattain) {
            return match field {
                Field::AbsoluteMaxObjectiveCount => matches!(self, Self::Fminimax),
                Field::EqualityGoalCount => matches!(self, Self::Fgoalattain),
                Field::TolX
                | Field::TolFun
                | Field::TolCon
                | Field::OptimalityTolerance
                | Field::FunctionTolerance
                | Field::MaxIter
                | Field::MaxFunEvals
                | Field::Display
                | Field::SpecifyObjectiveGradient
                | Field::SpecifyConstraintGradient
                | Field::FiniteDifferenceType
                | Field::FiniteDifferenceStepSize
                | Field::TypicalX
                | Field::OutputFcn
                | Field::MeritFunction => true,
                _ => false,
            };
        }
        if matches!(self, Self::Fmincon) {
            return matches!(
                field,
                Field::Algorithm
                    | Field::TolX
                    | Field::TolFun
                    | Field::OptimalityTolerance
                    | Field::TolCon
                    | Field::MaxIter
                    | Field::MaxFunEvals
                    | Field::Display
                    | Field::SpecifyObjectiveGradient
                    | Field::SpecifyConstraintGradient
                    | Field::FunctionTolerance
                    | Field::HessianFcn
                    | Field::HessianMultiplyFcn
                    | Field::HessPattern
                    | Field::MaxPCGIter
                    | Field::TolPCG
                    | Field::PrecondBandWidth
                    | Field::SubproblemAlgorithm
                    | Field::FiniteDifferenceType
                    | Field::FiniteDifferenceStepSize
                    | Field::TypicalX
                    | Field::OutputFcn
            );
        }
        if matches!(self, Self::Intlinprog) {
            return matches!(
                field,
                Field::AbsoluteGapTolerance
                    | Field::RelativeGapTolerance
                    | Field::TolCon
                    | Field::Display
                    | Field::MaxFeasiblePoints
                    | Field::MaxNodes
                    | Field::MaxTime
                    | Field::ObjectiveCutOff
                    | Field::OutputFcn
                    | Field::PlotFcn
            );
        }
        if matches!(self, Self::Lsqlin) {
            return matches!(
                field,
                Field::TolFun
                    | Field::TolX
                    | Field::ObjectiveLimit
                    | Field::TolCon
                    | Field::OptimalityTolerance
                    | Field::FunctionTolerance
                    | Field::MaxIter
                    | Field::Display
                    | Field::Algorithm
                    | Field::TypicalX
                    | Field::MaxPCGIter
                    | Field::TolPCG
                    | Field::PrecondBandWidth
                    | Field::SubproblemAlgorithm
            );
        }
        if matches!(self, Self::Quadprog) {
            return matches!(
                field,
                Field::TolFun
                    | Field::TolCon
                    | Field::MaxIter
                    | Field::Display
                    | Field::Algorithm
                    | Field::TolX
                    | Field::ObjectiveLimit
                    | Field::MaxTime
            );
        }
        if matches!(self, Self::Linprog) {
            return matches!(field, Field::MaxIter | Field::Display);
        }
        if matches!(self, Self::Coneprog) {
            return matches!(
                field,
                Field::TolX | Field::TolFun | Field::MaxIter | Field::Display | Field::Algorithm
            );
        }
        if matches!(self, Self::Lsqnonlin) {
            return matches!(
                field,
                Field::JacobianMultiplyFcn
                    | Field::TolX
                    | Field::TolFun
                    | Field::FunctionTolerance
                    | Field::OptimalityTolerance
                    | Field::MaxIter
                    | Field::MaxFunEvals
                    | Field::Display
                    | Field::Algorithm
                    | Field::SpecifyObjectiveGradient
                    | Field::FiniteDifferenceStepSize
                    | Field::TypicalX
                    | Field::SubproblemAlgorithm
                    | Field::PrecondBandWidth
                    | Field::MaxPCGIter
                    | Field::TolPCG
            );
        }
        match field {
            Field::PopulationType
            | Field::BatchUpdateInterval
            | Field::CheckpointFile
            | Field::MinSampleDistance
            | Field::MinSurrogatePoints
            | Field::PopulationSize
            | Field::EliteCount
            | Field::CrossoverFraction
            | Field::InitialPopulationMatrix
            | Field::InitialScoresMatrix
            | Field::InitialPopulationRange
            | Field::CrossoverFcn
            | Field::MutationFcn
            | Field::SelectionFcn
            | Field::FitnessScalingFcn
            | Field::NonlinearConstraintAlgorithm
            | Field::MigrationDirection
            | Field::MigrationFraction
            | Field::MigrationInterval
            | Field::StallTest => false,
            Field::AcceptanceFcn
            | Field::AnnealingFcn
            | Field::DataType
            | Field::HybridInterval
            | Field::InitialTemperature
            | Field::PlotInterval
            | Field::ReannealInterval
            | Field::TemperatureFcn
            | Field::CreationFcn
            | Field::HybridFcn
            | Field::DisplayInterval
            | Field::InertiaRange
            | Field::InitialPoints
            | Field::InitialSwarmSpan
            | Field::MaxStallIterations
            | Field::MaxStallTime
            | Field::MinNeighborsFraction
            | Field::SelfAdjustmentWeight
            | Field::SocialAdjustmentWeight
            | Field::SwarmSize
            | Field::MeshTolerance
            | Field::InitialMeshSize
            | Field::MaxMeshSize
            | Field::MeshContractionFactor
            | Field::MeshExpansionFactor
            | Field::MeshRotate
            | Field::AccelerateMesh
            | Field::ScaleMesh
            | Field::PollMethod
            | Field::PollOrderAlgorithm
            | Field::UseCompletePoll
            | Field::SearchFcn
            | Field::UseCompleteSearch
            | Field::TolBind
            | Field::InitialPenalty
            | Field::PenaltyFactor
            | Field::Cache
            | Field::CacheSize
            | Field::CacheTol
            | Field::UseVectorized => false,
            Field::Diagnostics
            | Field::DiffMaxChange
            | Field::DiffMinChange
            | Field::FunValCheck
            | Field::MaxSQPIter
            | Field::RelLineSrchBnd
            | Field::RelLineSrchBndDuration
            | Field::TolConSQP => false,
            Field::FiniteDifferenceStepSize | Field::TypicalX => matches!(
                self,
                Self::Fminunc | Self::Fsolve | Self::Lsqcurvefit | Self::Lsqnonlin | Self::Generic
            ),
            Field::AbsoluteMaxObjectiveCount | Field::EqualityGoalCount | Field::MeritFunction => {
                false
            }
            Field::AbsoluteGapTolerance
            | Field::RelativeGapTolerance
            | Field::MaxFeasiblePoints
            | Field::MaxNodes
            | Field::MaxTime
            | Field::ObjectiveCutOff
            | Field::PlotFcn => false,
            Field::HessianApproximation
            | Field::HessUpdate
            | Field::ObjectiveLimit
            | Field::UseParallel => matches!(self, Self::Fminunc | Self::Generic),
            Field::OptimalityTolerance
            | Field::FunctionTolerance
            | Field::HessianFcn
            | Field::HessianMultiplyFcn
            | Field::HessPattern
            | Field::MaxPCGIter
            | Field::TolPCG
            | Field::PrecondBandWidth
            | Field::SubproblemAlgorithm => matches!(self, Self::Fminunc | Self::Generic),
            Field::TolCon => matches!(self, Self::Generic),
            Field::TolX | Field::MaxIter | Field::MaxFunEvals | Field::Display => true,
            Field::TolFun => matches!(
                self,
                Self::Coneprog
                    | Self::Fminunc
                    | Self::Fsolve
                    | Self::Lsqcurvefit
                    | Self::Lsqnonlin
                    | Self::Generic
            ),
            Field::Algorithm => matches!(
                self,
                Self::Coneprog
                    | Self::Fminunc
                    | Self::Lsqcurvefit
                    | Self::Lsqnonlin
                    | Self::Generic
            ),
            Field::SpecifyObjectiveGradient => {
                matches!(self, Self::Fminunc | Self::Fsolve | Self::Generic)
            }
            Field::JacobianMultiplyFcn => matches!(self, Self::Generic),
            Field::SpecifyConstraintGradient => matches!(self, Self::Generic),
            Field::FiniteDifferenceType => matches!(self, Self::Generic),
            Field::OutputFcn => matches!(self, Self::Fminunc | Self::Generic),
        }
    }
}
