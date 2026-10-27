//! Callback and scalar-role metadata over the single field identity set.
use super::{OptimOptionField, OptimSolver};
impl OptimOptionField {
    pub const fn is_callback(self) -> bool {
        matches!(
            self,
            Self::OutputFcn
                | Self::JacobianMultiplyFcn
                | Self::HessianMultiplyFcn
                | Self::PlotFcn
                | Self::CreationFcn
                | Self::HybridFcn
                | Self::SearchFcn
                | Self::AcceptanceFcn
                | Self::AnnealingFcn
                | Self::TemperatureFcn
                | Self::CrossoverFcn
                | Self::MutationFcn
                | Self::SelectionFcn
                | Self::FitnessScalingFcn
        )
    }
    pub const fn trust_region_only(self) -> bool {
        matches!(
            self,
            Self::FunctionTolerance
                | Self::JacobianMultiplyFcn
                | Self::HessianFcn
                | Self::HessianMultiplyFcn
                | Self::HessPattern
                | Self::MaxPCGIter
                | Self::TolPCG
                | Self::PrecondBandWidth
                | Self::SubproblemAlgorithm
        )
    }
    pub const fn allows_zero(self, solver: OptimSolver) -> bool {
        if matches!(solver, OptimSolver::Lsqnonlin)
            && matches!(
                self,
                Self::TolX
                    | Self::TolFun
                    | Self::FunctionTolerance
                    | Self::OptimalityTolerance
                    | Self::MaxIter
                    | Self::MaxFunEvals
            )
        {
            return true;
        }
        if matches!(solver, OptimSolver::Quadprog)
            && matches!(self, Self::TolX | Self::TolCon | Self::TolFun)
        {
            return true;
        }
        if matches!(solver, OptimSolver::Lsqlin)
            && matches!(
                self,
                Self::TolFun | Self::OptimalityTolerance | Self::FunctionTolerance
            )
        {
            return true;
        }
        if matches!(solver, OptimSolver::Surrogateopt) {
            return matches!(
                self,
                Self::TolCon | Self::MaxFunEvals | Self::MaxTime | Self::MinSampleDistance
            );
        }
        if matches!(solver, OptimSolver::Ga) {
            return matches!(
                self,
                Self::EliteCount
                    | Self::MaxIter
                    | Self::MaxStallIterations
                    | Self::MaxTime
                    | Self::FunctionTolerance
                    | Self::TolCon
                    | Self::ObjectiveLimit
                    | Self::CrossoverFraction
                    | Self::MigrationFraction
            );
        }
        if matches!(solver, OptimSolver::Patternsearch) {
            return matches!(
                self,
                Self::TolX
                    | Self::FunctionTolerance
                    | Self::MaxIter
                    | Self::MaxFunEvals
                    | Self::MeshTolerance
                    | Self::MaxTime
            );
        }
        match self {
            Self::MaxStallTime => matches!(solver, OptimSolver::Particleswarm),
            Self::AbsoluteMaxObjectiveCount | Self::EqualityGoalCount => true,
            Self::PrecondBandWidth | Self::DiffMinChange | Self::RelLineSrchBnd => true,
            Self::AbsoluteGapTolerance | Self::RelativeGapTolerance | Self::MaxTime => true,
            Self::MaxIter => solver.accepts_zero_iterations(),
            Self::MaxFunEvals | Self::TolCon
                if matches!(
                    solver,
                    OptimSolver::Fmincon
                        | OptimSolver::Particleswarm
                        | OptimSolver::Fseminf
                        | OptimSolver::Fminimax
                        | OptimSolver::Fgoalattain
                ) =>
            {
                true
            }
            Self::TolX | Self::TolFun => {
                matches!(
                    solver,
                    OptimSolver::Fminunc
                        | OptimSolver::Fmincon
                        | OptimSolver::Particleswarm
                        | OptimSolver::Fseminf
                        | OptimSolver::Fminimax
                        | OptimSolver::Fgoalattain
                )
            }
            Self::OptimalityTolerance | Self::FunctionTolerance => {
                matches!(
                    solver,
                    OptimSolver::Fminunc
                        | OptimSolver::Fmincon
                        | OptimSolver::Particleswarm
                        | OptimSolver::Fseminf
                        | OptimSolver::Fminimax
                        | OptimSolver::Fgoalattain
                        | OptimSolver::Generic
                )
            }
            _ => false,
        }
    }
}
