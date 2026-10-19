/// Public `fitclinear` name-value identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitclinearOption {
    Learner,
    Lambda,
    Regularization,
    Solver,
    ObservationsIn,
    FitBias,
    ClassNames,
    PredictorNames,
    ResponseName,
    Weights,
    Prior,
    ScoreTransform,
    IterationLimit,
    PassLimit,
    Beta,
    Bias,
    BatchLimit,
    BetaTolerance,
    DeltaGradientTolerance,
    GradientTolerance,
    NumCheckConvergence,
    PostFitBias,
    Verbose,
    Cost,
    CrossVal,
    CvPartition,
    Holdout,
    KFold,
    Leaveout,
    OptimizeHyperparameters,
    HyperparameterOptimizationOptions,
}

impl FitclinearOption {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "learner" => Self::Learner,
            "lambda" => Self::Lambda,
            "regularization" => Self::Regularization,
            "solver" => Self::Solver,
            "observationsin" => Self::ObservationsIn,
            "fitbias" => Self::FitBias,
            "classnames" => Self::ClassNames,
            "predictornames" => Self::PredictorNames,
            "responsename" => Self::ResponseName,
            "weights" => Self::Weights,
            "prior" => Self::Prior,
            "scoretransform" => Self::ScoreTransform,
            "iterationlimit" => Self::IterationLimit,
            "passlimit" => Self::PassLimit,
            "beta" => Self::Beta,
            "bias" => Self::Bias,
            "batchlimit" => Self::BatchLimit,
            "betatolerance" => Self::BetaTolerance,
            "deltagradienttolerance" => Self::DeltaGradientTolerance,
            "gradienttolerance" => Self::GradientTolerance,
            "numcheckconvergence" => Self::NumCheckConvergence,
            "postfitbias" => Self::PostFitBias,
            "verbose" => Self::Verbose,
            "cost" => Self::Cost,
            "crossval" => Self::CrossVal,
            "cvpartition" => Self::CvPartition,
            "holdout" => Self::Holdout,
            "kfold" => Self::KFold,
            "leaveout" => Self::Leaveout,
            "optimizehyperparameters" => Self::OptimizeHyperparameters,
            "hyperparameteroptimizationoptions" => Self::HyperparameterOptimizationOptions,
            _ => return None,
        })
    }

    pub const fn unsupported(self) -> bool {
        matches!(
            self,
            Self::Cost
                | Self::CrossVal
                | Self::CvPartition
                | Self::Holdout
                | Self::KFold
                | Self::Leaveout
                | Self::OptimizeHyperparameters
                | Self::HyperparameterOptimizationOptions
        )
    }

    /// Accepted for script compatibility but not consumed by the current CPU solver.
    /// These payloads must not cause resident downloads.
    pub const fn ignored(self) -> bool {
        matches!(
            self,
            Self::BatchLimit
                | Self::BetaTolerance
                | Self::DeltaGradientTolerance
                | Self::GradientTolerance
                | Self::NumCheckConvergence
                | Self::PostFitBias
                | Self::Verbose
        )
    }
}

/// The admitted solver spelling is retained as a type; model output alone uses text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitclinearSolver {
    Auto,
    Sgd,
    Asgd,
    Bfgs,
    Lbfgs,
    Sparsa,
    Dual,
}

impl FitclinearSolver {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "sgd" => Self::Sgd,
            "asgd" => Self::Asgd,
            "bfgs" => Self::Bfgs,
            "lbfgs" => Self::Lbfgs,
            "sparsa" => Self::Sparsa,
            "dual" => Self::Dual,
            _ => return None,
        })
    }

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Sgd => "sgd",
            Self::Asgd => "asgd",
            Self::Bfgs => "bfgs",
            Self::Lbfgs => "lbfgs",
            Self::Sparsa => "sparsa",
            Self::Dual => "dual",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_and_solver_spelling_have_one_closed_parser() {
        assert_eq!(
            FitclinearOption::from_public_name("CLASSNAMES"),
            Some(FitclinearOption::ClassNames)
        );
        assert!(FitclinearOption::Cost.unsupported());
        assert_eq!(FitclinearOption::from_public_name("unknown"), None);
        assert_eq!(
            FitclinearSolver::from_public_name("LBFGS"),
            Some(FitclinearSolver::Lbfgs)
        );
        assert_eq!(FitclinearSolver::Lbfgs.public_name(), "lbfgs");
        assert_eq!(FitclinearSolver::from_public_name("unsupported"), None);
    }
}
