//! Public option identities; aliases are decoded once at the call boundary.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LassoOption {
    Alpha,
    Lambda,
    LambdaRatio,
    NumLambda,
    Standardize,
    Intercept,
    Weights,
    RelTol,
    MaxIter,
    Cv,
    PredictorNames,
    DfMax,
    UseCovariance,
    McReps,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LassoOptionRole {
    FloatingParameter,
    StructuralControl,
    BooleanControl,
}

impl LassoOption {
    pub fn from_public_name(name: &str) -> Option<Self> {
        let normalized = name
            .chars()
            .filter(|ch| *ch != '_' && *ch != '-')
            .flat_map(char::to_lowercase)
            .collect::<String>();
        Some(match normalized.as_str() {
            "alpha" => Self::Alpha,
            "lambda" => Self::Lambda,
            "lambdaratio" => Self::LambdaRatio,
            "numlambda" => Self::NumLambda,
            "standardize" => Self::Standardize,
            "intercept" => Self::Intercept,
            "weights" => Self::Weights,
            "reltol" => Self::RelTol,
            "maxiter" => Self::MaxIter,
            "cv" => Self::Cv,
            "predictornames" => Self::PredictorNames,
            "dfmax" => Self::DfMax,
            "usecovariance" => Self::UseCovariance,
            "mcreps" => Self::McReps,
            _ => return None,
        })
    }

    pub const fn integer_role(self) -> LassoOptionRole {
        match self {
            Self::Cv | Self::DfMax | Self::MaxIter | Self::McReps | Self::NumLambda => {
                LassoOptionRole::StructuralControl
            }
            Self::Intercept | Self::Standardize | Self::UseCovariance => {
                LassoOptionRole::BooleanControl
            }
            _ => LassoOptionRole::FloatingParameter,
        }
    }
}
