//! Generalized linear-model distribution and option identities.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LassoglmDistribution {
    Normal,
    Binomial,
    Poisson,
}

impl LassoglmDistribution {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match normalized(name).as_str() {
            "normal" | "gaussian" => Self::Normal,
            "binomial" => Self::Binomial,
            "poisson" => Self::Poisson,
            _ => return None,
        })
    }

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Binomial => "binomial",
            Self::Poisson => "poisson",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LassoglmLink {
    Identity,
    Logit,
    Log,
}

impl LassoglmLink {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match normalized(name).as_str() {
            "identity" => Self::Identity,
            "logit" => Self::Logit,
            "log" => Self::Log,
            _ => return None,
        })
    }

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Logit => "logit",
            Self::Log => "log",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LassoglmEstimateDisp {
    Off,
}

impl LassoglmEstimateDisp {
    pub fn from_public_name(name: &str) -> Option<Self> {
        (normalized(name) == "off").then_some(Self::Off)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LassoglmOption {
    Alpha,
    Lambda,
    LambdaRatio,
    NumLambda,
    Standardize,
    Intercept,
    Weights,
    Offset,
    BinomialSize,
    RelTol,
    MaxIter,
    Cv,
    PredictorNames,
    Options,
    Link,
    EstDisp,
    McReps,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LassoglmOptionRole {
    FloatingParameter,
    StructuralControl,
    BooleanControl,
    Aggregate,
}

impl LassoglmOption {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match normalized(name).as_str() {
            "alpha" => Self::Alpha,
            "lambda" => Self::Lambda,
            "lambdaratio" => Self::LambdaRatio,
            "numlambda" => Self::NumLambda,
            "standardize" => Self::Standardize,
            "intercept" | "constant" => Self::Intercept,
            "weights" => Self::Weights,
            "offset" => Self::Offset,
            "binomialsize" => Self::BinomialSize,
            "reltol" | "tolx" | "tolfun" => Self::RelTol,
            "maxiter" => Self::MaxIter,
            "cv" => Self::Cv,
            "predictornames" => Self::PredictorNames,
            "options" => Self::Options,
            "link" => Self::Link,
            "estdisp" => Self::EstDisp,
            "mcreps" => Self::McReps,
            _ => return None,
        })
    }

    pub const fn integer_role(self) -> LassoglmOptionRole {
        match self {
            Self::Cv | Self::MaxIter | Self::McReps | Self::NumLambda => {
                LassoglmOptionRole::StructuralControl
            }
            Self::Intercept | Self::Standardize => LassoglmOptionRole::BooleanControl,
            Self::Options => LassoglmOptionRole::Aggregate,
            _ => LassoglmOptionRole::FloatingParameter,
        }
    }
}

fn normalized(name: &str) -> String {
    name.chars()
        .filter(|ch| *ch != '_' && *ch != '-')
        .flat_map(char::to_lowercase)
        .collect()
}
