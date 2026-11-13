//! Canonical semantic identities for performance-curve controls.

fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| !ch.is_ascii_whitespace() && *ch != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerfcurveOptionName {
    XCrit,
    YCrit,
    NegClass,
    Weights,
    Prior,
    Cost,
    TVals,
    XVals,
    UseNearest,
    ProcessNaN,
    UnsupportedBootstrap,
}

impl PerfcurveOptionName {
    pub fn from_public(text: &str) -> Option<Self> {
        Some(match canonical(text).as_str() {
            "xcrit" => Self::XCrit,
            "ycrit" => Self::YCrit,
            "negclass" => Self::NegClass,
            "weights" => Self::Weights,
            "prior" => Self::Prior,
            "cost" => Self::Cost,
            "tvals" => Self::TVals,
            "xvals" => Self::XVals,
            "usenearest" => Self::UseNearest,
            "processnan" => Self::ProcessNaN,
            "nboot" | "alpha" | "bootstrap" | "bootstraps" | "nbootstraps" => {
                Self::UnsupportedBootstrap
            }
            _ => return None,
        })
    }

    pub fn canonical_spelling(text: &str) -> String {
        canonical(text)
    }

    pub const fn is_floating_numeric_control(self) -> bool {
        matches!(
            self,
            Self::Weights | Self::Prior | Self::Cost | Self::TVals | Self::XVals
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerfcurveCriterion {
    Fpr,
    Tpr,
    Tnr,
    Fnr,
    Precision,
    Npv,
    Accuracy,
    TruePositive,
    FalseNegative,
    FalsePositive,
    TrueNegative,
    PredictedPositive,
    RatePredictedPositive,
    RatePredictedNegative,
    ExpectedCost,
}

impl PerfcurveCriterion {
    pub fn from_public(text: &str) -> Option<Self> {
        Some(match canonical(text).as_str() {
            "fpr" | "fallout" => Self::Fpr,
            "tpr" | "sens" | "sensitivity" | "recall" | "reca" => Self::Tpr,
            "tnr" | "spec" | "specificity" => Self::Tnr,
            "fnr" | "miss" => Self::Fnr,
            "ppv" | "precision" | "prec" => Self::Precision,
            "npv" => Self::Npv,
            "accu" | "accuracy" | "acc" => Self::Accuracy,
            "tp" => Self::TruePositive,
            "fn" => Self::FalseNegative,
            "fp" => Self::FalsePositive,
            "tn" => Self::TrueNegative,
            "tpfp" | "tp+fp" => Self::PredictedPositive,
            "rpp" => Self::RatePredictedPositive,
            "rnp" => Self::RatePredictedNegative,
            "ecost" => Self::ExpectedCost,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerfcurveProcessNaN {
    Ignore,
    AddToFalse,
}

impl PerfcurveProcessNaN {
    pub fn from_public(text: &str) -> Option<Self> {
        Some(match canonical(text).as_str() {
            "ignore" => Self::Ignore,
            "addtofalse" => Self::AddToFalse,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerfcurvePriorKeyword {
    Empirical,
    Uniform,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerfcurveBoolean {
    False,
    True,
}

impl PerfcurveBoolean {
    pub fn from_public(text: &str) -> Option<Self> {
        Some(match canonical(text).as_str() {
            "true" | "on" | "yes" => Self::True,
            "false" | "off" | "no" => Self::False,
            _ => return None,
        })
    }

    pub const fn value(self) -> bool {
        matches!(self, Self::True)
    }
}

impl PerfcurvePriorKeyword {
    pub fn from_public(text: &str) -> Option<Self> {
        Some(match canonical(text).as_str() {
            "empirical" => Self::Empirical,
            "uniform" => Self::Uniform,
            _ => return None,
        })
    }
}
