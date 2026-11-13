//! Dependency-neutral real domains used by numeric call contracts.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealUnaryPredicate {
    /// Both endpoints of [-1, 1] are included.
    ClosedUnitInterval,
    AtLeastOne,
    Nonnegative,
    AtLeastMinusOne,
}

/// Evidence for an exact invocation's represented values. Indeterminate is
/// not a real-domain or complex-domain promise (for example NaN range probes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealUnaryClassification {
    Inside,
    Outside,
    Indeterminate,
}

impl RealUnaryPredicate {
    /// Classify the represented native values, without narrowing Double inputs.
    /// An outside value dominates NaN; NaN alone cannot establish a real domain.
    pub fn classify(self, values: impl IntoIterator<Item = f64>) -> RealUnaryClassification {
        let mut result = RealUnaryClassification::Inside;
        for value in values {
            if value.is_nan() {
                result = RealUnaryClassification::Indeterminate;
                continue;
            }
            let inside = match self {
                Self::Nonnegative => value >= 0.0,
                Self::AtLeastMinusOne => value >= -1.0,
                Self::AtLeastOne => value >= 1.0,
                Self::ClosedUnitInterval => (-1.0..=1.0).contains(&value),
            };
            if !inside {
                return RealUnaryClassification::Outside;
            }
        }
        result
    }
}
