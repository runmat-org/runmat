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
