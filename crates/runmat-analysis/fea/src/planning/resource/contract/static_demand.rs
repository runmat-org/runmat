//! Mandatory producer metadata reservation on the whole-step structural plan.
use super::{PreparedOutputResourceFacts, StructuralOperatorResourceDemand};
use serde::{Deserialize, Serialize};

/// Whole-step static numerical envelope plus the mandatory portable producer catalog.
/// Direct numerical calls reserve the same envelope, even when no transport is requested.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearPipelineResourceDemand {
    pub operator: StructuralOperatorResourceDemand,
    pub state_profiles_output: PreparedOutputResourceFacts,
}
