//! Mandatory producer metadata reservation on the whole-step structural plan.
use super::{
    PreparedInputResourceFacts, PreparedOutputResourceFacts, StructuralOperatorResourceDemand,
};
use serde::{Deserialize, Serialize};

/// Whole-step static numerical envelope plus the mandatory portable producer catalog.
/// Direct numerical calls reserve the same envelope, even when no transport is requested.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearPipelineResourceDemand {
    /// Fresh original compiler phase; cold values are claims rederived by execution.
    pub constraint_compilation_peak_bytes: super::ByteCount,
    pub operator: StructuralOperatorResourceDemand,
    /// Independently derived transport facts; absent only for direct numerical execution.
    pub inputs: Option<PreparedInputResourceFacts>,
    pub state_profiles_output: PreparedOutputResourceFacts,
}
