use super::*;
/// Retained native acoustic field copy: no solve, only source validation and partitioning.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticFieldRecoveryResourceDemand {
    pub request: FieldRecoveryRequestIdentity,
    pub source_result: SolverResultIdentity,
    pub topology_digest: Digest,
    pub construction_workspace_bytes: ByteCount,
    pub source_validation_workspace_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}
