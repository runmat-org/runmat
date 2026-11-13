use super::*;
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearFieldRecoveryResourceDemand {
    pub request: FieldRecoveryRequestIdentity,
    pub source_result: SolverResultIdentity,
    pub topology_digest: Digest,
    pub construction_workspace_bytes: ByteCount,
    pub recovery_workspace_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}
