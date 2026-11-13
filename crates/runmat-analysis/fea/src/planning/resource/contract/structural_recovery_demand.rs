use super::*;
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFieldRecoveryResourceDemand {
    pub request: FieldRecoveryRequestIdentity,
    pub source_result: SolverResultIdentity,
    pub topology_digest: Digest,
    pub constraint_plan_digest: Digest,
    pub constraint_compilation_peak_bytes: ByteCount,
    pub construction_workspace_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}
