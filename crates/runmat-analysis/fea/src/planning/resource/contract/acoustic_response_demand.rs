use super::*;
/// Cavity-mode or pressure-transient solve on the canonical acoustic operators.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticResponseResourceDemand {
    pub workload: AnalysisWorkloadIdentity,
    pub topology_digest: Digest,
    pub node_count: Count,
    pub element_count: Count,
    pub sample_count: Count,
    pub operator_support_upper_bound: Count,
    pub numerical_reservation_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub state_profiles_resident_bytes: ByteCount,
}
