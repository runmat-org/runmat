//! Exact edge/gauge sparse support and shared GMRES recurrence; no SPD proof.
use super::*;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagneticGmresResourceEnvelope {
    pub dimension: Count,
    pub restart: Count,
    pub maximum_iterations: Count,
    pub resident_bytes: ByteCount,
    pub operator_applications: Count,
    pub dot_calls: Count,
    pub axpy_calls: Count,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagnetostaticResourceDemand {
    pub workload: AnalysisWorkloadIdentity,
    pub topology_digest: Digest,
    pub system_plan_identity: Digest,
    pub node_count: Count,
    pub element_count: Count,
    pub edge_count_upper_bound: Count,
    pub quadrature_sample_count: Count,
    pub free_edge_count: Count,
    pub gauge_node_count: Count,
    pub physical_nonzeros: Count,
    pub saddle_nonzeros: Count,
    pub krylov: MagneticGmresResourceEnvelope,
    pub numerical_reservation_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub state_profiles_resident_bytes: ByteCount,
}
