//! Exact electric-phasor complex block support and shared general GMRES counts.
use super::*;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellGmresResourceEnvelope {
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
pub struct MaxwellHarmonicResourceDemand {
    pub workload: AnalysisWorkloadIdentity,
    pub topology_digest: Digest,
    pub system_plan_identity: Digest,
    pub node_count: Count,
    pub element_count: Count,
    pub edge_count_upper_bound: Count,
    pub quadrature_sample_count: Count,
    pub free_edge_count: Count,
    pub real_block_dimension: Count,
    pub block_nonzeros: Count,
    pub krylov: MaxwellGmresResourceEnvelope,
    pub numerical_reservation_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub state_profiles_resident_bytes: ByteCount,
}
