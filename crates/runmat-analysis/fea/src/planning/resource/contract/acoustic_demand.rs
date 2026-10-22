//! Canonical scalar Helmholtz CPU reservation (certified direct factorization), distinct from structural operator demand.
use super::*;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticHarmonicResourceDemand {
    pub workload: AnalysisWorkloadIdentity,
    pub topology_digest: Digest,
    pub node_count: Count,
    pub element_count: Count,
    pub operator_nonzeros_upper_bound: Count,
    pub block_dimension_upper_bound: Count,
    pub recovery_row_support_upper_bound: Count,
    pub sparse_solve_reservation_bytes: ByteCount,
    /// Requested-heap admission estimate, not measured peak/RSS or an allocator guarantee.
    pub numerical_reservation_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub state_profiles_resident_bytes: ByteCount,
}
