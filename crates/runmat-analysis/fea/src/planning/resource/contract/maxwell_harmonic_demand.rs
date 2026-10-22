//! Exact electric-phasor complex support and the sparse direct factorization envelope.
use super::*;

/// Sparse LU of the complex edge operator (plus terminal-drive unknowns).
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellDirectFactorEnvelope {
    pub dimension: Count,
    pub nonzeros: Count,
    /// Upper bound of L and U complex entries (nested-dissection n^(4/3) growth).
    pub factor_entries_upper_bound: Count,
    pub resident_bytes: ByteCount,
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
    pub complex_dimension: Count,
    pub nonzeros: Count,
    pub factorization: MaxwellDirectFactorEnvelope,
    pub numerical_reservation_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub state_profiles_resident_bytes: ByteCount,
}
