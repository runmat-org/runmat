//! Canonical CPU scalar electrical admission; no semantic publication is implied.
use super::*;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalStaticResourceDemand {
    pub workload: AnalysisWorkloadIdentity,
    pub topology_digest: Digest,
    pub node_count: Count,
    pub element_count: Count,
    pub operator_nonzeros_upper_bound: Count,
    /// Additive requested-heap estimate, including the shared bounded coercivity proof.
    pub numerical_reservation_bytes: ByteCount,
    pub solver_admission: SolverAdmissionResourcePolicy,
    /// The planned solve: sparse Cholesky when true, AMG-preconditioned CG otherwise.
    pub direct_solve: bool,
    /// The effective memory budget the solve was chosen under: min(authored ceiling, machine).
    pub solver_memory_budget: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub state_profiles_resident_bytes: ByteCount,
}
