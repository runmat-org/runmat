use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{ByteCount, Count, PreparedInputResourceFacts, PreparedOutputResourceFacts};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticFrequencyAssemblyResourceDemand {
    pub workload: crate::execution::AnalysisWorkloadIdentity,
    pub topology_digest: Digest,
    pub entry_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub workspace_bytes_upper_bound: ByteCount,
    pub result_output: PreparedOutputResourceFacts,
    pub publication_output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyResultAssemblyResourceDemand {
    pub scope_digest: Digest,
    pub entry_count: Count,
    pub entry_encoded_bytes: ByteCount,
    pub physical_dof_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub ordering_workspace_bytes_upper_bound: ByteCount,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientResultAssemblyResourceDemand {
    pub scope_digest: Digest,
    pub segment_count: Count,
    pub sample_count: Count,
    pub physical_dof_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub ordering_workspace_bytes_upper_bound: ByteCount,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalResponseSpectrumAssemblyResourceDemand {
    pub scope_digest: Digest,
    pub direction_count: Count,
    pub direction_encoded_bytes: ByteCount,
    pub physical_dof_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub ordering_workspace_bytes_upper_bound: ByteCount,
    pub field_recovery_workspace_bytes_upper_bound: ByteCount,
    pub missing_mass_static_solve_count: Count,
    pub missing_mass_reduced_dof_count: Count,
    pub missing_mass_system_workspace_bytes_upper_bound: ByteCount,
    pub missing_mass_total_iteration_count_upper_bound: Count,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub field_recovery_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRandomResponseAssemblyResourceDemand {
    pub scope_digest: Digest,
    pub task_count: Count,
    pub interval_count: Count,
    pub interval_encoded_bytes: ByteCount,
    pub response_value_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub ordering_workspace_bytes_upper_bound: ByteCount,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
}
