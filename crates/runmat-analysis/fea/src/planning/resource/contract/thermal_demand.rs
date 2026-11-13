//! Resource demand for the monolithic thermal solve and its prepared outputs.

use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::execution::AnalysisWorkloadIdentity;

use super::{ByteCount, Count, PreparedInputResourceFacts, PreparedOutputResourceFacts};

/// Original retained field-source closure and additional no-solve authentication workspace.
/// Decoded input storage is included in inputs, not counted again as numerical solve storage.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalFieldSourceResourceFacts {
    pub identity: Digest,
    pub encoded_payload_bytes: ByteCount,
    pub validation_workspace_bytes: ByteCount,
}

/// The family-owned numerical reservation is established before assembly and is separate from
/// imported inputs and result/receipt transport construction.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalSolveResourceDemand {
    pub workload: AnalysisWorkloadIdentity,
    pub topology_digest: Digest,
    pub numerical_reservation_bytes: ByteCount,
    pub sample_count: Count,
    pub field_source: Option<ThermalFieldSourceResourceFacts>,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
}
