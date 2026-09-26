//! Resource demand for the monolithic thermal solve and its prepared outputs.

use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::execution::AnalysisWorkloadIdentity;

use super::{ByteCount, Count, PreparedInputResourceFacts, PreparedOutputResourceFacts};

/// The family-owned numerical reservation is established before assembly and is separate from
/// imported inputs and result/receipt transport construction.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalSolveResourceDemand {
    pub workload: AnalysisWorkloadIdentity,
    pub topology_digest: Digest,
    pub numerical_reservation_bytes: ByteCount,
    pub sample_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
}
