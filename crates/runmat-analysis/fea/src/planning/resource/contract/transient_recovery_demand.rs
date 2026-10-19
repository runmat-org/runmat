use serde::{Deserialize, Serialize};

use crate::execution::SolverResultIdentity;

use super::{ByteCount, Count, PreparedInputResourceFacts, StructuralOperatorResourceDemand};

/// Host-side recovery of one completed, identity-bound transient numerical result.
/// This phase produces the public run result, not a prepared numerical object set.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientFieldRecoveryResourceDemand {
    pub result: SolverResultIdentity,
    pub sample_count: Count,
    pub physical_dof_count: Count,
    pub decoded_sample_bytes_upper_bound: ByteCount,
    pub retained_field_bytes_upper_bound: ByteCount,
    pub operator: StructuralOperatorResourceDemand,
    pub inputs: PreparedInputResourceFacts,
}
