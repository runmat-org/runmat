use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::execution::StructuralTransientBlockRequestIdentity;

use super::{
    Count, PreparedInputResourceFacts, PreparedOutputResourceFacts,
    StructuralOperatorResourceDemand,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientBlockResourceDemand {
    pub request: StructuralTransientBlockRequestIdentity,
    pub scope_digest: Digest,
    pub time_point_count: Count,
    pub operator: StructuralOperatorResourceDemand,
    pub inputs: PreparedInputResourceFacts,
    pub segment_output: PreparedOutputResourceFacts,
}
