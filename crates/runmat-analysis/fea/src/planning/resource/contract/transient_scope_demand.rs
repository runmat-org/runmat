use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    PreparedInputResourceFacts, PreparedOutputResourceFacts, StructuralOperatorResourceDemand,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientScopeResourceDemand {
    pub source_workload_digest: Digest,
    pub operator: StructuralOperatorResourceDemand,
    pub inputs: PreparedInputResourceFacts,
    pub scope_output: PreparedOutputResourceFacts,
}
