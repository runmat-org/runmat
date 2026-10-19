use super::{ByteCount, Count, PreparedInputResourceFacts, PreparedOutputResourceFacts};
use crate::execution::{AccuracyEvaluationRequestIdentity, SolverResultIdentity};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyEvaluationResourceDemand {
    pub request: AccuracyEvaluationRequestIdentity,
    pub source_result: SolverResultIdentity,
    pub criterion_count: Count,
    pub evidence_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub evaluator_workspace_bytes_upper_bound: ByteCount,
    pub output: PreparedOutputResourceFacts,
}
