//! Admission of the final result receipt, distinct from its numerical source plan.
use super::{ByteCount, PreparedInputResourceFacts, PreparedOutputResourceFacts};
use crate::execution::{PublicationRequestIdentity, SolverResultIdentity};
use crate::planning::resource::ResourceUsageEvidenceIdentity;
use crate::results::accuracy::AccuracyArtifactIdentity;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultPublicationResourceDemand {
    pub request: PublicationRequestIdentity,
    pub source_result: SolverResultIdentity,
    pub source_accuracy: AccuracyArtifactIdentity,
    pub source_usage: ResourceUsageEvidenceIdentity,
    pub receipt_encoded_bytes_upper_bound: ByteCount,
    pub receipt_materialization_bytes: ByteCount,
    pub accounting_interval_workspace_bytes: ByteCount,
    pub nested_plan_workspace_bytes: ByteCount,
    pub observation_task_index_workspace_bytes: ByteCount,
    pub observation_execution_workspace_bytes: ByteCount,
    pub source_validation_workspace_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}
