//! Source-bound admission of one persisted semantic state.
use super::{ByteCount, PreparedInputResourceFacts, PreparedOutputResourceFacts};
use crate::execution::{SemanticStatePublicationSource, SolverResultIdentity};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticStatePublicationResourceDemand {
    /// SHA-256 of the exact canonical publication request.
    pub request: Digest,
    pub source: SemanticStatePublicationSource,
    pub source_result: SolverResultIdentity,
    pub source_payload_digest: Digest,
    pub topology_digest: Digest,
    pub profile_catalog: Digest,
    pub state_encoded_bytes_upper_bound: ByteCount,
    pub state_materialization_resident_bytes_estimate: ByteCount,
    /// Separate from retained decoded inputs: canonical source verification may reconstruct
    /// statistical assemblies while the original source remains live.
    pub source_validation_workspace_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}
