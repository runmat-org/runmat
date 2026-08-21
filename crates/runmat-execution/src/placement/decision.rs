use runmat_types::RegionId;
use serde::{Deserialize, Serialize};

use crate::{Digest, ProgramRevision};

use super::ExecutionCandidateKind;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementRevision {
    pub program: Option<ProgramRevision>,
    pub catalog: Digest,
    pub compiler: Digest,
    pub provider: Digest,
    pub policy: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementSignature {
    pub region: Option<RegionId>,
    pub operation: String,
    /// Digest of exact shapes, representations, layouts, residency, requested
    /// outputs, and guards observed at dispatch.
    pub runtime_facts: Digest,
    pub revision: PlacementRevision,
}

impl PlacementSignature {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.operation.is_empty()
            || self.operation.len() > 128
            || self.operation.chars().any(char::is_control)
        {
            return Err("placement operation must be 1..=128 bytes without control characters");
        }
        Ok(())
    }

    pub fn cache_key(&self) -> Digest {
        let encoded = serde_json::to_vec(self)
            .expect("placement signature consists only of infallible serializable fields");
        Digest::sha256(encoded)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedExecutionCandidate {
    pub node: String,
    pub candidate: String,
    pub kind: ExecutionCandidateKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementDecision {
    pub signature: PlacementSignature,
    pub selections: Vec<SelectedExecutionCandidate>,
    pub predicted_total_ns: u64,
    pub from_cache: bool,
    pub used_local_fallback: bool,
    pub explored_states: u32,
    pub pruned_states: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementFeedback {
    pub signature: PlacementSignature,
    pub candidate: String,
    /// End-to-end elapsed time for this exact candidate/signature. Detailed
    /// stage timings belong to executor/provider observations and must not be
    /// invented when only the complete duration is known.
    pub total_elapsed_ns: u64,
    pub succeeded: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum PlacementInvalidation {
    All,
    Program { revision: ProgramRevision },
    Provider { digest: Digest },
    Policy { digest: Digest },
    Signature { key: Digest },
}
