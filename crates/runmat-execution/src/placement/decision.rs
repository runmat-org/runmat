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
    pub authority: PlacementExecutionAuthorityRevision,
    pub policy: Digest,
}

/// The execution authority whose capability revision makes one complete-route
/// sample reusable. Host services are not GPU providers and cannot be
/// invalidated as though their digests were provider capability snapshots.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "digest")]
pub enum PlacementExecutionAuthorityRevision {
    Provider(Digest),
    HostService(Digest),
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
        Digest::sha256_serialized(self)
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
    pub kind: ExecutionCandidateKind,
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
    HostService { digest: Digest },
    Policy { digest: Digest },
    Signature { key: Digest },
}

impl PlacementInvalidation {
    /// The same typed match governs cached feedback, pending receipts, and
    /// still-running whole-work observation permits.
    pub fn matches_signature(&self, signature: &PlacementSignature) -> bool {
        match self {
            Self::All => true,
            Self::Program { revision } => signature.revision.program.as_ref() == Some(revision),
            Self::Provider { digest } => {
                signature.revision.authority
                    == PlacementExecutionAuthorityRevision::Provider(*digest)
            }
            Self::HostService { digest } => {
                signature.revision.authority
                    == PlacementExecutionAuthorityRevision::HostService(*digest)
            }
            Self::Policy { digest } => signature.revision.policy == *digest,
            Self::Signature { key } => signature.cache_key() == *key,
        }
    }
}
