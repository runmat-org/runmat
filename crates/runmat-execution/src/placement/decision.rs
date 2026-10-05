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
        PlacementOperationId::new(&self.operation)?;
        if self
            .revision
            .program
            .as_ref()
            .is_some_and(|program| program.validate().is_err())
        {
            return Err("placement program revision is invalid");
        }
        Ok(())
    }

    pub fn cache_key(&self) -> Digest {
        Digest::sha256_serialized(self).expect("placement signature serializes")
    }

    /// Compact invalidation facts for an in-flight selection. The program
    /// revision uses its canonical identity rather than cloning its retained
    /// domain-contribution names into each observation registration.
    pub fn permit_identity(&self) -> Result<PlacementPermitIdentity, crate::ContractError> {
        Ok(PlacementPermitIdentity {
            signature: self.cache_key(),
            program: self
                .revision
                .program
                .as_ref()
                .map(|revision| revision.identity_digest())
                .transpose()?,
            authority: self.revision.authority,
            policy: self.revision.policy,
        })
    }
}

/// The validated operation spelling retained by a local placement decision.
/// The portable signature still owns its String at the wire/profile boundary.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PlacementOperationId {
    bytes: [u8; 128],
    length: u8,
}

impl PlacementOperationId {
    pub fn new(operation: &str) -> Result<Self, &'static str> {
        if operation.is_empty() || operation.len() > 128 || operation.chars().any(char::is_control)
        {
            return Err("placement operation must be 1..=128 bytes without control characters");
        }
        let mut bytes = [0; 128];
        bytes[..operation.len()].copy_from_slice(operation.as_bytes());
        Ok(Self {
            bytes,
            length: operation.len() as u8,
        })
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..usize::from(self.length)])
            .expect("validated operation retains UTF-8 bytes")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementPermitIdentity {
    pub signature: Digest,
    pub program: Option<Digest>,
    pub authority: PlacementExecutionAuthorityRevision,
    pub policy: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedExecutionCandidate {
    pub node: String,
    pub candidate: String,
    pub kind: ExecutionCandidateKind,
}

/// Allocation-free local identity for a candidate selected by the same
/// whole-work policy. Presentation Strings are built only at a portable
/// decision/feedback boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactSelectedExecutionCandidate {
    pub node: PlacementOperationId,
    pub candidate: Digest,
    pub kind: ExecutionCandidateKind,
}

impl CompactSelectedExecutionCandidate {
    pub fn portable(self) -> SelectedExecutionCandidate {
        SelectedExecutionCandidate {
            node: self.node.as_str().to_owned(),
            candidate: self.candidate.to_string(),
            kind: self.kind,
        }
    }
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

    pub fn matches_permit_identity(&self, identity: &PlacementPermitIdentity) -> bool {
        match self {
            Self::All => true,
            Self::Program { revision } => revision
                .identity_digest()
                .is_ok_and(|digest| identity.program == Some(digest)),
            Self::Provider { digest } => {
                identity.authority == PlacementExecutionAuthorityRevision::Provider(*digest)
            }
            Self::HostService { digest } => {
                identity.authority == PlacementExecutionAuthorityRevision::HostService(*digest)
            }
            Self::Policy { digest } => identity.policy == *digest,
            Self::Signature { key } => identity.signature == *key,
        }
    }
}
