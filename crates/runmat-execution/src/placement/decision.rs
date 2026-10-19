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
    /// Valid only within its owning PlacementSession; never profile-persisted.
    RuntimeBindingSession(Digest),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementSignature {
    pub region: Option<RegionId>,
    pub operation: String,
    /// Digest of exact shapes, representations, layouts, residency, requested
    /// outputs, and guards observed at dispatch.
    pub runtime_facts: Digest,
    /// Provider capability namespace for typed invalidation of a session-bound
    /// decision. It is not an execution permit or a provider instance handle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_provenance: Option<Digest>,
    /// Canonical public-output roles and graph joins for multi-output plans.
    /// Absent on legacy one-output selections, preserving their wire identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_output_authority: Option<Digest>,
    pub revision: PlacementRevision,
}

/// Borrowed semantic revision for a managed invocation. Portable profile and
/// wire representations continue to own `PlacementRevision`; this view
/// serializes with the same field names/order without cloning a program's
/// domain-contribution vector at each selection.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlacementRevisionRef<'a> {
    pub program: Option<&'a ProgramRevision>,
    pub catalog: Digest,
    pub compiler: Digest,
    pub authority: PlacementExecutionAuthorityRevision,
    pub policy: Digest,
}

impl<'a> From<&'a PlacementRevision> for PlacementRevisionRef<'a> {
    fn from(revision: &'a PlacementRevision) -> Self {
        Self {
            program: revision.program.as_ref(),
            catalog: revision.catalog,
            compiler: revision.compiler,
            authority: revision.authority,
            policy: revision.policy,
        }
    }
}

/// Allocation-free signature view for one managed selection. It is not a
/// second wire schema: its serialized bytes and permit digest must match the
/// owned portable signature exactly.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlacementSignatureRef<'a> {
    pub region: Option<RegionId>,
    pub operation: &'a str,
    pub runtime_facts: Digest,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_provenance: Option<Digest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_output_authority: Option<Digest>,
    pub revision: PlacementRevisionRef<'a>,
}

impl<'a> From<&'a PlacementSignature> for PlacementSignatureRef<'a> {
    fn from(signature: &'a PlacementSignature) -> Self {
        Self {
            region: signature.region,
            operation: &signature.operation,
            runtime_facts: signature.runtime_facts,
            provider_provenance: signature.provider_provenance,
            graph_output_authority: signature.graph_output_authority,
            revision: (&signature.revision).into(),
        }
    }
}

impl PlacementSignatureRef<'_> {
    pub fn validate(self) -> Result<(), &'static str> {
        PlacementOperationId::new(self.operation)?;
        if self
            .revision
            .program
            .is_some_and(|program| program.validate().is_err())
        {
            return Err("placement program revision is invalid");
        }
        Ok(())
    }

    pub fn cache_key(self) -> Digest {
        Digest::sha256_serialized(&self).expect("placement signature view serializes")
    }

    pub fn permit_identity(self) -> Result<PlacementPermitIdentity, crate::ContractError> {
        Ok(PlacementPermitIdentity {
            signature: self.cache_key(),
            program: self
                .revision
                .program
                .map(ProgramRevision::identity_digest)
                .transpose()?,
            authority: self.revision.authority,
            provider_provenance: self.provider_provenance,
            policy: self.revision.policy,
        })
    }

    pub fn to_owned(&self) -> PlacementSignature {
        PlacementSignature {
            region: self.region,
            operation: self.operation.to_owned(),
            runtime_facts: self.runtime_facts,
            provider_provenance: self.provider_provenance,
            graph_output_authority: self.graph_output_authority,
            revision: PlacementRevision {
                program: self.revision.program.cloned(),
                catalog: self.revision.catalog,
                compiler: self.revision.compiler,
                authority: self.revision.authority,
                policy: self.revision.policy,
            },
        }
    }
}

impl PlacementSignature {
    pub fn validate(&self) -> Result<(), &'static str> {
        PlacementSignatureRef::from(self).validate()
    }

    pub fn cache_key(&self) -> Digest {
        PlacementSignatureRef::from(self).cache_key()
    }

    /// Compact invalidation facts for an in-flight selection. The program
    /// revision uses its canonical identity rather than cloning its retained
    /// domain-contribution names into each observation registration.
    pub fn permit_identity(&self) -> Result<PlacementPermitIdentity, crate::ContractError> {
        PlacementSignatureRef::from(self).permit_identity()
    }
}

#[cfg(test)]
mod borrowed_signature_tests {
    use super::*;
    use crate::{DomainContribution, ProgramEnvironment};

    #[test]
    fn borrowed_and_owned_signature_identity_share_exact_wire_bytes_with_program_domains() {
        let program = ProgramRevision::new(
            Digest::sha256(b"graph"),
            Digest::sha256(b"source"),
            ProgramEnvironment::new(
                2,
                2,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                "runmat",
            )
            .unwrap(),
        )
        .unwrap()
        .with_domain_contribution(
            DomainContribution::new("filesystem", Digest::sha256(b"provider")).unwrap(),
        )
        .unwrap();
        let owned = PlacementSignature {
            region: Some(RegionId {
                function: runmat_types::ProgramFunctionId(5),
                ordinal: 7,
            }),
            operation: "readmatrix".into(),
            runtime_facts: Digest::sha256(b"numeric-text"),
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: Some(program),
                catalog: Digest::sha256(b"placement-catalog"),
                compiler: Digest::sha256(b"placement-compiler"),
                authority: PlacementExecutionAuthorityRevision::HostService(Digest::sha256(
                    b"native-file",
                )),
                policy: Digest::sha256(b"policy"),
            },
        };
        let borrowed = PlacementSignatureRef::from(&owned);
        assert_eq!(
            serde_json::to_vec(&borrowed).unwrap(),
            serde_json::to_vec(&owned).unwrap()
        );
        assert_eq!(borrowed.cache_key(), owned.cache_key());
        assert_eq!(
            borrowed.cache_key(),
            Digest::sha256_serialized(&owned).unwrap()
        );
        assert_eq!(
            borrowed.permit_identity().unwrap(),
            owned.permit_identity().unwrap()
        );
        assert_eq!(borrowed.to_owned(), owned);

        let without_program = PlacementSignature {
            region: None,
            operation: "host.effect".into(),
            runtime_facts: Digest::sha256(b"effect"),
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                catalog: Digest::sha256(b"catalog"),
                compiler: Digest::sha256(b"compiler"),
                authority: PlacementExecutionAuthorityRevision::Provider(Digest::sha256(
                    b"provider",
                )),
                policy: Digest::sha256(b"policy"),
            },
        };
        let borrowed = PlacementSignatureRef::from(&without_program);
        assert_eq!(
            serde_json::to_vec(&borrowed).unwrap(),
            serde_json::to_vec(&without_program).unwrap()
        );
        assert_eq!(
            borrowed.cache_key(),
            Digest::sha256_serialized(&without_program).unwrap()
        );
        assert_eq!(
            borrowed.permit_identity().unwrap(),
            without_program.permit_identity().unwrap()
        );
    }

    #[test]
    fn absent_provider_provenance_preserves_legacy_wire_and_present_value_separates_keys() {
        #[derive(serde::Serialize)]
        struct LegacySignature<'a> {
            region: Option<RegionId>,
            operation: &'a str,
            runtime_facts: Digest,
            #[serde(skip_serializing_if = "Option::is_none")]
            graph_output_authority: Option<Digest>,
            revision: PlacementRevisionRef<'a>,
        }

        let digest = Digest::sha256(b"provider namespace");
        let mut current = PlacementSignature {
            region: None,
            operation: "fusion.reduction".into(),
            runtime_facts: digest,
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                catalog: digest,
                compiler: digest,
                authority: PlacementExecutionAuthorityRevision::RuntimeBindingSession(digest),
                policy: digest,
            },
        };
        let legacy = LegacySignature {
            region: current.region,
            operation: &current.operation,
            runtime_facts: current.runtime_facts,
            graph_output_authority: current.graph_output_authority,
            revision: (&current.revision).into(),
        };
        let legacy_bytes = serde_json::to_vec(&legacy).unwrap();
        assert_eq!(serde_json::to_vec(&current).unwrap(), legacy_bytes);
        assert_eq!(current.cache_key(), Digest::sha256(&legacy_bytes));
        assert_eq!(
            serde_json::from_slice::<PlacementSignature>(&legacy_bytes).unwrap(),
            current
        );

        current.provider_provenance = Some(digest);
        assert_ne!(current.cache_key(), Digest::sha256(&legacy_bytes));
        assert_eq!(
            current.permit_identity().unwrap().provider_provenance,
            Some(digest)
        );
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
    pub provider_provenance: Option<Digest>,
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
    RuntimeBindingSession { digest: Digest },
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
                    || signature.provider_provenance == Some(*digest)
            }
            Self::HostService { digest } => {
                signature.revision.authority
                    == PlacementExecutionAuthorityRevision::HostService(*digest)
            }
            Self::RuntimeBindingSession { digest } => {
                signature.revision.authority
                    == PlacementExecutionAuthorityRevision::RuntimeBindingSession(*digest)
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
                    || identity.provider_provenance == Some(*digest)
            }
            Self::HostService { digest } => {
                identity.authority == PlacementExecutionAuthorityRevision::HostService(*digest)
            }
            Self::RuntimeBindingSession { digest } => {
                identity.authority
                    == PlacementExecutionAuthorityRevision::RuntimeBindingSession(*digest)
            }
            Self::Policy { digest } => identity.policy == *digest,
            Self::Signature { key } => identity.signature == *key,
        }
    }
}
