//! Executor-neutral evidence before one complete invocation becomes a numeric
//! placement graph. An absent cost or resource bound is never zero demand.

use std::collections::BTreeSet;
use std::io::{self, Write};

use super::{
    CandidateCostEvidence, CandidateExecutionLocation, CandidateOutputPresentation,
    CandidateOutputResidency, CandidatePreparationState, CandidateResourceDemand,
    ExecutionCandidateKind, PlacementGraphLimits, PlacementResourceSnapshot, PlacementSignature,
    PlacementSignatureRef, SelectedExecutionCandidate,
};
use crate::{Digest, DigestStream};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WholeWorkConstraint {
    AnyLegal,
    ProviderRequired { device_id: u32 },
}

/// One already legal, executable alternative for the same admitted semantic
/// workload. `route_facts` is candidate-specific: physical stages, transfer,
/// preparation and output presentation. The common `PlacementSignature`
/// identifies the workload shared by CPU and provider alternatives.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WholeWorkCandidate {
    pub kind: ExecutionCandidateKind,
    pub execution_location: CandidateExecutionLocation,
    pub preparation: CandidatePreparationState,
    pub output_residency: CandidateOutputResidency,
    pub presentation: CandidateOutputPresentation,
    pub route_facts: Digest,
    pub cost: CandidateCostEvidence,
    /// Complete-route resource demand. Unknown byte dimensions remain typed;
    /// provider allocation-limit preflight is separate from peak-memory facts.
    pub resources: CandidateResourceDemand,
    pub transactional_results: bool,
}

/// Read-only route identity: no resources, executor, or selection authority.
pub struct WholeWorkRouteIdentity<'a> {
    pub kind: ExecutionCandidateKind,
    pub execution_location: CandidateExecutionLocation,
    pub preparation: CandidatePreparationState,
    pub output_residency: CandidateOutputResidency,
    pub presentation: &'a CandidateOutputPresentation,
    pub route_facts: Digest,
}
impl WholeWorkRouteIdentity<'_> {
    pub fn digest(&self) -> Digest {
        let mut encoded = DigestStream::new();
        encoded.update(b"runmat.whole-work.candidate.v2\0");
        encoded.update(&[match self.kind {
            ExecutionCandidateKind::SharedRuntime => 0,
            ExecutionCandidateKind::OriginalVmRegion => 9,
            ExecutionCandidateKind::OriginalNativeRegion => 10,
            ExecutionCandidateKind::GenericNativeCpu => 1,
            ExecutionCandidateKind::SpecializedNativeCpu => 2,
            ExecutionCandidateKind::VectorizedNativeCpu => 3,
            ExecutionCandidateKind::CpuLibrary => 4,
            ExecutionCandidateKind::ProviderOperation => 5,
            ExecutionCandidateKind::ProviderLibrary => 6,
            ExecutionCandidateKind::ProviderGraph => 7,
            ExecutionCandidateKind::ProviderFusion => 8,
        }]);
        match self.execution_location {
            CandidateExecutionLocation::Host => encoded.update(&[0]),
            CandidateExecutionLocation::NativeSource => encoded.update(&[4]),
            CandidateExecutionLocation::Provider { device_id } => {
                encoded.update(&[1]);
                encoded.update(&device_id.to_le_bytes());
            }
            CandidateExecutionLocation::VmRegion { device_id } => {
                encoded.update(&[2]);
                encoded.update(&device_id.to_le_bytes());
            }
            CandidateExecutionLocation::NativeRegion { device_id } => {
                encoded.update(&[3]);
                encoded.update(&device_id.to_le_bytes());
            }
        }
        encoded.update(&[match self.preparation {
            CandidatePreparationState::Ready => 0,
            CandidatePreparationState::Warm => 1,
            CandidatePreparationState::Cold => 2,
            CandidatePreparationState::Preparing => 3,
            CandidatePreparationState::Unknown => 4,
        }]);
        match self.output_residency {
            CandidateOutputResidency::Host => encoded.update(&[0]),
            CandidateOutputResidency::Provider { device_id } => {
                encoded.update(&[1]);
                encoded.update(&device_id.to_le_bytes());
            }
            CandidateOutputResidency::Mirrored { device_id } => {
                encoded.update(&[2]);
                encoded.update(&device_id.to_le_bytes());
            }
            CandidateOutputResidency::ProviderAndHost {
                device_id,
                host_bytes,
            } => {
                encoded.update(&[5]);
                encoded.update(&device_id.to_le_bytes());
                serde_json::to_writer(&mut encoded, &host_bytes)
                    .expect("typed host bound serializes");
            }
            CandidateOutputResidency::Unknown => encoded.update(&[3]),
            CandidateOutputResidency::ConditionalHostOrProvider {
                device_id,
                host_bytes,
            } => {
                encoded.update(&[4]);
                encoded.update(&device_id.to_le_bytes());
                serde_json::to_writer(&mut encoded, &host_bytes)
                    .expect("typed host bound serializes");
            }
        }
        let mut stage_length = JsonLength::default();
        serde_json::to_writer(&mut stage_length, &self.presentation)
            .expect("typed presentation stage must serialize");
        encoded.update(&stage_length.0.to_le_bytes());
        serde_json::to_writer(&mut encoded, &self.presentation)
            .expect("typed presentation stage must serialize");
        encoded.update(self.route_facts.bytes());
        encoded.finish()
    }
}

impl WholeWorkCandidate {
    /// Opaque placement identity derived only from typed route facts. Neither
    /// user layer labels nor unstable allocation IDs are consulted by policy.
    pub fn selected(&self, node: &str) -> SelectedExecutionCandidate {
        SelectedExecutionCandidate {
            node: node.to_owned(),
            candidate: self.identity_digest().to_string(),
            kind: self.kind,
        }
    }

    /// The canonical route identity without the presentation String objects
    /// owned by a selected decision. This is useful for exact post-selection
    /// validation before another node/candidate String would be allocated.
    pub fn identity_digest(&self) -> Digest {
        WholeWorkRouteIdentity {
            kind: self.kind,
            execution_location: self.execution_location,
            preparation: self.preparation,
            output_residency: self.output_residency,
            presentation: &self.presentation,
            route_facts: self.route_facts,
        }
        .digest()
    }
    fn validate(&self) -> Result<(), &'static str> {
        if !self.kind.matches_location(self.execution_location) {
            return Err("whole-work candidate kind and location disagree");
        }
        self.presentation
            .validate_for(self.execution_location, self.output_residency)?;
        match (self.execution_location, self.output_residency) {
            (
                CandidateExecutionLocation::Provider {
                    device_id: execution,
                },
                CandidateOutputResidency::Provider { device_id: output }
                | CandidateOutputResidency::Mirrored { device_id: output }
                | CandidateOutputResidency::ProviderAndHost {
                    device_id: output, ..
                }
                | CandidateOutputResidency::ConditionalHostOrProvider {
                    device_id: output, ..
                },
            ) if execution != output => {
                return Err("whole-work output resides on a foreign device");
            }
            _ => {}
        }
        if let CandidateCostEvidence::Estimated(estimate) = self.cost {
            if estimate.checked_risk_adjusted_ns().is_none() {
                return Err("whole-work candidate risk-adjusted cost overflows");
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct JsonLength(u64);

impl Write for JsonLength {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::other("candidate presentation encoding is too long"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WholeWorkPlacementRequest {
    pub signature: PlacementSignature,
    pub constraint: WholeWorkConstraint,
    pub candidates: Vec<WholeWorkCandidate>,
    pub resources: PlacementResourceSnapshot,
    pub deterministic: bool,
    pub require_transactional_results: bool,
}

impl WholeWorkPlacementRequest {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.as_ref().validate()
    }

    pub fn as_ref(&self) -> WholeWorkPlacementRequestRef<'_> {
        WholeWorkPlacementRequestRef {
            signature: (&self.signature).into(),
            constraint: self.constraint,
            candidates: &self.candidates,
            resources: &self.resources,
            deterministic: self.deterministic,
            require_transactional_results: self.require_transactional_results,
        }
    }
}

/// Borrowed policy input. A caller with one route can pass a stack array,
/// while the portable owned request remains the serialization/wire carrier.
#[derive(Clone, Copy)]
pub struct WholeWorkPlacementRequestRef<'a> {
    pub signature: PlacementSignatureRef<'a>,
    pub constraint: WholeWorkConstraint,
    pub candidates: &'a [WholeWorkCandidate],
    pub resources: &'a PlacementResourceSnapshot,
    pub deterministic: bool,
    pub require_transactional_results: bool,
}

impl WholeWorkPlacementRequestRef<'_> {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.signature.validate()?;
        self.constraint
            .validate_candidates(self.candidates, self.resources)
    }
}

impl WholeWorkConstraint {
    /// Validate the same physical set without requiring an observation/session
    /// signature. Context-free selection gains no execution or learning grant.
    pub fn validate_candidates(
        self,
        candidates: &[WholeWorkCandidate],
        resources: &PlacementResourceSnapshot,
    ) -> Result<(), &'static str> {
        resources.validate()?;
        if resources.providers.len() > PlacementGraphLimits::default().max_providers as usize {
            return Err("whole-work resources exceed provider limit");
        }
        if candidates.is_empty()
            || candidates.len() > PlacementGraphLimits::default().max_candidates_per_node as usize
        {
            return Err("whole-work candidate count is invalid");
        }
        // The common provider/optional-VM pair needs no heap-backed set.
        // Larger candidate sets retain the existing duplicate policy below.
        let mut identities = BTreeSet::new();
        for candidate in candidates {
            candidate.validate()?;
            if let WholeWorkConstraint::ProviderRequired { device_id } = self {
                if !matches!(
                    candidate.execution_location,
                    CandidateExecutionLocation::Provider { device_id: selected }
                        | CandidateExecutionLocation::VmRegion { device_id: selected }
                        | CandidateExecutionLocation::NativeRegion { device_id: selected }
                        if selected == device_id
                ) {
                    return Err("provider-required work contains a host or foreign-device route");
                }
            }
            if candidates.len() > 2 && !identities.insert(candidate.identity_digest()) {
                return Err("whole-work candidate route identity is duplicated");
            }
        }
        if matches!(candidates, [left, right] if left.identity_digest() == right.identity_digest())
        {
            return Err("whole-work candidate route identity is duplicated");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WholeWorkDecisionKind {
    Compared,
    Exploratory,
    ProviderRequired,
    /// Only one legal route or only provider routes were offered, so no
    /// host-versus-provider profitability comparison was made.
    Uncompared,
    /// A legal route ran without sufficient cross-route pricing evidence
    /// (including when a cohort probe was already in flight). A valid selected
    /// route estimate and genuine completion observation remain usable.
    Unmeasured,
}

/// Portable result. A process-local probe lease belongs to the runtime service
/// adapter, not to this contract or its serialized placement evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WholeWorkDecision {
    pub signature: PlacementSignature,
    pub selected: SelectedExecutionCandidate,
    pub kind: WholeWorkDecisionKind,
    /// The selected route's valid estimate, when available. Missing prices
    /// and exploration never become a fabricated zero or huge number.
    pub predicted_total_ns: Option<u64>,
}

/// Outcome of one genuine invocation. Cancellation and abandonment are not
/// provider failures and must never enter the failure-count profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WholeWorkObservationOutcome {
    Success,
    OperationalFailure,
    Cancelled,
    Abandoned,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WholeWorkObservation {
    pub signature: PlacementSignature,
    pub selected: SelectedExecutionCandidate,
    pub elapsed_ns: u64,
    pub outcome: WholeWorkObservationOutcome,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_native_keeps_distinct_location_and_explicit_intent_excludes_both_cpu_kinds() {
        let mut original = candidate(
            ExecutionCandidateKind::OriginalNativeRegion,
            b"native-source",
        );
        original.execution_location = CandidateExecutionLocation::NativeRegion { device_id: 7 };
        original.output_residency = CandidateOutputResidency::Unknown;
        original.transactional_results = false;
        assert!(original.validate().is_ok());
        let resources = PlacementResourceSnapshot::default();
        let constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
        assert!(constraint
            .validate_candidates(std::slice::from_ref(&original), &resources)
            .is_ok());
        original.execution_location = CandidateExecutionLocation::VmRegion { device_id: 7 };
        assert!(
            original.validate().is_err(),
            "native source is not a VM installation"
        );
        for kind in [
            ExecutionCandidateKind::SpecializedNativeCpu,
            ExecutionCandidateKind::VectorizedNativeCpu,
        ] {
            let cpu = candidate(kind, b"host");
            assert!(constraint.validate_candidates(&[cpu], &resources).is_err());
        }
    }
    use crate::{
        CandidateCostUnavailable, EstimateConfidence, EstimateSource, ExecutionCostComponents,
        ExecutionCostEstimate, PlacementRevision, ProgramRevision, ResourceByteBound,
    };

    fn signature() -> PlacementSignature {
        PlacementSignature {
            region: None,
            operation: "whole-network".into(),
            runtime_facts: Digest::sha256(b"admitted-workload"),
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None::<ProgramRevision>,
                catalog: Digest::sha256(b"catalog"),
                compiler: Digest::sha256(b"compiler"),
                authority: crate::PlacementExecutionAuthorityRevision::Provider(Digest::sha256(
                    b"provider",
                )),
                policy: Digest::sha256(b"policy"),
            },
        }
    }

    #[test]
    fn streamed_placement_signature_keeps_legacy_cache_key() {
        let mut value = signature();
        let old = Digest::sha256(serde_json::to_vec(&value).unwrap());
        assert_eq!(value.cache_key(), old);
        value.region = Some(runmat_types::RegionId {
            function: runmat_types::ProgramFunctionId(3),
            ordinal: 11,
        });
        value.runtime_facts = Digest::sha256(b"different shape and residency");
        value.revision.authority = crate::PlacementExecutionAuthorityRevision::HostService(
            Digest::sha256(b"host service"),
        );
        assert_eq!(
            value.cache_key(),
            Digest::sha256(serde_json::to_vec(&value).unwrap())
        );
    }

    fn candidate(kind: ExecutionCandidateKind, route: &[u8]) -> WholeWorkCandidate {
        let provider = kind.is_provider();
        WholeWorkCandidate {
            kind,
            execution_location: if provider {
                CandidateExecutionLocation::Provider { device_id: 7 }
            } else {
                CandidateExecutionLocation::Host
            },
            preparation: CandidatePreparationState::Warm,
            output_residency: if provider {
                CandidateOutputResidency::Provider { device_id: 7 }
            } else {
                CandidateOutputResidency::Host
            },
            presentation: CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256(route),
            cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoObservations),
            resources: CandidateResourceDemand::default(),
            transactional_results: true,
        }
    }

    fn legacy_candidate_identity(route: &WholeWorkCandidate) -> String {
        let mut legacy = b"runmat.whole-work.candidate.v2\0".to_vec();
        legacy.push(match route.kind {
            ExecutionCandidateKind::SharedRuntime => 0,
            ExecutionCandidateKind::OriginalVmRegion => 9,
            ExecutionCandidateKind::OriginalNativeRegion => 10,
            ExecutionCandidateKind::GenericNativeCpu => 1,
            ExecutionCandidateKind::SpecializedNativeCpu => 2,
            ExecutionCandidateKind::VectorizedNativeCpu => 3,
            ExecutionCandidateKind::CpuLibrary => 4,
            ExecutionCandidateKind::ProviderOperation => 5,
            ExecutionCandidateKind::ProviderLibrary => 6,
            ExecutionCandidateKind::ProviderGraph => 7,
            ExecutionCandidateKind::ProviderFusion => 8,
        });
        match route.execution_location {
            CandidateExecutionLocation::Host => legacy.push(0),
            CandidateExecutionLocation::NativeSource => legacy.push(4),
            CandidateExecutionLocation::Provider { device_id } => {
                legacy.push(1);
                legacy.extend_from_slice(&device_id.to_le_bytes());
            }
            CandidateExecutionLocation::VmRegion { device_id } => {
                legacy.push(2);
                legacy.extend_from_slice(&device_id.to_le_bytes());
            }
            CandidateExecutionLocation::NativeRegion { device_id } => {
                legacy.push(3);
                legacy.extend_from_slice(&device_id.to_le_bytes());
            }
        }
        legacy.push(match route.preparation {
            CandidatePreparationState::Ready => 0,
            CandidatePreparationState::Warm => 1,
            CandidatePreparationState::Cold => 2,
            CandidatePreparationState::Preparing => 3,
            CandidatePreparationState::Unknown => 4,
        });
        match route.output_residency {
            CandidateOutputResidency::Host => legacy.push(0),
            CandidateOutputResidency::Provider { device_id } => {
                legacy.push(1);
                legacy.extend_from_slice(&device_id.to_le_bytes());
            }
            CandidateOutputResidency::Mirrored { device_id } => {
                legacy.push(2);
                legacy.extend_from_slice(&device_id.to_le_bytes());
            }
            CandidateOutputResidency::ProviderAndHost {
                device_id,
                host_bytes,
            } => {
                legacy.push(5);
                legacy.extend_from_slice(&device_id.to_le_bytes());
                legacy.extend_from_slice(&serde_json::to_vec(&host_bytes).unwrap());
            }
            CandidateOutputResidency::Unknown => legacy.push(3),
            CandidateOutputResidency::ConditionalHostOrProvider {
                device_id,
                host_bytes,
            } => {
                legacy.push(4);
                legacy.extend_from_slice(&device_id.to_le_bytes());
                legacy.extend_from_slice(&serde_json::to_vec(&host_bytes).unwrap());
            }
        }
        let presentation = serde_json::to_vec(&route.presentation).unwrap();
        legacy.extend_from_slice(&(presentation.len() as u64).to_le_bytes());
        legacy.extend_from_slice(&presentation);
        legacy.extend_from_slice(route.route_facts.bytes());
        Digest::sha256(legacy).to_string()
    }

    #[test]
    fn streamed_candidate_identity_preserves_legacy_bytes_for_every_fixed_tag() {
        let kinds = [
            ExecutionCandidateKind::SharedRuntime,
            ExecutionCandidateKind::GenericNativeCpu,
            ExecutionCandidateKind::SpecializedNativeCpu,
            ExecutionCandidateKind::VectorizedNativeCpu,
            ExecutionCandidateKind::CpuLibrary,
            ExecutionCandidateKind::ProviderOperation,
            ExecutionCandidateKind::ProviderLibrary,
            ExecutionCandidateKind::ProviderGraph,
            ExecutionCandidateKind::ProviderFusion,
        ];
        let preparations = [
            CandidatePreparationState::Ready,
            CandidatePreparationState::Warm,
            CandidatePreparationState::Cold,
            CandidatePreparationState::Preparing,
            CandidatePreparationState::Unknown,
        ];
        let residencies = [
            CandidateOutputResidency::Host,
            CandidateOutputResidency::Provider { device_id: 7 },
            CandidateOutputResidency::Mirrored { device_id: 7 },
            CandidateOutputResidency::Unknown,
            CandidateOutputResidency::ConditionalHostOrProvider {
                device_id: 7,
                host_bytes: ResourceByteBound::Known(37),
            },
            CandidateOutputResidency::ConditionalHostOrProvider {
                device_id: 7,
                host_bytes: ResourceByteBound::Unavailable(
                    crate::ResourceBoundUnavailable::NoRetainedValueBound,
                ),
            },
        ];
        for kind in kinds {
            for preparation in preparations {
                for output_residency in residencies {
                    let mut route = candidate(kind, b"exact-route");
                    route.preparation = preparation;
                    route.output_residency = output_residency;
                    assert_eq!(
                        route.selected("readmatrix").candidate,
                        legacy_candidate_identity(&route),
                    );
                    assert_eq!(
                        route.selected("readmatrix").candidate.parse::<Digest>(),
                        Ok(route.identity_digest())
                    );
                }
            }
        }
    }

    #[test]
    fn streamed_candidate_identity_preserves_nontrivial_presentation() {
        let mut route = candidate(ExecutionCandidateKind::ProviderOperation, b"presentation");
        route.presentation = CandidateOutputPresentation::ProviderDownload {
            device_id: 7,
            provider_scratch_bytes: ResourceByteBound::Known(8192),
            queue_slots: 2,
        };
        assert_eq!(
            route.selected("download").candidate,
            legacy_candidate_identity(&route)
        );
        route.presentation = CandidateOutputPresentation::ProviderUpload {
            device_id: 7,
            transfer_bytes: Some(128),
            host_result_bytes: Some(ResourceByteBound::Known(128)),
            host_scratch_bytes: ResourceByteBound::Known(7),
            provider_scratch_bytes: ResourceByteBound::Known(11),
            queue_slots: 1,
            preparation: CandidatePreparationState::Cold,
        };
        assert_eq!(
            route.selected("upload").candidate,
            legacy_candidate_identity(&route)
        );
    }

    #[test]
    fn borrowed_single_route_matches_owned_validation_without_candidate_vec() {
        let multi = WholeWorkPlacementRequest {
            signature: signature(),
            constraint: WholeWorkConstraint::AnyLegal,
            candidates: vec![candidate(ExecutionCandidateKind::SharedRuntime, b"host")],
            resources: PlacementResourceSnapshot::default(),
            deterministic: true,
            require_transactional_results: false,
        };
        let single = WholeWorkPlacementRequestRef {
            signature: (&multi.signature).into(),
            constraint: multi.constraint,
            candidates: std::slice::from_ref(&multi.candidates[0]),
            resources: &multi.resources,
            deterministic: multi.deterministic,
            require_transactional_results: multi.require_transactional_results,
        };
        assert_eq!(single.validate(), multi.validate());
        assert_eq!(single.candidates.len(), 1);
    }

    #[test]
    fn common_workload_key_retains_distinct_typed_candidate_routes() {
        let host = candidate(ExecutionCandidateKind::SharedRuntime, b"host");
        let provider = candidate(ExecutionCandidateKind::ProviderGraph, b"device");
        let request = WholeWorkPlacementRequest {
            signature: signature(),
            constraint: WholeWorkConstraint::AnyLegal,
            candidates: vec![host.clone(), provider.clone()],
            resources: PlacementResourceSnapshot::default(),
            deterministic: true,
            require_transactional_results: true,
        };
        assert!(request.validate().is_ok());
        assert_ne!(
            host.selected(&request.signature.operation),
            provider.selected(&request.signature.operation)
        );
        let mut cold = provider;
        cold.preparation = CandidatePreparationState::Cold;
        assert_ne!(
            cold.selected(&request.signature.operation),
            request.candidates[1].selected(&request.signature.operation)
        );
        let mut unknown = request.candidates[1].clone();
        unknown.preparation = CandidatePreparationState::Unknown;
        assert_ne!(
            unknown.selected(&request.signature.operation),
            request.candidates[1].selected(&request.signature.operation)
        );
    }

    #[test]
    fn provider_required_rejects_host_and_foreign_device_routes() {
        let request = WholeWorkPlacementRequest {
            signature: signature(),
            constraint: WholeWorkConstraint::ProviderRequired { device_id: 7 },
            candidates: vec![candidate(ExecutionCandidateKind::SharedRuntime, b"host")],
            resources: PlacementResourceSnapshot::default(),
            deterministic: true,
            require_transactional_results: true,
        };
        assert!(request.validate().is_err());

        let mut original = candidate(ExecutionCandidateKind::OriginalVmRegion, b"installed");
        original.execution_location = CandidateExecutionLocation::VmRegion { device_id: 7 };
        original.output_residency = CandidateOutputResidency::Unknown;
        original.transactional_results = false;
        let source_ordered = WholeWorkPlacementRequest {
            candidates: vec![original.clone()],
            require_transactional_results: false,
            ..request
        };
        assert!(source_ordered.validate().is_ok());
        original.execution_location = CandidateExecutionLocation::VmRegion { device_id: 8 };
        assert!(WholeWorkPlacementRequest {
            candidates: vec![original],
            ..source_ordered
        }
        .validate()
        .is_err());
    }

    #[test]
    fn candidate_rejects_foreign_output_device_and_host_provider_presentation() {
        let mut provider = candidate(ExecutionCandidateKind::ProviderGraph, b"provider");
        provider.output_residency = CandidateOutputResidency::Mirrored { device_id: 8 };
        assert!(provider.validate().is_err());
        let mut host = candidate(ExecutionCandidateKind::SharedRuntime, b"host");
        host.output_residency = CandidateOutputResidency::Provider { device_id: 7 };
        assert!(host.validate().is_err());
        host.presentation = CandidateOutputPresentation::ProviderUpload {
            device_id: 7,
            transfer_bytes: Some(16),
            host_result_bytes: None,
            host_scratch_bytes: 0.into(),
            provider_scratch_bytes: 0.into(),
            queue_slots: 1,
            preparation: CandidatePreparationState::Ready,
        };
        assert!(host.validate().is_ok());
        let direct = candidate(ExecutionCandidateKind::SharedRuntime, b"host");
        assert_ne!(host.selected("work"), direct.selected("work"));
        // A synchronous provider still owns its output; no fictitious queue
        // slot is needed to validate that physical presentation route.
        if let CandidateOutputPresentation::ProviderUpload { queue_slots, .. } =
            &mut host.presentation
        {
            *queue_slots = 0;
        }
        assert!(host.validate().is_ok());
        host.presentation = CandidateOutputPresentation::ProviderUpload {
            device_id: 8,
            transfer_bytes: Some(16),
            host_result_bytes: None,
            host_scratch_bytes: 0.into(),
            provider_scratch_bytes: 0.into(),
            queue_slots: 1,
            preparation: CandidatePreparationState::Ready,
        };
        assert!(host.validate().is_err());
        host.presentation = CandidateOutputPresentation::ProviderUpload {
            device_id: 7,
            transfer_bytes: Some(16),
            host_result_bytes: None,
            host_scratch_bytes: 0.into(),
            provider_scratch_bytes: 0.into(),
            queue_slots: 1,
            preparation: CandidatePreparationState::Unknown,
        };
        assert!(host.validate().is_err());
    }

    #[test]
    fn conditional_provider_upload_requires_exact_owner_and_host_result_evidence() {
        let mut provider = candidate(ExecutionCandidateKind::ProviderOperation, b"provider");
        provider.presentation = CandidateOutputPresentation::ConditionalProviderUpload {
            device_id: 7,
            transfer_bytes: Some(16),
            host_result_bytes: Some(16.into()),
            host_scratch_bytes: 8.into(),
            provider_scratch_bytes: 8.into(),
            queue_slots: 1,
        };
        assert!(provider.validate().is_ok());
        let selected = provider.selected("imfilter");
        if let CandidateOutputPresentation::ConditionalProviderUpload {
            host_result_bytes, ..
        } = &mut provider.presentation
        {
            *host_result_bytes = None;
        }
        assert!(provider.validate().is_err());
        if let CandidateOutputPresentation::ConditionalProviderUpload {
            host_result_bytes,
            device_id,
            ..
        } = &mut provider.presentation
        {
            *host_result_bytes = Some(16.into());
            *device_id = 8;
        }
        assert!(provider.validate().is_err());
        provider.presentation = CandidateOutputPresentation::Direct;
        assert_ne!(selected, provider.selected("imfilter"));
    }

    #[test]
    fn risk_adjustment_overflow_is_rejected_before_provider_ranking() {
        let mut provider = candidate(ExecutionCandidateKind::ProviderGraph, b"provider");
        provider.cost = CandidateCostEvidence::Estimated(ExecutionCostEstimate {
            components: ExecutionCostComponents {
                execution_ns: u64::MAX,
                ..ExecutionCostComponents::default()
            },
            scratch_bytes: Some(0),
            confidence: EstimateConfidence::Prior,
            source: EstimateSource::Provider,
        });
        assert!(provider.validate().is_err());
    }
}
