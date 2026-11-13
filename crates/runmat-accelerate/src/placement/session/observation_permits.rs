//! In-flight complete-call observations belong to the selection epoch. A
//! profile restore or matching invalidation revokes them before late receipts
//! can repopulate a replaced profile. Weak registration does not retain calls.

use runmat_execution::{
    Digest, ExecutionCandidateKind, PlacementDecision, PlacementInvalidation,
    PlacementPermitIdentity, PlacementSignature,
};
use runmat_runtime::context::{RuntimeGraphObservationPermit, RuntimeWholeWorkObservationPermit};
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Weak,
};

#[derive(Default)]
pub(super) struct ObservationPermits {
    active: RefCell<Vec<(PlacementPermitIdentity, Weak<Permit>)>>,
}

#[derive(Debug)]
struct Permit {
    current: AtomicBool,
    graph: Option<GraphSelection>,
}

#[derive(Debug)]
struct GraphSelection {
    signature: Digest,
    candidate: String,
    kind: ExecutionCandidateKind,
}

impl RuntimeWholeWorkObservationPermit for Permit {
    fn is_current(&self) -> bool {
        self.current.load(Ordering::Acquire)
    }
}

impl RuntimeGraphObservationPermit for Permit {
    fn authorizes(
        &self,
        signature: &PlacementSignature,
        candidate: &str,
        kind: ExecutionCandidateKind,
    ) -> bool {
        self.current.load(Ordering::Acquire)
            && self.graph.as_ref().is_some_and(|selected| {
                selected.signature == signature.cache_key()
                    && selected.candidate == candidate
                    && selected.kind == kind
            })
    }
}

impl ObservationPermits {
    pub(super) fn register(
        &self,
        signature: &PlacementSignature,
    ) -> Result<Arc<dyn RuntimeWholeWorkObservationPermit>, runmat_execution::ContractError> {
        Ok(self.register_with(signature, None)?)
    }

    pub(super) fn register_graph(
        &self,
        decision: &PlacementDecision,
    ) -> Option<Arc<dyn RuntimeGraphObservationPermit>> {
        let [selected] = decision.selections.as_slice() else {
            return None;
        };
        self.register_with(
            &decision.signature,
            Some(GraphSelection {
                signature: decision.signature.cache_key(),
                candidate: selected.candidate.clone(),
                kind: selected.kind,
            }),
        )
        .ok()
        .map(|permit| permit as Arc<dyn RuntimeGraphObservationPermit>)
    }

    fn register_with(
        &self,
        signature: &PlacementSignature,
        graph: Option<GraphSelection>,
    ) -> Result<Arc<Permit>, runmat_execution::ContractError> {
        let identity = signature.permit_identity()?;
        let permit = Arc::new(Permit {
            current: AtomicBool::new(true),
            graph,
        });
        let mut active = self.active.borrow_mut();
        active.retain(|(_, weak)| weak.strong_count() != 0);
        active.push((identity, Arc::downgrade(&permit)));
        Ok(permit)
    }

    pub(super) fn invalidate(&self, invalidation: &PlacementInvalidation) {
        self.active.borrow_mut().retain(|(signature, weak)| {
            let Some(permit) = weak.upgrade() else {
                return false;
            };
            if invalidation.matches_permit_identity(signature) {
                permit.current.store(false, Ordering::Release);
                false
            } else {
                true
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{
        Digest, DomainContribution, PlacementExecutionAuthorityRevision, PlacementRevision,
        ProgramEnvironment, ProgramRevision,
    };

    #[test]
    fn completed_selections_leave_no_session_length_weak_index() {
        let digest = Digest::sha256(b"host-service");
        let signature = PlacementSignature {
            region: None,
            operation: "unzip".into(),
            runtime_facts: digest,
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                catalog: digest,
                compiler: digest,
                authority: PlacementExecutionAuthorityRevision::HostService(digest),
                policy: digest,
            },
        };
        let permits = ObservationPermits::default();
        for _ in 0..128 {
            drop(permits.register(&signature).unwrap());
            assert_eq!(permits.active.borrow().len(), 1);
        }
        permits.invalidate(&PlacementInvalidation::All);
        assert!(permits.active.borrow().is_empty());
    }

    #[test]
    fn compact_registration_retains_exact_program_and_authority_invalidation() {
        let digest = Digest::sha256(b"authority");
        let program = ProgramRevision::new(
            Digest::sha256(b"graph"),
            Digest::sha256(b"source"),
            ProgramEnvironment::new(1, 1, digest, digest, "runmat").unwrap(),
        )
        .unwrap()
        .with_domain_contribution(DomainContribution::new("test.domain", digest).unwrap())
        .unwrap();
        let signature = PlacementSignature {
            region: None,
            operation: "readmatrix".into(),
            runtime_facts: digest,
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: Some(program.clone()),
                catalog: digest,
                compiler: digest,
                authority: PlacementExecutionAuthorityRevision::HostService(digest),
                policy: digest,
            },
        };
        let permits = ObservationPermits::default();
        let active = permits.register(&signature).unwrap();
        permits.invalidate(&PlacementInvalidation::Provider { digest });
        assert!(active.is_current());
        permits.invalidate(&PlacementInvalidation::Program {
            revision: ProgramRevision::new(
                Digest::sha256(b"different graph"),
                Digest::sha256(b"source"),
                ProgramEnvironment::new(1, 1, digest, digest, "runmat").unwrap(),
            )
            .unwrap(),
        });
        assert!(active.is_current());
        permits.invalidate(&PlacementInvalidation::Program { revision: program });
        assert!(!active.is_current());
        assert!(permits.active.borrow().is_empty());
    }

    #[test]
    fn provider_invalidation_revokes_only_matching_session_bound_permit() {
        let provider_a = Digest::sha256(b"provider A capability");
        let provider_b = Digest::sha256(b"provider B capability");
        let mut signature = PlacementSignature {
            region: None,
            operation: "fusion.reduction".into(),
            runtime_facts: Digest::sha256(b"typed facts"),
            provider_provenance: Some(provider_a),
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                catalog: provider_a,
                compiler: provider_a,
                authority: PlacementExecutionAuthorityRevision::RuntimeBindingSession(
                    Digest::sha256(b"shared VM session"),
                ),
                policy: provider_a,
            },
        };
        let permits = ObservationPermits::default();
        let first = permits.register(&signature).unwrap();
        signature.provider_provenance = Some(provider_b);
        let second = permits.register(&signature).unwrap();
        permits.invalidate(&PlacementInvalidation::Provider { digest: provider_a });
        assert!(!first.is_current());
        assert!(second.is_current());
        permits.invalidate(&PlacementInvalidation::Provider { digest: provider_b });
        assert!(!second.is_current());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn graph_permit_binds_exact_selected_candidate_and_revokes_by_provider() {
        let provider = Digest::sha256(b"typed provider capability");
        let mut signature = PlacementSignature {
            region: None,
            operation: "fusion.reduction".into(),
            runtime_facts: Digest::sha256(b"typed graph"),
            provider_provenance: Some(provider),
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                catalog: provider,
                compiler: provider,
                authority: PlacementExecutionAuthorityRevision::RuntimeBindingSession(
                    Digest::sha256(b"VM owner"),
                ),
                policy: provider,
            },
        };
        let decision = PlacementDecision {
            signature: signature.clone(),
            selections: vec![runmat_execution::SelectedExecutionCandidate {
                node: "sum".into(),
                candidate: "physical.single".into(),
                kind: ExecutionCandidateKind::ProviderFusion,
            }],
            predicted_total_ns: 1,
            from_cache: false,
            used_local_fallback: false,
            explored_states: 1,
            pruned_states: 0,
        };
        let permits = ObservationPermits::default();
        let permit = permits.register_graph(&decision).unwrap();
        assert!(permit.authorizes(
            &signature,
            "physical.single",
            ExecutionCandidateKind::ProviderFusion,
        ));
        assert!(!permit.authorizes(
            &signature,
            "physical.two",
            ExecutionCandidateKind::ProviderFusion,
        ));
        assert!(!permit.authorizes(
            &signature,
            "physical.single",
            ExecutionCandidateKind::SharedRuntime,
        ));
        signature.runtime_facts = Digest::sha256(b"different facts");
        assert!(!permit.authorizes(
            &signature,
            "physical.single",
            ExecutionCandidateKind::ProviderFusion,
        ));
        permits.invalidate(&PlacementInvalidation::Provider { digest: provider });
        assert!(!permit.authorizes(
            &decision.signature,
            "physical.single",
            ExecutionCandidateKind::ProviderFusion,
        ));
    }
}
