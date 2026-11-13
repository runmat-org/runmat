use super::*;

impl PlacementSession {
    /// Receipt fixtures select through the canonical graph planner. A provider
    /// route includes its mandatory CPU fallback; the returned permit is never
    /// fabricated from a feedback tuple.
    pub(crate) fn test_graph_observation_permit(
        &self,
        signature: PlacementSignature,
        candidate: &str,
        kind: ExecutionCandidateKind,
    ) -> Arc<dyn runmat_runtime::context::RuntimeGraphObservationPermit> {
        use runmat_execution::{
            CandidateExecutionLocation, CandidateOutputPresentation, CandidateOutputResidency,
            CandidatePreparationState, CandidateResourceDemand, EstimateConfidence,
            ExecutionCandidateDescriptor, PlacementGraph, PlacementGraphCandidate,
            PlacementGraphLimits, PlacementGraphNode, PlacementResourceSnapshot,
            ProviderResourceSnapshot, ResourceByteBound,
        };
        let provider = kind.is_provider();
        let device_id = 7;
        let selected_candidate = PlacementGraphCandidate {
            descriptor: ExecutionCandidateDescriptor {
                identity: candidate.into(),
                region: signature.region,
                kind,
                execution_location: if provider {
                    CandidateExecutionLocation::Provider { device_id }
                } else {
                    CandidateExecutionLocation::Host
                },
                preparation: CandidatePreparationState::Warm,
                cost: runmat_execution::ExecutionCostEstimate {
                    components: ExecutionCostComponents {
                        execution_ns: 10,
                        ..Default::default()
                    },
                    scratch_bytes: Some(0),
                    confidence: EstimateConfidence::Exact,
                    source: EstimateSource::Synthetic,
                },
                output_residency: if provider {
                    CandidateOutputResidency::Provider { device_id }
                } else {
                    CandidateOutputResidency::Host
                },
                presentation: CandidateOutputPresentation::Direct,
                guards: Vec::new(),
            },
            resources: CandidateResourceDemand {
                cpu_millicores: if provider { 0 } else { 1_000 },
                retained_bytes: ResourceByteBound::Known(0),
                scratch_bytes: ResourceByteBound::Known(0),
                provider_host_workspace: None,
                provider_input_stages: None,
                queue_slots: if provider { 1 } else { 0 },
                peak_live: None,
            },
            output_facts: None,
            transactional_results: true,
        };
        let mut candidates = vec![selected_candidate];
        if provider {
            candidates.push(PlacementGraphCandidate {
                descriptor: ExecutionCandidateDescriptor {
                    identity: "test.cpu-fallback".into(),
                    region: signature.region,
                    kind: ExecutionCandidateKind::GenericNativeCpu,
                    execution_location: CandidateExecutionLocation::Host,
                    preparation: CandidatePreparationState::Warm,
                    cost: runmat_execution::ExecutionCostEstimate {
                        components: ExecutionCostComponents {
                            execution_ns: 100_000,
                            ..Default::default()
                        },
                        scratch_bytes: Some(0),
                        confidence: EstimateConfidence::Exact,
                        source: EstimateSource::Synthetic,
                    },
                    output_residency: CandidateOutputResidency::Host,
                    presentation: CandidateOutputPresentation::Direct,
                    guards: Vec::new(),
                },
                resources: CandidateResourceDemand {
                    cpu_millicores: 1_000,
                    retained_bytes: ResourceByteBound::Known(0),
                    scratch_bytes: ResourceByteBound::Known(0),
                    provider_host_workspace: None,
                    provider_input_stages: None,
                    queue_slots: 0,
                    peak_live: None,
                },
                output_facts: None,
                transactional_results: true,
            });
        }
        let selected =
            runmat_runtime::context::RuntimePlacementService::plan_with_observation_permit(
                self,
                PlacementPlanRequest {
                    signature: signature.clone(),
                    graph: PlacementGraph {
                        nodes: vec![PlacementGraphNode {
                            identity: signature.operation.clone(),
                            candidates,
                        }],
                        edges: Vec::new(),
                    },
                    limits: PlacementGraphLimits::default(),
                    resources: PlacementResourceSnapshot {
                        cpu_millicores_available: 1_000,
                        memory_available_bytes: None,
                        cancellation_requested: false,
                        providers: if provider {
                            vec![ProviderResourceSnapshot {
                                device_id,
                                capacity_bytes: Some(1_024),
                                live_bytes: 0,
                                reclaimable_bytes: 0,
                                scratch_available_bytes: Some(1_024),
                                queue_depth: Some(0),
                                queue_limit: Some(1),
                                lost: false,
                                epoch: 1,
                            }]
                        } else {
                            Vec::new()
                        },
                        epoch: 1,
                    },
                    deterministic: true,
                    require_transactional_results: true,
                },
            )
            .expect("fixture has a feasible selected route and canonical CPU fallback");
        assert_eq!(selected.decision.selections.len(), 1);
        assert_eq!(selected.decision.selections[0].candidate, candidate);
        assert_eq!(selected.decision.selections[0].kind, kind);
        let permit = selected
            .observation_permit
            .expect("session issues graph permit");
        assert!(permit.authorizes(&signature, candidate, kind));
        permit
    }
}
