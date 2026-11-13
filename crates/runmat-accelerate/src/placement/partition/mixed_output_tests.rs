use super::{
    candidate_total_ns, next_retained_state, predecessor_transition_ns, RetentionSchedule,
    SearchState,
};
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputPresentation, CandidateOutputResidency,
    CandidatePreparationState, CandidateResourceDemand, Digest, EstimateConfidence, EstimateSource,
    ExecutionCandidateDescriptor, ExecutionCandidateKind, ExecutionCostComponents,
    ExecutionCostEstimate, PlacementExecutionAuthorityRevision, PlacementGraph,
    PlacementGraphCandidate, PlacementGraphEdge, PlacementGraphLimits, PlacementGraphNode,
    PlacementGraphOutputFact, PlacementPlanRequest, PlacementResourceSnapshot, PlacementRevision,
    PlacementSignature, ProviderResourceSnapshot, ResourceByteBound, SourceOutputOrdinal,
};

fn candidate(identity: &str, location: CandidateExecutionLocation) -> PlacementGraphCandidate {
    let provider = matches!(location, CandidateExecutionLocation::Provider { .. });
    let device_id = match location {
        CandidateExecutionLocation::Provider { device_id } => device_id,
        CandidateExecutionLocation::VmRegion { device_id }
        | CandidateExecutionLocation::NativeRegion { device_id } => device_id,
        CandidateExecutionLocation::Host | CandidateExecutionLocation::NativeSource => 0,
    };
    PlacementGraphCandidate {
        descriptor: ExecutionCandidateDescriptor {
            identity: identity.into(),
            region: None,
            kind: if provider {
                ExecutionCandidateKind::ProviderFusion
            } else {
                ExecutionCandidateKind::SharedRuntime
            },
            execution_location: location,
            preparation: CandidatePreparationState::Ready,
            cost: ExecutionCostEstimate {
                components: ExecutionCostComponents {
                    execution_ns: 1,
                    ..ExecutionCostComponents::default()
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
            retained_bytes: ResourceByteBound::Known(0),
            scratch_bytes: ResourceByteBound::Known(0),
            ..CandidateResourceDemand::default()
        },
        output_facts: None,
        transactional_results: true,
    }
}

fn graph() -> PlacementPlanRequest {
    let mut mixed = candidate(
        "producer.mixed",
        CandidateExecutionLocation::Provider { device_id: 7 },
    );
    mixed.descriptor.output_residency = CandidateOutputResidency::ProviderAndHost {
        device_id: 7,
        host_bytes: ResourceByteBound::Known(8),
    };
    mixed.descriptor.presentation = CandidateOutputPresentation::ProviderProducedReadback {
        device_id: 7,
        transfer_bytes: 4,
        provider_scratch_bytes: ResourceByteBound::Known(4),
        queue_slots: 1,
    };
    mixed.resources.retained_bytes = ResourceByteBound::Known(4);
    mixed.output_facts = Some(vec![
        PlacementGraphOutputFact {
            residency: CandidateOutputResidency::Provider { device_id: 7 },
            retained_bytes: ResourceByteBound::Known(4),
        },
        PlacementGraphOutputFact {
            residency: CandidateOutputResidency::Host,
            retained_bytes: ResourceByteBound::Known(8),
        },
    ]);
    let mut host = candidate("producer.host", CandidateExecutionLocation::Host);
    host.output_facts = Some(vec![
        PlacementGraphOutputFact {
            residency: CandidateOutputResidency::Host,
            retained_bytes: ResourceByteBound::Known(0),
        },
        PlacementGraphOutputFact {
            residency: CandidateOutputResidency::Host,
            retained_bytes: ResourceByteBound::Known(0),
        },
    ]);
    let digest = Digest::sha256(b"mixed-output-test");
    PlacementPlanRequest {
        signature: PlacementSignature {
            region: None,
            operation: "mixed-output-test".into(),
            runtime_facts: digest,
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                catalog: digest,
                compiler: digest,
                authority: PlacementExecutionAuthorityRevision::Provider(digest),
                policy: digest,
            },
        },
        graph: PlacementGraph {
            nodes: vec![
                PlacementGraphNode {
                    identity: "producer".into(),
                    candidates: vec![host, mixed],
                },
                PlacementGraphNode {
                    identity: "consumer".into(),
                    candidates: vec![candidate("consumer.host", CandidateExecutionLocation::Host)],
                },
            ],
            edges: Vec::new(),
        },
        limits: PlacementGraphLimits::default(),
        resources: PlacementResourceSnapshot {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(1_024),
            cancellation_requested: false,
            providers: vec![ProviderResourceSnapshot {
                device_id: 7,
                capacity_bytes: Some(1_024),
                live_bytes: 0,
                reclaimable_bytes: 0,
                scratch_available_bytes: Some(1_024),
                queue_depth: Some(0),
                queue_limit: Some(2),
                lost: false,
                epoch: 1,
            }],
            epoch: 1,
        },
        deterministic: true,
        require_transactional_results: true,
    }
}

fn edge(output: u32) -> PlacementGraphEdge {
    PlacementGraphEdge {
        from: 0,
        to: 1,
        source_output: Some(SourceOutputOrdinal::new(output)),
        bytes: 4,
        host_to_provider_ns: 11,
        provider_to_host_ns: 13,
        cross_provider_ns: 17,
    }
}

#[test]
fn source_output_role_selects_exact_transition_cost() {
    let mut request = graph();
    request.graph.edges.push(edge(0));
    request.bind_graph_output_authority();
    request.validate().unwrap();
    assert_eq!(
        predecessor_transition_ns(&request, 1, &[1], CandidateExecutionLocation::Host),
        Some(13),
    );
    assert_eq!(
        predecessor_transition_ns(
            &request,
            1,
            &[1],
            CandidateExecutionLocation::Provider { device_id: 7 }
        ),
        Some(0),
    );
    request.graph.edges[0] = edge(1);
    assert_eq!(
        predecessor_transition_ns(&request, 1, &[1], CandidateExecutionLocation::Host),
        Some(0),
    );
    assert_eq!(
        predecessor_transition_ns(
            &request,
            1,
            &[1],
            CandidateExecutionLocation::Provider { device_id: 7 }
        ),
        Some(11),
    );
    assert_eq!(
        predecessor_transition_ns(
            &request,
            1,
            &[1],
            CandidateExecutionLocation::Provider { device_id: 8 }
        ),
        Some(11),
    );
    request.graph.edges.push(edge(0));
    request.bind_graph_output_authority();
    request.validate().unwrap();
    assert_eq!(
        predecessor_transition_ns(&request, 1, &[1], CandidateExecutionLocation::Host),
        Some(13),
    );
    assert_eq!(
        predecessor_transition_ns(
            &request,
            1,
            &[1],
            CandidateExecutionLocation::Provider { device_id: 7 }
        ),
        Some(11),
    );
    assert_eq!(
        predecessor_transition_ns(
            &request,
            1,
            &[1],
            CandidateExecutionLocation::Provider { device_id: 8 }
        ),
        Some(28),
    );
}

#[test]
fn both_output_roles_remain_retained_until_their_last_graph_consumer() {
    let mut request = graph();
    request.graph.edges = vec![edge(0), edge(1)];
    request.bind_graph_output_authority();
    request.validate().unwrap();
    let retention = RetentionSchedule::new(&request);
    let empty = SearchState {
        selections: Vec::new(),
        total_ns: 0,
        host_retained_bytes: Default::default(),
        provider_retained_bytes: Default::default(),
    };
    let producer = &request.graph.nodes[0].candidates[1];
    let (host, providers) = next_retained_state(&request, &retention, 0, &empty, producer).unwrap();
    assert_eq!(host.known_bytes(), 8);
    assert_eq!(providers[&7].known_bytes(), 4);
    let after_producer = SearchState {
        selections: vec![1],
        total_ns: 0,
        host_retained_bytes: host,
        provider_retained_bytes: providers,
    };
    let consumer = &request.graph.nodes[1].candidates[0];
    let (host, providers) =
        next_retained_state(&request, &retention, 1, &after_producer, consumer).unwrap();
    assert_eq!(host.known_bytes(), 0);
    assert!(providers.is_empty());
}

#[test]
fn host_output_releases_before_later_provider_output_consumer() {
    let mut request = graph();
    request.graph.nodes.push(PlacementGraphNode {
        identity: "late_consumer".into(),
        candidates: vec![candidate(
            "late_consumer.host",
            CandidateExecutionLocation::Host,
        )],
    });
    request.graph.nodes[2].candidates[0]
        .resources
        .retained_bytes = ResourceByteBound::Known(6);
    request.graph.edges = vec![
        PlacementGraphEdge {
            from: 0,
            to: 1,
            source_output: Some(SourceOutputOrdinal::new(1)),
            ..edge(1)
        },
        PlacementGraphEdge {
            from: 0,
            to: 2,
            source_output: Some(SourceOutputOrdinal::new(0)),
            ..edge(0)
        },
    ];
    request.resources.memory_available_bytes = Some(10);
    request.bind_graph_output_authority();
    request.validate().unwrap();
    let retention = RetentionSchedule::new(&request);
    let empty = SearchState {
        selections: Vec::new(),
        total_ns: 0,
        host_retained_bytes: Default::default(),
        provider_retained_bytes: Default::default(),
    };
    let (host, providers) = next_retained_state(
        &request,
        &retention,
        0,
        &empty,
        &request.graph.nodes[0].candidates[1],
    )
    .unwrap();
    let after_producer = SearchState {
        selections: vec![1],
        total_ns: 0,
        host_retained_bytes: host,
        provider_retained_bytes: providers,
    };
    let (host, providers) = next_retained_state(
        &request,
        &retention,
        1,
        &after_producer,
        &request.graph.nodes[1].candidates[0],
    )
    .unwrap();
    assert_eq!(host.known_bytes(), 0);
    assert_eq!(providers[&7].known_bytes(), 4);
    let after_first_consumer = SearchState {
        selections: vec![1, 0],
        total_ns: 0,
        host_retained_bytes: host,
        provider_retained_bytes: providers,
    };
    assert!(candidate_total_ns(
        &request.graph.nodes[2].candidates[0],
        &request,
        &after_first_consumer
    )
    .is_some());
    let aggregate_still_live = SearchState {
        host_retained_bytes: after_producer.host_retained_bytes,
        ..after_first_consumer.clone()
    };
    assert!(candidate_total_ns(
        &request.graph.nodes[2].candidates[0],
        &request,
        &aggregate_still_live
    )
    .is_none());
    let (host, providers) = next_retained_state(
        &request,
        &retention,
        2,
        &after_first_consumer,
        &request.graph.nodes[2].candidates[0],
    )
    .unwrap();
    assert_eq!(host.known_bytes(), 6);
    assert!(providers.is_empty());
}

#[test]
fn repeated_consumers_retain_one_output_once_until_its_last_use() {
    let mut request = graph();
    request.graph.nodes.push(PlacementGraphNode {
        identity: "late_consumer".into(),
        candidates: vec![candidate(
            "late_consumer.host",
            CandidateExecutionLocation::Host,
        )],
    });
    request.graph.edges = vec![edge(0), PlacementGraphEdge { to: 2, ..edge(0) }];
    request.bind_graph_output_authority();
    request.validate().unwrap();
    let retention = RetentionSchedule::new(&request);
    let empty = SearchState {
        selections: Vec::new(),
        total_ns: 0,
        host_retained_bytes: Default::default(),
        provider_retained_bytes: Default::default(),
    };
    let (host, providers) = next_retained_state(
        &request,
        &retention,
        0,
        &empty,
        &request.graph.nodes[0].candidates[1],
    )
    .unwrap();
    assert_eq!(host.known_bytes(), 0);
    assert_eq!(providers[&7].known_bytes(), 4);
    let after_producer = SearchState {
        selections: vec![1],
        total_ns: 0,
        host_retained_bytes: host,
        provider_retained_bytes: providers,
    };
    let (host, providers) = next_retained_state(
        &request,
        &retention,
        1,
        &after_producer,
        &request.graph.nodes[1].candidates[0],
    )
    .unwrap();
    assert_eq!(providers[&7].known_bytes(), 4);
    let after_first_consumer = SearchState {
        selections: vec![1, 0],
        total_ns: 0,
        host_retained_bytes: host,
        provider_retained_bytes: providers,
    };
    let (_, providers) = next_retained_state(
        &request,
        &retention,
        2,
        &after_first_consumer,
        &request.graph.nodes[2].candidates[0],
    )
    .unwrap();
    assert!(providers.is_empty());
}
