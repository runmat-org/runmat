use super::*;

fn fact(residency: CandidateOutputResidency, bytes: u64) -> PlacementGraphOutputFact {
    PlacementGraphOutputFact {
        residency,
        retained_bytes: ResourceByteBound::Known(bytes),
    }
}

fn validate(request: &mut PlacementPlanRequest) -> Result<(), &'static str> {
    request.bind_graph_output_authority();
    request.validate()
}

fn paired_request() -> PlacementPlanRequest {
    let mut request = super::tests::request();
    let first = &mut request.graph.nodes[0];
    first.candidates[0].output_facts = Some(vec![
        fact(CandidateOutputResidency::Host, 0),
        fact(CandidateOutputResidency::Host, 0),
    ]);
    first.candidates[1].descriptor.output_residency = CandidateOutputResidency::ProviderAndHost {
        device_id: 7,
        host_bytes: ResourceByteBound::Known(8),
    };
    first.candidates[1].descriptor.presentation =
        crate::CandidateOutputPresentation::ProviderProducedReadback {
            device_id: 7,
            transfer_bytes: 4,
            provider_scratch_bytes: ResourceByteBound::Known(4),
            queue_slots: 1,
        };
    first.candidates[1].resources.retained_bytes = ResourceByteBound::Known(4);
    first.candidates[1].output_facts = Some(vec![
        fact(CandidateOutputResidency::Provider { device_id: 7 }, 4),
        fact(CandidateOutputResidency::Host, 8),
    ]);
    request.graph.nodes.push(PlacementGraphNode {
        identity: "consumer".into(),
        candidates: vec![super::tests::candidate("consumer.host", false)],
    });
    request
}

fn edge(ordinal: Option<u32>) -> PlacementGraphEdge {
    PlacementGraphEdge {
        from: 0,
        to: 1,
        source_output: ordinal.map(SourceOutputOrdinal::new),
        bytes: 4,
        host_to_provider_ns: 11,
        provider_to_host_ns: 13,
        cross_provider_ns: 17,
    }
}

#[test]
fn legacy_single_output_wire_omits_new_fields_and_keeps_admission() {
    let mut request = super::tests::request();
    request.graph.nodes.push(PlacementGraphNode {
        identity: "consumer".into(),
        candidates: vec![super::tests::candidate("consumer.host", false)],
    });
    request.graph.edges.push(edge(None));
    request.validate().unwrap();
    let wire = serde_json::to_string(&request).unwrap();
    assert!(!wire.contains("source_output"));
    assert!(!wire.contains("output_facts"));
    assert!(!wire.contains("graph_output_authority"));
    let decoded: PlacementPlanRequest = serde_json::from_str(&wire).unwrap();
    assert_eq!(decoded, request);
    decoded.validate().unwrap();
}

#[test]
fn mixed_edges_require_exact_ordinals_and_allow_distinct_outputs_to_one_consumer() {
    let mut request = paired_request();
    request.graph.edges.push(edge(None));
    assert_eq!(
        validate(&mut request),
        Err("mixed placement edge requires a source-output ordinal")
    );
    request.graph.edges[0] = edge(Some(2));
    assert_eq!(
        validate(&mut request),
        Err("placement edge source-output ordinal is out of range")
    );
    request.graph.edges[0] = edge(Some(0));
    request.graph.edges.push(edge(Some(1)));
    validate(&mut request).unwrap();
    let decoded: PlacementPlanRequest =
        serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
    assert_eq!(
        decoded.graph.edges[0].source_output,
        Some(SourceOutputOrdinal::new(0))
    );
    assert_eq!(decoded, request);
    request.graph.edges.push(edge(Some(1)));
    assert_eq!(
        validate(&mut request),
        Err("placement graph edges must be unique")
    );
}

#[test]
fn malformed_or_unproven_output_roles_fail_before_graph_placement() {
    let mut request = paired_request();
    request.graph.edges.push(edge(Some(1)));
    request.graph.nodes[0].candidates[1].output_facts = None;
    assert_eq!(
        validate(&mut request),
        Err("placement source candidates lack matching output roles")
    );
    request.graph.nodes[0].candidates[1].output_facts = Some(vec![
        fact(CandidateOutputResidency::Provider { device_id: 7 }, 4),
        fact(CandidateOutputResidency::Provider { device_id: 7 }, 8),
    ]);
    assert_eq!(
        validate(&mut request),
        Err("mixed placement output roles do not match their provider owner")
    );
    request.graph.nodes[0].candidates[1].output_facts = Some(vec![
        fact(CandidateOutputResidency::Provider { device_id: 8 }, 4),
        fact(CandidateOutputResidency::Host, 8),
    ]);
    assert_eq!(
        validate(&mut request),
        Err("mixed placement output roles do not match their provider owner")
    );
    request.graph.nodes[0].candidates[1].output_facts = Some(vec![
        fact(CandidateOutputResidency::Provider { device_id: 7 }, 4),
        fact(CandidateOutputResidency::Host, 8),
    ]);
    request.graph.nodes[0].candidates[0].output_facts = None;
    assert_eq!(
        validate(&mut request),
        Err("placement candidates disagree on public output count")
    );
}

#[test]
fn aggregate_mixed_sink_needs_no_fabricated_public_output_order() {
    let mut request = paired_request();
    request.graph.nodes.pop();
    request.graph.nodes[0].candidates[1].output_facts = None;
    validate(&mut request).expect("single-node whole-work envelope has no source-output edge");
    request.graph.nodes.push(PlacementGraphNode {
        identity: "independent_sink".into(),
        candidates: vec![super::tests::candidate("independent_sink.host", false)],
    });
    assert_eq!(
        validate(&mut request),
        Err("placement source candidates lack matching output roles")
    );
    request.graph.nodes[0].candidates.swap(0, 1);
    assert_eq!(
        validate(&mut request),
        Err("placement source candidates lack matching output roles")
    );
    request.graph.edges.push(edge(Some(0)));
    assert_eq!(
        validate(&mut request),
        Err("placement source candidates lack matching output roles")
    );
}

#[test]
fn multi_node_terminal_aggregate_requires_public_output_roles() {
    let mut request = paired_request();
    request.graph.nodes[1].candidates = request.graph.nodes[0].candidates.clone();
    request.graph.nodes[1].candidates[1].output_facts = None;
    assert_eq!(
        validate(&mut request),
        Err("placement source candidates lack matching output roles")
    );
}

#[test]
fn output_byte_bounds_must_sum_to_each_aggregate_authority() {
    let mut request = paired_request();
    request.graph.nodes[0].candidates[1]
        .output_facts
        .as_mut()
        .unwrap()[0]
        .retained_bytes = ResourceByteBound::Known(3);
    assert_eq!(
        validate(&mut request),
        Err("mixed placement output byte bounds do not match aggregate resources")
    );
    request.graph.nodes[0].candidates[1]
        .output_facts
        .as_mut()
        .unwrap()[0]
        .retained_bytes = ResourceByteBound::Known(4);
    request.graph.nodes[0].candidates[1]
        .output_facts
        .as_mut()
        .unwrap()[1]
        .retained_bytes = ResourceByteBound::Known(7);
    assert_eq!(
        validate(&mut request),
        Err("mixed placement output byte bounds do not match aggregate resources")
    );
}

#[test]
fn output_authority_changes_signature_and_rejects_unbound_or_stale_graphs() {
    let mut request = paired_request();
    assert_eq!(
        request.validate(),
        Err("placement graph output authority does not match signature")
    );
    validate(&mut request).unwrap();
    let old = request.signature.clone();
    let old_key = old.cache_key();
    request.graph.nodes[0].candidates[1]
        .output_facts
        .as_mut()
        .unwrap()[0]
        .retained_bytes = ResourceByteBound::Known(5);
    request.graph.nodes[0].candidates[1]
        .resources
        .retained_bytes = ResourceByteBound::Known(5);
    assert_eq!(
        request.validate(),
        Err("placement graph output authority does not match signature")
    );
    validate(&mut request).unwrap();
    assert_ne!(request.signature.cache_key(), old_key);
    assert_ne!(
        request.signature.permit_identity().unwrap(),
        old.permit_identity().unwrap()
    );
    request.graph.edges.push(edge(Some(0)));
    assert_eq!(
        request.validate(),
        Err("placement graph output authority does not match signature")
    );
    validate(&mut request).unwrap();
    assert_ne!(request.signature.cache_key(), old_key);
}
