use runmat_execution::{
    ExecutionCandidateKind, PlacementFeedback, PlacementGraph, PlacementGraphEdge,
    PlacementGraphLimits, PlacementGraphNode, PlacementPlanRequest,
};

use super::super::tests::{candidate, resources, signature};
use super::super::{ExplorationMode, PlacementSession, PlacementSessionConfig};

fn three_route_request(token: &str) -> PlacementPlanRequest {
    PlacementPlanRequest {
        signature: signature(token),
        graph: PlacementGraph {
            nodes: vec![PlacementGraphNode {
                identity: "reduction".into(),
                candidates: vec![
                    candidate(
                        "gpu.a",
                        ExecutionCandidateKind::ProviderFusion,
                        10_000,
                        64,
                        true,
                    ),
                    candidate(
                        "gpu.b",
                        ExecutionCandidateKind::ProviderFusion,
                        12_000,
                        64,
                        true,
                    ),
                    candidate(
                        "cpu",
                        ExecutionCandidateKind::SharedRuntime,
                        30_000,
                        0,
                        true,
                    ),
                ],
            }],
            edges: Vec::new(),
        },
        limits: PlacementGraphLimits::default(),
        resources: resources(1_024),
        deterministic: false,
        require_transactional_results: true,
    }
}

fn exploratory_session() -> PlacementSession {
    PlacementSession::new(PlacementSessionConfig {
        exploration: ExplorationMode::TransactionalEvery { interval: 1 },
        exploration_max_overhead_basis_points: u32::MAX,
        ..PlacementSessionConfig::default()
    })
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn three_admitted_routes_each_receive_completed_evidence_before_repeating() {
    let session = exploratory_session();
    let request = three_route_request("three-route-evidence");
    let mut selected = Vec::new();
    for _ in 0..4 {
        let decision = session.plan(request.clone()).expect("admitted route");
        let choice = &decision.selections[0];
        selected.push(choice.candidate.clone());
        session
            .observe(PlacementFeedback {
                signature: request.signature.clone(),
                candidate: choice.candidate.clone(),
                kind: choice.kind,
                total_elapsed_ns: match choice.candidate.as_str() {
                    "gpu.b" => 12_000,
                    "cpu" => 30_000,
                    _ => panic!("the best route is not an exploration alternative"),
                },
                succeeded: true,
            })
            .expect("completed route feedback");
    }
    assert_eq!(selected, ["gpu.b", "cpu", "gpu.b", "cpu"]);
    let best = session.plan(request.clone()).expect("trained best route");
    assert_eq!(best.selections[0].candidate, "gpu.a");
    assert!(!best.from_cache);
    let cached = session.plan(request).expect("trained cached route");
    assert_eq!(cached.selections, best.selections);
    assert!(cached.from_cache);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn canonical_cost_order_breaks_unobserved_ties_after_candidate_reordering() {
    let mut request = three_route_request("stable-alternate-order");
    request.graph.nodes[0].candidates[1]
        .descriptor
        .cost
        .components
        .execution_ns = 20_000;
    request.graph.nodes[0].candidates[2]
        .descriptor
        .cost
        .components
        .execution_ns = 20_000;
    let original = exploratory_session()
        .plan(request.clone())
        .expect("tied alternatives");
    request.graph.nodes[0].candidates.swap(1, 2);
    let selected = exploratory_session().plan(request).expect("admitted route");
    assert_eq!(selected.selections, original.selections);
    assert_eq!(selected.selections[0].candidate, "cpu");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn two_node_exploration_preserves_transition_cost_and_changed_candidate_evidence() {
    let mut request = three_route_request("two-node-transition");
    request.graph.nodes = vec![
        PlacementGraphNode {
            identity: "first".into(),
            candidates: vec![
                candidate(
                    "gpu.first",
                    ExecutionCandidateKind::ProviderFusion,
                    10_000,
                    64,
                    true,
                ),
                candidate(
                    "cpu.first",
                    ExecutionCandidateKind::SharedRuntime,
                    30_000,
                    0,
                    true,
                ),
            ],
        },
        PlacementGraphNode {
            identity: "second".into(),
            candidates: vec![
                candidate(
                    "gpu.second",
                    ExecutionCandidateKind::ProviderFusion,
                    10_000,
                    64,
                    true,
                ),
                candidate(
                    "cpu.second",
                    ExecutionCandidateKind::SharedRuntime,
                    30_000,
                    0,
                    true,
                ),
            ],
        },
    ];
    request.graph.edges = vec![PlacementGraphEdge {
        from: 0,
        to: 1,
        source_output: None,
        bytes: 64,
        host_to_provider_ns: 1_000,
        provider_to_host_ns: 1_000,
        cross_provider_ns: 1_000,
    }];
    let session = exploratory_session();
    let first = session.plan(request.clone()).expect("first mixed route");
    assert_eq!(first.predicted_total_ns, 41_000);
    assert_eq!(
        first
            .selections
            .iter()
            .filter(|choice| choice.kind.is_provider())
            .count(),
        1
    );
    for choice in &first.selections {
        session
            .observe(PlacementFeedback {
                signature: request.signature.clone(),
                candidate: choice.candidate.clone(),
                kind: choice.kind,
                total_elapsed_ns: 41_000,
                succeeded: true,
            })
            .expect("completed selected-node feedback");
    }
    let second = session.plan(request).expect("other mixed route");
    assert_eq!(second.predicted_total_ns, 41_000);
    assert_eq!(
        second
            .selections
            .iter()
            .filter(|choice| choice.kind.is_provider())
            .count(),
        1
    );
    assert_ne!(second.selections, first.selections);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn resource_and_transactional_admission_exclude_ineligible_alternatives() {
    let mut resource_bound = three_route_request("resource-bound-alternate");
    resource_bound.graph.nodes[0].candidates[1]
        .resources
        .retained_bytes = 2_048.into();
    let selected = exploratory_session()
        .plan(resource_bound)
        .expect("CPU remains admitted");
    assert_eq!(selected.selections[0].candidate, "cpu");

    let mut nontransactional = three_route_request("transactional-alternate");
    nontransactional.graph.nodes[0].candidates[1].transactional_results = false;
    let selected = exploratory_session()
        .plan(nontransactional)
        .expect("CPU remains transactional");
    assert_eq!(selected.selections[0].candidate, "cpu");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cache_hit_and_deterministic_requests_do_not_explore() {
    let session = exploratory_session();
    let request = three_route_request("cached-alternate");
    let first = session.plan(request.clone()).expect("first plan");
    assert_eq!(first.selections[0].candidate, "gpu.b");
    let cached = session.plan(request).expect("cached plan");
    assert!(cached.from_cache);
    assert_eq!(cached.selections, first.selections);

    let mut deterministic = three_route_request("deterministic-alternate");
    deterministic.deterministic = true;
    let selected = exploratory_session()
        .plan(deterministic)
        .expect("deterministic plan");
    assert_eq!(selected.selections[0].candidate, "gpu.a");
}
