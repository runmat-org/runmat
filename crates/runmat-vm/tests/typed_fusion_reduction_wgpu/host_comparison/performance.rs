//! Explicit performance qualification for the two representative folded
//! reduction sizes. These tests deliberately fail on a machine whose measured
//! route ordering does not match the qualification contract; they never alter
//! production costs or force either route.

use super::*;
use runmat_accelerate::placement::{ExplorationMode, PlacementSessionConfig};
use runmat_execution::{
    CandidateObservationEvidence, EstimateConfidence, EstimateSource, ExecutionCandidateKind,
    ExecutionCostComponents, ExecutionCostEstimate, PlacementFeedback, PlacementSignature,
};

struct QualifiedWinner {
    choice: MeasuredChoice,
    cpu_candidate: String,
    provider_candidate: String,
    cpu_elapsed_ns: u64,
    cpu_confidence: EstimateConfidence,
    cpu_risk_adjusted_ns: u64,
    provider_elapsed_ns: u64,
    provider_confidence: EstimateConfidence,
    provider_risk_adjusted_ns: u64,
    cpu_receipts: usize,
    provider_receipts: usize,
    invocations: usize,
}

pub(super) fn observed_cost(
    evidence: CandidateObservationEvidence,
) -> (u64, EstimateConfidence, u64) {
    let CandidateObservationEvidence::Observed(observed) = evidence else {
        panic!("performance route lacks completed observation evidence");
    };
    let estimate = ExecutionCostEstimate {
        components: ExecutionCostComponents {
            execution_ns: observed.elapsed_ns,
            ..ExecutionCostComponents::default()
        },
        scratch_bytes: None,
        confidence: observed.confidence,
        source: EstimateSource::Observation,
    };
    (
        observed.elapsed_ns,
        observed.confidence,
        estimate
            .checked_risk_adjusted_ns()
            .expect("observed route cost must be representable"),
    )
}

fn completed_receipt_count(
    receipts: &[PlacementFeedback],
    signature: &PlacementSignature,
    candidate: &str,
    kind: ExecutionCandidateKind,
) -> usize {
    receipts
        .iter()
        .filter(|feedback| {
            feedback.signature == *signature
                && feedback.candidate == candidate
                && feedback.kind == kind
                && feedback.succeeded
                && feedback.total_elapsed_ns > 0
        })
        .count()
}

fn train_measured_routes(workload: ComparisonWorkload) -> QualifiedWinner {
    let bytecode = test_helpers::compile_source(workload.source())
        .expect("compile representative folded minimum");
    assert_compiled_folded_graph(&bytecode, 1);
    let session = Rc::new(PlacementSession::new(PlacementSessionConfig {
        exploration: ExplorationMode::TransactionalEvery { interval: 2 },
        exploration_max_overhead_basis_points: u32::MAX,
        ..PlacementSessionConfig::default()
    }));
    let placement = Rc::new(MeasuredPlacement {
        session: session.clone(),
        choices: RefCell::new(Vec::new()),
        delivered: Arc::new(Mutex::new(Vec::new())),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));

    let mut winner = None;
    for invocation in 1..=32 {
        let choice = execute_workload(workload, &bytecode, &runtime, placement.as_ref());
        let qualifying_receipts = if let (Some(cpu), Some(provider)) = (
            choice.cpu_observed_candidate.as_ref(),
            choice.warm_provider_observed_candidate.as_ref(),
        ) {
            let receipts = placement.delivered.lock().unwrap();
            [
                (cpu, ExecutionCandidateKind::SharedRuntime),
                (provider, ExecutionCandidateKind::ProviderFusion),
            ]
            .into_iter()
            .all(|(candidate, kind)| {
                completed_receipt_count(
                    &receipts[..choice.receipts_before],
                    &choice.signature,
                    candidate,
                    kind,
                ) >= 2
            })
        } else {
            false
        };
        if choice.from_cache && choice.selected_observed_before && qualifying_receipts {
            winner = Some((choice, invocation));
            break;
        }
    }
    let (choice, invocations) = winner.unwrap_or_else(|| {
        panic!(
            "{} did not train both real routes and reach a cached nonexploration decision",
            workload.name()
        )
    });
    let cpu_candidate = choice.cpu_observed_candidate.clone().unwrap();
    let provider_candidate = choice.warm_provider_observed_candidate.clone().unwrap();
    let (cpu_elapsed_ns, cpu_confidence, cpu_risk_adjusted_ns) = observed_cost(
        choice
            .cpu_observation_before
            .expect("CPU evidence was captured before the winning plan"),
    );
    let (provider_elapsed_ns, provider_confidence, provider_risk_adjusted_ns) = observed_cost(
        choice
            .warm_provider_observation_before
            .expect("provider evidence was captured before the winning plan"),
    );
    let receipts = placement.delivered.lock().unwrap();
    let completed = &receipts[..choice.receipts_before];
    let cpu_receipts = completed_receipt_count(
        completed,
        &choice.signature,
        &cpu_candidate,
        ExecutionCandidateKind::SharedRuntime,
    );
    let provider_receipts = completed_receipt_count(
        completed,
        &choice.signature,
        &provider_candidate,
        ExecutionCandidateKind::ProviderFusion,
    );
    assert!(
        cpu_receipts >= 2,
        "two exact completed CPU FullCalls before planning"
    );
    assert!(
        provider_receipts >= 2,
        "two exact completed provider FullCalls before planning"
    );
    drop(receipts);
    choice
        .resources
        .validate()
        .expect("selected performance resource snapshot is canonical");
    assert!(!choice.resources.cancellation_requested);

    QualifiedWinner {
        choice,
        cpu_candidate,
        provider_candidate,
        cpu_elapsed_ns,
        cpu_confidence,
        cpu_risk_adjusted_ns,
        provider_elapsed_ns,
        provider_confidence,
        provider_risk_adjusted_ns,
        cpu_receipts,
        provider_receipts,
        invocations,
    }
}

fn emit_evidence(workload: ComparisonWorkload, winner: &QualifiedWinner) {
    let report = serde_json::json!({
        "schema": "runmat.s1.folded-reduction-performance.v1",
        "workload": workload.name(),
        "shape": workload.input_shape(),
        "input_bytes": workload.input_shape()[0] * workload.input_shape()[1] * 4,
        "signature": winner.choice.signature.cache_key(),
        "provider_provenance": winner.choice.signature.provider_provenance,
        "resources": &winner.choice.resources,
        "offered_candidates": &winner.choice.offers,
        "invocations": winner.invocations,
        "cpu": {
            "candidate": &winner.cpu_candidate,
            "receipts": winner.cpu_receipts,
            "elapsed_ns": winner.cpu_elapsed_ns,
            "confidence": format!("{:?}", winner.cpu_confidence),
            "risk_adjusted_ns": winner.cpu_risk_adjusted_ns,
        },
        "provider": {
            "candidate": &winner.provider_candidate,
            "receipts": winner.provider_receipts,
            "elapsed_ns": winner.provider_elapsed_ns,
            "confidence": format!("{:?}", winner.provider_confidence),
            "risk_adjusted_ns": winner.provider_risk_adjusted_ns,
        },
        "selected_kind": winner.choice.kind,
        "selected_candidate": &winner.choice.candidate,
        "selected_predicted_total_ns": winner.choice.predicted_total_ns,
        "from_cache": winner.choice.from_cache,
    });
    eprintln!("S1_FOLDED_REDUCTION_PERFORMANCE={report}");
}

fn assert_selected_cost_matches_observation(winner: &QualifiedWinner) {
    let selected_risk = match winner.choice.kind {
        ExecutionCandidateKind::SharedRuntime => winner.cpu_risk_adjusted_ns,
        ExecutionCandidateKind::ProviderFusion => winner.provider_risk_adjusted_ns,
        kind => panic!("unexpected selected performance route {kind:?}"),
    };
    assert_eq!(winner.choice.predicted_total_ns, selected_risk);
}

#[test]
#[ignore = "performance qualification: requires optimized native execution and actual WGPU hardware"]
fn representative_8x8_observes_cpu_faster_and_selects_cpu() {
    runmat_gc::gc_test_context(|| {
        register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for CPU-faster qualification");
        let workload = ComparisonWorkload::Latency8x8;
        let winner = train_measured_routes(workload);
        emit_evidence(workload, &winner);
        assert!(winner.cpu_risk_adjusted_ns < winner.provider_risk_adjusted_ns);
        assert_eq!(winner.choice.kind, ExecutionCandidateKind::SharedRuntime);
        assert_eq!(winner.choice.candidate, winner.cpu_candidate);
        assert_selected_cost_matches_observation(&winner);
    });
}

#[test]
#[ignore = "performance qualification: requires optimized native execution and actual WGPU hardware"]
fn representative_2048x2048_observes_gpu_faster_and_selects_provider() {
    runmat_gc::gc_test_context(|| {
        register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for GPU-faster qualification");
        let workload = ComparisonWorkload::Throughput2048x2048;
        let winner = train_measured_routes(workload);
        emit_evidence(workload, &winner);
        assert!(winner.provider_risk_adjusted_ns < winner.cpu_risk_adjusted_ns);
        assert_eq!(winner.choice.kind, ExecutionCandidateKind::ProviderFusion);
        assert_eq!(winner.choice.candidate, winner.provider_candidate);
        assert_selected_cost_matches_observation(&winner);
    });
}
