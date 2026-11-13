//! Rehashed adversarial evidence, not stale-identity rejection.
use super::*;
use crate::execution::PublicationIdentity;
use crate::results::publication::validate_result_publication_resource_usage;

fn rebound_request(
    request: &ResultPublicationRequest,
    usage: &ResourceUsageEvidence,
) -> ResultPublicationRequest {
    let identity = Digest::sha256(
        runmat_canonical_codec::encode_contract(
            b"runmat-analysis-identity/v1\0",
            "result-publication-request",
            &(
                request.producer,
                request.source_result,
                request.source_observations,
                request.accuracy_request,
                request.source_accuracy,
                usage.primary_plan,
                usage.identity(),
            ),
            CanonicalLimits::new(1024 * 1024, 65_536, 65_536, 32),
        )
        .unwrap(),
    );
    let mut wire = serde_json::to_value(request).unwrap();
    wire["identity"] = serde_json::to_value(identity).unwrap();
    wire["source_resource_plan"] = serde_json::to_value(usage.primary_plan).unwrap();
    wire["source_resource_usage"] = serde_json::to_value(usage.identity()).unwrap();
    let request: ResultPublicationRequest = serde_json::from_value(wire).unwrap();
    request.validate().unwrap();
    request
}

fn rebound_receipt(
    receipt: &PublishedAnalysisResult,
    request: &ResultPublicationRequest,
) -> PublishedAnalysisResult {
    let identity = PublicationIdentity::new(
        request.identity(),
        request.source_result,
        request.source_observations.map(|id| id.digest()),
        request.source_accuracy.digest(),
        request.source_resource_plan.digest(),
        request.source_resource_usage.digest(),
    )
    .unwrap();
    let mut wire = serde_json::to_value(receipt).unwrap();
    wire["identity"] = serde_json::to_value(identity).unwrap();
    wire["request"] = serde_json::to_value(request.identity()).unwrap();
    wire["source_resource_plan"] = serde_json::to_value(request.source_resource_plan).unwrap();
    wire["source_resource_usage"] = serde_json::to_value(request.source_resource_usage).unwrap();
    serde_json::from_value(wire).unwrap()
}

fn terminal_usage(
    request: &ResultPublicationRequest,
    accuracy: &AccuracyArtifact,
    valid_usage: &ResourceUsageEvidence,
    baseline: &ResourceUsageEvidence,
    rebound: &ResultPublicationRequest,
) -> ResourceUsageEvidence {
    let profile = published_result_encoding_profile(request, accuracy, valid_usage).unwrap();
    let valid_plan = build_result_publication_resource_plan(
        request,
        accuracy,
        valid_usage,
        None,
        source_facts(
            request,
            accuracy,
            valid_usage,
            profile.encoded_bytes_upper_bound,
        ),
    )
    .unwrap();
    // Rebuild the plan's canonical identity too. A source-aware admission boundary must reject
    // this graph even though its scalar demand, allocations, totals and hashes are self-consistent.
    let mut input = valid_plan.input;
    input.phase = FeaResourcePhase::ResultPublication(rebound.identity());
    let FeaResourceDemand::ResultPublication(demand) = &mut input.demand else {
        panic!("publication demand")
    };
    demand.request = rebound.identity();
    demand.source_usage = baseline.identity();
    let plan = build_resource_plan(input).unwrap();
    let mut attempts = baseline.attempts.clone();
    attempts.push(ResourceUsageAttempt {
        attempt_id: AttemptId::derive(&[b"terminal-publication"]),
        worker_id: WorkerId::derive(&[b"worker"]),
        plan: plan.identity,
        resources: ResourceAccounting::for_reserved_interval(plan.request.clone(), 1, 0, 0, 0)
            .unwrap(),
    });
    let mut plans = baseline
        .plans
        .iter()
        .filter(|p| p.identity != baseline.primary_plan)
        .cloned()
        .collect::<Vec<_>>();
    plans.push(plan);
    record_resource_graph_usage(
        baseline.primary_plan().unwrap(),
        plans,
        baseline.workload_digest,
        baseline.result_digest,
        attempts,
        baseline.actual.scheduler.wall_millis + 1,
    )
    .unwrap()
}

#[test]
fn unrelated_accuracy_cannot_satisfy_publication_even_after_rehashing_every_boundary() {
    let (request, accuracy, usage) = fixture(1, "residual", true);
    let (_, _, foreign) = fixture(1, "different", true);
    assert_eq!(usage.result_digest, foreign.result_digest);
    assert_eq!(usage.attempts.len(), foreign.attempts.len());
    foreign.validate_identity().unwrap();
    assert!(ResultPublicationRequest::new(&accuracy, None, &foreign).is_err());
    let rebound = rebound_request(&request, &foreign);
    assert!(PublishedAnalysisResult::new(&rebound, &accuracy, &foreign).is_err());
    assert!(published_result_encoding_profile(&rebound, &accuracy, &foreign).is_err());

    let receipt = PublishedAnalysisResult::new(&request, &accuracy, &usage).unwrap();
    let forged = rebound_receipt(&receipt, &rebound);
    forged.validate().unwrap();
    assert!(forged.validate_sources(&accuracy, &foreign).is_err());
    let terminal = terminal_usage(&request, &accuracy, &usage, &foreign, &rebound);
    assert!(terminal.overruns.is_empty());
    let error =
        validate_result_publication_resource_usage(&forged, &foreign, &terminal, 1).unwrap_err();
    assert!(error
        .to_string()
        .contains("source-bound accuracy scheduler attempt"));
}

#[test]
fn accuracy_request_in_receipt_is_bound_to_the_original_publication_request() {
    let (request, accuracy, usage) = fixture(1, "residual", true);
    let (other, _, _) = fixture(1, "different", true);
    let mut receipt = PublishedAnalysisResult::new(&request, &accuracy, &usage).unwrap();
    let mut omitted = serde_json::to_value(&receipt).unwrap();
    omitted.as_object_mut().unwrap().remove("accuracy_request");
    assert!(serde_json::from_value::<PublishedAnalysisResult>(omitted).is_err());
    let mut previous_schema = serde_json::to_value(&receipt).unwrap();
    previous_schema["schema_version"] = serde_json::json!(2);
    assert!(serde_json::from_value::<PublishedAnalysisResult>(previous_schema).is_err());
    receipt.accuracy_request = other.accuracy_request;
    assert!(receipt.validate().is_err());
    assert!(serde_json::from_value::<PublishedAnalysisResult>(
        serde_json::to_value(&receipt).unwrap()
    )
    .is_err());
}

#[test]
fn matching_request_with_substituted_result_cannot_hide_beside_valid_accounting() {
    let (request, accuracy, usage) = fixture(1, "residual", true);
    let mut input = usage.primary_plan().unwrap().input.clone();
    let FeaResourceDemand::AccuracyEvaluation(demand) = &mut input.demand else {
        panic!("accuracy demand")
    };
    demand.source_result = SolverResultIdentity::from_step_result(
        request.producer,
        Digest::sha256(b"different result"),
    )
    .unwrap();
    let foreign_plan = build_resource_plan(input).unwrap();
    let mut attempts = usage.attempts.clone();
    attempts.push(ResourceUsageAttempt {
        attempt_id: AttemptId::derive(&[b"foreign-accuracy"]),
        worker_id: WorkerId::derive(&[b"worker"]),
        plan: foreign_plan.identity,
        resources: ResourceAccounting::for_reserved_interval(
            foreign_plan.request.clone(),
            1,
            0,
            0,
            0,
        )
        .unwrap(),
    });
    let mixed = record_resource_graph_usage(
        usage.primary_plan().unwrap(),
        vec![foreign_plan],
        usage.workload_digest,
        usage.result_digest,
        attempts,
        2,
    )
    .unwrap();
    mixed.validate_identity().unwrap();
    assert!(mixed.overruns.is_empty());
    assert!(ResultPublicationRequest::new(&accuracy, None, &mixed).is_err());
}

#[test]
fn repeated_accuracy_attempts_are_preserved_in_terminal_usage() {
    let (request, accuracy, usage) = fixture(3, "residual", true);
    let receipt = PublishedAnalysisResult::new(&request, &accuracy, &usage).unwrap();
    receipt.validate_sources(&accuracy, &usage).unwrap();
    let terminal = terminal_usage(&request, &accuracy, &usage, &usage, &request);
    validate_result_publication_resource_usage(&receipt, &usage, &terminal, 1).unwrap();
    assert_eq!(terminal.attempts.len(), 4);
    for original in &usage.attempts {
        assert!(terminal.attempts.contains(original));
    }
}
