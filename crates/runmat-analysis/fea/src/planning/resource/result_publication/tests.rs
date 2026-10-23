use super::*;
use crate::execution::{
    AccuracyEvaluationRequest, AccuracySourceKind, SequenceIdentity, SolverResultIdentity,
    StepIdentity,
};
use crate::results::accuracy::{
    accuracy_evaluation_resource_profile, evaluate_accuracy_artifact, AccuracyEvidence,
    AccuracyEvidenceScope, AccuracyMeasure,
};
use crate::results::publication::PublishedAnalysisResult;
use runmat_analysis_core::{
    AccuracyPlan, AccuracyTarget, ErrorContributionKind, ProcedureStep, RequiredAccuracyEvidence,
};
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use runmat_execution::{
    identity::{AttemptId, WorkerId},
    resource::ResourceAccounting,
    Digest,
};

mod accuracy_usage;

fn facts(output: u64) -> PreparedPhaseResourceFacts {
    PreparedPhaseResourceFacts {
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(10_000),
            transport_closure_bytes: ByteCount(20_000),
            decoded_resident_bytes_admission_budget: ByteCount(1_000_000),
        },
        output: projected_canonical_output_resource_facts(output).unwrap(),
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(4096),
        },
    }
}

fn source_facts(
    request: &ResultPublicationRequest,
    accuracy: &AccuracyArtifact,
    usage: &ResourceUsageEvidence,
    output: u64,
) -> PreparedPhaseResourceFacts {
    let payloads = [
        request.canonical_bytes().unwrap(),
        accuracy.canonical_bytes().unwrap(),
        usage.canonical_bytes().unwrap(),
    ];
    let mut facts = facts(output);
    facts.inputs.encoded_payload_bytes =
        ByteCount(payloads.iter().map(|bytes| bytes.len() as u64).sum());
    facts.inputs.transport_closure_bytes = ByteCount(facts.inputs.encoded_payload_bytes.0 + 16_384);
    facts.inputs.decoded_resident_bytes_admission_budget = ByteCount(
        payloads
            .iter()
            .map(|bytes| {
                runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(
                    bytes,
                )
                .unwrap()
            })
            .sum(),
    );
    facts
}

fn fixture(
    attempts: usize,
    criterion: &str,
    satisfied: bool,
) -> (
    ResultPublicationRequest,
    AccuracyArtifact,
    ResourceUsageEvidence,
) {
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::new(Digest::sha256(b"sequence")).unwrap(),
        &ProcedureStep::structural_static("producer"),
    )
    .unwrap();
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256(b"result")).unwrap();
    let mut plan = AccuracyPlan::baseline("publication-accuracy");
    plan.criteria[0].criterion_id = criterion.into();
    let request = AccuracyEvaluationRequest::new(
        producer,
        AccuracySourceKind::StructuralLinear,
        result,
        None,
        plan,
    )
    .unwrap();
    let evidence = if satisfied {
        vec![AccuracyEvidence::new(
            AccuracyTarget::GlobalEquilibrium,
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ResidualBound,
            AccuracyMeasure::Relative(1.0e-9),
            vec!["reference residual".into()],
            AccuracyEvidenceScope::Step(producer),
        )
        .unwrap()]
    } else {
        vec![]
    };
    let profile = accuracy_evaluation_resource_profile(&request, &evidence).unwrap();
    let primary = build_accuracy_resource_plan(
        &request,
        &evidence,
        facts(profile.output_encoded_bytes_upper_bound),
    )
    .unwrap();
    let accuracy = evaluate_accuracy_artifact(&request, evidence).unwrap();
    let attempts = (0..attempts)
        .map(|i| ResourceUsageAttempt {
            attempt_id: AttemptId::derive(&[b"publication-source", &(i as u64).to_le_bytes()]),
            worker_id: WorkerId::derive(&[b"worker"]),
            plan: primary.identity,
            resources: ResourceAccounting::for_reserved_interval(
                primary.request.clone(),
                1,
                0,
                0,
                0,
            )
            .unwrap(),
        })
        .collect::<Vec<_>>();
    let usage = record_resource_usage(
        &primary,
        Digest::sha256(b"workload"),
        result.digest(),
        attempts,
        1,
    )
    .unwrap();
    let publication = ResultPublicationRequest::new(&accuracy, None, &usage).unwrap();
    (publication, accuracy, usage)
}

#[test]
fn borrowed_receipt_profile_matches_shared_actual_serializer_and_canonical_plan() {
    for criterion in ["residual".to_string(), "r".repeat(2048)] {
        let (request, accuracy, usage) = fixture(1, &criterion, true);
        let projected = published_result_encoding_profile(&request, &accuracy, &usage).unwrap();
        let actual = PublishedAnalysisResult::new(&request, &accuracy, &usage).unwrap();
        let actual_profile = profile_serialized(
            &actual,
            CanonicalLimits::new(8 * 1024 * 1024, 1_000_000, 65_536, 64),
        )
        .unwrap();
        assert_eq!(projected, actual_profile);
        let bytes = actual.canonical_bytes().unwrap();
        assert!(bytes.len() as u64 <= projected.encoded_bytes_upper_bound);
        assert_eq!(
            actual,
            PublishedAnalysisResult::from_canonical_bytes(&bytes).unwrap()
        );
        let plan = build_result_publication_resource_plan(
            &request,
            &accuracy,
            &usage,
            None,
            source_facts(
                &request,
                &accuracy,
                &usage,
                projected.encoded_bytes_upper_bound,
            ),
        )
        .unwrap();
        assert_eq!(
            plan,
            FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap()
        );
        assert_eq!(usage.primary_plan().unwrap().identity, usage.primary_plan);
    }
}

#[test]
fn receipt_projection_and_constructor_reject_substitutions_and_blocked_gate() {
    let (request, accuracy, usage) = fixture(1, "residual", true);
    let (_, other_accuracy, _) = fixture(1, "other", true);
    assert!(published_result_encoding_profile(&request, &other_accuracy, &usage).is_err());
    let (_, _, other_usage) = fixture(2, "residual", true);
    assert!(published_result_encoding_profile(&request, &accuracy, &other_usage).is_err());
    let (blocked_request, blocked_accuracy, blocked_usage) = fixture(1, "residual", false);
    assert!(
        published_result_encoding_profile(&blocked_request, &blocked_accuracy, &blocked_usage)
            .is_err()
    );
    assert!(
        PublishedAnalysisResult::new(&blocked_request, &blocked_accuracy, &blocked_usage).is_err()
    );
}

#[test]
fn receipt_and_validation_underreservation_are_rejected() {
    let (request, accuracy, usage) = fixture(1, "residual", true);
    let profile = published_result_encoding_profile(&request, &accuracy, &usage).unwrap();
    let mut output_facts = source_facts(
        &request,
        &accuracy,
        &usage,
        profile.encoded_bytes_upper_bound,
    );
    output_facts.output.encoded_payload_bytes_upper_bound.0 -= 1;
    assert!(build_result_publication_resource_plan(
        &request,
        &accuracy,
        &usage,
        None,
        output_facts
    )
    .is_err());
    let mut output_facts = source_facts(
        &request,
        &accuracy,
        &usage,
        profile.encoded_bytes_upper_bound,
    );
    output_facts
        .output
        .construction_resident_bytes_admission_budget
        .0 = runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(
        profile.encoded_bytes_upper_bound,
        1,
    )
    .unwrap()
        + output_facts.output.transport_closure_bytes_upper_bound.0
        - 1;
    assert!(build_result_publication_resource_plan(
        &request,
        &accuracy,
        &usage,
        None,
        output_facts
    )
    .is_err());
    let plan = build_result_publication_resource_plan(
        &request,
        &accuracy,
        &usage,
        None,
        source_facts(
            &request,
            &accuracy,
            &usage,
            profile.encoded_bytes_upper_bound,
        ),
    )
    .unwrap();
    let mut input = plan.input;
    let FeaResourceDemand::ResultPublication(demand) = &mut input.demand else {
        panic!("publication demand")
    };
    assert_eq!(
        demand.observation_task_index_workspace_bytes.0,
        (usage.plans.len() * std::mem::size_of::<&FeaResourcePlan>()) as u64
    );
    assert_eq!(demand.observation_execution_workspace_bytes, ByteCount(0));
    let mut omitted_workspace = input.clone();
    let FeaResourceDemand::ResultPublication(omitted) = &mut omitted_workspace.demand else {
        panic!("publication demand")
    };
    omitted.observation_execution_workspace_bytes = ByteCount(1);
    assert!(build_resource_plan(omitted_workspace).is_err());
    let FeaResourceDemand::ResultPublication(demand) = &mut input.demand else {
        panic!("publication demand")
    };
    demand.source_validation_workspace_bytes.0 -= 1;
    assert!(build_resource_plan(input).is_err());
}

#[test]
fn validation_reservation_scales_with_measured_attempt_inventory_not_receipt_size() {
    let build = |attempts| {
        let (request, accuracy, usage) = fixture(attempts, "residual", true);
        let profile = published_result_encoding_profile(&request, &accuracy, &usage).unwrap();
        build_result_publication_resource_plan(
            &request,
            &accuracy,
            &usage,
            None,
            source_facts(
                &request,
                &accuracy,
                &usage,
                profile.encoded_bytes_upper_bound,
            ),
        )
        .unwrap()
    };
    let small = build(1);
    let large = build(256);
    let FeaResourceDemand::ResultPublication(small_demand) = &small.input.demand else {
        panic!("publication")
    };
    let FeaResourceDemand::ResultPublication(large_demand) = &large.input.demand else {
        panic!("publication")
    };
    assert_eq!(
        small_demand.receipt_encoded_bytes_upper_bound,
        large_demand.receipt_encoded_bytes_upper_bound
    );
    assert!(
        large_demand.accounting_interval_workspace_bytes
            > small_demand.accounting_interval_workspace_bytes
    );
    assert!(large.request.memory_bytes > small.request.memory_bytes);
}
