use super::*;
use runmat_analysis_fea::execution::AccuracySourceKind;

pub(super) fn execute(solved: &SolvedPipeline, observations: &ObservationArtifactManifest) {
    let host = &solved.prepared.host;
    let step = host.workload.step;
    let accuracy_request = AccuracyEvaluationRequest::new(
        step,
        AccuracySourceKind::StructuralLinear,
        solved.source_result,
        Some(observations),
        solved.solve.formulation_requirement,
        solved.solve.model.sequence.steps[0].accuracy.clone(),
    )
    .unwrap();
    let accuracy_request = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::AccuracyEvaluationRequest,
        accuracy_request.identity().digest(),
        &accuracy_request.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let observation_artifact = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationCatalog,
        observations.identity.digest(),
        &observations.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let accuracy_inputs = vec![
        accuracy_request,
        solved.numerical_receipt.clone(),
        observation_artifact,
    ];
    let accuracy_host = test_host_for_inputs(
        host,
        AnalysisWorkKind::EvaluateAccuracy,
        &accuracy_inputs,
        vec![AnalysisNumericalObjectKind::AccuracyReport],
        Some(1),
    );
    let accuracy_outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::EvaluateAccuracy,
            host: &accuracy_host,
            inputs: &accuracy_inputs,
        })
        .unwrap();
    let accuracy =
        AccuracyArtifact::from_canonical_bytes(accuracy_outputs[0].whole_payload().unwrap())
            .unwrap();
    assert_eq!(
        accuracy.evaluation().report().status,
        AccuracyStatus::Satisfied
    );
    assert_eq!(accuracy.gate().decision, PublicationDecision::Allow);
    assert_eq!(accuracy.source_result, solved.source_result);

    let accuracy_artifact = runmat_analysis_execution::prepare_analysis_objects(
        accuracy_outputs[0].kind,
        accuracy_outputs[0].logical_identity,
        accuracy_outputs[0].whole_payload().unwrap(),
        4096,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let resource_usage = record_resource_usage(
        &solved.prepared.resource_plan,
        host.workload.identity.digest(),
        solved.source_result.digest(),
        vec![ResourceUsageAttempt {
            attempt_id: runmat_execution::identity::AttemptId::derive(&[b"kernel-attempt"]),
            worker_id: runmat_execution::identity::WorkerId::derive(&[b"kernel-worker"]),
            plan: solved.prepared.resource_plan.identity,
            resources: ResourceAccounting::for_reserved_interval(
                solved.prepared.resource_plan.request.clone(),
                1,
                0,
                0,
                0,
            )
            .unwrap(),
        }],
        1,
    )
    .unwrap();
    let publication_request = ResultPublicationRequest::new(
        &accuracy,
        Some(observations),
        &solved.prepared.resource_plan,
        &resource_usage,
    )
    .unwrap();
    let publication_request = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::PublicationRequest,
        publication_request.identity().digest(),
        &publication_request.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let resource_usage = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourceUsage,
        resource_usage.identity().digest(),
        &resource_usage.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let publication_inputs = vec![
        publication_request,
        accuracy_artifact,
        solved.prepared.inputs[3].clone(),
        resource_usage,
        accuracy_inputs[2].clone(),
    ];
    let publication_host = test_host_for_input_references(
        host,
        AnalysisWorkKind::PublishResult,
        &publication_inputs,
        vec![AnalysisNumericalObjectKind::PublishedResult],
        &[1, 3],
    );
    let publication_outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::PublishResult,
            host: &publication_host,
            inputs: &publication_inputs,
        })
        .unwrap();
    let published = PublishedAnalysisResult::from_canonical_bytes(
        publication_outputs[0].whole_payload().unwrap(),
    )
    .unwrap();
    assert_eq!(published.source_result, solved.source_result);
    assert_eq!(published.source_observations, Some(observations.identity));
    assert_eq!(published.source_accuracy, accuracy.identity());

    let mut substituted_accuracy_inputs = accuracy_inputs;
    substituted_accuracy_inputs[2].manifest.logical_identity = digest("substituted-observations");
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::EvaluateAccuracy,
            host: &accuracy_host,
            inputs: &substituted_accuracy_inputs,
        })
        .is_err());
}
