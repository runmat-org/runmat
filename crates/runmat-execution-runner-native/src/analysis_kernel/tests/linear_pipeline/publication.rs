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
    let mut accuracy_inputs = vec![
        accuracy_request,
        solved.numerical_receipt.clone(),
        observation_artifact,
    ];
    let plan = runmat_analysis_execution::build_prepared_accuracy_resource_plan(
        &AccuracyEvaluationRequest::from_canonical_bytes(&accuracy_inputs[0].payload()).unwrap(),
        &runmat_analysis_fea::execution::AccuracySourceReceipt::from_canonical_bytes(
            AccuracySourceKind::StructuralLinear,
            &accuracy_inputs[1].payload(),
        )
        .unwrap(),
        &accuracy_inputs,
    )
    .unwrap();
    accuracy_inputs.push(
        runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            plan.identity.digest(),
            &plan.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap(),
    );
    let accuracy_host = test_host_for_inputs(
        host,
        AnalysisWorkKind::EvaluateAccuracy,
        &accuracy_inputs,
        vec![AnalysisNumericalObjectKind::AccuracyReport],
        Some(1),
    );
    let mut exact_ceiling = plan.input.clone();
    exact_ceiling.accuracy_ceiling.maximum_resident_bytes = plan.request.memory_bytes;
    runmat_analysis_fea::planning::resource::build_resource_plan(exact_ceiling.clone()).unwrap();
    exact_ceiling.accuracy_ceiling.maximum_resident_bytes -= 1;
    assert!(runmat_analysis_fea::planning::resource::build_resource_plan(exact_ceiling).is_err());

    let mut altered_input = plan.input.clone();
    altered_input.worker.fixed_memory_bytes.0 += 1;
    let altered_plan =
        runmat_analysis_fea::planning::resource::build_resource_plan(altered_input).unwrap();
    let mut altered_inputs = accuracy_inputs.clone();
    altered_inputs[3] = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        altered_plan.identity.digest(),
        &altered_plan.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let altered_host = test_host_for_inputs(
        host,
        AnalysisWorkKind::EvaluateAccuracy,
        &altered_inputs,
        vec![AnalysisNumericalObjectKind::AccuracyReport],
        Some(1),
    );
    let error = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::EvaluateAccuracy,
            host: &altered_host,
            inputs: &altered_inputs,
        })
        .unwrap_err();
    assert!(error.to_string().contains("accuracy resource plan differs"));
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
