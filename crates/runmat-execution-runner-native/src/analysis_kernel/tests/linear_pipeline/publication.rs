use super::*;
use runmat_analysis_fea::execution::{AccuracySourceKind, SequenceIdentity};

pub(super) fn execute(
    solved: &SolvedPipeline,
    observations: &runmat_analysis_fea::results::observation::ObservationPublicationCatalog,
) {
    let host = &solved.prepared.host;
    let prepared = runmat_analysis_execution::prepare_accuracy(
        runmat_analysis_execution::AccuracyPreparation {
            solve_host: host.clone(),
            source_kind: AccuracySourceKind::StructuralLinear,
            source_numerical_receipt: solved.numerical_receipt.clone(),
            source_observations: Some(observations.clone()),
            source_context_inputs: solved
                .prepared
                .inputs
                .iter()
                .filter(|input| {
                    matches!(
                        input.manifest.kind,
                        AnalysisNumericalObjectKind::SolverInvocation
                            | AnalysisNumericalObjectKind::Mesh
                    )
                })
                .cloned()
                .collect(),
            sequence: SequenceIdentity::from_solver_contract(&solved.solve.model.sequence).unwrap(),
            step: solved.solve.model.sequence.steps[0].clone(),
            formulation_requirement: solved.solve.formulation_requirement,
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
        },
    )
    .unwrap();
    let accuracy_inputs = prepared.inputs;
    let plan = prepared.resource_plan;
    let accuracy_host = prepared.host;
    let plan_index = accuracy_inputs
        .iter()
        .position(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap();
    let observations_index = accuracy_inputs
        .iter()
        .position(|input| input.manifest.kind == AnalysisNumericalObjectKind::ObservationCatalog)
        .unwrap();
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
    altered_inputs[plan_index] = runmat_analysis_execution::prepare_analysis_objects(
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
    let original_request =
        AccuracyEvaluationRequest::from_canonical_bytes(&accuracy_inputs[0].payload()).unwrap();
    let mut weakened_plan = original_request.plan.clone();
    weakened_plan.criteria[0].tolerance = runmat_analysis_core::AccuracyTolerance::Relative(0.99);
    let weakened = AccuracyEvaluationRequest::new(
        original_request.producer,
        original_request.source_kind,
        original_request.source_result,
        Some(observations),
        original_request.formulation_requirement,
        weakened_plan,
    )
    .unwrap();
    let mut weakened_inputs = accuracy_inputs.clone();
    weakened_inputs[0] = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::AccuracyEvaluationRequest,
        weakened.identity().digest(),
        &weakened.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let weakened_host = test_host_for_inputs(
        host,
        AnalysisWorkKind::EvaluateAccuracy,
        &weakened_inputs,
        vec![AnalysisNumericalObjectKind::AccuracyReport],
        Some(1),
    );
    let error = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::EvaluateAccuracy,
            host: &weakened_host,
            inputs: &weakened_inputs,
        })
        .unwrap_err();
    assert!(error.to_string().contains("authored requirements"));
    let missing_context: Vec<_> = accuracy_inputs
        .iter()
        .filter(|input| input.manifest.kind != AnalysisNumericalObjectKind::AccuracySourceContext)
        .cloned()
        .collect();
    let missing_host = test_host_for_inputs(
        host,
        AnalysisWorkKind::EvaluateAccuracy,
        &missing_context,
        vec![AnalysisNumericalObjectKind::AccuracyReport],
        Some(1),
    );
    let error = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::EvaluateAccuracy,
            host: &missing_host,
            inputs: &missing_context,
        })
        .unwrap_err();
    assert!(error.to_string().contains("unique source inputs"));
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
    let publication_request =
        ResultPublicationRequest::new(&accuracy, Some(observations), &resource_usage).unwrap();
    let publication_request_objects = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::PublicationRequest,
        publication_request.identity().digest(),
        &publication_request.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let resource_usage_objects = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourceUsage,
        resource_usage.identity().digest(),
        &resource_usage.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut publication_inputs = vec![
        publication_request_objects,
        accuracy_artifact,
        resource_usage_objects,
        accuracy_inputs[observations_index].clone(),
    ];
    let admission_plan =
        runmat_analysis_execution::build_prepared_result_publication_resource_plan(
            &publication_request,
            &accuracy,
            &resource_usage,
            &publication_inputs,
        )
        .unwrap();
    assert_ne!(
        admission_plan.identity,
        solved.prepared.resource_plan.identity
    );
    publication_inputs.push(
        runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            admission_plan.identity.digest(),
            &admission_plan.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap(),
    );
    let publication_host = test_host_for_input_references(
        host,
        AnalysisWorkKind::PublishResult,
        &publication_inputs,
        vec![AnalysisNumericalObjectKind::PublishedResult],
        &[1, 2],
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
    assert_eq!(published.source_observations, Some(observations.identity()));
    assert_eq!(published.source_accuracy, accuracy.identity());

    // Neither the source solve's valid plan nor a self-consistently rehashed publication
    // plan is authority for this task: the worker must independently derive its exact plan.
    let mut altered_input = admission_plan.input.clone();
    altered_input.worker.fixed_memory_bytes.0 += 1;
    let altered_plan =
        runmat_analysis_fea::planning::resource::build_resource_plan(altered_input).unwrap();
    for substituted_plan in [&solved.prepared.resource_plan, &altered_plan] {
        let mut substituted_inputs = publication_inputs.clone();
        substituted_inputs[4] = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            substituted_plan.identity.digest(),
            &substituted_plan.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let substituted_host = test_host_for_input_references(
            host,
            AnalysisWorkKind::PublishResult,
            &substituted_inputs,
            vec![AnalysisNumericalObjectKind::PublishedResult],
            &[1, 2],
        );
        let error = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                output_policy: output_policy(),
                kind: AnalysisWorkKind::PublishResult,
                host: &substituted_host,
                inputs: &substituted_inputs,
            })
            .unwrap_err();
        assert!(error.to_string().contains("publication inputs differ"));
    }

    let mut substituted_accuracy_inputs = accuracy_inputs;
    substituted_accuracy_inputs[observations_index]
        .manifest
        .logical_identity = digest("substituted-observations");
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::EvaluateAccuracy,
            host: &accuracy_host,
            inputs: &substituted_accuracy_inputs,
        })
        .is_err());
}
