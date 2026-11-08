use super::*;

pub(super) fn execute(solved: &SolvedPipeline, recovered: &RecoveredPipeline) -> ObservedPipeline {
    let definitions = recovered
        .artifact
        .fields()
        .iter()
        .map(|field| field.definition().clone())
        .collect::<Vec<_>>();
    let solver_mesh = SolverMeshArtifact::canonical_decode(&recovered.inputs[3].payload()).unwrap();
    let observation_plan =
        runmat_analysis_fea::results::observation::resolve_structural_observation_plan(
            &solved.solve.model.sequence.steps[0],
            &solved.solve.model,
            Some(&solver_mesh),
            &solved.constraint_plan,
            &definitions,
            runmat_analysis_fea::results::observation::ObservationLimits::default(),
        )
        .unwrap();
    let recovery_request =
        FieldRecoveryRequest::from_canonical_bytes(&recovered.inputs[0].payload()).unwrap();
    let catalog = runmat_analysis_fea::results::observation::ObservationFieldCatalog::structural(
        recovery_request.clone(),
        recovered.artifact.manifest().unwrap(),
    )
    .unwrap();
    let selected_fields = catalog.canonical_selected_fields(
        &observation_plan,
        runmat_analysis_fea::results::observation::ObservationSelectionCoordinate::StructuralRecovery,
    ).unwrap();
    let field_selection =
        runmat_analysis_fea::results::observation::ObservationFieldSelection::structural(
            &recovery_request,
            &recovered.artifact.manifest().unwrap(),
            selected_fields,
        )
        .unwrap();
    let source_request =
        ObservationEvaluationRequest::new(field_selection, observation_plan).unwrap();
    let catalog_objects = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationFieldCatalog,
        catalog.identity(),
        &catalog.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let reference_catalog = RecoveredFieldCatalog::new_for_topology(
        recovered.artifact.topology_digest,
        recovered.artifact.entity_counts,
        recovered.artifact.fields().to_vec(),
    )
    .unwrap();
    let reference_result =
        runmat_analysis_fea::results::observation::evaluate_structural_observations(
            &source_request.plan,
            &solved.solve.model,
            Some(&solver_mesh),
            &reference_catalog,
        )
        .unwrap()
        .remove(0);
    let partition_plan = ObservationPartitionPlan::derive(
        &source_request.plan,
        &source_request.plan.observations[0].spec.observation_id,
        solved.solve.observation_execution,
    )
    .unwrap();
    let source_field = recovered
        .artifact
        .fields()
        .iter()
        .find(|field| {
            field.definition().identity()
                == source_request.plan.observations[0].source_field_definition_identity
        })
        .unwrap();
    let source_request_objects = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationEvaluationRequest,
        source_request.identity().digest(),
        &source_request.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let plan_objects = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationPartitionPlan,
        partition_plan.identity.digest(),
        &partition_plan.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut contribution_objects = Vec::new();
    let mut partition_reduction_plans = Vec::new();
    let mut partition_reduction_attempts = Vec::new();
    for partition in &partition_plan.partitions {
        let required_indices = runmat_analysis_fea::results::observation::structural_observation_partition_source_indices(
            &source_request.plan,
            &partition_plan,
            partition,
            &solved.solve.model,
            Some(&solver_mesh),
        )
        .unwrap();
        // Workers consume published canonical blocks, not newly sliced field identities.
        let source_fields = runmat_analysis_fea::results::field::RecoveredFieldPartitionSet::new(
            runmat_analysis_fea::results::field::canonical_field_partitions(source_field)
                .unwrap()
                .into_iter()
                .filter(|block| {
                    block
                        .source_indices
                        .iter()
                        .any(|index| required_indices.binary_search(index).is_ok())
                })
                .collect(),
        )
        .unwrap();
        let partition_request = ObservationPartitionEvaluationRequest::new(
            &source_request,
            &partition_plan,
            partition,
            &source_fields,
            None,
        )
        .unwrap();
        let partition_request_objects = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ObservationPartitionEvaluationRequest,
            partition_request.identity().digest(),
            &partition_request.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let field_partition_objects = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::FieldPartition,
            source_fields.identity.digest(),
            &source_fields.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let mut inputs = vec![
            partition_request_objects,
            source_request_objects.clone(),
            plan_objects.clone(),
            field_partition_objects,
            catalog_objects.clone(),
            recovered.inputs[1].clone(),
            recovered.inputs[3].clone(),
        ];
        let input_facts = prepared_input_resource_facts(&inputs).unwrap();
        let payload_ceiling =
            runmat_analysis_fea::planning::resource::observation_partition_payload_bytes_upper_bound(
                &source_request,
                &partition_request,
                &partition_plan,
                partition,
                &source_fields,
                false,
            )
            .unwrap();
        let resource_plan =
            build_observation_partition_resource_plan(ObservationPartitionResourcePlanning {
                source: &source_request,
                request: &partition_request,
                partition_plan: &partition_plan,
                partition,
                source_fields: &source_fields,
                inputs: input_facts,
                output: runmat_analysis_execution::projected_output_resource_facts(payload_ceiling)
                    .unwrap(),
                output_set: runmat_analysis_execution::projected_output_set_resource_facts(),
                accelerator: None,
                source_workspace:
                    runmat_analysis_fea::planning::resource::ObservationPartitionSourceWorkspace::InputEnvelope,
            })
            .unwrap();
        inputs.push(resource_plan_objects(&resource_plan));
        let host = test_host_for_inputs(
            &recovered.host,
            AnalysisWorkKind::EvaluateObservationPartition,
            &inputs,
            vec![AnalysisNumericalObjectKind::ObservationPartitionContribution],
            None,
        );
        assert_substituted_resource_plan_rejected(
            &host,
            AnalysisWorkKind::EvaluateObservationPartition,
            &inputs,
            &resource_plan,
        );
        let mut missing_catalog = inputs.clone();
        missing_catalog.retain(|input| {
            input.manifest.kind != AnalysisNumericalObjectKind::ObservationFieldCatalog
        });
        let missing_catalog_host = test_host_for_inputs(
            &recovered.host,
            AnalysisWorkKind::EvaluateObservationPartition,
            &missing_catalog,
            vec![AnalysisNumericalObjectKind::ObservationPartitionContribution],
            None,
        );
        assert!(
            NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    output_policy: output_policy(),
                    kind: AnalysisWorkKind::EvaluateObservationPartition,
                    host: &missing_catalog_host,
                    inputs: &missing_catalog,
                })
                .is_err(),
            "a valid selection is not a substitute for its producing catalog"
        );
        let outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                output_policy: output_policy(),
                kind: AnalysisWorkKind::EvaluateObservationPartition,
                host: &host,
                inputs: &inputs,
            })
            .unwrap();
        contribution_objects.push(prepare_kernel_output(&outputs[0]));
        partition_reduction_attempts.push(executed_attempt(&resource_plan));
        partition_reduction_plans.push(resource_plan);
    }

    let contributions = contribution_objects
        .iter()
        .map(|objects| {
            ObservationPartitionContribution::from_canonical_bytes(&objects.payload()).unwrap()
        })
        .collect::<Vec<_>>();
    let reduction_request =
        ObservationPartitionReductionRequest::new(&source_request, &partition_plan, &contributions)
            .unwrap();
    let reduction_request_objects = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationPartitionReductionRequest,
        reduction_request.identity().digest(),
        &reduction_request.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut reduction_inputs = vec![
        reduction_request_objects,
        source_request_objects,
        plan_objects,
    ];
    reduction_inputs.extend(contribution_objects);
    let input_facts = prepared_input_resource_facts(&reduction_inputs).unwrap();
    let payload_ceiling =
        runmat_analysis_fea::planning::resource::observation_reduction_payload_bytes_upper_bound(
            &input_facts,
            &contributions,
        )
        .unwrap();
    let reduction_resource_plan = build_observation_reduction_resource_plan(
        &source_request,
        &reduction_request,
        &partition_plan,
        &contributions,
        input_facts,
        runmat_analysis_execution::projected_output_resource_facts(payload_ceiling).unwrap(),
        runmat_analysis_execution::projected_output_set_resource_facts(),
    )
    .unwrap();
    reduction_inputs.push(resource_plan_objects(&reduction_resource_plan));
    let reduction_host = test_host_for_inputs(
        &recovered.host,
        AnalysisWorkKind::ReduceObservationPartitions,
        &reduction_inputs,
        vec![AnalysisNumericalObjectKind::ObservationResult],
        None,
    );
    assert_substituted_resource_plan_rejected(
        &reduction_host,
        AnalysisWorkKind::ReduceObservationPartitions,
        &reduction_inputs,
        &reduction_resource_plan,
    );
    let reduction_outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::ReduceObservationPartitions,
            host: &reduction_host,
            inputs: &reduction_inputs,
        })
        .unwrap();
    partition_reduction_attempts.push(executed_attempt(&reduction_resource_plan));
    partition_reduction_plans.push(reduction_resource_plan);
    let distributed_result =
        runmat_analysis_fea::results::observation::ObservationResult::from_canonical_bytes(
            reduction_outputs[0].whole_payload().unwrap(),
        )
        .unwrap();
    assert_eq!(
        distributed_result.logical_identity,
        reference_result.logical_identity
    );
    assert_eq!(distributed_result.partition_evidence.contributions.len(), 2);

    let assembly_request =
        ObservationAssemblyRequest::new(&source_request, std::slice::from_ref(&distributed_result))
            .unwrap();
    let assembly_request_objects = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationAssemblyRequest,
        assembly_request.identity().digest(),
        &assembly_request.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let distributed_result_objects = prepare_kernel_output(&reduction_outputs[0]);
    let mut assembly_inputs = vec![
        assembly_request_objects,
        reduction_inputs[1].clone(),
        distributed_result_objects,
    ];
    let input_facts = prepared_input_resource_facts(&assembly_inputs).unwrap();
    let payload_ceiling =
        runmat_analysis_fea::planning::resource::observation_artifact_payload_bytes_upper_bound(
            &input_facts,
            1,
        )
        .unwrap();
    let assembly_resource_plan = build_observation_assembly_resource_plan(
        &source_request,
        &assembly_request,
        std::slice::from_ref(&distributed_result),
        input_facts,
        runmat_analysis_execution::projected_output_resource_facts(payload_ceiling).unwrap(),
        runmat_analysis_execution::projected_output_set_resource_facts(),
    )
    .unwrap();
    assembly_inputs.push(resource_plan_objects(&assembly_resource_plan));
    let assembly_host = test_host_for_inputs(
        &recovered.host,
        AnalysisWorkKind::AssembleObservations,
        &assembly_inputs,
        vec![AnalysisNumericalObjectKind::Observation],
        None,
    );
    assert_substituted_resource_plan_rejected(
        &assembly_host,
        AnalysisWorkKind::AssembleObservations,
        &assembly_inputs,
        &assembly_resource_plan,
    );
    let assembly_outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::AssembleObservations,
            host: &assembly_host,
            inputs: &assembly_inputs,
        })
        .unwrap();
    // This direct-kernel fixture models reserved accounting, as its solve/accuracy stages do.
    // Retain the exact plan actually executed; do not reconstruct a fictitious assembly task.
    let assembly_attempt = ResourceUsageAttempt {
        attempt_id: runmat_execution::identity::AttemptId::derive(&[
            b"kernel-observation-assembly-attempt",
        ]),
        worker_id: runmat_execution::identity::WorkerId::derive(&[b"kernel-worker"]),
        plan: assembly_resource_plan.identity,
        resources: ResourceAccounting::for_reserved_interval(
            assembly_resource_plan.request.clone(),
            1,
            0,
            0,
            0,
        )
        .unwrap(),
    };
    let artifact_objects = prepare_kernel_output(&assembly_outputs[0]);
    let manifest = runmat_analysis_fea::results::observation::ObservationArtifactManifest::from_canonical_bytes(
        &artifact_objects
            .partition_payload(artifact_objects.manifest.logical_identity)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(manifest.results.len(), 1);
    assert_eq!(
        manifest.results[0].result,
        reference_result.logical_identity
    );
    let catalog = runmat_analysis_fea::results::observation::ObservationPublicationCatalog::new(
        source_request.producer,
        source_request.plan.clone(),
        catalog,
        &solved.solve.model.sequence.steps[0],
        vec![
            runmat_analysis_fea::results::observation::ObservationPublicationEntry::Sample(
                Box::new(
                    runmat_analysis_fea::results::observation::ObservationSamplePublication {
                        request: source_request,
                        manifest,
                        partition_evidence: vec![distributed_result.partition_evidence.clone()],
                    },
                ),
            ),
        ],
        runmat_analysis_fea::execution::ObservationHistorySourceContext::Structural {
            source: runmat_analysis_fea::execution::StructuralObservationSource::Linear(
                &solved.solve,
            ),
            mesh: Some(&solver_mesh),
        },
    )
    .unwrap();
    ObservedPipeline {
        catalog,
        assembly_plan: assembly_resource_plan,
        assembly_attempt,
        partition_reduction_plans,
        partition_reduction_attempts,
    }
}

// The native kernel ran these exact plans. As elsewhere in this direct-kernel fixture,
// reserved intervals model accounting; they are not scheduler-measured resource use.
fn executed_attempt(
    plan: &runmat_analysis_fea::planning::resource::FeaResourcePlan,
) -> ResourceUsageAttempt {
    ResourceUsageAttempt {
        attempt_id: runmat_execution::identity::AttemptId::derive(&[
            b"kernel-observation-executed-attempt",
            plan.identity.digest().bytes(),
        ]),
        worker_id: runmat_execution::identity::WorkerId::derive(&[b"kernel-worker"]),
        plan: plan.identity,
        resources: ResourceAccounting::for_reserved_interval(plan.request.clone(), 1, 0, 0, 0)
            .unwrap(),
    }
}

fn resource_plan_objects(
    plan: &runmat_analysis_fea::planning::resource::FeaResourcePlan,
) -> runmat_analysis_execution::PreparedAnalysisObjects {
    runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        plan.identity.digest(),
        &plan.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}

fn assert_substituted_resource_plan_rejected(
    host: &AnalysisHostWorkload,
    kind: AnalysisWorkKind,
    inputs: &[runmat_analysis_execution::PreparedAnalysisObjects],
    plan: &runmat_analysis_fea::planning::resource::FeaResourcePlan,
) {
    let mut substituted_inputs = inputs.to_vec();
    let mut substituted_plan_input = plan.input.clone();
    substituted_plan_input.worker.cpu_millicores += 1;
    let substituted_plan =
        runmat_analysis_fea::planning::resource::build_resource_plan(substituted_plan_input)
            .unwrap();
    *substituted_inputs.last_mut().unwrap() = resource_plan_objects(&substituted_plan);
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind,
            host,
            inputs: &substituted_inputs,
        })
        .is_err());
}
