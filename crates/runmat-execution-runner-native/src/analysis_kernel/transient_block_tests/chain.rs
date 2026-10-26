use super::*;

pub(super) fn assert_two_block_chain(
    workload: StructuralTransientWorkload,
    mesh: runmat_meshing_core::SolverMeshArtifact,
) {
    let scope = runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
        &workload, &mesh, None,
    )
    .unwrap();
    let limits = ObjectInventoryLimits::default();
    let namespace = ObjectNamespace::InputValue;
    let workload_objects = prepare_structural_transient_workload_objects(
        &workload,
        scope.source_workload,
        256,
        namespace,
        limits,
    )
    .unwrap();
    let mesh_bytes = mesh.canonical_encode().unwrap();
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        &mesh_bytes,
        256,
        namespace,
        limits,
    )
    .unwrap();
    let first_request = StructuralTransientBlockRequest::new(scope.clone(), None, 2).unwrap();
    let first_request_objects =
        prepare_structural_transient_block_request_objects(&first_request, 256, namespace, limits)
            .unwrap();
    let first = prepare_structural_transient_block_task(StructuralTransientBlockPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        workload: workload_objects.clone(),
        mesh: mesh_objects.clone(),
        request: first_request_objects,
        predecessor: None,
        modal_state: None,
        artifact_access: access(),
        program_revision: revision(),
        chunk_bytes: 256,
        inventory_limits: limits,
    })
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let store_root = directory.path().join("objects");
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let first_response = execute_durable_block(&mut store, &first);
    drop(store);

    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let first_import =
        import_structural_transient_block(&store, first_response, &first, limits).unwrap();
    let checkpoint = first_import.segment.manifest().terminal_checkpoint.clone();
    let second_request =
        StructuralTransientBlockRequest::new(scope.clone(), Some(&checkpoint), 4).unwrap();
    let second_request_objects =
        prepare_structural_transient_block_request_objects(&second_request, 256, namespace, limits)
            .unwrap();
    let checkpoint_objects = prepare_structural_transient_checkpoint_objects(
        &checkpoint,
        256,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    let second = prepare_structural_transient_block_task(StructuralTransientBlockPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        workload: workload_objects.clone(),
        mesh: mesh_objects.clone(),
        request: second_request_objects,
        predecessor: Some(checkpoint_objects),
        modal_state: None,
        artifact_access: access(),
        program_revision: revision(),
        chunk_bytes: 256,
        inventory_limits: limits,
    })
    .unwrap();
    let second_response = execute_durable_block(&mut store, &second);
    drop(store);

    let store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let second_import =
        import_structural_transient_block(&store, second_response, &second, limits).unwrap();

    assert_eq!(first_import.segment.samples().len(), 2);
    assert_eq!(second_import.segment.samples().len(), 2);
    assert_eq!(
        second_import
            .segment
            .manifest()
            .terminal_checkpoint
            .continuation
            .schedule_ordinal,
        4
    );
    assert_eq!(
        second_import
            .segment
            .manifest()
            .terminal_checkpoint
            .predecessor,
        Some(checkpoint.identity)
    );

    let result_task =
        prepare_structural_transient_result_task(StructuralTransientResultPreparation {
            blocks: vec![second_import, first_import],
            artifact_access: access(),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
        })
        .unwrap();
    assert_eq!(
        result_task.resource_plan.input.phase,
        FeaResourcePhase::StructuralTransientResultAssembly(result_task.request.identity)
    );
    let FeaResourceDemand::StructuralTransientResultAssembly(demand) =
        &result_task.resource_plan.input.demand
    else {
        panic!("transient fan-in must carry its own resource demand");
    };
    assert_eq!(demand.segment_count.0, 2);
    assert_eq!(demand.sample_count.0, 5);
    assert_eq!(
        result_task.resource_plan_objects.manifest.logical_identity,
        result_task.resource_plan.identity.digest()
    );
    let mut understated_input = result_task.resource_plan.input.clone();
    understated_input.worker.fixed_memory_bytes =
        ByteCount(understated_input.worker.fixed_memory_bytes.0 - 1);
    let understated = build_resource_plan(understated_input).unwrap();
    let understated_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        understated.identity.digest(),
        &understated.canonical_bytes().unwrap(),
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let mut substituted_inputs = result_task.inputs.clone();
    substituted_inputs[1] = understated_objects;
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: runmat_analysis_execution::AnalysisOutputPolicy::new(
                4096,
                ObjectInventoryLimits::default()
            )
            .unwrap(),
            kind: AnalysisWorkKind::AssembleStructuralTransientResult,
            host: &result_task.host,
            inputs: &substituted_inputs,
        })
        .is_err());
    let mut understated_input_facts = result_task.resource_plan.input.clone();
    let FeaResourceDemand::StructuralTransientResultAssembly(demand) =
        &mut understated_input_facts.demand
    else {
        unreachable!()
    };
    demand.inputs.decoded_resident_bytes_admission_budget.0 -= 1;
    let understated = build_resource_plan(understated_input_facts).unwrap();
    substituted_inputs[1] = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        understated.identity.digest(),
        &understated.canonical_bytes().unwrap(),
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let error = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: runmat_analysis_execution::AnalysisOutputPolicy::new(
                4096,
                ObjectInventoryLimits::default(),
            )
            .unwrap(),
            kind: AnalysisWorkKind::AssembleStructuralTransientResult,
            host: &result_task.host,
            inputs: &substituted_inputs,
        })
        .err()
        .unwrap();
    assert!(error.to_string().contains("input facts"));
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let result_response = execute_durable_result(&mut store, &result_task);
    drop(store);

    let store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let imported =
        import_structural_transient_result(&store, result_response, &result_task, limits).unwrap();
    assert_eq!(imported.result.samples().len(), 5);
    for condition in &workload.model.initial_conditions {
        if let Some(reference) = condition.field_artifact_ref() {
            let field = workload
                .initial_fields
                .iter()
                .find(|field| field.reference() == reference)
                .unwrap();
            let admitted = field
                .admit(
                    condition,
                    Digest::from_bytes(*mesh.canonical_digest.bytes()),
                    mesh.topology.nodes.len(),
                )
                .unwrap();
            let initial = &imported.result.samples()[0].state;
            let actual = match condition {
                runmat_analysis_core::InitialCondition::Displacement(_) => &initial.displacement,
                runmat_analysis_core::InitialCondition::Velocity(_) => &initial.velocity,
                _ => unreachable!(),
            };
            assert_eq!(actual.as_slice(), admitted.values());
        }
    }

    assert_eq!(
        imported.result.samples()[0].schedule_ordinal,
        0,
        "the first computed physical state must survive block partitioning"
    );
    assert_eq!(imported.numerical_receipt.blocks.len(), 2);
    assert_eq!(
        imported.numerical_receipt.source_result,
        imported.result.manifest().identity
    );

    let one_request = StructuralTransientBlockRequest::new(scope, None, 4).unwrap();
    let one_request_objects =
        prepare_structural_transient_block_request_objects(&one_request, 256, namespace, limits)
            .unwrap();
    let one_block = prepare_structural_transient_block_task(StructuralTransientBlockPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        workload: workload_objects,
        mesh: mesh_objects,
        request: one_request_objects,
        predecessor: None,
        modal_state: None,
        artifact_access: access(),
        program_revision: revision(),
        chunk_bytes: 256,
        inventory_limits: limits,
    })
    .unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let one_response = execute_durable_block(&mut store, &one_block);
    drop(store);
    let store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let one_import =
        import_structural_transient_block(&store, one_response, &one_block, limits).unwrap();
    let one_result_task =
        prepare_structural_transient_result_task(StructuralTransientResultPreparation {
            blocks: vec![one_import],
            artifact_access: access(),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
        })
        .unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let one_result_response = execute_durable_result(&mut store, &one_result_task);
    drop(store);
    let store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &store_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let one_result =
        import_structural_transient_result(&store, one_result_response, &one_result_task, limits)
            .unwrap();

    assert_eq!(one_result.result, imported.result);
    assert_ne!(
        one_result.numerical_receipt.identity, imported.numerical_receipt.identity,
        "execution partitioning belongs to the receipt, not the physical result"
    );
}
