//! One original native solve feeds cold observations, complete state and truthful accuracy.
use super::*;
use runmat_analysis_fea::{
    execution::{
        AccuracySourceKind, CanonicalSemanticState, ObservationHistorySourceContext,
        SequenceIdentity,
    },
    results::{
        accuracy::AccuracyStatus,
        field::RecoveredFieldSample,
        observation::{
            ObservationArtifact, ObservationArtifactInput, ObservationFieldCatalog,
            ObservationLimits, ObservationPublicationCatalog, ObservationPublicationEntry,
            ObservationSamplePublication,
        },
    },
};

fn observation() -> ObservationRequest {
    ObservationRequest::Field(ObservationSpec {
        observation_id: ObservationId("native-U".into()),
        quantity: ObservationQuantity::FluidVelocity,
        value_shape: ObservationValueShape::Vector3,
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::Cell,
        output_location: FieldLocation::Cell,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::Final,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1_000_000,
            peak_resident_bytes: 16 * 1024 * 1024,
            stored_bytes: 16 * 1024 * 1024,
            transfer_bytes: 16 * 1024 * 1024,
            collective_count: 0,
        }),
    })
}

// witness: FEA-FLUID-EXECUTION-001/publication @
//   incompressible_laminar_steady/canonical_tetrahedra/durable;
//   boundary: Native durable complete-state/observation chain rejects missing mandatory spatial
//   evidence; not a qualified convergence certificate.
#[test]
fn original_native_fluid_observation_complete_state_and_accuracy_cold_chain() {
    for (workload, mesh, original) in [
        task(true, 8192),
        super::super::closed_reference::closed_task(),
    ] {
        verify_original_publication(workload, mesh, original);
    }
}

fn verify_original_publication(
    mut workload: FluidSteadyWorkload,
    mesh: SolverMeshArtifact,
    original: PreparedFluidSteadyTask,
) {
    workload.model.sequence.steps[0].observations = vec![observation()];
    workload.model.sequence.steps[0].accuracy.criteria = [
        (
            "residual",
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ResidualBound,
        ),
        (
            "closure",
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ConservationBalance,
        ),
        (
            "spatial",
            ErrorContributionKind::SpatialDiscretization,
            RequiredAccuracyEvidence::ConvergenceStudy,
        ),
    ]
    .into_iter()
    .map(|(id, contribution, evidence)| AccuracyCriterion {
        criterion_id: id.into(),
        target: if contribution == ErrorContributionKind::SpatialDiscretization {
            AccuracyTarget::NumericalResult
        } else {
            AccuracyTarget::GlobalEquilibrium
        },
        contribution,
        evidence,
        tolerance: AccuracyTolerance::Relative(1e-8),
        mandatory: true,
    })
    .collect();
    let task = prepare_fluid_steady_task(FluidSteadyTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: original.host.workload.study.digest(),
        frozen_project_digest: None,
        program_revision: original.program.request.recipe.program_revision.clone(),
        artifact_access: original.host.artifact_access.clone(),
        workload: workload.clone(),
        mesh: mesh.clone(),
        output_policy: original.output_policy,
        required_capabilities: Default::default(),
    })
    .unwrap();
    let policy = task.output_policy;
    let limits = policy.inventory_limits();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cold-publication");
    let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
    let response = run(&mut store, &task.program, &task.host, policy);
    drop(store);
    let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
    let imported = import_fluid_steady_result(&store, response, &task, limits).unwrap();
    let input = |kind| {
        task.inputs
            .iter()
            .find(|i| i.manifest.kind == kind)
            .unwrap()
            .clone()
    };
    let recovery = prepare_fluid_field_recovery(FluidFieldRecoveryPreparation {
        source_host: task.host.clone(),
        workload: input(AnalysisNumericalObjectKind::FluidSteadyWorkload),
        mesh: input(AnalysisNumericalObjectKind::Mesh),
        source_resource_plan: input(AnalysisNumericalObjectKind::ResourcePlan),
        source_result: imported.objects.clone(),
        program_revision: task.program.request.recipe.program_revision.clone(),
        output_policy: policy,
    })
    .unwrap();
    let response = run(&mut store, &recovery.program, &recovery.host, policy);
    drop(store);
    let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
    let fields = import_fluid_recovered_fields(&store, response, &recovery, limits).unwrap();
    let fanout = prepare_observation_fanout(
        &store,
        ObservationPreparation {
            source_host: recovery.host.clone(),
            source: ObservationPublishedSource::VolumeMeshRecovery {
                recovery_request: recovery.request.clone(),
                fields: fields.clone(),
            },
            source_workload: ObservationWorkloadSource::VolumeMesh(input(
                AnalysisNumericalObjectKind::FluidSteadyWorkload,
            )),
            mesh: Some(input(AnalysisNumericalObjectKind::Mesh)),
            accelerator_requests: Vec::new(),
            sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap(),
            step: workload.model.sequence.steps[0].clone(),
            limits: ObservationLimits::default(),
            program_revision: task.program.request.recipe.program_revision.clone(),
            chunk_bytes: policy.chunk_bytes(),
            inventory_limits: limits,
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        },
        RecoveredFieldSample::Final,
        None,
    )
    .unwrap();
    let mut contributions = Vec::new();
    for partition in &fanout.partitions {
        let response = run(&mut store, &partition.program, &partition.host, policy);
        let cold = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        contributions.push(
            import_observation_partition_contribution(&cold, response, partition, limits).unwrap(),
        );
    }
    let reduction = prepare_observation_reduction(ObservationReductionPreparation {
        source_host: fanout.partitions[0].host.clone(),
        source_request: fanout.source_request.clone(),
        partition_plan: fanout.partitions[0].partition_plan.clone(),
        contributions,
        program_revision: task.program.request.recipe.program_revision.clone(),
        chunk_bytes: policy.chunk_bytes(),
        inventory_limits: limits,
    })
    .unwrap();
    let response = run(&mut store, &reduction.program, &reduction.host, policy);
    let cold = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
    let observed = import_reduced_observation(&cold, response, &reduction, limits)
        .unwrap()
        .result;
    // The observation reports cells in mesh element order; the solver stores them in
    // finite-volume cell order.
    let topology = runmat_analysis_fea::topology::FiniteVolumeTopology::derive(
        &runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap(),
        workload.geometry_policy,
    )
    .unwrap();
    let mut by_element = topology
        .cells
        .iter()
        .zip(
            imported
                .result
                .original()
                .velocity_m_per_s
                .canonical_values(),
        )
        .map(|(cell, velocity)| (cell.source_element_index, *velocity))
        .collect::<Vec<_>>();
    by_element.sort_by_key(|(element, _)| *element);
    assert_eq!(
        observed.field.as_host_f64().unwrap(),
        by_element
            .into_iter()
            .flat_map(|(_, velocity)| velocity)
            .collect::<Vec<_>>()
    );
    let evaluation = &fanout.source_request;
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: evaluation.identity(),
        producer: evaluation.producer,
        source_selection: evaluation.selection.identity(),
        source: evaluation.selection.source(),
        topology_digest: runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh)
            .unwrap()
            .projection_digest,
        plan_identity: evaluation.plan.identity,
        results: vec![observed],
    })
    .unwrap();
    let catalog = ObservationPublicationCatalog::new(
        imported.result.producer_identity(),
        evaluation.plan.clone(),
        ObservationFieldCatalog::volume_mesh_recovery(
            recovery.request.clone(),
            fields.manifest.clone(),
            SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap(),
            workload.model.sequence.steps[0].clone(),
        )
        .unwrap(),
        &workload.model.sequence.steps[0],
        vec![ObservationPublicationEntry::Sample(Box::new(
            ObservationSamplePublication::new(evaluation.clone(), &artifact).unwrap(),
        ))],
        ObservationHistorySourceContext::VolumeMesh { mesh: &mesh },
    )
    .unwrap();
    for declaration in &workload.model.sequence.steps[0].state_outputs {
        let mut publication = prepare_fluid_state_publication(FluidStatePublicationPreparation {
            source_host: task.host.clone(),
            source_inputs: task.inputs.clone(),
            source_result: imported.objects.clone(),
            declaration: declaration.clone(),
            program_revision: task.program.request.recipe.program_revision.clone(),
            chunk_bytes: policy.chunk_bytes(),
            inventory_limits: limits,
        })
        .unwrap();
        // Upstream source plans may coexist, but duplicate current output authority is ambiguous.
        let mut duplicate_inputs = publication.inputs.clone();
        duplicate_inputs.push(publication.inputs.last().unwrap().clone());
        let mut duplicate_bindings = publication.host.workload.inputs.clone();
        duplicate_bindings.push(publication.host.workload.inputs.last().unwrap().clone());
        let duplicate_workload = AnalysisWorkloadRequest::new(
            publication.host.workload.study,
            publication.host.workload.step,
            publication.host.workload.execution_plan,
            AnalysisWorkKind::PublishState,
            duplicate_bindings,
            publication.host.workload.outputs.clone(),
            publication.host.workload.required_capabilities.clone(),
        )
        .unwrap();
        let duplicate_host = AnalysisHostWorkload::new(
            duplicate_workload,
            publication.host.artifact_access.clone(),
            publication.host.output_policy,
        )
        .unwrap();
        let duplicate_program = prepare_analysis_program(
            &duplicate_host,
            task.program.request.recipe.program_revision.clone(),
            &duplicate_inputs,
        )
        .unwrap();
        duplicate_program.write_to(&mut store).unwrap();
        let rejected = execute_serial_analysis(
            &duplicate_program.request,
            &mut store,
            &NativeAnalysisKernel,
            policy.chunk_bytes(),
            limits,
        )
        .unwrap_err();
        assert!(rejected
            .to_string()
            .contains("exactly one output-governing resource plan"));
        let response = run(&mut store, &publication.program, &publication.host, policy);
        let cold = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let original_plan = publication.resource_plan.clone();
        let mut understated = original_plan.input.clone();
        let FeaResourceDemand::SemanticStatePublication(demand) = &mut understated.demand else {
            panic!("publication phase")
        };
        demand.state_materialization_resident_bytes_estimate =
            demand.state_encoded_bytes_upper_bound;
        publication.resource_plan =
            runmat_analysis_fea::planning::resource::build_resource_plan(understated).unwrap();
        assert!(
            import_semantic_state_completion(&cold, response.clone(), &publication, limits)
                .is_err()
        );
        publication.resource_plan = original_plan;
        let mut understated = publication.resource_plan.input.clone();
        let FeaResourceDemand::SemanticStatePublication(demand) = &mut understated.demand else {
            panic!("publication phase")
        };
        demand.inputs.decoded_resident_bytes_admission_budget =
            runmat_analysis_fea::planning::resource::ByteCount(
                demand.inputs.encoded_payload_bytes.0,
            );
        let forged =
            runmat_analysis_fea::planning::resource::build_resource_plan(understated).unwrap();
        let mut forged_inputs = publication.inputs.clone();
        *forged_inputs.last_mut().unwrap() = prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            forged.identity.digest(),
            &forged.canonical_bytes().unwrap(),
            policy.chunk_bytes(),
            runmat_execution_artifact::ObjectNamespace::InputValue,
            limits,
        )
        .unwrap();
        assert!(require_fluid_state_source_decode_budget(&forged_inputs, &forged, limits).is_err());
        let state =
            import_semantic_state_completion(&cold, response, &publication, limits).unwrap();
        let decoded =
            CanonicalSemanticState::from_canonical_bytes(&state.objects.payload()).unwrap();
        assert_eq!(
            decoded.payload.fluid().unwrap().original_result().unwrap(),
            imported.result
        );
        let mut corrupted = publication.inputs.clone();
        corrupted
            .iter_mut()
            .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::SolverResult)
            .unwrap()
            .objects[1]
            .bytes[0] ^= 1;
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::PublishState,
                host: &publication.host,
                inputs: &corrupted,
                output_policy: policy
            })
            .is_err());
    }
    let accuracy = prepare_accuracy(AccuracyPreparation {
        solve_host: task.host.clone(),
        source_kind: AccuracySourceKind::FluidSteady,
        source_numerical_receipt: imported.numerical_receipt_objects.clone(),
        source_observations: Some(catalog),
        source_context_inputs: vec![
            input(AnalysisNumericalObjectKind::FluidSteadyWorkload),
            input(AnalysisNumericalObjectKind::Mesh),
            imported.objects.clone(),
        ],
        sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap(),
        step: workload.model.sequence.steps[0].clone(),
        program_revision: task.program.request.recipe.program_revision.clone(),
        chunk_bytes: policy.chunk_bytes(),
        inventory_limits: limits,
    })
    .unwrap();
    let response = run(&mut store, &accuracy.program, &accuracy.host, policy);
    drop(store);
    let cold = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
    let report = import_accuracy(&cold, response, &accuracy, limits).unwrap();
    assert_eq!(
        report.artifact.evaluation().report().status,
        AccuracyStatus::Indeterminate
    );
    assert_eq!(report.artifact.evidence().len(), 2);
}
