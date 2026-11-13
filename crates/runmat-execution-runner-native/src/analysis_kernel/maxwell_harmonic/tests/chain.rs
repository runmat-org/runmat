//! Actual native task chain, not a completed-study root or spatial qualification witness.
use super::*;
use crate::NativeAnalysisKernel;
use runmat_analysis_execution::*;
use runmat_analysis_fea::{
    execution::{
        AccuracySourceKind, CanonicalSemanticState, ObservationHistorySourceContext,
        SemanticStatePublicationSource, SequenceIdentity,
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
use runmat_execution_artifact::{cache::FilesystemObjectStore, ProgramExecutionResponse};

fn run(
    store: &mut FilesystemObjectStore,
    program: &PreparedAnalysisProgram,
    host: &AnalysisHostWorkload,
    policy: AnalysisOutputPolicy,
) -> ProgramExecutionResponse {
    program.write_to(store).unwrap();
    let done = execute_serial_analysis(
        &program.request,
        store,
        &NativeAnalysisKernel,
        policy.chunk_bytes(),
        policy.inventory_limits(),
    )
    .unwrap();
    AnalysisHostResponse::completed(host, &done)
        .unwrap()
        .program_response()
}

fn observation() -> ObservationRequest {
    ObservationRequest::Field(ObservationSpec {
        observation_id: ObservationId("complex-E".into()),
        quantity: ObservationQuantity::ElectricField,
        value_shape: ObservationValueShape::Vector3,
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::IntegrationPoint,
        output_location: FieldLocation::IntegrationPoint,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Complex,
        sampling: SamplingPolicy::Final,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Complex128,
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

#[test]
fn both_orders_native_solve_recovery_complex_observation_state_and_accuracy_cold_chain() {
    for quadratic in [false, true] {
        let (mut w, mesh, original) = task(quadratic, 8192);
        w.model.sequence.steps[0].observations = vec![observation()];
        w.model.sequence.steps[0].accuracy.criteria = [
            (
                "residual",
                ErrorContributionKind::Algebraic,
                RequiredAccuracyEvidence::ResidualBound,
            ),
            (
                "power",
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
        let task = prepare_maxwell_harmonic_task(MaxwellHarmonicTaskPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: original.host.workload.study.digest(),
            frozen_project_digest: None,
            program_revision: original.program.request.recipe.program_revision.clone(),
            artifact_access: original.host.artifact_access.clone(),
            workload: w.clone(),
            mesh: mesh.clone(),
            output_policy: original.output_policy,
            required_capabilities: Default::default(),
        })
        .unwrap();
        let policy = task.output_policy;
        let limits = policy.inventory_limits();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cold");
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let response = run(&mut store, &task.program, &task.host, policy);
        drop(store);
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let imported = import_maxwell_harmonic_result(&store, response, &task, limits).unwrap();
        imported
            .numerical_receipt
            .validate_against(&w, &imported.result, &mesh)
            .unwrap();
        let input = |kind| {
            task.inputs
                .iter()
                .find(|i| i.manifest.kind == kind)
                .unwrap()
                .clone()
        };
        let recovery = prepare_maxwell_field_recovery(MaxwellFieldRecoveryPreparation {
            source_host: task.host.clone(),
            workload: input(AnalysisNumericalObjectKind::MaxwellHarmonicWorkload),
            mesh: input(AnalysisNumericalObjectKind::Mesh),
            source_result: imported.objects.clone(),
            program_revision: task.program.request.recipe.program_revision.clone(),
            output_policy: policy,
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        })
        .unwrap();
        let response = run(&mut store, &recovery.program, &recovery.host, policy);
        drop(store);
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let fields = import_maxwell_recovered_fields(&store, response, &recovery, limits).unwrap();
        assert_eq!(fields.manifest.fields.len(), 11);
        let sampling = imported.result.sampling_topology().clone();
        let fanout = prepare_observation_fanout(
            &store,
            ObservationPreparation {
                source_host: recovery.host.clone(),
                source: ObservationPublishedSource::SampledRecovery {
                    recovery_request: recovery.request.clone(),
                    fields: fields.clone(),
                    sampling: sampling.clone(),
                },
                source_workload: ObservationWorkloadSource::MaxwellHarmonic(input(
                    AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
                )),
                mesh: Some(input(AnalysisNumericalObjectKind::Mesh)),
                accelerator_requests: Vec::new(),
                sequence: SequenceIdentity::from_solver_contract(&w.model.sequence).unwrap(),
                step: w.model.sequence.steps[0].clone(),
                limits: ObservationLimits::default(),
                program_revision: task.program.request.recipe.program_revision.clone(),
                chunk_bytes: policy.chunk_bytes(),
                inventory_limits: limits,
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            },
            RecoveredFieldSample::Final,
            None,
        )
        .unwrap();
        assert!(!fanout.partitions.is_empty());
        let mut contributions = Vec::new();
        for partition in &fanout.partitions {
            let response = run(&mut store, &partition.program, &partition.host, policy);
            let cold = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            contributions.push(
                import_observation_partition_contribution(&cold, response, partition, limits)
                    .unwrap(),
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
        let expected = imported
            .result
            .response()
            .samples
            .iter()
            .flat_map(|s| s.electric_field_v_per_m.canonical_value())
            .flat_map(|v| [v.real, v.imaginary])
            .collect::<Vec<_>>();
        assert_eq!(
            observed.complex_representation,
            ComplexRepresentation::Complex
        );
        assert_eq!(observed.field.as_host_f64().unwrap(), expected);
        // Bind publication metadata only to genuine native partition/reduction output above.
        let evaluation = &fanout.source_request;
        let artifact = ObservationArtifact::new(ObservationArtifactInput {
            evaluation_request: evaluation.identity(),
            producer: evaluation.producer,
            source_selection: evaluation.selection.identity(),
            source: evaluation.selection.source(),
            topology_digest: sampling.identity().unwrap().discretization_digest(),
            plan_identity: evaluation.plan.identity,
            results: vec![observed],
        })
        .unwrap();
        let catalog = ObservationPublicationCatalog::new(
            imported.result.producer_identity(),
            evaluation.plan.clone(),
            ObservationFieldCatalog::sampled_recovery(
                recovery.request.clone(),
                fields.manifest.clone(),
                sampling.clone(),
            )
            .unwrap(),
            &w.model.sequence.steps[0],
            vec![ObservationPublicationEntry::Sample(Box::new(
                ObservationSamplePublication::new(evaluation.clone(), &artifact).unwrap(),
            ))],
            ObservationHistorySourceContext::MaxwellHarmonic {
                workload: &w,
                mesh: &mesh,
            },
        )
        .unwrap();
        let catalog = ObservationPublicationCatalog::from_canonical_bytes(
            &catalog.canonical_bytes().unwrap(),
        )
        .unwrap();
        let declarations = imported.result.state_publication_requests().unwrap();
        assert_eq!(declarations.len(), 1);
        let request = &declarations[0];
        let publication = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: task.host.clone(),
            source_result: imported.objects.clone(),
            source_profiles: imported.source_profiles.clone(),
            source: SemanticStatePublicationSource::MaxwellNativeFields,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            program_revision: task.program.request.recipe.program_revision.clone(),
            chunk_bytes: policy.chunk_bytes(),
            inventory_limits: limits,
        })
        .unwrap();
        let response = run(&mut store, &publication.program, &publication.host, policy);
        let cold = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let state =
            import_semantic_state_completion(&cold, response, &publication, limits).unwrap();
        let decoded =
            CanonicalSemanticState::from_canonical_bytes(&state.objects.payload()).unwrap();
        assert_eq!(decoded.source_result, imported.result.identity().unwrap());
        assert_eq!(
            decoded.payload.maxwell_harmonic().unwrap().result.as_ref(),
            &imported.result
        );
        let accuracy = prepare_accuracy(AccuracyPreparation {
            solve_host: task.host.clone(),
            source_kind: AccuracySourceKind::MaxwellHarmonic,
            source_numerical_receipt: imported.numerical_receipt_objects.clone(),
            source_observations: Some(catalog),
            source_context_inputs: vec![
                input(AnalysisNumericalObjectKind::MaxwellHarmonicWorkload),
                input(AnalysisNumericalObjectKind::Mesh),
                imported.objects.clone(),
            ],
            sequence: SequenceIdentity::from_solver_contract(&w.model.sequence).unwrap(),
            step: w.model.sequence.steps[0].clone(),
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
}
