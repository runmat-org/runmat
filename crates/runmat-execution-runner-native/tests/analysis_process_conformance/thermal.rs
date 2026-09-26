//! Placement parity conformance, not a distributed thermal solve qualification.
use super::publication_chain::{ProcessDriver, PublicationChainDriver};
use runmat_analysis_core::{
    AnalysisSequence, FormulationRequirement, ProcedureStep, Temperature, ThermalDomain,
};
use runmat_analysis_execution::{
    import_semantic_state, import_thermal_completion, import_thermal_result,
    prepare_semantic_state_publication, prepare_thermal_task, AnalysisOutputPolicy,
    ImportedThermalResult, SemanticStatePublicationPreparation, ThermalTaskPreparation,
};
use runmat_analysis_fea::{
    execution::{SemanticStatePublicationSource, ThermalWorkload},
    results::field::RecoveredFieldPartition,
    ComputeBackend, ThermalProcedureControl,
};
use runmat_execution_artifact::ArtifactAccess;
use runmat_execution_runner_native::{
    execute_analysis_program_request, NativeAnalysisHostLimits, NativeAnalysisKernel,
    NativeProgramSession,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

/// Reuse the harness's isolated stdio worker and authoritative resource-planned submission.
pub(super) async fn run(session: &NativeProgramSession) {
    let mut store = session.object_store();
    run_driver(
        &mut ProcessDriver {
            session,
            store: &mut store,
        },
        "analysis-process-conformance",
        super::revision(),
    )
    .await;
}

pub(super) async fn run_driver<D: PublicationChainDriver>(
    driver: &mut D,
    authorization_scope: &str,
    revision: runmat_execution::ProgramRevision,
) {
    let limits = NativeAnalysisHostLimits::default();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut model = super::fixture_model(super::FixtureId::CantileverLinearStatic);
        model.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        model.sequence = AnalysisSequence::from_steps(
            "isolated-thermal",
            vec![ProcedureStep::thermal_transient("heat")],
        );
        let mut energy = model.sequence.steps[0].accuracy.criteria[0].clone();
        energy.criterion_id = "energy_balance".into();
        energy.evidence = runmat_analysis_core::RequiredAccuracyEvidence::ConservationBalance;
        model.sequence.steps[0].accuracy.criteria.push(energy);
        let workload = ThermalWorkload {
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            formulation_requirement: FormulationRequirement::Experimental,
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
            residual_target: 1e-4,
        };
        let mut prepared = prepare_thermal_task(ThermalTaskPreparation {
            authored_study_digest: super::digest("isolated-thermal-study"),
            frozen_project_digest: None,
            program_revision: revision.clone(),
            artifact_access: ArtifactAccess {
                authorization_scope: authorization_scope.into(),
                encryption_context: super::digest("isolated-thermal-encryption"),
            },
            workload: workload.clone(),
            mesh: canonical_tetrahedron_solver_mesh(order),
            output_policy: AnalysisOutputPolicy::new(limits.chunk_bytes, limits.inventory).unwrap(),
            required_capabilities: Default::default(),
        })
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let mut store = runmat_execution_runner_native::FilesystemObjectStore::open(
            directory.path().join("inline"),
            limits.inventory.max_object_bytes,
        )
        .unwrap();
        prepared.program.write_to(&mut store).unwrap();
        let request = prepared.program.request.clone();
        let process = driver
            .execute(
                &prepared.host,
                &prepared.program,
                Some(&prepared.resource_plan),
            )
            .await;
        let output_policy =
            AnalysisOutputPolicy::new(limits.chunk_bytes, limits.inventory).unwrap();
        let mut wrong_allocation = process.resources.clone();
        let mut allocation = wrong_allocation.resources.allocation.clone();
        allocation.memory_bytes += 1;
        wrong_allocation.resources =
            runmat_execution::resource::ResourceAccounting::for_reserved_interval(
                allocation,
                process.resources.resources.usage.wall_millis,
                process.resources.resources.artifact_bytes,
                process.resources.resources.usage.egress_bytes,
                process.resources.resources.usage.relay_bytes,
            )
            .unwrap();
        assert!(import_thermal_completion(
            driver.store(),
            process.response.clone(),
            wrong_allocation,
            &prepared,
            output_policy,
        )
        .is_err());
        let admitted_plan = prepared.resource_plan.clone();
        prepared.resource_plan.input.phase =
            runmat_analysis_fea::planning::resource::FeaResourcePhase::ThermalSolve(
                prepared.host.workload.identity,
            );
        assert!(import_thermal_completion(
            driver.store(),
            process.response.clone(),
            process.resources.clone(),
            &prepared,
            output_policy,
        )
        .is_err());
        prepared.resource_plan = admitted_plan;
        let mut wrong_bytes = process.resources.clone();
        wrong_bytes.resources.artifact_bytes += 1;
        assert!(import_thermal_completion(
            driver.store(),
            process.response.clone(),
            wrong_bytes,
            &prepared,
            output_policy,
        )
        .is_err());
        let mut missing_object = process.response.clone();
        if let runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
            result_objects,
            ..
        } = &mut missing_object
        {
            result_objects.pop();
        }
        assert!(import_thermal_completion(
            driver.store(),
            missing_object,
            process.resources.clone(),
            &prepared,
            output_policy,
        )
        .is_err());
        let completion = import_thermal_completion(
            driver.store(),
            process.response,
            process.resources,
            &prepared,
            output_policy,
        )
        .unwrap();
        let process = completion.result;
        let usage = completion.solve_usage;
        assert_eq!(usage.evidence.attempts.len(), 1);
        assert_eq!(
            usage.evidence.workload_digest,
            prepared.host.workload.identity.digest()
        );
        assert_eq!(
            usage.evidence.result_digest,
            process.manifest.identity().digest()
        );
        let inline_response =
            execute_analysis_program_request(&request, &mut store, &NativeAnalysisKernel, limits);
        let inline =
            import_thermal_result(&store, inline_response, &prepared, limits.inventory).unwrap();
        assert_eq!(process.manifest, inline.manifest);
        assert_eq!(process.numerical_receipt, inline.numerical_receipt);
        assert_eq!(process.objects, inline.objects);
        assert_eq!(
            process.numerical_receipt_objects,
            inline.numerical_receipt_objects
        );
        process
            .numerical_receipt
            .validate_against(&workload, &process.manifest)
            .unwrap();
        assert_equilibrium(&process);
        let accuracy = runmat_analysis_execution::prepare_accuracy(
            runmat_analysis_execution::AccuracyPreparation {
                solve_host: prepared.host.clone(),
                source_kind: runmat_analysis_fea::execution::AccuracySourceKind::Thermal,
                source_numerical_receipt: process.numerical_receipt_objects.clone(),
                source_observations: None,
                sequence: runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
                    &workload.model.sequence,
                )
                .unwrap(),
                step: workload.model.sequence.steps[0].clone(),
                formulation_requirement: workload.formulation_requirement,
                program_revision: revision.clone(),
                chunk_bytes: limits.chunk_bytes,
                inventory_limits: limits.inventory,
            },
        )
        .unwrap();
        let response = driver
            .execute(&accuracy.host, &accuracy.program, None)
            .await;
        let report = runmat_analysis_execution::import_accuracy(
            driver.store(),
            response.response,
            &accuracy,
            limits.inventory,
        )
        .unwrap();
        assert_eq!(
            report.artifact.gate().status,
            runmat_analysis_fea::results::accuracy::AccuracyStatus::Satisfied
        );
        assert_eq!(report.artifact.evidence().len(), 2);
        // This publication binds solve-attempt usage only. Accuracy, state and publication
        // tasks currently have no FEA phase plans and are not claimed as measured here.
        let result_publication = runmat_analysis_execution::prepare_result_publication(
            runmat_analysis_execution::ResultPublicationPreparation {
                accuracy_host: accuracy.host.clone(),
                source_accuracy: report.objects,
                source_resource_plan: prepared.inputs.iter().find(|input| {
                    input.manifest.kind == runmat_analysis_fea::execution::AnalysisNumericalObjectKind::ResourcePlan
                }).unwrap().clone(),
                source_resource_usage: usage.objects,
                source_observations: None,
                program_revision: revision.clone(),
                chunk_bytes: limits.chunk_bytes,
                inventory_limits: limits.inventory,
            },
        ).unwrap();
        let response = driver
            .execute(&result_publication.host, &result_publication.program, None)
            .await;
        let published = runmat_analysis_execution::import_published_result(
            driver.store(),
            response.response,
            &result_publication,
            limits.inventory,
        )
        .unwrap()
        .result;
        assert_eq!(published.source_result, process.manifest.identity());
        assert_eq!(published.source_resource_usage, usage.evidence.identity());
        assert_eq!(
            published.source_resource_plan,
            prepared.resource_plan.identity
        );
        assert_eq!(
            published.formulation_requirement,
            FormulationRequirement::Experimental
        );
        assert_eq!(
            published.accuracy_status,
            runmat_analysis_fea::results::accuracy::AccuracyStatus::Satisfied
        );
        let publication = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: prepared.host.clone(),
            source_result: process.objects,
            source: SemanticStatePublicationSource::ThermalPrimaryField,
            declaration: workload.model.sequence.steps[0].state_outputs[0].clone(),
            topology_digest: process.manifest.topology_digest,
            program_revision: revision.clone(),
            chunk_bytes: limits.chunk_bytes,
            inventory_limits: limits.inventory,
        })
        .unwrap();
        let response = driver
            .execute(&publication.host, &publication.program, None)
            .await;
        let state = import_semantic_state(
            driver.store(),
            response.response,
            &publication,
            limits.inventory,
        )
        .unwrap();
        assert_eq!(state.kind, runmat_analysis_core::StateKind::Thermal);
        assert_eq!(state.topology_digest, process.manifest.topology_digest);
        let fields = state.payload.fields().unwrap();
        assert_eq!(fields.len(), 1);
        let values = fields[0].as_host_f64().unwrap();
        assert!(!values.is_empty());
        assert!(values.iter().all(|value| (value - 293.15).abs() < 1e-8));
    }
}

fn assert_equilibrium(result: &ImportedThermalResult) {
    assert_eq!(result.manifest.time_points_s.len(), 3);
    assert_eq!(result.manifest.fields.len(), 3);
    // The independent reference is constant-temperature equilibrium in an unforced insulated
    // body: T = 293.15 K, grad(T) = 0, q = 0, and d(stored energy)/dt = 0.
    for fields in &result.manifest.fields {
        for (field_index, field) in fields.iter().enumerate() {
            assert!(!field.partitions.is_empty());
            let expected = if field_index == 0 { 293.15 } else { 0.0 };
            for descriptor in &field.partitions {
                let bytes = result
                    .objects
                    .partition_payload(descriptor.partition.digest())
                    .unwrap();
                let partition = RecoveredFieldPartition::from_canonical_bytes(&bytes).unwrap();
                assert!(!partition.storage.as_host_f64().unwrap().is_empty());
                assert!(partition
                    .storage
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .all(|value| (value - expected).abs() < 1e-8));
            }
        }
    }
}
