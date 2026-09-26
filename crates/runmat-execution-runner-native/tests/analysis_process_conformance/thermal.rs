//! Local process isolation conformance, not a distributed thermal solve qualification.
use runmat_analysis_core::{
    AnalysisSequence, FormulationRequirement, ProcedureStep, Temperature, ThermalDomain,
};
use runmat_analysis_execution::{
    import_thermal_result, prepare_thermal_task, AnalysisOutputPolicy, ImportedThermalResult,
    ThermalTaskPreparation,
};
use runmat_analysis_fea::{
    execution::ThermalWorkload, results::field::RecoveredFieldPartition, ComputeBackend,
    ThermalProcedureControl,
};
use runmat_execution_artifact::ArtifactAccess;
use runmat_execution_runner_native::{
    execute_analysis_program_request, NativeAnalysisHostLimits, NativeAnalysisKernel,
    NativeProgramSession,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

/// Reuse the harness's isolated stdio worker and authoritative resource-planned submission.
pub(super) async fn run(session: &NativeProgramSession) {
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
        let workload = ThermalWorkload {
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            formulation_requirement: FormulationRequirement::Experimental,
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
            residual_target: 1e-4,
        };
        let prepared = prepare_thermal_task(ThermalTaskPreparation {
            authored_study_digest: super::digest("isolated-thermal-study"),
            frozen_project_digest: None,
            program_revision: super::revision(),
            artifact_access: ArtifactAccess {
                authorization_scope: "analysis-process-conformance".into(),
                encryption_context: super::digest("isolated-thermal-encryption"),
            },
            workload: workload.clone(),
            mesh: canonical_tetrahedron_solver_mesh(order),
            output_policy: AnalysisOutputPolicy::new(limits.chunk_bytes, limits.inventory).unwrap(),
            required_capabilities: Default::default(),
        })
        .unwrap();
        let mut store = session.object_store();
        prepared.program.write_to(&mut store).unwrap();
        let request = prepared.program.request.clone();
        let submission = super::analysis_submission(
            session.scope_id(),
            session.pool_id(),
            &prepared.host,
            &request,
            Some(&prepared.resource_plan),
            prepared.resource_plan.request.clone(),
        );
        let task = session
            .submit_graph(vec![(request.clone(), submission)])
            .unwrap()
            .pop()
            .unwrap();
        let process = super::wait_for_native_task(task).await;
        let process = import_thermal_result(
            &store,
            super::success_response(&process),
            &prepared,
            limits.inventory,
        )
        .unwrap();
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
