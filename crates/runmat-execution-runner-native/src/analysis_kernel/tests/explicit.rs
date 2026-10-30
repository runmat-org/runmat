use super::*;

#[test]
fn native_kernel_executes_connector_observation_admitted_by_portable_preparation() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model
        .connectors
        .push(runmat_analysis_core::AnalysisConnector {
            connector_id: "tip_support_spring".into(),
            kind: runmat_analysis_core::ConnectorKind::IsotropicGroundSpring {
                target: runmat_analysis_core::RegionTarget::point("node:2"),
                stiffness: runmat_analysis_core::Stiffness::new(1.0e6),
            },
        });
    let mut observation = structural_displacement_observation();
    let spec = observation.spec_mut().expect("probe observation");
    spec.observation_id = runmat_analysis_core::ObservationId("tip_support_force".into());
    spec.quantity = runmat_analysis_core::ObservationQuantity::ReactionForce;
    spec.target = runmat_analysis_core::ObservationTarget::Connector {
        connector_id: "tip_support_spring".into(),
    };
    spec.source_location = runmat_analysis_core::FieldLocation::Connector;
    spec.output_location = runmat_analysis_core::FieldLocation::Connector;
    model.sequence.steps[0].observations = vec![observation];
    let solve = StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::serial(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("connector-observation-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "study:connector-observation".into(),
            encryption_context: digest("connector-observation-encryption"),
        },
        workload: solve,
        mesh: None,
        state_input: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .expect("portable preparation must admit the connector observation");
    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::Solve,
            host: &prepared.host,
            inputs: &prepared.inputs,
        })
        .expect("native worker must execute the admitted connector observation");
    let result =
        StructuralLinearWorkloadResult::from_canonical_bytes(outputs[0].whole_payload().unwrap())
            .expect("canonical solver result");
    let connector = result
        .run
        .structural_connector_result
        .as_ref()
        .expect("connector response");
    let observation = result
        .run
        .observations
        .first()
        .expect("connector observation");
    assert_eq!(observation.observation_id.0, "tip_support_force");
    assert_eq!(
        observation.field.as_host_f64().unwrap(),
        connector.responses[0].resultant_force_newtons().as_slice(),
    );
}

fn explicit_structural_program() -> (
    StructuralLinearWorkload,
    runmat_analysis_execution::PreparedStructuralLinearProgram,
) {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model.sequence.steps[0].observations = vec![structural_displacement_observation()];
    let solve = StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("explicit-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "study:explicit".into(),
            encryption_context: digest("explicit-encryption"),
        },
        workload: solve.clone(),
        mesh: None,
        state_input: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap();
    (solve, prepared)
}

#[test]
fn driver_analysis_requires_allocation_before_kernel_effects() {
    use runmat_execution_artifact::ProgramExecutionResponse;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingKernel(AtomicUsize);
    impl AnalysisKernel for CountingKernel {
        fn execute(
            &self,
            invocation: AnalysisKernelInvocation<'_>,
        ) -> runmat_analysis_execution::AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            NativeAnalysisKernel.execute(invocation)
        }
    }

    let (_, prepared) = explicit_structural_program();
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        ObjectInventoryLimits::default().max_object_bytes,
    )
    .unwrap();
    prepared.program.write_to(&mut store).unwrap();
    let required = &prepared.resource_plan.request;
    let mut allocation = runmat_execution_transport_native::control::ResourceRequest {
        cpu_millicores: u64::from(required.cpu_millicores) - 1,
        memory_bytes: required.memory_bytes,
        scratch_bytes: required.scratch_bytes,
        maximum_wall_millis: required.max_wall_millis,
        accelerators: required.accelerators.clone(),
    };
    let kernel = CountingKernel(AtomicUsize::new(0));
    let response = crate::remote::execute_driver_analysis(
        &prepared.program.request,
        &mut store,
        &kernel,
        &allocation,
    )
    .unwrap();
    assert!(
        matches!(response, ProgramExecutionResponse::Failure { ref message }
        if message.contains("CPU reservation")),
        "{response:?}"
    );
    assert_eq!(kernel.0.load(Ordering::SeqCst), 0);

    allocation.cpu_millicores = u64::from(required.cpu_millicores);
    let response = crate::remote::execute_driver_analysis(
        &prepared.program.request,
        &mut store,
        &kernel,
        &allocation,
    )
    .unwrap();
    assert!(
        !matches!(response, ProgramExecutionResponse::Failure { .. }),
        "{response:?}"
    );
    response
        .validate_against(&prepared.program.request)
        .unwrap();
    assert_eq!(kernel.0.load(Ordering::SeqCst), 1);
}

#[test]
fn native_kernel_executes_explicit_structural_workload_without_mesh_input() {
    let (solve, prepared) = explicit_structural_program();
    assert_eq!(prepared.inputs.len(), 3);
    assert!(prepared
        .inputs
        .iter()
        .all(|input| input.manifest.kind != AnalysisNumericalObjectKind::Mesh));

    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::Solve,
            host: &prepared.host,
            inputs: &prepared.inputs,
        })
        .unwrap();
    let result =
        StructuralLinearWorkloadResult::from_canonical_bytes(outputs[0].whole_payload().unwrap())
            .unwrap();
    assert!(matches!(
        result.constraint_result.topology,
        CanonicalStructuralTopologyIdentity::Explicit { .. }
    ));
    assert!(result.run.fields.iter().any(|field| {
        field.field_id == runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
            && field.as_host_f64().is_some_and(|values| {
                !values.is_empty() && values.iter().all(|value| value.is_finite())
            })
    }));

    let solver_result = prepare_kernel_output(&outputs[0]);
    // Field recovery runs under its own governed admission: the preparation carries the
    // recovery request, the original invocation, the solver result and the resource plan.
    let recovery = runmat_analysis_execution::prepare_structural_field_recovery(
        runmat_analysis_execution::StructuralFieldRecoveryPreparation {
            source_host: prepared.host.clone(),
            source_invocation: prepared
                .inputs
                .iter()
                .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation)
                .unwrap()
                .clone(),
            source_result: solver_result,
            mesh: None,
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        },
    )
    .unwrap();
    let recovery_inputs = recovery.inputs.clone();
    assert!(recovery_inputs
        .iter()
        .all(|input| input.manifest.kind != AnalysisNumericalObjectKind::Mesh));
    let recovery_host = recovery.host.clone();
    let recovered_outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::RecoverField,
            host: &recovery_host,
            inputs: &recovery_inputs,
        })
        .unwrap();
    let recovered_objects = prepare_kernel_output(&recovered_outputs[0]);
    let recovered = decode_recovered_field_artifact(&recovered_objects).unwrap();
    assert_eq!(
        recovered.topology_digest,
        result.constraint_result.topology.discretization_digest()
    );
    assert_eq!(recovered.entity_counts.volume_elements, 0);
    assert!(recovered.entity_counts.beam_elements > 0);

    let mut substituted_workload = solve.clone();
    substituted_workload.model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    substituted_workload.step_id = substituted_workload.model.sequence.steps[0].step_id.clone();
    let substituted_invocation = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverInvocation,
        digest("substituted-explicit-invocation"),
        &substituted_workload.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut substituted_recovery_inputs = recovery_inputs.clone();
    let invocation_index = substituted_recovery_inputs
        .iter()
        .position(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation)
        .unwrap();
    substituted_recovery_inputs[invocation_index] = substituted_invocation.clone();
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::RecoverField,
            host: &recovery_host,
            inputs: &substituted_recovery_inputs,
        })
        .is_err());
}
