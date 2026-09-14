use super::*;

#[test]
fn native_kernel_executes_explicit_structural_workload_without_mesh_input() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model.sequence.steps[0].observations = vec![structural_displacement_observation()];
    let solve = StructuralLinearWorkload {
        schema_version: 1,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_runtime::analysis::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        initial_guess_transition: None,
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
        initial_guess: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
    })
    .unwrap();
    assert_eq!(prepared.inputs.len(), 3);
    assert!(prepared
        .inputs
        .iter()
        .all(|input| input.manifest.kind != AnalysisNumericalObjectKind::Mesh));

    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::Solve,
            host: &prepared.host,
            inputs: &prepared.inputs,
        })
        .unwrap();
    let result = StructuralLinearWorkloadResult::from_canonical_bytes(&outputs[0].payload).unwrap();
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
    let source_result = SolverResultIdentity::new(
        prepared.host.workload.identity,
        solver_result.manifest.payload_digest,
    )
    .unwrap();
    let recovery_request = FieldRecoveryRequest::new(
        prepared.host.workload.step,
        FieldRecoverySource::SolverResult(source_result),
        result.constraint_result.topology,
    )
    .unwrap();
    let recovery_request = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        recovery_request.identity().digest(),
        &recovery_request.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let recovery_inputs = vec![recovery_request, prepared.inputs[0].clone(), solver_result];
    assert!(recovery_inputs
        .iter()
        .all(|input| input.manifest.kind != AnalysisNumericalObjectKind::Mesh));
    let recovery_host = test_host_for_inputs(
        &prepared.host,
        AnalysisWorkKind::RecoverField,
        &recovery_inputs,
        vec![AnalysisNumericalObjectKind::Field],
        Some(2),
    );
    let recovered_outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
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
    assert_eq!(recovered.entity_counts.solid_elements, 0);
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
    substituted_recovery_inputs[1] = substituted_invocation.clone();
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::RecoverField,
            host: &recovery_host,
            inputs: &substituted_recovery_inputs,
        })
        .is_err());
}
