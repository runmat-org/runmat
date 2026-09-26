use super::*;

pub(super) fn execute(solved: &SolvedPipeline) {
    rejects_structural_bytes_for_thermal_publication(solved);
    publishes_retained_thermal_result(solved);
    let host = &solved.prepared.host;
    let step = host.workload.step;
    let publication = SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::StructuralPrimaryFields,
        step,
        solved.solve.model.sequence.steps[0].state_outputs[0].clone(),
        solved.prepared.inputs[1].manifest.logical_identity,
        solved.source_result,
    )
    .unwrap();
    let publication_bytes = publication.canonical_bytes().unwrap();
    let publication = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::StatePublicationRequest,
        Digest::sha256(&publication_bytes),
        &publication_bytes,
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let state_inputs = vec![publication, solved.solver_result.clone()];
    let state_host = test_host_for_inputs(
        host,
        AnalysisWorkKind::PublishState,
        &state_inputs,
        vec![AnalysisNumericalObjectKind::SemanticState],
        None,
    );
    let state_outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::PublishState,
            host: &state_host,
            inputs: &state_inputs,
        })
        .unwrap();
    let state = CanonicalSemanticState::from_canonical_bytes(&state_outputs[0].payload).unwrap();
    assert_eq!(
        state.payload.fields().unwrap()[0].field_id,
        runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
    );
    assert_eq!(
        state.topology_digest,
        solved.prepared.inputs[1].manifest.logical_identity
    );

    let linearization_declaration = solved.solve.model.sequence.steps[0]
        .state_outputs
        .iter()
        .find(|output| output.kind == runmat_analysis_core::StateKind::GeometricStiffness)
        .unwrap()
        .clone();
    let linearization_request = SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::StructuralLinearization,
        step,
        linearization_declaration,
        solved.prepared.inputs[1].manifest.logical_identity,
        solved.source_result,
    )
    .unwrap();
    let request_bytes = linearization_request.canonical_bytes().unwrap();
    let request = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::StatePublicationRequest,
        Digest::sha256(&request_bytes),
        &request_bytes,
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let inputs = vec![request, solved.solver_result.clone()];
    let publication_host = test_host_for_inputs(
        host,
        AnalysisWorkKind::PublishState,
        &inputs,
        vec![AnalysisNumericalObjectKind::SemanticState],
        None,
    );
    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::PublishState,
            host: &publication_host,
            inputs: &inputs,
        })
        .unwrap();
    let linearization_state =
        CanonicalSemanticState::from_canonical_bytes(&outputs[0].payload).unwrap();
    let linearization = linearization_state
        .payload
        .structural_linearization()
        .unwrap();
    assert_eq!(linearization.topology_digest(), state.topology_digest);
    assert_eq!(linearization.source_run(), &solved.result.run.identity);

    let consumer_step = solved
        .solve
        .model
        .sequence
        .steps
        .iter()
        .find(|candidate| candidate.step_id.as_str() == "branch")
        .unwrap();
    let mut warm_solve = solved.solve.clone();
    warm_solve.step_id = consumer_step.step_id.clone();
    let warm_state = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::SemanticState,
        state.identity.digest(),
        &state.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let warm_prepared = prepare_structural_sequence_step(StructuralLinearProgramPreparation {
        authored_study_digest: digest("study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: host.artifact_access.clone(),
        workload: warm_solve,
        mesh: Some(
            SolverMeshArtifact::canonical_decode(&solved.prepared.inputs[1].payload()).unwrap(),
        ),
        initial_guess: Some(warm_state),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
    })
    .unwrap();
    let warm_outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::Solve,
            host: &warm_prepared.host,
            inputs: &warm_prepared.inputs,
        })
        .unwrap();
    let warm_result =
        StructuralLinearWorkloadResult::from_canonical_bytes(&warm_outputs[0].payload).unwrap();
    assert_eq!(
        solved
            .result
            .run
            .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT),
        warm_result
            .run
            .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
    );
    assert!(warm_result.run.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "FEA_CONVERGENCE" && diagnostic.message.contains("iterations=0 ")
    }));

    let mut substituted = warm_prepared.inputs.clone();
    substituted[4].manifest.logical_identity = digest("substituted-state");
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::Solve,
            host: &warm_prepared.host,
            inputs: &substituted,
        })
        .is_err());
}

fn rejects_structural_bytes_for_thermal_publication(solved: &SolvedPipeline) {
    let declaration =
        runmat_analysis_core::ProcedureStep::thermal_steady("heat").state_outputs[0].clone();
    let request = SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::ThermalPrimaryField,
        solved.prepared.host.workload.step,
        declaration,
        solved.prepared.inputs[1].manifest.logical_identity,
        solved.source_result,
    )
    .unwrap();
    let bytes = request.canonical_bytes().unwrap();
    let request = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::StatePublicationRequest,
        Digest::sha256(&bytes),
        &bytes,
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let inputs = vec![request, solved.solver_result.clone()];
    let host = test_host_for_inputs(
        &solved.prepared.host,
        AnalysisWorkKind::PublishState,
        &inputs,
        vec![AnalysisNumericalObjectKind::SemanticState],
        None,
    );
    let error = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::PublishState,
            host: &host,
            inputs: &inputs,
        })
        .err()
        .expect("structural bytes must not masquerade as a thermal result");
    assert!(
        error.to_string().contains("prefix") || error.to_string().contains("contract"),
        "{error}"
    );
}

fn publishes_retained_thermal_result(solved: &SolvedPipeline) {
    use runmat_analysis_core::{AnalysisSequence, ProcedureStep, Temperature, ThermalDomain};
    use runmat_analysis_fea::{ComputeBackend, ThermalProcedureControl, ThermalSolveOptions};
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence = AnalysisSequence::from_steps(
        "thermal_native",
        vec![ProcedureStep::thermal_transient("heat")],
    );
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet4,
    );
    let result = runmat_analysis_fea::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
            solver_mesh: Some(mesh.clone()),
            ..ThermalSolveOptions::default()
        },
    )
    .unwrap();
    let expected = CanonicalSemanticState::from_thermal_result(&model, &mesh, &result).unwrap();
    let request = SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::ThermalPrimaryField,
        result.producer,
        model.sequence.steps[0].state_outputs[0].clone(),
        result.topology_digest,
        result.source_result_identity().unwrap(),
    )
    .unwrap();
    let request_bytes = request.canonical_bytes().unwrap();
    let inputs = vec![
        runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::StatePublicationRequest,
            Digest::sha256(&request_bytes),
            &request_bytes,
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap(),
        runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::SolverResult,
            request.source_result.digest(),
            &result.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap(),
    ];
    let host = test_host_for_inputs(
        &solved.prepared.host,
        AnalysisWorkKind::PublishState,
        &inputs,
        vec![AnalysisNumericalObjectKind::SemanticState],
        None,
    );
    let output = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::PublishState,
            host: &host,
            inputs: &inputs,
        })
        .unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].logical_identity, expected.identity.digest());
    assert_eq!(
        CanonicalSemanticState::from_canonical_bytes(&output[0].payload).unwrap(),
        expected
    );
}
