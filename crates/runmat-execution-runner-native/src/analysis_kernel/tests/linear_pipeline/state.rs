use super::*;

pub(super) fn execute(solved: &SolvedPipeline) {
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
