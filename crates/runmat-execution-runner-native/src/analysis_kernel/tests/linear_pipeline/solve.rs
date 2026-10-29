use super::*;

pub(super) fn execute() -> SolvedPipeline {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut producer_step = runmat_analysis_core::ProcedureStep::structural_static("preload");
    producer_step.observations = vec![structural_nodal_von_mises_observation()];
    producer_step
        .state_outputs
        .push(runmat_analysis_core::StateOutput {
            state_id: runmat_analysis_core::StateId("preload-linearization".into()),
            kind: runmat_analysis_core::StateKind::GeometricStiffness,
            completeness: runmat_analysis_core::StateCompleteness::Complete,
            persistence: runmat_analysis_core::StatePersistence::UntilDependentsComplete,
        });
    let state_id = producer_step.state_outputs[0].state_id.clone();
    let mut consumer_step = runmat_analysis_core::ProcedureStep::structural_static("branch");
    consumer_step
        .state_inputs
        .push(runmat_analysis_core::StateInput {
            input_id: "warm-start".into(),
            required_kind: runmat_analysis_core::StateKind::StructuralPrimary,
            source: runmat_analysis_core::StateSource::Upstream {
                producer_step: producer_step.step_id.clone(),
                state_id: state_id.clone(),
            },
            use_as: runmat_analysis_core::StateUse::InitialGuess,
            conflict: runmat_analysis_core::InitialStateConflictPolicy::Reject,
        });
    let authored_transition = runmat_analysis_core::StateTransition {
        transition_id: runmat_analysis_core::TransitionId("warm-start".into()),
        producer_step: producer_step.step_id.clone(),
        consumer_step: consumer_step.step_id.clone(),
        state_id,
        method: runmat_analysis_core::StateTransitionMethod::InitialGuessOnly,
        topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
        reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
    };
    model.sequence = runmat_analysis_core::AnalysisSequence {
        sequence_id: "stateful-sequence".into(),
        steps: vec![producer_step, consumer_step],
        dependencies: vec![runmat_analysis_core::SequenceDependency {
            predecessor: authored_transition.producer_step.clone(),
            successor: authored_transition.consumer_step.clone(),
            reason: runmat_analysis_core::SequenceDependencyReason::StateTransfer,
        }],
        transitions: vec![authored_transition],
        branches: Vec::new(),
    };
    let step_id = model.sequence.steps[0].step_id.clone();
    let solve = StructuralLinearWorkload {
        schema_version: 2,
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::canonical(
                std::num::NonZeroU32::new(2).unwrap(),
            ),
        require_solver_mesh_for_solid: false,
        state_input: None,
    };
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (face_index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            if face_index == 0 { "root" } else { "tip" }.into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    if let Some(assignment) = solve.model.material_assignments.first() {
        let region = runmat_meshing_core::fixtures::fixture_entity_for_selection(
            assignment
                .selected_region_id()
                .expect("fixture uses selected material region"),
        )
        .expect("fixture material target is a fixture selector");
        mesh.topology.volume_elements[0].region_id = region.clone();
        mesh.topology.regions[0].region_id = region;
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "study:test".into(),
            encryption_context: digest("encryption"),
        },
        workload: solve.clone(),
        mesh: Some(mesh),
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
    assert!(!result.run.fields_are_empty());
    assert!(result.linearization.is_some());
    let constraint_plan =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&prepared.inputs[2].payload())
            .unwrap();
    result
        .constraint_result
        .validate_against(&constraint_plan)
        .unwrap();
    assert!(result
        .run
        .diagnostics
        .iter()
        .all(|diagnostic| !diagnostic.message.contains("artifact_path")));

    let mut substituted_plan = prepared.inputs.clone();
    substituted_plan[2].manifest.logical_identity = digest("substituted-constraint-plan");
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::Solve,
            host: &prepared.host,
            inputs: &substituted_plan,
        })
        .is_err());

    let solver_result = prepare_kernel_output(&outputs[0]);
    let numerical_receipt = prepare_kernel_output(&outputs[1]);
    let source_result = SolverResultIdentity::new(
        prepared.host.workload.identity,
        solver_result.manifest.payload_digest,
    )
    .unwrap();
    assert_eq!(
        StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(
            &numerical_receipt.payload()
        )
        .unwrap()
        .source_result,
        source_result
    );

    SolvedPipeline {
        solve,
        prepared,
        result,
        constraint_plan,
        solver_result,
        numerical_receipt,
        source_result,
    }
}
