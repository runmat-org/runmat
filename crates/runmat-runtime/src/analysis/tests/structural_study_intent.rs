mod original_fields;
use super::*;

#[test]
fn response_spectrum_intent_derives_required_modal_residual_directions() {
    let _guard = analysis_test_guard();
    let spec = sample_response_spectrum_study_spec("31".repeat(32));
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-response-spectrum".into()), None),
    )
    .expect("response-spectrum intent should prepare without a solver mesh");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads,
    } = intent.workloads()
    else {
        panic!("response-spectrum intent must remain a heterogeneous sequence")
    };
    let [modal_step, response_step] = intent.model().sequence.steps.as_slice() else {
        panic!("response-spectrum fixture must have two steps")
    };
    let runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction { workload } =
        &workloads[&modal_step.step_id]
    else {
        panic!("first response-spectrum step must extract modes")
    };
    assert_eq!(
        workload.residual_vector_directions(),
        &[runmat_analysis_core::SpectrumDirection {
            frame: runmat_analysis_core::CoordinateFrame::Global,
            vector: [1.0, 0.0, 0.0],
        }]
    );
    assert!(matches!(
        workloads.get(&response_step.step_id),
        Some(runmat_analysis_fea::execution::StructuralStepWorkload::ModalResponseSpectrum { .. })
    ));
    let restored = PreparedStructuralStudyIntent::from_continuation_bytes(
        &intent.continuation_bytes().unwrap(),
    )
    .expect("response-spectrum intent should round trip");
    assert_eq!(restored.workloads(), intent.workloads());
}

#[test]
fn random_response_intent_is_typed_path_free_and_round_trips() {
    let _guard = analysis_test_guard();
    let spec = sample_random_response_study_spec("32".repeat(32));
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-random-response".into()), None),
    )
    .expect("random-response intent should prepare without a solver mesh");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads,
    } = intent.workloads()
    else {
        panic!("random-response intent must remain a heterogeneous sequence")
    };
    let [modal_step, response_step] = intent.model().sequence.steps.as_slice() else {
        panic!("random-response fixture must have two steps")
    };
    assert!(matches!(
        workloads.get(&modal_step.step_id),
        Some(runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction { .. })
    ));
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::RandomResponse { template }) =
        workloads.get(&response_step.step_id)
    else {
        panic!("random-response consumer must retain its typed template")
    };
    let AnalysisSolverOptions::StructuralRandomResponse(options) =
        &spec.execution.steps[1].solver_options
    else {
        panic!("random-response fixture must retain typed solver options")
    };
    assert_eq!(template.integration_policy, options.integration);
    assert_eq!(
        template.maximum_intervals_per_task,
        options.maximum_intervals_per_task
    );
    let restored = PreparedStructuralStudyIntent::from_continuation_bytes(
        &intent.continuation_bytes().unwrap(),
    )
    .expect("random-response intent should round trip");
    assert_eq!(restored.workloads(), intent.workloads());
}

#[test]
fn structural_study_intent_is_path_free_and_does_not_require_a_mesh() {
    let _guard = analysis_test_guard();
    let mut spec = sample_linear_static_study_spec();
    let mut model = spec.model.clone();
    set_model_contact(&mut model, 0.0);
    spec.model = model;
    spec.geometry.source.path = "/author/machine/private/part.step".to_string();
    spec.geometry.source.sha256 = "11".repeat(32);
    spec.solver_mesh_artifact_path = Some("/missing/local/mesh.cbor".to_string());
    let AnalysisSolverOptions::StructuralLinear(options) =
        &mut spec.execution.steps[0].solver_options
    else {
        panic!("sample study must be structural-linear");
    };
    options.solver_mesh_artifact_path = Some("/another/missing/mesh.cbor".to_string());

    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-intent".into()), Some("request-intent".into())),
    )
    .expect("pre-mesh intent preparation must not load a solver mesh");
    let workload = only_workload(&intent);
    assert!(workload
        .model
        .sequence
        .steps
        .iter()
        .any(|step| step.step_id == workload.step_id));
    assert_eq!(spec.execution.steps[0].backend, workload.backend);
    assert_eq!(intent.meshing_contacts().len(), 1);
    assert_eq!(
        intent.meshing_contacts()[0].side_a_face_selectors,
        vec![intent.model().interfaces[0].primary.region_id.clone()]
    );
    assert_eq!(
        intent.meshing_contacts()[0].side_b_face_selectors,
        vec![intent.model().interfaces[0].secondary.region_id.clone()]
    );

    let bytes = intent
        .continuation_bytes()
        .expect("path-free intent continuation should encode");
    let encoded = String::from_utf8(bytes.clone()).expect("continuation is JSON");
    assert!(!encoded.contains("/author/machine"));
    assert!(!encoded.contains("/missing/local"));
    assert!(!encoded.contains("/another/missing"));

    let restored = PreparedStructuralStudyIntent::from_continuation_bytes(&bytes)
        .expect("path-free intent continuation should restore");
    assert_eq!(restored.study_id(), intent.study_id());
    assert_eq!(
        restored.authored_study_digest(),
        intent.authored_study_digest()
    );
    assert_eq!(restored.model(), intent.model());
    assert_eq!(restored.workloads(), intent.workloads());
}

#[test]
fn structural_study_intent_identity_is_independent_of_local_paths() {
    let _guard = analysis_test_guard();
    let mut first = sample_linear_static_study_spec();
    first.geometry.source.sha256 = "22".repeat(32);
    let mut second = first.clone();
    first.geometry.source.path = "/first/part.step".to_string();
    second.geometry.source.path = "/second/part.step".to_string();
    first.solver_mesh_artifact_path = Some("/first/mesh.cbor".to_string());
    second.solver_mesh_artifact_path = Some("/second/mesh.cbor".to_string());
    if let AnalysisSolverOptions::StructuralLinear(options) =
        &mut second.execution.steps[0].solver_options
    {
        options.solver_mesh_artifact_path = Some("/second/options-mesh.cbor".to_string());
    }
    let authored_study_digest = analysis_study_content_digest(&second);

    let first =
        analysis_prepare_structural_study_intent_op(&first, OperationContext::new(None, None))
            .expect("first intent should prepare");
    let second =
        analysis_prepare_structural_study_intent_op(&second, OperationContext::new(None, None))
            .expect("second intent should prepare");
    assert_eq!(
        first.authored_study_digest(),
        second.authored_study_digest()
    );
    assert_eq!(first.authored_study_digest(), authored_study_digest);
    assert_eq!(first.workloads(), second.workloads());
}

#[test]
fn structural_buckling_uses_one_typed_workload_contract_for_local_and_remote_placement() {
    let _guard = analysis_test_guard();
    let mut spec = sample_buckling_study_spec("11".repeat(32));
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-buckling-intent".into()), None),
    )
    .expect("buckling intent should prepare without a local mesh");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads,
    } = intent.workloads()
    else {
        panic!("buckling intent must retain its heterogeneous sequence")
    };
    let step_ids = intent.workloads().step_ids().unwrap();
    assert!(matches!(
        workloads.get(&step_ids[0]),
        Some(runmat_analysis_fea::execution::StructuralStepWorkload::Linear { .. })
    ));
    assert!(matches!(
        workloads.get(&step_ids[1]),
        Some(runmat_analysis_fea::execution::StructuralStepWorkload::Buckling { .. })
    ));
    let restored = PreparedStructuralStudyIntent::from_continuation_bytes(
        &intent.continuation_bytes().unwrap(),
    )
    .expect("buckling continuation should restore");
    assert_eq!(restored.workloads(), intent.workloads());

    let mut substituted = intent.clone();
    let AnalysisSolverOptions::StructuralLinear(options) =
        &mut substituted.steps[0].authored_execution.solver_options
    else {
        panic!("buckling producer must retain structural-linear options")
    };
    options.preconditioner_mode = PreconditionerMode::Ilu;
    substituted.intent_digest = substituted.semantic_digest().unwrap();
    assert!(substituted.continuation_bytes().is_err());

    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("local-buckling-preparation-mesh");
    spec.solver_mesh_artifact_path = Some(mesh_path.display().to_string());
    let prepared = analysis_prepare_structural_study_op(
        &spec,
        OperationContext::new(Some("trace-local-buckling-preparation".into()), None),
    )
    .expect("local buckling study should use the same workload contract");
    assert_eq!(prepared.workloads(), intent.workloads());
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn structural_buckling_completion_publishes_ordinary_runtime_results() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("buckling-runtime-completion");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    let meshing_output = ready_buckling_meshing_output();
    let spec =
        sample_buckling_study_spec(digest_hex(meshing_output.artifact.geometry.source_digest));
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-buckling-completion".into()), None),
    )
    .expect("buckling intent should prepare");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
        mut workloads,
    } = intent.workloads()
    else {
        panic!("buckling completion requires a heterogeneous sequence")
    };
    let [preload_step_id, buckling_step_id] = intent
        .model()
        .sequence
        .steps
        .iter()
        .map(|step| step.step_id.clone())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
        workload: preload_workload,
    } = workloads.remove(&preload_step_id).unwrap()
    else {
        panic!("first buckling workload must be the preload")
    };
    let runmat_analysis_fea::execution::StructuralStepWorkload::Buckling { template } =
        workloads.remove(&buckling_step_id).unwrap()
    else {
        panic!("second buckling workload must be the stability solve")
    };
    let model = preload_workload.model.clone();
    let constraints =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
            &model,
            Some(&meshing_output.artifact),
        )
        .unwrap();
    let topology_digest =
        runmat_execution::Digest::from_bytes(*meshing_output.artifact.canonical_digest.bytes());
    let preload =
        runmat_analysis_fea::run_linear_static_with_prepared_constraint_plan_and_linearization(
            &model,
            ComputeBackend::Cpu,
            runmat_analysis_fea::LinearStaticSolveOptions {
                solver_mesh: Some(meshing_output.artifact.clone()),
                require_solver_mesh_for_solid: true,
                ..runmat_analysis_fea::LinearStaticSolveOptions::default()
            },
            &constraints,
            topology_digest,
            None,
        )
        .unwrap();
    let constraint_result =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintResult::recover(
            &constraints,
            &meshing_output.artifact,
            &preload.run.fields,
        )
        .unwrap();
    let preload =
        runmat_analysis_fea::execution::StructuralLinearWorkloadResult::with_linearization(
            preload.run,
            constraint_result,
            preload.linearization,
        )
        .unwrap();
    let sequence =
        runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(&model.sequence)
            .unwrap();
    let producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence,
        &model.sequence.steps[0],
    )
    .unwrap();
    let preload_numerical_receipt =
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::new(
            linear_test_workload_identity(&preload_workload, &meshing_output.artifact),
            &preload,
        )
        .unwrap();
    let source_result = preload_numerical_receipt.source_result();
    let declaration = model.sequence.steps[0]
        .state_outputs
        .iter()
        .find(|output| output.kind == runmat_analysis_core::StateKind::GeometricStiffness)
        .unwrap()
        .clone();
    let state = runmat_analysis_fea::execution::CanonicalSemanticState::new(
        runmat_analysis_fea::execution::CanonicalSemanticStateInput {
            producer,
            declaration,
            topology_digest,
            source_result,
            payload:
                runmat_analysis_fea::execution::CanonicalSemanticStatePayload::StructuralLinearization {
                    linearization: Box::new(preload.linearization.clone().unwrap()),
                },
        },
    )
    .unwrap();
    let transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &template.model,
        &template.step_id,
        &state,
        runmat_analysis_core::StateUse::Prestress,
        topology_digest,
    )
    .unwrap();
    let buckling_workload = (*template).bind(transition).unwrap();
    let buckling_result = runmat_analysis_fea::execute_structural_buckling_workload(
        &buckling_workload,
        &meshing_output.artifact,
        &state,
    )
    .unwrap();
    let buckling_receipt = runmat_analysis_fea::execution::StructuralBucklingNumericalReceipt::new(
        buckling_test_workload_identity(&buckling_workload, &meshing_output.artifact),
        &buckling_result,
    )
    .unwrap();
    let envelope = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        StructuralStudyNumericalResult::StructuralBuckling(Box::new(
            StructuralStudyBucklingNumericalResult {
                frozen_project_digest: None,
                preload_step_id,
                buckling_step_id,
                preload_numerical_receipt,
                preload_result: preload,
                preload_solver_result_identity: source_result,
                prestress_state: state,
                buckling_result,
                buckling_numerical_receipt: buckling_receipt,
            },
        )),
        AnalysisStudyCompletionIdentity::new("server-run-buckling").unwrap(),
    )
    .expect("buckling result should enter ordinary study publication");

    assert_eq!(envelope.data.steps.len(), 2);
    assert!(envelope
        .data
        .states
        .iter()
        .any(|state| state.kind == runmat_analysis_core::StateKind::GeometricStiffness));
    let persisted = storage::load_run_result(&envelope.data.steps[1].run_id)
        .unwrap()
        .expect("buckling run should be persisted");
    let buckling = persisted
        .buckling_results
        .as_ref()
        .expect("ordinary runtime result should retain buckling modes");
    assert_eq!(buckling.mode_indices, vec![1]);
    assert_eq!(buckling.load_factors.len(), 1);
    assert_eq!(buckling.mode_shapes.len(), 1);
    assert!(persisted
        .run
        .fields
        .iter()
        .all(|field| field.field_id != buckling.mode_shapes[0].field_id));
    assert!(analysis_run_field_ids(&persisted).contains(&buckling.mode_shapes[0].field_id));
    let queried = analysis_results_op(
        &persisted,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    assert_eq!(queried.data.summary.buckling_mode_count, 1);
    assert_eq!(
        queried.data.summary.available_buckling_mode_indices,
        vec![1]
    );
    assert!(queried.data.summary.minimum_buckling_load_factor.is_some());
    assert!(queried.data.summary.max_buckling_residual_norm.is_some());
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn structural_prestressed_modal_completion_publishes_exact_state_and_modes() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("prestressed-modal-runtime-completion");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    let meshing_output = ready_buckling_meshing_output();
    let spec = sample_prestressed_modal_study_spec(digest_hex(
        meshing_output.artifact.geometry.source_digest,
    ));
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-prestressed-modal-completion".into()), None),
    )
    .expect("prestressed-modal intent should prepare");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
        mut workloads,
    } = intent.workloads()
    else {
        panic!("prestressed-modal completion requires a heterogeneous sequence")
    };
    let [preload_step_id, modal_step_id] = intent
        .model()
        .sequence
        .steps
        .iter()
        .map(|step| step.step_id.clone())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
        workload: preload_workload,
    } = workloads.remove(&preload_step_id).unwrap()
    else {
        panic!("first prestressed-modal workload must be the preload")
    };
    let runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal { template } =
        workloads.remove(&modal_step_id).unwrap()
    else {
        panic!("second prestressed-modal workload must extract modes")
    };
    let model = preload_workload.model.clone();
    let constraints =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
            &model,
            Some(&meshing_output.artifact),
        )
        .unwrap();
    let topology_digest =
        runmat_execution::Digest::from_bytes(*meshing_output.artifact.canonical_digest.bytes());
    let preload =
        runmat_analysis_fea::run_linear_static_with_prepared_constraint_plan_and_linearization(
            &model,
            ComputeBackend::Cpu,
            runmat_analysis_fea::LinearStaticSolveOptions {
                solver_mesh: Some(meshing_output.artifact.clone()),
                require_solver_mesh_for_solid: true,
                ..runmat_analysis_fea::LinearStaticSolveOptions::default()
            },
            &constraints,
            topology_digest,
            None,
        )
        .unwrap();
    let constraint_result =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintResult::recover(
            &constraints,
            &meshing_output.artifact,
            &preload.run.fields,
        )
        .unwrap();
    let preload =
        runmat_analysis_fea::execution::StructuralLinearWorkloadResult::with_linearization(
            preload.run,
            constraint_result,
            preload.linearization,
        )
        .unwrap();
    let sequence =
        runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(&model.sequence)
            .unwrap();
    let producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence,
        &model.sequence.steps[0],
    )
    .unwrap();
    let preload_numerical_receipt =
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::new(
            linear_test_workload_identity(&preload_workload, &meshing_output.artifact),
            &preload,
        )
        .unwrap();
    let source_result = preload_numerical_receipt.source_result();
    let declaration = model.sequence.steps[0]
        .state_outputs
        .iter()
        .find(|output| output.kind == runmat_analysis_core::StateKind::GeometricStiffness)
        .unwrap()
        .clone();
    let state = runmat_analysis_fea::execution::CanonicalSemanticState::new(
        runmat_analysis_fea::execution::CanonicalSemanticStateInput {
            producer,
            declaration,
            topology_digest,
            source_result,
            payload:
                runmat_analysis_fea::execution::CanonicalSemanticStatePayload::StructuralLinearization {
                    linearization: Box::new(preload.linearization.clone().unwrap()),
                },
        },
    )
    .unwrap();
    let transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &template.model,
        &template.step_id,
        &state,
        runmat_analysis_core::StateUse::Prestress,
        topology_digest,
    )
    .unwrap();
    let modal_workload = (*template).bind(transition).unwrap();
    let modal_result = runmat_analysis_fea::execute_structural_prestressed_modal_workload(
        &modal_workload,
        &meshing_output.artifact,
        &state,
    )
    .unwrap();
    let modal_receipt = runmat_analysis_fea::execution::StructuralModalNumericalReceipt::new(
        modal_test_workload_identity(&modal_workload, &meshing_output.artifact),
        &modal_result,
    )
    .unwrap();
    let mode_count = modal_result.basis().modes().len();
    let envelope = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        StructuralStudyNumericalResult::StructuralPrestressedModal(Box::new(
            StructuralStudyPrestressedModalNumericalResult {
                frozen_project_digest: None,
                preload_step_id,
                modal_step_id,
                preload_numerical_receipt,
                preload_result: preload,
                preload_solver_result_identity: source_result,
                prestress_state: state,
                modal_result,
                modal_numerical_receipt: modal_receipt,
            },
        )),
        AnalysisStudyCompletionIdentity::new("server-run-prestressed-modal").unwrap(),
    )
    .expect("prestressed-modal result should enter ordinary study publication");

    assert_eq!(envelope.data.steps.len(), 2);
    for expected in [
        runmat_analysis_core::StateKind::StructuralPrimary,
        runmat_analysis_core::StateKind::GeometricStiffness,
        runmat_analysis_core::StateKind::ModalBasis,
    ] {
        assert!(envelope
            .data
            .states
            .iter()
            .any(|state| state.kind == expected));
    }
    let persisted = storage::load_run_result(&envelope.data.steps[1].run_id)
        .unwrap()
        .expect("prestressed-modal run should be persisted");
    let modal = persisted
        .modal_results
        .expect("ordinary runtime result should retain prestressed modes");
    assert_eq!(modal.mode_shapes.len(), mode_count);
    assert_eq!(modal.basis.modes().len(), mode_count);
    assert_eq!(modal.frequencies_hz().len(), mode_count);
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn structural_prestressed_frequency_intent_is_typed_and_restartable() {
    let meshing_output = ready_buckling_meshing_output();
    let spec = sample_prestressed_frequency_study_spec(digest_hex(
        meshing_output.artifact.geometry.source_digest,
    ));
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-prestressed-frequency-intent".into()), None),
    )
    .expect("prestressed-frequency intent should prepare");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads,
    } = intent.workloads()
    else {
        panic!("prestressed-frequency intent requires a heterogeneous sequence")
    };
    assert!(matches!(
        workloads.get(&runmat_analysis_core::ProcedureStepId(
            "step_preload".into()
        )),
        Some(runmat_analysis_fea::execution::StructuralStepWorkload::Linear { .. })
    ));
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency {
        template,
    }) = workloads.get(&runmat_analysis_core::ProcedureStepId(
        "step_prestressed_frequency".into(),
    ))
    else {
        panic!("prestressed-frequency intent must retain its unresolved typed template")
    };
    assert!(matches!(
        template.solver,
        runmat_analysis_fea::execution::StructuralPrestressedFrequencySolverTemplate::Direct { .. }
    ));
    assert_eq!(template.plan.frequencies_hz(), &[20.0, 40.0]);
    let restored = PreparedStructuralStudyIntent::from_continuation_bytes(
        &intent.continuation_bytes().unwrap(),
    )
    .unwrap();
    assert_eq!(restored.intent_digest, intent.intent_digest);
    assert_eq!(restored.workloads(), intent.workloads());
}

#[test]
fn structural_prestressed_modal_frequency_intent_is_typed_and_restartable() {
    let meshing_output = ready_buckling_meshing_output();
    let spec = sample_prestressed_modal_frequency_study_spec(digest_hex(
        meshing_output.artifact.geometry.source_digest,
    ));
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(
            Some("trace-prestressed-modal-frequency-intent".into()),
            None,
        ),
    )
    .expect("prestressed modal-frequency intent should prepare");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads,
    } = intent.workloads()
    else {
        panic!("prestressed modal-frequency intent requires a heterogeneous sequence")
    };
    assert_eq!(workloads.len(), 3);
    assert!(matches!(
        workloads.get(&runmat_analysis_core::ProcedureStepId(
            "step_prestressed_modes".into()
        )),
        Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal { .. })
    ));
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency {
        template,
    }) = workloads.get(&runmat_analysis_core::ProcedureStepId(
        "step_prestressed_frequency".into(),
    ))
    else {
        panic!("prestressed modal-frequency intent must retain its harmonic template")
    };
    assert_eq!(
        template.solver,
        runmat_analysis_fea::execution::StructuralPrestressedFrequencySolverTemplate::ModalReduced
    );
    let restored = PreparedStructuralStudyIntent::from_continuation_bytes(
        &intent.continuation_bytes().unwrap(),
    )
    .unwrap();
    assert_eq!(restored.intent_digest, intent.intent_digest);
    assert_eq!(restored.workloads(), intent.workloads());
}

#[test]
fn structural_study_intent_admits_one_typed_direct_transient_workload() {
    let _guard = analysis_test_guard();
    let spec = sample_direct_transient_study_spec("44".repeat(32), 9, 4);
    let step = spec.model.sequence.steps[0].clone();

    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-direct-transient-intent".into()), None),
    )
    .expect("direct transient intent should prepare without a local mesh");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralTransient {
        workload,
        block_plan,
    } = intent.workloads()
    else {
        panic!("direct transient study produced the wrong workload kind")
    };
    assert_eq!(workload.step_id, step.step_id);
    assert_eq!(workload.time_step_policy.interval_count(), 9);
    assert_eq!(block_plan.maximum_intervals_per_block(), 4);
    assert_eq!(
        block_plan.terminal_schedule_ordinals(9).unwrap(),
        vec![4, 8, 9]
    );

    let restored = PreparedStructuralStudyIntent::from_continuation_bytes(
        &intent.continuation_bytes().unwrap(),
    )
    .expect("direct transient intent continuation should restore");
    assert_eq!(restored.workloads(), intent.workloads());

    let mut substituted = intent.clone();
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralTransient {
        workload,
        ..
    } = &mut substituted.workloads
    else {
        unreachable!("test prepared a direct transient")
    };
    workload.residual_target *= 2.0;
    substituted.intent_digest = substituted.semantic_digest().unwrap();
    assert!(substituted.continuation_bytes().is_err());
}

#[test]
fn structural_modal_intent_recovers_and_publishes_all_mode_shapes_without_resolving_again() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("remote-modal-completion");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    let meshing_output = ready_minimal_meshing_output();
    let spec = sample_modal_study_spec(digest_hex(meshing_output.artifact.geometry.source_digest));
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-modal-completion".into()), None),
    )
    .expect("modal intent should prepare");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralModal { workload } =
        intent.workloads()
    else {
        panic!("test requires a standalone modal workload")
    };
    let result = runmat_analysis_fea::pipeline::modal::execute_structural_modal_workload(
        &workload,
        &meshing_output.artifact,
    )
    .expect("portable modal extraction should succeed");
    let task_workload = modal_test_workload_identity(&workload, &meshing_output.artifact);
    let numerical_receipt = runmat_analysis_fea::execution::StructuralModalNumericalReceipt::new(
        task_workload,
        &result,
    )
    .unwrap();
    let mode_count = result.basis().modes().len();
    let source_result = numerical_receipt.source_result();
    let step_id = workload.step_id().clone();
    let envelope = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        StructuralStudyNumericalResult::StructuralModal(Box::new(
            StructuralStudyModalNumericalResult {
                frozen_project_digest: None,
                step_id,
                result,
                numerical_receipt,
            },
        )),
        AnalysisStudyCompletionIdentity::new("server-run-modal").unwrap(),
    )
    .expect("portable modal result should enter normal study publication");

    assert_eq!(envelope.data.steps.len(), 1);
    assert_eq!(envelope.data.states.len(), 1);
    assert_eq!(envelope.data.states[0].source_result, source_result);
    let persisted = storage::load_run_result(&envelope.data.steps[0].run_id)
        .unwrap()
        .expect("modal run should be persisted");
    let modal = persisted
        .modal_results
        .expect("runtime modal result should retain recovered shapes");
    assert_eq!(modal.mode_shapes.len(), mode_count);
    assert_eq!(modal.basis.modes().len(), mode_count);
    assert!(modal
        .mode_shapes
        .iter()
        .all(|field| field.shape.len() == 2 && field.shape[1] == 3));
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn prepared_local_modal_study_uses_the_same_portable_recovery_contract() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("local-modal-completion");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    let default_spec = sample_linear_static_study_spec();
    let mut spec = sample_modal_study_spec(default_spec.geometry.source.sha256);
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("local-modal-completion-mesh");
    spec.solver_mesh_artifact_path = Some(mesh_path.display().to_string());
    let prepared = analysis_prepare_structural_study_op(
        &spec,
        OperationContext::new(Some("trace-local-modal-completion".into()), None),
    )
    .expect("local modal study should prepare");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralModal { workload } =
        prepared.workloads()
    else {
        panic!("prepared local study should contain one modal workload")
    };
    let result = runmat_analysis_fea::pipeline::modal::execute_structural_modal_workload(
        &workload,
        prepared.solver_mesh(),
    )
    .unwrap();
    let receipt = runmat_analysis_fea::execution::StructuralModalNumericalReceipt::new(
        modal_test_workload_identity(&workload, prepared.solver_mesh()),
        &result,
    )
    .unwrap();
    let step_id = workload.step_id().clone();
    let envelope = analysis_complete_structural_study_op(
        prepared,
        Ok(StructuralStudyNumericalResult::StructuralModal(Box::new(
            StructuralStudyModalNumericalResult {
                frozen_project_digest: None,
                step_id,
                result,
                numerical_receipt: receipt,
            },
        ))),
    )
    .expect("local modal result should complete through shared study publication");
    assert_eq!(envelope.data.steps.len(), 1);
    assert!(storage::load_run_result(&envelope.data.steps[0].run_id)
        .unwrap()
        .unwrap()
        .modal_results
        .is_some());
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&mesh_root);
}

#[test]
fn modal_transient_intent_publishes_the_exact_basis_and_both_runtime_results() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("remote-modal-transient-completion");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    let meshing_output = ready_minimal_meshing_output();
    let spec = sample_modal_transient_study_spec(digest_hex(
        meshing_output.artifact.geometry.source_digest,
    ));
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-modal-transient-completion".into()), None),
    )
    .expect("modal-transient intent should prepare");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
        mut workloads,
    } = intent.workloads()
    else {
        panic!("test requires heterogeneous modal-transient workloads")
    };
    let [modal_step_id, transient_step_id] = intent
        .model()
        .sequence
        .steps
        .iter()
        .map(|step| step.step_id.clone())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
        workload: modal_workload,
    } = workloads.remove(&modal_step_id).unwrap()
    else {
        panic!("first workload must extract modes")
    };
    let runmat_analysis_fea::execution::StructuralStepWorkload::ModalTransient {
        template,
        block_plan,
    } = workloads.remove(&transient_step_id).unwrap()
    else {
        panic!("second workload must be modal transient")
    };
    let modal_model = modal_workload.model().clone();
    let modal_workload = *modal_workload;
    let modal_result = runmat_analysis_fea::pipeline::modal::execute_structural_modal_workload(
        &modal_workload,
        &meshing_output.artifact,
    )
    .unwrap();
    let modal_numerical_receipt =
        runmat_analysis_fea::execution::StructuralModalNumericalReceipt::new(
            modal_test_workload_identity(&modal_workload, &meshing_output.artifact),
            &modal_result,
        )
        .unwrap();
    let sequence = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &modal_model.sequence,
    )
    .unwrap();
    let modal_step = modal_model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == modal_step_id)
        .unwrap();
    let producer =
        runmat_analysis_fea::execution::StepIdentity::from_solver_contract(sequence, modal_step)
            .unwrap();
    let topology_digest =
        runmat_execution::Digest::from_bytes(*meshing_output.artifact.canonical_digest.bytes());
    let basis_state = runmat_analysis_fea::execution::CanonicalSemanticState::new(
        runmat_analysis_fea::execution::CanonicalSemanticStateInput {
            producer,
            declaration: modal_step.state_outputs[0].clone(),
            topology_digest,
            source_result: modal_numerical_receipt.source_result(),
            payload: runmat_analysis_fea::execution::CanonicalSemanticStatePayload::ModalBasis {
                basis: Box::new(modal_result.basis().clone()),
            },
        },
    )
    .unwrap();
    let mut template = *template;
    template.model = modal_model;
    let transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &template.model,
        &template.step_id,
        &basis_state,
        runmat_analysis_core::StateUse::ModalBasis,
        topology_digest,
    )
    .unwrap();
    let transient_workload = template.bind(transition).unwrap();
    let scope = runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
        &transient_workload,
        &meshing_output.artifact,
        Some(&basis_state),
    )
    .unwrap();
    let terminal_schedule_ordinal = u64::try_from(scope.schedule.points.len() - 1).unwrap();
    let mut predecessor = None;
    let mut segments = Vec::new();
    for terminal in block_plan
        .terminal_schedule_ordinals(terminal_schedule_ordinal)
        .unwrap()
    {
        let request = runmat_analysis_fea::execution::StructuralTransientBlockRequest::new(
            scope.clone(),
            predecessor.as_ref(),
            terminal,
        )
        .unwrap();
        let segment = runmat_analysis_fea::pipeline::transient::run_transient_block(
            runmat_analysis_fea::pipeline::transient::StructuralTransientRunBlockInput {
                workload: &transient_workload,
                solver_mesh: &meshing_output.artifact,
                request: &request,
                predecessor: predecessor.as_ref(),
                modal_state: Some(&basis_state),
                observation_cursors: Vec::new(),
            },
        )
        .unwrap();
        predecessor = Some(segment.manifest().terminal_checkpoint.clone());
        segments.push(segment);
    }
    let assembly =
        runmat_analysis_fea::execution::StructuralTransientResultAssemblyRequest::new(&segments)
            .unwrap();
    let transient_result =
        runmat_analysis_fea::execution::StructuralTransientSolverResult::assemble(
            &assembly, &segments,
        )
        .unwrap();
    let transient_numerical_receipt =
        runmat_analysis_fea::execution::StructuralTransientNumericalReceipt::new(
            &transient_result,
            &assembly,
            &segments,
        )
        .unwrap();
    let envelope = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        StructuralStudyNumericalResult::StructuralModalTransient(Box::new(
            StructuralStudyModalTransientNumericalResult {
                frozen_project_digest: None,
                modal_step_id,
                transient_step_id,
                modal_result,
                modal_numerical_receipt,
                basis_state,
                transient_result,
                transient_numerical_receipt,
            },
        )),
        AnalysisStudyCompletionIdentity::new("server-run-modal-transient").unwrap(),
    )
    .expect("modal-transient result should enter normal study publication");

    assert_eq!(envelope.data.steps.len(), 2);
    assert!(envelope.data.states.len() >= 4);
    let modal = storage::load_run_result(&envelope.data.steps[0].run_id)
        .unwrap()
        .unwrap();
    assert!(modal.modal_results.is_some());
    let transient = storage::load_run_result(&envelope.data.steps[1].run_id)
        .unwrap()
        .unwrap();
    assert!(transient.transient_results.is_some());
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn structural_study_intent_continuation_rejects_semantic_tampering() {
    let _guard = analysis_test_guard();
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = "33".repeat(32);
    let intent =
        analysis_prepare_structural_study_intent_op(&spec, OperationContext::new(None, None))
            .expect("intent should prepare");
    let bytes = intent.continuation_bytes().expect("intent should encode");
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["study_id"] = serde_json::Value::String("substituted-study".into());
    let tampered = serde_json::to_vec(&value).unwrap();

    assert!(PreparedStructuralStudyIntent::from_continuation_bytes(&tampered).is_err());
    assert!(PreparedStructuralStudyIntent::from_continuation_bytes(&[]).is_err());

    let mut wrong_schema: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let supported_schema = wrong_schema["schema_version"].as_u64().unwrap();
    wrong_schema["schema_version"] =
        serde_json::Value::from(supported_schema.checked_add(1).unwrap());
    assert!(PreparedStructuralStudyIntent::from_continuation_bytes(
        &serde_json::to_vec(&wrong_schema).unwrap()
    )
    .is_err());

    let mut unknown_field: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    unknown_field["unexpected"] = serde_json::Value::Bool(true);
    assert!(PreparedStructuralStudyIntent::from_continuation_bytes(
        &serde_json::to_vec(&unknown_field).unwrap()
    )
    .is_err());

    let mut invalid_settings: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    invalid_settings["meshing_settings"]["target_edge_length_m"] = serde_json::Value::from(0.0);
    assert!(PreparedStructuralStudyIntent::from_continuation_bytes(
        &serde_json::to_vec(&invalid_settings).unwrap()
    )
    .is_err());

    let mut substituted_workload = intent.clone();
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::LinearSequence { workloads } =
        &mut substituted_workload.workloads
    else {
        unreachable!("test prepared a linear sequence")
    };
    workloads.values_mut().next().unwrap().preconditioner =
        runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind::Ilu0;
    substituted_workload.intent_digest = substituted_workload.semantic_digest().unwrap();
    assert!(substituted_workload.continuation_bytes().is_err());
}

#[test]
fn structural_study_intent_completion_reenters_shared_quality_and_persistence() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("remote-study-intent-completion");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    let source_store_root = root.join("published-original-fields");
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: source_store_root.clone(),
    })
    .unwrap();
    let meshing_output = ready_minimal_meshing_output();
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = digest_hex(meshing_output.artifact.geometry.source_digest);
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(
            Some("trace-remote-completion".into()),
            Some("request-remote-completion".into()),
        ),
    )
    .expect("remote study intent should prepare");
    let executor = CanonicalStructuralExecutor {
        calls: AtomicU64::new(0),
    };
    let workload = only_workload(&intent);
    let numerical_result = executor
        .execute(&workload, &meshing_output.artifact)
        .expect("portable structural solve should succeed");
    let numerical_result = sequence_result(&intent, &meshing_output.artifact, numerical_result);
    let retry_intent = intent.clone();
    let retry_mesh = meshing_output.clone();
    let retry_result = numerical_result.clone();

    let envelope = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        numerical_result,
        AnalysisStudyCompletionIdentity::new("server-run-completion").unwrap(),
    )
    .expect("remote study result should reenter runtime completion");
    let retried = analysis_complete_structural_study_intent_op(
        retry_intent,
        retry_mesh,
        retry_result,
        AnalysisStudyCompletionIdentity::new("server-run-completion").unwrap(),
    )
    .expect("retrying one server allocation should converge on its completed run");

    assert_eq!(
        envelope.trace_id.as_deref(),
        Some("trace-remote-completion")
    );
    assert_eq!(envelope.data.study_id, spec.study_id);
    assert_eq!(retried.data, envelope.data);
    assert_eq!(storage::list_run_results().unwrap().len(), 1);
    let stored = storage::load_run_result(&envelope.data.steps[0].run_id)
        .expect("completed remote run lookup should succeed")
        .expect("completed remote run must be stored");
    original_fields::assert_static_source(&stored, &source_store_root);
    let mut conflicting = stored.clone();
    let runmat_analysis_core::AnalysisFieldValues::HostF64(values) =
        &mut conflicting.run.fields[0].values
    else {
        panic!("portable CPU result must retain host field values");
    };
    values[0] += 1.0;
    assert!(storage::persist_run_result(&conflicting).is_err());

    let round_trip: AnalysisRunResult =
        serde_json::from_slice(&serde_json::to_vec(&stored).unwrap()).unwrap();
    assert_eq!(round_trip, stored, "analysis runs must round-trip exactly");

    let filesystem_store = storage::FilesystemAnalysisArtifactStore::new(root.join("run-store"));
    storage::AnalysisArtifactStore::persist_run(&filesystem_store, &stored)
        .expect("first filesystem publication should succeed");
    storage::AnalysisArtifactStore::persist_run(&filesystem_store, &stored)
        .expect("identical filesystem publication should be idempotent");
    assert!(storage::AnalysisArtifactStore::persist_run(&filesystem_store, &conflicting).is_err());
    assert!(envelope.data.solver_mesh_artifact_path.is_some());
    assert!(envelope.data.meshing_evidence_artifact_path.is_some());
    assert!(Path::new(&envelope.data.evidence_artifact_path).is_file());
    assert!(Path::new(
        envelope
            .data
            .solver_mesh_artifact_path
            .as_deref()
            .expect("mesh artifact path")
    )
    .is_file());
    assert!(Path::new(
        envelope
            .data
            .meshing_evidence_artifact_path
            .as_deref()
            .expect("meshing evidence path")
    )
    .is_file());
    drop(_runtime_guard);
    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn structural_study_intent_completion_rejects_a_different_geometry_source() {
    let _guard = analysis_test_guard();
    let meshing_output = ready_minimal_meshing_output();
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = "77".repeat(32);
    let intent =
        analysis_prepare_structural_study_intent_op(&spec, OperationContext::new(None, None))
            .expect("remote study intent should prepare");
    let executor = CanonicalStructuralExecutor {
        calls: AtomicU64::new(0),
    };
    let workload = only_workload(&intent);
    let numerical_result = executor
        .execute(&workload, &meshing_output.artifact)
        .expect("portable structural solve should succeed");
    let numerical_result = sequence_result(&intent, &meshing_output.artifact, numerical_result);

    let error = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        numerical_result,
        AnalysisStudyCompletionIdentity::new("server-run-source-mismatch").unwrap(),
    )
    .expect_err("a mesh from another geometry source must fail closed");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_STUDY.REMOTE_MESH_IDENTITY_MISMATCH"
    );
}

#[test]
fn structural_study_intent_completion_rejects_a_different_meshing_request() {
    let _guard = analysis_test_guard();
    let mut meshing_output = ready_minimal_meshing_output();
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = digest_hex(meshing_output.artifact.geometry.source_digest);
    let intent =
        analysis_prepare_structural_study_intent_op(&spec, OperationContext::new(None, None))
            .expect("remote study intent should prepare");
    meshing_output.artifact.resolved_request.deterministic_seed ^= 1;
    meshing_output.artifact.canonical_digest = StableDigest::ZERO;
    meshing_output.artifact.seal_canonical_digest().unwrap();
    meshing_output.evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(
        &meshing_output.artifact,
    );
    let executor = CanonicalStructuralExecutor {
        calls: AtomicU64::new(0),
    };
    let workload = only_workload(&intent);
    let numerical_result = executor
        .execute(&workload, &meshing_output.artifact)
        .expect("portable structural solve should succeed");
    let numerical_result = sequence_result(&intent, &meshing_output.artifact, numerical_result);

    let error = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        numerical_result,
        AnalysisStudyCompletionIdentity::new("server-run-request-mismatch").unwrap(),
    )
    .expect_err("a mesh from another canonical request must fail closed");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_STUDY.REMOTE_MESH_IDENTITY_MISMATCH"
    );
}

fn sample_direct_transient_study_spec(
    geometry_sha256: String,
    interval_count: usize,
    maximum_intervals_per_block: u64,
) -> AnalysisStudySpec {
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = geometry_sha256;
    let mut model = spec.model.clone();
    let step = ProcedureStep::structural_transient("step_direct_transient");
    model.sequence = AnalysisSequence::from_steps("sequence_direct_transient", vec![step.clone()]);
    spec.model = model;
    spec.execution = AnalysisExecutionPlan {
        steps: vec![AnalysisExecutionStep {
            step_id: step.step_id,
            backend: ComputeBackend::Cpu,
            accelerator_admission: Default::default(),
            solver_options: AnalysisSolverOptions::StructuralTransient(
                AnalysisTransientRunOptions {
                    time_step_policy: AnalysisStructuralTimeStepPolicy::Fixed { interval_count },
                    maximum_intervals_per_block,
                    ..AnalysisTransientRunOptions::solid_recommended()
                },
            ),
        }],
    };
    spec
}

fn sample_modal_study_spec(geometry_sha256: String) -> AnalysisStudySpec {
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = geometry_sha256;
    let mut model = spec.model.clone();
    let mut step = ProcedureStep::structural_modal("step_modal");
    step.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 1 };
    model.sequence = AnalysisSequence::from_steps("sequence_modal", vec![step.clone()]);
    spec.model = model;
    spec.execution = AnalysisExecutionPlan {
        steps: vec![AnalysisExecutionStep {
            step_id: step.step_id,
            backend: ComputeBackend::Cpu,
            accelerator_admission: Default::default(),
            solver_options: AnalysisSolverOptions::StructuralModes(AnalysisModalRunOptions {
                truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(1),
                convergence: runmat_analysis_fea::ModalConvergencePolicy::exploratory(),
                ..AnalysisModalRunOptions::default()
            }),
        }],
    };
    spec
}

fn sample_modal_transient_study_spec(geometry_sha256: String) -> AnalysisStudySpec {
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = geometry_sha256;
    let mut model = spec.model.clone();
    let mut modal = ProcedureStep::structural_modal("step_modal");
    modal.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 1 };
    let mut transient = ProcedureStep::structural_modal_transient(
        "step_modal_transient",
        modal.step_id.as_str(),
        "modal_basis",
    );
    transient.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 0.1,
        interval_count: 10,
    };
    let mut sequence = AnalysisSequence::from_steps(
        "sequence_modal_transient",
        vec![modal.clone(), transient.clone()],
    );
    sequence.dependencies[0].reason = runmat_analysis_core::SequenceDependencyReason::StateTransfer;
    sequence
        .transitions
        .push(runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId("modal_to_transient".into()),
            producer_step: modal.step_id.clone(),
            consumer_step: transient.step_id.clone(),
            state_id: runmat_analysis_core::StateId("modal_basis".into()),
            method: runmat_analysis_core::StateTransitionMethod::Reuse,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        });
    model.sequence = sequence;
    spec.model = model;
    spec.execution = AnalysisExecutionPlan {
        steps: vec![
            AnalysisExecutionStep {
                step_id: modal.step_id,
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::StructuralModes(AnalysisModalRunOptions {
                    truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(1),
                    convergence: runmat_analysis_fea::ModalConvergencePolicy::exploratory(),
                    ..AnalysisModalRunOptions::default()
                }),
            },
            AnalysisExecutionStep {
                step_id: transient.step_id,
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::StructuralTransient(
                    AnalysisTransientRunOptions {
                        time_step_policy: AnalysisStructuralTimeStepPolicy::Fixed {
                            interval_count: 4,
                        },
                        maximum_intervals_per_block: 2,
                        ..AnalysisTransientRunOptions::solid_recommended()
                    },
                ),
            },
        ],
    };
    spec
}

fn sample_response_spectrum_study_spec(geometry_sha256: String) -> AnalysisStudySpec {
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = geometry_sha256;
    let mut model = spec.model.clone();
    let mut modal = ProcedureStep::structural_modal("step_modal");
    modal.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 3 };
    let spectrum = runmat_analysis_core::ResponseSpectrum {
        spectrum_id: runmat_analysis_core::ResponseSpectrumId("spectrum_x".into()),
        abscissa: runmat_analysis_core::SpectrumAbscissa::FrequencyHertz(
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Frequency, _>::new(
                vec![0.1, 100.0],
            ),
        ),
        ordinates: runmat_analysis_core::NumericalDataSource::authored(
            vec![1.0, 1.0],
            "sha256:response-spectrum-intent",
        ),
        damping: runmat_analysis_core::SpectrumDamping::Ratio(0.05),
        response: runmat_analysis_core::SpectrumResponse::RelativeDisplacement,
        envelope: runmat_analysis_core::SpectrumEnvelope::Maximax,
        interpolation: runmat_analysis_core::SpectrumInterpolation::LinearFrequency,
        extrapolation: runmat_analysis_core::SpectrumExtrapolation::HoldEndpoint,
        direction: runmat_analysis_core::SpectrumDirection {
            frame: runmat_analysis_core::CoordinateFrame::Global,
            vector: [1.0, 0.0, 0.0],
        },
        modal_combination: runmat_analysis_core::ModalCombination::AbsoluteSum,
        directional_combination: runmat_analysis_core::DirectionalCombination::Srss,
        truncation: runmat_analysis_core::ModalTruncationPolicy {
            minimum_cumulative_mass_ratio: 0.9,
            correction: runmat_analysis_core::ModalTruncationCorrection::ResidualMode,
        },
        rigid_response: runmat_analysis_core::RigidResponsePolicy::Disabled,
        construction: runmat_analysis_core::SpectrumConstruction::AuthoredDesign,
        evidence: runmat_analysis_core::SpectrumEvidence {
            sampling_adequacy_ratio: 1.0,
            max_interpolation_error: None,
        },
    };
    let response = ProcedureStep::structural_modal_response_spectrum(
        "step_response",
        vec![spectrum.spectrum_id.clone()],
        modal.step_id.as_str(),
        "modal_basis",
    );
    let mut sequence = AnalysisSequence::from_steps(
        "sequence_response_spectrum",
        vec![modal.clone(), response.clone()],
    );
    sequence.dependencies[0].reason = runmat_analysis_core::SequenceDependencyReason::StateTransfer;
    sequence
        .transitions
        .push(runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId("modal_to_response".into()),
            producer_step: modal.step_id.clone(),
            consumer_step: response.step_id.clone(),
            state_id: runmat_analysis_core::StateId("modal_basis".into()),
            method: runmat_analysis_core::StateTransitionMethod::Reuse,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        });
    model.sequence = sequence;
    model.response_spectra = vec![spectrum];
    spec.model = model;
    spec.execution = AnalysisExecutionPlan {
        steps: vec![
            AnalysisExecutionStep {
                step_id: modal.step_id,
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::StructuralModes(AnalysisModalRunOptions {
                    truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(3),
                    convergence: runmat_analysis_fea::ModalConvergencePolicy::exploratory(),
                    ..AnalysisModalRunOptions::default()
                }),
            },
            AnalysisExecutionStep {
                step_id: response.step_id,
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::StructuralResponseSpectrum(
                    AnalysisResponseSpectrumRunOptions::default(),
                ),
            },
        ],
    };
    spec
}

fn sample_random_response_study_spec(geometry_sha256: String) -> AnalysisStudySpec {
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = geometry_sha256;
    let mut model = spec.model.clone();
    let mut modal = ProcedureStep::structural_modal("step_modal");
    modal.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 3 };
    let excitation = runmat_analysis_core::RandomExcitationSpectrum {
        excitation_id: runmat_analysis_core::RandomExcitationId("broadband_x".into()),
        frequencies_hz: runmat_types::physical::Quantity::new(vec![0.1, 10.0, 100.0]),
        channels: vec![runmat_analysis_core::RandomExcitationChannel {
            channel_id: runmat_analysis_core::RandomExcitationChannelId("base_x".into()),
            quantity: runmat_analysis_core::RandomExcitationQuantity::BaseAcceleration,
            direction: runmat_analysis_core::SpectrumDirection {
                frame: runmat_analysis_core::CoordinateFrame::Global,
                vector: [1.0, 0.0, 0.0],
            },
        }],
        spectral_density: runmat_analysis_core::NumericalDataSource::authored(
            vec![
                runmat_analysis_core::CrossPowerSpectralDensity::diagonal(
                    &runmat_types::physical::Quantity::new([1.0e-4])
                );
                3
            ],
            "sha256:random-response-intent",
        ),
        convention: runmat_analysis_core::SpectralDensityConvention::OneSidedPerHertz,
        interpolation: runmat_analysis_core::RandomSpectrumInterpolation::LinearFrequency,
        extrapolation: runmat_analysis_core::RandomSpectrumExtrapolation::Reject,
        evidence: runmat_analysis_core::RandomSpectrumEvidence {
            max_absolute_density_error: Some(
                runmat_types::physical::AccelerationPowerSpectralDensity::new(0.0),
            ),
            equivalent_degrees_of_freedom: None,
        },
    };
    let response = ProcedureStep::structural_random_response(
        "step_response",
        excitation.excitation_id.clone(),
        runmat_analysis_core::StructuralRandomResponseSettings {
            modal_damping_ratio: 0.05,
            minimum_cumulative_mass_ratio: 0.9,
        },
        modal.step_id.as_str(),
        "modal_basis",
    );
    let mut sequence = AnalysisSequence::from_steps(
        "sequence_random_response",
        vec![modal.clone(), response.clone()],
    );
    sequence.dependencies[0].reason = runmat_analysis_core::SequenceDependencyReason::StateTransfer;
    sequence
        .transitions
        .push(runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId("modal_to_random".into()),
            producer_step: modal.step_id.clone(),
            consumer_step: response.step_id.clone(),
            state_id: runmat_analysis_core::StateId("modal_basis".into()),
            method: runmat_analysis_core::StateTransitionMethod::Reuse,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        });
    model.sequence = sequence;
    model.random_excitations = vec![excitation];
    spec.model = model;
    spec.execution = AnalysisExecutionPlan {
        steps: vec![
            AnalysisExecutionStep {
                step_id: modal.step_id,
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::StructuralModes(AnalysisModalRunOptions {
                    truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(3),
                    convergence: runmat_analysis_fea::ModalConvergencePolicy::exploratory(),
                    ..AnalysisModalRunOptions::default()
                }),
            },
            AnalysisExecutionStep {
                step_id: response.step_id,
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::StructuralRandomResponse(
                    AnalysisRandomResponseRunOptions {
                        integration: runmat_analysis_fea::solve::dynamics::RandomResponseIntegrationPolicy {
                            minimum_cumulative_mass_ratio: 0.0,
                            ..runmat_analysis_fea::solve::dynamics::RandomResponseIntegrationPolicy::production_default()
                        },
                        maximum_intervals_per_task: 2,
                        ..AnalysisRandomResponseRunOptions::default()
                    },
                ),
            },
        ],
    };
    spec
}

fn sample_buckling_study_spec(geometry_sha256: String) -> AnalysisStudySpec {
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = geometry_sha256;
    let mut model = spec.model.clone();
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        panic!("buckling fixture must carry a force signal")
    };
    signal.representation = runmat_analysis_core::SignalRepresentation::Constant(
        runmat_analysis_core::SignalValue::Vector3([
            runmat_analysis_core::Force::new(0.0),
            runmat_analysis_core::Force::new(0.0),
            runmat_analysis_core::Force::new(-1.0),
        ]),
    );
    let preload = ProcedureStep::structural_preload("step_preload");
    let mut buckling = ProcedureStep::structural_buckling(
        "step_buckling",
        preload.step_id.as_str(),
        "geometric_stiffness",
    );
    buckling.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 1 };
    let mut sequence =
        AnalysisSequence::from_steps("sequence_buckling", vec![preload.clone(), buckling.clone()]);
    sequence.dependencies[0].reason = runmat_analysis_core::SequenceDependencyReason::StateTransfer;
    sequence
        .transitions
        .push(runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId("preload_to_buckling".into()),
            producer_step: preload.step_id.clone(),
            consumer_step: buckling.step_id.clone(),
            state_id: runmat_analysis_core::StateId("geometric_stiffness".into()),
            method: runmat_analysis_core::StateTransitionMethod::Reuse,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        });
    model.sequence = sequence;
    spec.model = model;
    spec.execution = AnalysisExecutionPlan {
        steps: vec![
            AnalysisExecutionStep {
                step_id: preload.step_id,
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::StructuralLinear(
                    AnalysisRunOptions::default(),
                ),
            },
            AnalysisExecutionStep {
                step_id: buckling.step_id,
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::StructuralBuckling(
                    AnalysisBucklingRunOptions {
                        solver:
                            runmat_analysis_fea::execution::StructuralBucklingSolverPolicy::new(
                                1, 1.0e-8, 256,
                            )
                            .unwrap(),
                        ..AnalysisBucklingRunOptions::default()
                    },
                ),
            },
        ],
    };
    spec
}

fn sample_prestressed_modal_study_spec(geometry_sha256: String) -> AnalysisStudySpec {
    let mut spec = sample_buckling_study_spec(geometry_sha256);
    let model = &mut spec.model;
    let preload = model.sequence.steps[0].clone();
    let mut modal = ProcedureStep::structural_prestressed_modal(
        "step_prestressed_modal",
        preload.step_id.as_str(),
        "geometric_stiffness",
    );
    modal.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 1 };
    let mut sequence = AnalysisSequence::from_steps(
        "sequence_prestressed_modal",
        vec![preload.clone(), modal.clone()],
    );
    sequence.dependencies[0].reason = runmat_analysis_core::SequenceDependencyReason::StateTransfer;
    sequence
        .transitions
        .push(runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId(
                "preload_to_prestressed_modal".into(),
            ),
            producer_step: preload.step_id.clone(),
            consumer_step: modal.step_id.clone(),
            state_id: runmat_analysis_core::StateId("geometric_stiffness".into()),
            method: runmat_analysis_core::StateTransitionMethod::Reuse,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        });
    model.sequence = sequence;
    spec.execution.steps[1] = AnalysisExecutionStep {
        step_id: modal.step_id,
        backend: ComputeBackend::Cpu,
        accelerator_admission: Default::default(),
        solver_options: AnalysisSolverOptions::StructuralModes(AnalysisModalRunOptions {
            truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(1),
            convergence: runmat_analysis_fea::ModalConvergencePolicy::exploratory(),
            ..AnalysisModalRunOptions::default()
        }),
    };
    spec
}

fn sample_prestressed_frequency_study_spec(geometry_sha256: String) -> AnalysisStudySpec {
    let mut spec = sample_buckling_study_spec(geometry_sha256);
    let model = &mut spec.model;
    let preload = model.sequence.steps[0].clone();
    let mut frequency = ProcedureStep::structural_prestressed_harmonic(
        "step_prestressed_frequency",
        preload.step_id.as_str(),
        "geometric_stiffness",
    );
    frequency.independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    let mut sequence = AnalysisSequence::from_steps(
        "sequence_prestressed_frequency",
        vec![preload.clone(), frequency.clone()],
    );
    sequence.dependencies[0].reason = runmat_analysis_core::SequenceDependencyReason::StateTransfer;
    sequence
        .transitions
        .push(runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId(
                "preload_to_prestressed_frequency".into(),
            ),
            producer_step: preload.step_id.clone(),
            consumer_step: frequency.step_id.clone(),
            state_id: runmat_analysis_core::StateId("geometric_stiffness".into()),
            method: runmat_analysis_core::StateTransitionMethod::Reuse,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        });
    model.sequence = sequence;
    spec.execution.steps[1] = AnalysisExecutionStep {
        step_id: frequency.step_id,
        backend: ComputeBackend::Cpu,
        accelerator_admission: Default::default(),
        solver_options: AnalysisSolverOptions::StructuralFrequency(
            AnalysisStructuralFrequencyRunOptions::default(),
        ),
    };
    spec
}

fn sample_prestressed_modal_frequency_study_spec(geometry_sha256: String) -> AnalysisStudySpec {
    let mut spec = sample_prestressed_frequency_study_spec(geometry_sha256);
    let model = &mut spec.model;
    let preload = model.sequence.steps[0].clone();
    let mut modal = ProcedureStep::structural_prestressed_modal(
        "step_prestressed_modes",
        preload.step_id.as_str(),
        "geometric_stiffness",
    );
    modal.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 3 };
    let mut frequency = model.sequence.steps[1].clone();
    frequency.solution_space = runmat_analysis_core::SolutionSpace::ModalReduced;
    frequency
        .state_inputs
        .push(runmat_analysis_core::StateInput {
            input_id: "modal_basis".into(),
            required_kind: runmat_analysis_core::StateKind::ModalBasis,
            source: runmat_analysis_core::StateSource::Upstream {
                producer_step: modal.step_id.clone(),
                state_id: runmat_analysis_core::StateId("modal_basis".into()),
            },
            use_as: runmat_analysis_core::StateUse::ModalBasis,
            conflict: runmat_analysis_core::InitialStateConflictPolicy::Reject,
        });
    let mut sequence = AnalysisSequence::from_steps(
        "sequence_prestressed_modal_frequency",
        vec![preload.clone(), modal.clone(), frequency.clone()],
    );
    for dependency in &mut sequence.dependencies {
        dependency.reason = runmat_analysis_core::SequenceDependencyReason::StateTransfer;
    }
    sequence
        .dependencies
        .push(runmat_analysis_core::SequenceDependency {
            predecessor: preload.step_id.clone(),
            successor: frequency.step_id.clone(),
            reason: runmat_analysis_core::SequenceDependencyReason::StateTransfer,
        });
    sequence.transitions = vec![
        runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId(
                "preload_to_prestressed_modal".into(),
            ),
            producer_step: preload.step_id.clone(),
            consumer_step: modal.step_id.clone(),
            state_id: runmat_analysis_core::StateId("geometric_stiffness".into()),
            method: runmat_analysis_core::StateTransitionMethod::Reuse,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        },
        runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId(
                "preload_to_prestressed_frequency".into(),
            ),
            producer_step: preload.step_id.clone(),
            consumer_step: frequency.step_id.clone(),
            state_id: runmat_analysis_core::StateId("geometric_stiffness".into()),
            method: runmat_analysis_core::StateTransitionMethod::Reuse,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        },
        runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId(
                "prestressed_modal_to_frequency".into(),
            ),
            producer_step: modal.step_id.clone(),
            consumer_step: frequency.step_id.clone(),
            state_id: runmat_analysis_core::StateId("modal_basis".into()),
            method: runmat_analysis_core::StateTransitionMethod::Reuse,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        },
    ];
    model.sequence = sequence;
    spec.execution.steps.insert(
        1,
        AnalysisExecutionStep {
            step_id: modal.step_id,
            backend: ComputeBackend::Cpu,
            accelerator_admission: Default::default(),
            solver_options: AnalysisSolverOptions::StructuralModes(AnalysisModalRunOptions {
                truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(3),
                convergence: runmat_analysis_fea::ModalConvergencePolicy::exploratory(),
                ..AnalysisModalRunOptions::default()
            }),
        },
    );
    let frequency_execution = spec
        .execution
        .steps
        .iter_mut()
        .find(|execution| execution.step_id == frequency.step_id)
        .expect("prestressed frequency execution must exist");
    frequency_execution.solver_options =
        AnalysisSolverOptions::StructuralFrequency(AnalysisStructuralFrequencyRunOptions {
            solver: AnalysisStructuralFrequencySolver::ModalReduced,
            ..AnalysisStructuralFrequencyRunOptions::default()
        });
    spec
}

fn modal_test_workload_identity(
    workload: &runmat_analysis_fea::execution::StructuralModalWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> runmat_analysis_fea::execution::AnalysisWorkloadIdentity {
    test_analysis_workload_identity(
        workload.model(),
        workload.procedure_step().unwrap(),
        runmat_analysis_fea::execution::AnalysisWorkKind::ExtractStructuralModes,
        workload.identity().digest(),
        mesh,
        "runtime-modal-completion-test",
    )
}

fn linear_test_workload_identity(
    workload: &runmat_analysis_fea::execution::StructuralLinearWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> runmat_analysis_fea::execution::AnalysisWorkloadIdentity {
    test_analysis_workload_identity(
        &workload.model,
        workload.procedure_step().unwrap(),
        runmat_analysis_fea::execution::AnalysisWorkKind::Solve,
        runmat_execution::Digest::sha256(workload.canonical_bytes().unwrap()),
        mesh,
        "runtime-linear-completion-test",
    )
}

fn buckling_test_workload_identity(
    workload: &runmat_analysis_fea::execution::StructuralBucklingWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> runmat_analysis_fea::execution::AnalysisWorkloadIdentity {
    test_analysis_workload_identity(
        workload.model(),
        workload.procedure_step().unwrap(),
        runmat_analysis_fea::execution::AnalysisWorkKind::SolveStructuralBucklingModes,
        workload.identity().digest(),
        mesh,
        "runtime-buckling-completion-test",
    )
}

fn test_analysis_workload_identity(
    model: &runmat_analysis_core::AnalysisModel,
    procedure_step: &runmat_analysis_core::ProcedureStep,
    work_kind: runmat_analysis_fea::execution::AnalysisWorkKind,
    contract_digest: runmat_execution::Digest,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    label: &str,
) -> runmat_analysis_fea::execution::AnalysisWorkloadIdentity {
    use runmat_analysis_fea::execution::{
        AnalysisWorkloadIdentity, ConstraintPlanIdentity, DiscretizationIdentity,
        ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity, ProblemIdentity,
        SequenceIdentity, StepIdentity, StudyIdentity,
    };
    let digest = runmat_execution::Digest::sha256(label);
    let study = StudyIdentity::from_authored_digest(digest, None).unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let step = StepIdentity::from_solver_contract(sequence, procedure_step).unwrap();
    let constraints = ConstraintPlanIdentity::new(step, digest, digest).unwrap();
    let problem = ProblemIdentity::new(step, digest, None, constraints, Vec::new()).unwrap();
    let mesh_digest = runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
    let discretization = DiscretizationIdentity::new(problem, mesh_digest, digest).unwrap();
    let operator = OperatorIdentity::new(discretization, digest).unwrap();
    let numerical = NumericalPlanIdentity::new(operator, digest).unwrap();
    let execution = ExecutionPlanIdentity::new(numerical, digest).unwrap();
    AnalysisWorkloadIdentity::new(
        study,
        step,
        execution,
        work_kind,
        digest,
        vec![contract_digest, mesh_digest],
    )
    .unwrap()
}

fn ready_minimal_meshing_output() -> crate::meshing::MeshingExecutionOutput {
    let mut artifact =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    artifact.resolved_request = structural_study::resolve_intent_meshing_request(Some(
        runmat_meshing_core::MeshingRequestSettings::default(),
    ))
    .unwrap()
    .unwrap();
    artifact.topology.boundary_faces[0]
        .provenance
        .push(persistent_face("root"));
    artifact.topology.boundary_faces[1]
        .provenance
        .push(persistent_face("tip"));
    for face in &mut artifact.topology.boundary_faces {
        face.provenance.sort();
    }
    artifact.canonical_digest = StableDigest::ZERO;
    artifact
        .seal_canonical_digest()
        .expect("seal remote solver mesh fixture");
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&artifact);
    crate::meshing::MeshingExecutionOutput { artifact, evidence }
}

fn ready_buckling_meshing_output() -> crate::meshing::MeshingExecutionOutput {
    let mut output = ready_minimal_meshing_output();
    let tip = persistent_face("tip");
    output.artifact.topology.boundary_faces[1]
        .provenance
        .retain(|source| source != &tip);
    output.artifact.topology.boundary_faces[3]
        .provenance
        .push(tip);
    output.artifact.topology.boundary_faces[3].provenance.sort();
    output.artifact.canonical_digest = StableDigest::ZERO;
    output.artifact.seal_canonical_digest().unwrap();
    output.evidence =
        runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&output.artifact);
    output
}

fn digest_hex(digest: StableDigest) -> String {
    digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn only_workload(
    intent: &PreparedStructuralStudyIntent,
) -> runmat_analysis_fea::execution::StructuralLinearWorkload {
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::LinearSequence { workloads } =
        intent.workloads()
    else {
        panic!("test requires a structural-linear sequence intent")
    };
    assert_eq!(workloads.len(), 1);
    workloads.into_values().next().unwrap()
}

fn sequence_result(
    intent: &PreparedStructuralStudyIntent,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    result: runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
) -> StructuralStudyNumericalResult {
    let workload = only_workload(intent);
    let _study = runmat_analysis_fea::execution::StudyIdentity::from_authored_digest(
        intent.authored_study_digest(),
        None,
    )
    .unwrap();
    let sequence = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &workload.model.sequence,
    )
    .unwrap();
    let step = workload.procedure_step().unwrap();
    let producer =
        runmat_analysis_fea::execution::StepIdentity::from_solver_contract(sequence, step).unwrap();
    let source_workload = linear_test_workload_identity(&workload, mesh);
    let numerical_receipt =
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::new(
            source_workload,
            &result,
        )
        .unwrap();
    let solver_result_identity = numerical_receipt.source_result();
    let fields = result
        .run
        .fields
        .iter()
        .filter(|field| !field.is_empty() && field.as_host_f64().is_some())
        .cloned()
        .collect::<Vec<_>>();
    let states = step
        .state_outputs
        .iter()
        .map(|declaration| {
            runmat_analysis_fea::execution::CanonicalSemanticState::new(
                runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                    producer,
                    declaration: declaration.clone(),
                    topology_digest: runmat_execution::Digest::from_bytes(
                        *mesh.canonical_digest.bytes(),
                    ),
                    source_result: solver_result_identity,
                    payload:
                        runmat_analysis_fea::execution::CanonicalSemanticStatePayload::Fields {
                            fields: fields.clone(),
                        },
                },
            )
            .unwrap()
        })
        .collect();
    StructuralStudyNumericalResult::LinearSequence(StructuralStudySequenceNumericalResult {
        frozen_project_digest: None,
        effective_memory_ceiling_bytes: None,
        steps: vec![StructuralStudyStepNumericalResult {
            step_id: workload.step_id.clone(),
            solver_result_identity,
            result,
            numerical_receipt,
            static_solve_path: None,
        }],
        states,
    })
}
