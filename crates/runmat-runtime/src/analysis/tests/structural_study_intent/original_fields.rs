//! Authored publication to cold original-field input; no producer re-solve.
use super::*;

pub(super) fn assert_static_source(stored: &AnalysisRunResult, store_root: &Path) {
    let original = stored
        .run
        .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .expect("static solve must retain displacement");
    let binding = storage::export_initial_field(&stored.run_id, &original.field_id, 128 << 20)
        .expect("automatic static catalog must be readable")
        .expect("published displacement must be an original source");
    let recovered = binding.artifact().fields().first().unwrap();
    assert_eq!(
        recovered.storage().as_host_f64().unwrap(),
        original.as_host_f64().unwrap()
    );
    assert_eq!(binding.reference().artifact_id, stored.run_id);
    assert_eq!(
        recovered.definition().quantity(),
        runmat_analysis_core::ObservationQuantity::Displacement
    );
    assert_eq!(
        recovered.sample(),
        runmat_analysis_fea::results::field::RecoveredFieldSample::Final
    );
    assert!(
        storage::export_initial_field(&stored.run_id, "unpublished", 128 << 20)
            .unwrap()
            .is_none()
    );
    let reopened = storage::FilesystemAnalysisArtifactStore::new(store_root.to_path_buf());
    let cold = storage::AnalysisArtifactStore::export_initial_field(
        &reopened,
        &stored.run_id,
        &original.field_id,
        128 << 20,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        cold, binding,
        "cold static source must retain exact values and identity"
    );
}

#[test]
fn structural_study_intent_completes_a_partitioned_direct_transient_with_state() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        witness(order);
    }
}

fn witness(order: ElementOrder) {
    let root = temp_artifact_root(&format!("remote-direct-transient-completion-{order:?}"));
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    let store_root = root.join("original-source-store");
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: store_root.clone(),
    })
    .unwrap();
    let meshing_output = mesh_output(order);
    let mut spec = sample_direct_transient_study_spec(
        digest_hex(meshing_output.artifact.geometry.source_digest),
        4,
        2,
    );
    spec.meshing_settings.as_mut().unwrap().element_order = order;
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-direct-transient-completion".into()), None),
    )
    .expect("direct transient intent should prepare");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralTransient {
        workload,
        block_plan,
    } = intent.workloads()
    else {
        panic!("test requires a direct transient workload")
    };
    let scope = runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
        &workload,
        &meshing_output.artifact,
        None,
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
                workload: &workload,
                solver_mesh: &meshing_output.artifact,
                request: &request,
                predecessor: predecessor.as_ref(),
                modal_state: None,
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
    let result = runmat_analysis_fea::execution::StructuralTransientSolverResult::assemble(
        &assembly, &segments,
    )
    .unwrap();
    let numerical_receipt =
        runmat_analysis_fea::execution::StructuralTransientNumericalReceipt::new(
            &result, &assembly, &segments,
        )
        .unwrap();
    let result_identity = result.manifest().identity;
    let envelope = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        StructuralStudyNumericalResult::StructuralTransient(Box::new(
            StructuralStudyTransientNumericalResult {
                frozen_project_digest: None,
                step_id: workload.step_id.clone(),
                result,
                numerical_receipt,
            },
        )),
        AnalysisStudyCompletionIdentity::new("server-run-direct-transient").unwrap(),
    )
    .expect("partitioned direct transient should enter normal study publication");

    assert_eq!(envelope.data.steps.len(), 1);
    assert_eq!(
        envelope.data.states.len(),
        spec.model.sequence.steps[0].state_outputs.len()
    );
    assert!(envelope
        .data
        .states
        .iter()
        .all(|state| state.source_result == result_identity
            && Path::new(&state.artifact_path).is_file()));
    assert_eq!(storage::list_run_results().unwrap().len(), 1);
    let source_run_id = envelope.data.steps[0].run_id.clone();
    let source_run = storage::load_run_result(&source_run_id).unwrap().unwrap();
    let history = source_run.transient_results.as_ref().unwrap();
    let displacement = history.displacement_snapshots.last().unwrap().clone();
    let velocity = history.velocity_snapshots.last().unwrap().clone();
    assert!(displacement
        .as_host_f64()
        .unwrap()
        .iter()
        .any(|value| *value != 0.0));
    drop(source_run);
    drop(envelope);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: store_root,
    })
    .unwrap();
    let u = storage::export_initial_field(&source_run_id, &displacement.field_id, 128 << 20)
        .unwrap()
        .unwrap();
    let v = storage::export_initial_field(&source_run_id, &velocity.field_id, 128 << 20)
        .unwrap()
        .unwrap();
    assert_eq!(u.artifact().fields()[0].storage(), &displacement);
    assert_eq!(v.artifact().fields()[0].storage(), &velocity);
    let mut consumer = spec.clone();
    consumer.study_id = "cold-original-field-consumer".into();
    consumer.model.initial_conditions = vec![
        runmat_analysis_core::InitialCondition::Displacement(
            runmat_analysis_core::InitialConditionValue::<runmat_analysis_core::Displacement> {
                initial_condition_id: "original-u".into(),
                target: runmat_analysis_core::InitialConditionTarget::WholeModel,
                frame: runmat_analysis_core::CoordinateFrame::Global,
                value: runmat_analysis_core::InitialValue::FieldArtifact(u.reference().clone()),
                applies_before_step_id: workload.step_id.as_str().into(),
            },
        ),
        runmat_analysis_core::InitialCondition::Velocity(
            runmat_analysis_core::InitialConditionValue::<runmat_analysis_core::Velocity> {
                initial_condition_id: "original-v".into(),
                target: runmat_analysis_core::InitialConditionTarget::WholeModel,
                frame: runmat_analysis_core::CoordinateFrame::Global,
                value: runmat_analysis_core::InitialValue::FieldArtifact(v.reference().clone()),
                applies_before_step_id: workload.step_id.as_str().into(),
            },
        ),
    ];
    let intent =
        analysis_prepare_structural_study_intent_op(&consumer, OperationContext::new(None, None))
            .unwrap();
    let cold = PreparedStructuralStudyIntent::from_continuation_bytes(
        &intent.continuation_bytes().unwrap(),
    )
    .unwrap();
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralTransient {
        workload,
        ..
    } = cold.workloads()
    else {
        panic!("direct consumer workload")
    };
    assert_eq!(workload.initial_fields, vec![u, v]);
    let mesh = mesh_output(order).artifact;
    let scope = runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
        &workload, &mesh, None,
    )
    .unwrap();
    let request =
        runmat_analysis_fea::execution::StructuralTransientBlockRequest::new(scope, None, 2)
            .unwrap();
    let block = runmat_analysis_fea::pipeline::transient::run_transient_block(
        runmat_analysis_fea::pipeline::transient::StructuralTransientRunBlockInput {
            workload: &workload,
            solver_mesh: &mesh,
            request: &request,
            predecessor: None,
            modal_state: None,
            observation_cursors: Vec::new(),
        },
    )
    .unwrap();
    let initial = block.initial_sample().unwrap();
    assert_eq!(
        initial.state.displacement,
        displacement.as_host_f64().unwrap()
    );
    assert_eq!(initial.state.velocity, velocity.as_host_f64().unwrap());
    storage::reset_artifact_store_for_tests();
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
}

fn mesh_output(order: ElementOrder) -> crate::meshing::MeshingExecutionOutput {
    let mut artifact = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
    artifact.resolved_request = structural_study::resolve_intent_meshing_request(Some(
        runmat_meshing_core::MeshingRequestSettings {
            element_order: order,
            ..Default::default()
        },
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
    artifact.seal_canonical_digest().unwrap();
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&artifact);
    crate::meshing::MeshingExecutionOutput { artifact, evidence }
}
