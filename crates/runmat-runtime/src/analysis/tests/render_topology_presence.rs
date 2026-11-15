//! A published step records whether its run saved render topology, from the topology handed to
//! the store, and a field scene of a run without it fails with a stable code.
use super::*;

#[test]
fn a_published_study_step_with_render_topology_records_persisted() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("render-topology-persisted");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let mut spec = sample_linear_static_study_spec();
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("render-topology-persisted-mesh");
    spec.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());

    let published = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect("study run should succeed")
        .data;

    let step = &published.steps[0];
    assert_eq!(
        step.render_topology,
        Some(RenderTopologyPresence::Persisted)
    );
    // The field scene reads the same fact: the run's saved render tiles.
    assert!(storage::load_render_tile_index(&step.run_id)
        .unwrap()
        .is_some());
    let mut wire = serde_json::to_value(step).unwrap();
    assert_eq!(wire["render_topology"], "persisted");
    // A record published before the field reads as unknown, never as persisted.
    wire.as_object_mut().unwrap().remove("render_topology");
    let older: AnalysisStudyStepRunData = serde_json::from_value(wire).unwrap();
    assert_eq!(older.render_topology, None);
    let document =
        analysis_document_study_result("study.fea".into(), published.clone(), Vec::new()).unwrap();
    assert!(!document.field_bindings.is_empty());
    for binding in &document.field_bindings {
        assert_eq!(
            binding.render_topology,
            Some(RenderTopologyPresence::Persisted)
        );
        let mut wire = serde_json::to_value(binding).unwrap();
        assert_eq!(wire["renderTopology"], "persisted");
        wire.as_object_mut().unwrap().remove("renderTopology");
        let older: AnalysisFieldBinding = serde_json::from_value(wire).unwrap();
        assert_eq!(older.render_topology, None);
    }
    #[cfg(feature = "plot-core")]
    crate::analysis::field_scene::build_scene_view(
        &step.run_id,
        None,
        "undeformed",
        1.0,
        false,
        None,
        &Default::default(),
    )
    .expect("a run with persisted render topology draws a field scene");
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn a_run_saved_without_render_topology_records_absent_and_refuses_a_field_scene_by_code() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("render-topology-absent-mesh");
    let model = sample_solid_model();
    let options = AnalysisRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
        preconditioner_mode: PreconditionerMode::Auto,
        quality_policy: QualityPolicy::Balanced,
        solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
    };
    let mut run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        options,
        OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    // A run whose solver mesh is not shipped publishes with no render topology.
    run.render_topology = None;
    let store: Arc<dyn storage::AnalysisArtifactStore> =
        Arc::new(storage::InMemoryAnalysisArtifactStore::new());
    store.persist_run(&run).unwrap();
    storage::set_artifact_store_for_tests(store);

    // Publication states what it hands the store, and the store saved no render tiles.
    assert_eq!(
        RenderTopologyPresence::of(run.render_topology.as_ref()),
        RenderTopologyPresence::Absent
    );
    assert!(storage::load_render_tile_index(&run.run_id)
        .unwrap()
        .is_none());
    assert_eq!(
        serde_json::to_value(RenderTopologyPresence::Absent).unwrap(),
        "absent"
    );

    #[cfg(feature = "plot-core")]
    {
        let error = crate::analysis::field_scene::build_scene_view(
            &run.run_id,
            None,
            "undeformed",
            1.0,
            false,
            None,
            &Default::default(),
        )
        .expect_err("a run without render topology has no field scene");
        assert_eq!(
            error,
            crate::analysis::field_scene::FieldSceneError::RenderTopologyMissing {
                run_id: run.run_id.clone()
            }
        );
        assert_eq!(error.code(), Some("RM.FEA.RENDER_TOPOLOGY_MISSING"));
        let diagnostic = error.diagnostic().expect("coded diagnostic");
        assert_eq!(diagnostic["identifier"], "RM.FEA.RENDER_TOPOLOGY_MISSING");
        assert_eq!(diagnostic["runId"], run.run_id.as_str());
    }
    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&mesh_root);
}
