use super::*;

#[test]
fn repeatable_study_evidence_is_addressed_by_its_validated_content() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("content-addressed-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let study_fingerprint = runmat_execution::Digest::sha256(b"same-study").to_string();

    let first = persist_content_addressed_study_evidence(
        &study_fingerprint,
        "validate",
        serde_json::json!({"valid": false, "message": "first contract"}),
    )
    .unwrap();
    let repeated = persist_content_addressed_study_evidence(
        &study_fingerprint,
        "validate",
        serde_json::json!({"valid": false, "message": "first contract"}),
    )
    .unwrap();
    let revised = persist_content_addressed_study_evidence(
        &study_fingerprint,
        "validate",
        serde_json::json!({"valid": false, "message": "revised contract"}),
    )
    .unwrap();

    assert_eq!(first, repeated);
    assert_ne!(first, revised);
    assert!(first.contains("validate_sha256_"));
    assert!(revised.contains("validate_sha256_"));
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn study_run_evidence_separates_logical_run_identity_from_attempt_content() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("run-addressed-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);

    let study_fingerprint = runmat_execution::Digest::sha256(b"same-study").to_string();
    let first = persist_content_addressed_study_evidence(
        &study_fingerprint,
        &run_evidence_stage("server-run-a"),
        serde_json::json!({"run_id": "server-run-a", "attempt": "worker-a"}),
    )
    .unwrap();
    let repeated = persist_content_addressed_study_evidence(
        &study_fingerprint,
        &run_evidence_stage("server-run-a"),
        serde_json::json!({"run_id": "server-run-a", "attempt": "worker-a"}),
    )
    .unwrap();
    let retried = persist_content_addressed_study_evidence(
        &study_fingerprint,
        &run_evidence_stage("server-run-a"),
        serde_json::json!({"run_id": "server-run-a", "attempt": "worker-b"}),
    )
    .unwrap();
    let second = persist_content_addressed_study_evidence(
        &study_fingerprint,
        &run_evidence_stage("server-run-b"),
        serde_json::json!({"run_id": "server-run-b"}),
    )
    .unwrap();

    assert_eq!(first, repeated);
    assert_ne!(first, retried);
    assert_ne!(first, second);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&fs::read(&first).unwrap()).unwrap()["run_id"],
        "server-run-a"
    );
    let first_envelope: serde_json::Value =
        serde_json::from_slice(&fs::read(&first).unwrap()).unwrap();
    assert_eq!(
        first_envelope["$evidence"]["study_fingerprint"],
        study_fingerprint
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&fs::read(second).unwrap()).unwrap()["run_id"],
        "server-run-b"
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&fs::read(retried).unwrap()).unwrap()
            ["attempt"],
        "worker-b"
    );
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn study_evidence_rejects_a_valid_json_envelope_bound_to_the_wrong_stage() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("study-evidence-wrong-envelope");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let study_fingerprint = runmat_execution::Digest::sha256(b"bound-study").to_string();
    let path = persist_immutable_study_evidence(
        &study_fingerprint,
        "plan",
        serde_json::json!({"study_id": "bound-study"}),
    )
    .unwrap();
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    envelope["$evidence"]["stage"] = serde_json::json!("validate");
    fs::write(&path, serde_json::to_vec_pretty(&envelope).unwrap()).unwrap();

    assert!(persist_immutable_study_evidence(
        &study_fingerprint,
        "plan",
        serde_json::json!({"study_id": "bound-study"}),
    )
    .is_err());
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&fs::read(&path).unwrap()).unwrap()
            ["$evidence"]["stage"],
        "validate"
    );
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn authored_output_fields_are_chunked_and_selectively_readable() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("authored-output-field-page");
    let mut model = sample_solid_model();
    let step_id = model.sequence.steps[0].step_id.0.clone();
    runmat_analysis_core::attach_authored_outputs(
        &[runmat_analysis_core::AuthoredOutput {
            output_id: "published_displacement".into(),
            expression: runmat_analysis_core::AuthoredOutputExpression::AnalysisField {
                step_id,
                quantity: runmat_analysis_core::ObservationQuantity::Displacement,
                location: runmat_analysis_core::FieldLocation::Node,
            },
            policy: runmat_analysis_core::AuthoredOutputPolicy::default(),
        }],
        &mut model.sequence,
    )
    .unwrap();
    let generated = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..AnalysisRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .unwrap();
    let mut run = generated.data;
    run.run_id = "run_authored_output_page".to_string();
    let observation = run
        .run
        .observations
        .iter()
        .find(|observation| observation.observation_id.0 == "published_displacement")
        .expect("authored output was not evaluated");
    let expected = observation.field.as_host_f64().unwrap().to_vec();

    let store_root = mesh_root.join("store");
    let store = storage::FilesystemAnalysisArtifactStore::new(store_root.clone());
    storage::AnalysisArtifactStore::persist_run(&store, &run).unwrap();
    let metadata = storage::AnalysisArtifactStore::load_run_metadata(&store, &run.run_id)
        .unwrap()
        .expect("persisted run metadata");
    assert!(metadata.render_topology.is_none());
    if let Some(source_topology) = &run.render_topology {
        let index = storage::AnalysisArtifactStore::load_render_tile_index(&store, &run.run_id)
            .unwrap()
            .expect("persisted render tile index");
        assert!(!index.meshes.is_empty());
        let selection = storage::AnalysisArtifactStore::plan_render_view(&store, &run.run_id, None)
            .unwrap()
            .expect("persisted render view plan");
        assert_eq!(selection.mesh_count, index.meshes.len());
        assert!(!selection.nodes.is_empty());
        let root = index.meshes[0].lod_root.expect("render LOD root");
        let overview =
            storage::AnalysisArtifactStore::load_render_lod_node(&store, &run.run_id, 0, root)
                .unwrap()
                .expect("persisted render LOD node");
        assert!(!overview.triangles.is_empty());
        assert!(index.field_summaries.iter().any(|summary| {
            summary.field_id == "published_displacement" && summary.minimum <= summary.maximum
        }));
        let restored = storage::AnalysisArtifactStore::load_render_topology(&store, &run.run_id)
            .unwrap()
            .expect("persisted render topology");
        assert_eq!(&restored, source_topology);
    }
    assert!(metadata.run.observations.is_empty());
    assert!(metadata
        .run
        .fields
        .iter()
        .all(|field| field.as_host_f64() == Some(&[])));
    let inventory = storage::AnalysisArtifactStore::load_field_inventory(&store, &run.run_id)
        .unwrap()
        .expect("persisted field inventory");
    assert!(inventory.iter().any(|field| {
        field.field_id == "published_displacement" && field.as_host_f64() == Some(&[])
    }));
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: store_root,
    })
    .unwrap();
    #[cfg(feature = "plot-core")]
    if run.render_topology.is_some() {
        let (scene, scale, _exact) = crate::analysis::field_scene::build_scene_view(
            &run.run_id,
            Some("published_displacement"),
            "undeformed",
            1.0,
            false,
            None,
        )
        .expect("render selected field from persisted LOD tiles");
        assert_eq!(scale, 0.0);
        assert!(scene.triangle_count() > 0);
        assert!(scene.overlay.as_ref().unwrap().scalar_legend.is_some());
    }
    let mut metadata_query = AnalysisResultsQuery::metadata_only();
    metadata_query.include_fields = vec!["published_displacement".to_string()];
    let public_metadata = analysis_results_by_run_id_op(
        &run.run_id,
        metadata_query,
        OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    assert_eq!(public_metadata.field_descriptors.len(), 1);
    assert_eq!(
        public_metadata.field_descriptors[0].field_id,
        "published_displacement"
    );
    let lookup = storage::AnalysisArtifactStore::load_field_page(
        &store,
        &run.run_id,
        &AnalysisFieldSelection::new("published_displacement", None),
        AnalysisFieldRequestOptions::default(),
    )
    .unwrap();
    let storage::AnalysisFieldPageLookup::Page(page) = lookup else {
        panic!("authored output field was not selectively readable")
    };
    assert_eq!(page.values, AnalysisFieldPageValues::Real(expected.clone()));
    let reloaded = storage::AnalysisArtifactStore::load_run(&store, &run.run_id)
        .unwrap()
        .expect("persisted run");
    assert_eq!(reloaded.render_topology, run.render_topology);
    let reloaded_output = reloaded
        .run
        .observations
        .iter()
        .find(|observation| observation.observation_id.0 == "published_displacement")
        .expect("reloaded authored output");
    assert_eq!(reloaded_output.field.as_host_f64().unwrap(), expected);
    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn filesystem_run_fields_are_chunked_and_pages_read_only_intersecting_chunks() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("filesystem-chunked-field-pages");
    let _ = fs::remove_dir_all(&root);
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let mut run = generated.data;
    run.run_id = "run_chunked_field_pages".to_string();
    let field = run.run.fields.first_mut().expect("linear result field");
    let value_count = runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES
        / std::mem::size_of::<f64>()
        + 128;
    let expected_values = (0..value_count)
        .map(|index| index as f64 + 0.25)
        .collect::<Vec<_>>();
    field.shape = vec![value_count];
    field.values = AnalysisFieldValues::HostF64(expected_values.clone());
    let field_id = field.field_id.clone();
    let mut expected_metadata = analysis_results_op(
        &run,
        AnalysisResultsQuery::metadata_only(),
        OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    for descriptor in &mut expected_metadata.field_descriptors {
        if descriptor.storage == AnalysisFieldStorage::HostF64 {
            descriptor.storage_ref.kind = "chunked_field".to_string();
        }
    }

    let store = storage::FilesystemAnalysisArtifactStore::new(root.clone());
    storage::AnalysisArtifactStore::persist_run(&store, &run).unwrap();
    assert_eq!(
        storage::AnalysisArtifactStore::load_run(&store, &run.run_id)
            .unwrap()
            .unwrap(),
        run
    );

    let artifact_path = root.join("runs/run_chunked_field_pages.json");
    let artifact: serde_json::Value =
        serde_json::from_slice(&fs::read(&artifact_path).unwrap()).unwrap();
    assert_eq!(
        artifact["metadata"]["run"]["fields"][0]["values"]["host_f64"],
        serde_json::json!([])
    );
    let field_manifest = &artifact["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|stored| stored["field_id"] == field_id)
        .expect("stored field manifest")["manifest"];
    assert_eq!(field_manifest["kind"], "field");
    assert_eq!(field_manifest["chunks"].as_array().unwrap().len(), 2);

    let default_page = match storage::AnalysisArtifactStore::load_field_page(
        &store,
        &run.run_id,
        &AnalysisFieldSelection::new(&field_id, None),
        AnalysisFieldRequestOptions::default(),
    )
    .unwrap()
    {
        storage::AnalysisFieldPageLookup::Page(page) => *page,
        _ => panic!("field page must exist"),
    };
    assert_eq!(
        default_page.count,
        super::ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT
    );
    assert_eq!(
        default_page.values,
        AnalysisFieldPageValues::Real(
            expected_values[..super::ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT].to_vec()
        )
    );
    let suffix = field_id
        .rsplit('.')
        .next()
        .expect("field ID has a terminal component");
    let suffix_page = match storage::AnalysisArtifactStore::load_field_page(
        &store,
        &run.run_id,
        &AnalysisFieldSelection::new(suffix, None),
        AnalysisFieldRequestOptions {
            offset: Some(0),
            limit: Some(1),
        },
    )
    .unwrap()
    {
        storage::AnalysisFieldPageLookup::Page(page) => *page,
        _ => panic!("unambiguous field suffix must resolve"),
    };
    assert_eq!(suffix_page.field_id, field_id);

    let capped_page = match storage::AnalysisArtifactStore::load_field_page(
        &store,
        &run.run_id,
        &AnalysisFieldSelection::new(&field_id, None),
        AnalysisFieldRequestOptions {
            offset: Some(32),
            limit: Some(usize::MAX),
        },
    )
    .unwrap()
    {
        storage::AnalysisFieldPageLookup::Page(page) => *page,
        _ => panic!("field page must exist"),
    };
    assert_eq!(capped_page.count, super::ANALYSIS_FIELD_DEFAULT_PAGE_SIZE);

    let first_digest = field_manifest["chunks"][0]["digest"]
        .as_str()
        .unwrap()
        .strip_prefix("sha256:")
        .unwrap();
    fs::write(root.join("objects").join(first_digest), b"corrupt").unwrap();
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .unwrap();
    let metadata = analysis_results_by_run_id_op(
        &run.run_id,
        AnalysisResultsQuery::metadata_only(),
        OperationContext::new(None, None),
    )
    .expect("metadata access must not read field chunks")
    .data;
    assert_eq!(metadata, expected_metadata);
    assert!(metadata.fields.is_empty());
    assert!(metadata
        .field_descriptors
        .iter()
        .all(|descriptor| descriptor.storage_ref.kind == "chunked_field"));
    analysis_trends_op(
        AnalysisTrendsQuery { window_size: 8 },
        OperationContext::new(None, None),
    )
    .expect("trend metadata must not read field chunks");
    assert!(storage::AnalysisArtifactStore::load_run(&store, &run.run_id).is_err());
    let second_chunk_offset =
        runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES / std::mem::size_of::<f64>();
    let later_page = match storage::AnalysisArtifactStore::load_field_page(
        &store,
        &run.run_id,
        &AnalysisFieldSelection::new(&field_id, None),
        AnalysisFieldRequestOptions {
            offset: Some(second_chunk_offset),
            limit: Some(16),
        },
    )
    .unwrap()
    {
        storage::AnalysisFieldPageLookup::Page(page) => *page,
        _ => panic!("field page must exist"),
    };
    assert_eq!(
        later_page.values,
        AnalysisFieldPageValues::Real(
            expected_values[second_chunk_offset..second_chunk_offset + 16].to_vec()
        )
    );
    let public_page = analysis_field_page_by_run_id_op(
        &run.run_id,
        AnalysisFieldSelection::new(&field_id, None),
        AnalysisFieldRequestOptions {
            offset: Some(second_chunk_offset),
            limit: Some(16),
        },
        OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    assert_eq!(public_page, later_page);
    assert_eq!(
        analysis_field_page_by_run_id_op(
            &run.run_id,
            AnalysisFieldSelection::new("missing.field", None),
            AnalysisFieldRequestOptions::default(),
            OperationContext::new(None, None),
        )
        .unwrap_err()
        .error_code,
        "RM.FEA.RESULTS.FIELD_NOT_FOUND"
    );
    assert!(storage::AnalysisArtifactStore::load_field_page(
        &store,
        &run.run_id,
        &AnalysisFieldSelection::new(&field_id, None),
        AnalysisFieldRequestOptions {
            offset: Some(0),
            limit: Some(1),
        },
    )
    .is_err());

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn run_storage_rejects_ambiguous_duplicate_field_ids() {
    let _guard = analysis_test_guard();
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let mut run = generated.data;
    run.run_id = "run_duplicate_field_ids".to_string();
    run.run
        .fields
        .push(run.run.fields.first().expect("linear result field").clone());

    let error = storage::AnalysisArtifactStore::persist_run(
        &storage::InMemoryAnalysisArtifactStore::new(),
        &run,
    )
    .unwrap_err();
    assert!(error.contains("appears more than once"));
}

#[test]
fn conventional_field_paging_rejects_an_ambiguous_complete_suffix() {
    let _guard = analysis_test_guard();
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let mut run = generated.data;
    run.run_id = "run_ambiguous_field_suffix".to_string();
    let [first, second, ..] = run.run.fields.as_mut_slice() else {
        panic!("linear result must contain at least two fields")
    };
    first.field_id = "first.shared".to_string();
    second.field_id = "second.shared".to_string();
    let store = storage::InMemoryAnalysisArtifactStore::new();
    storage::AnalysisArtifactStore::persist_run(&store, &run).unwrap();

    let lookup = storage::AnalysisArtifactStore::load_field_page(
        &store,
        &run.run_id,
        &AnalysisFieldSelection::new("shared", None),
        AnalysisFieldRequestOptions::default(),
    )
    .unwrap();
    let storage::AnalysisFieldPageLookup::FieldSelectionAmbiguous(fields) = lookup else {
        panic!("ambiguous field suffix must fail closed")
    };
    assert_eq!(fields, ["first.shared", "second.shared"]);
}

#[test]
fn run_storage_rejects_process_local_device_field_references() {
    let _guard = analysis_test_guard();
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let transfer = AnalysisStudyStepResult {
        run: generated.clone(),
        resolved_solver_options: AnalysisSolverOptions::StructuralLinear(Default::default()),
    };
    analysis_validate_study_step_transfer(&transfer)
        .expect("host fields transfer without changing backend");
    let mut run = generated.data;
    run.run_id = "run_device_field_reference".to_string();
    let field = run.run.fields.first_mut().expect("linear result field");
    let element_count = field.shape.iter().product();
    field.values = AnalysisFieldValues::DeviceRef(runmat_analysis_core::DeviceFieldRef {
        backend: "test".to_string(),
        token: "process-local-token".to_string(),
        element_count,
    });
    let mut transferred = transfer;
    transferred.run.data = run.clone();
    let denied = analysis_validate_study_step_transfer(&transferred).unwrap_err();
    assert_eq!(denied.error_code, "RM.FEA.WORKER.PROCESS_LOCAL_FIELD");

    let error = storage::AnalysisArtifactStore::persist_run(
        &storage::InMemoryAnalysisArtifactStore::new(),
        &run,
    )
    .unwrap_err();
    assert!(error.contains("process-local reference"));
    assert!(error.contains("typed runtime value admission"));
}

#[cfg(unix)]
#[test]
fn study_evidence_rejects_symlink_targets_and_owned_directories() {
    use std::os::unix::fs::symlink;

    let _guard = analysis_test_guard();
    let root = temp_artifact_root("study-evidence-symlinks");
    let outside = temp_artifact_root("study-evidence-symlinks-outside");
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&outside);
    fs::create_dir_all(&root).unwrap();
    fs::create_dir_all(&outside).unwrap();
    let _runtime_guard = scoped_study_artifact_root(&root);
    let study_fingerprint = runmat_execution::Digest::sha256(b"symlink-study").to_string();
    let digest = study_fingerprint.trim_start_matches("sha256:");
    let study_dir = root.join(format!("sha256_{digest}"));
    fs::create_dir_all(&study_dir).unwrap();
    let outside_file = outside.join("outside.json");
    fs::write(&outside_file, b"outside").unwrap();
    symlink(&outside_file, study_dir.join("plan.json")).unwrap();
    assert!(persist_immutable_study_evidence(
        &study_fingerprint,
        "plan",
        serde_json::json!({"study_id": "symlink-study"}),
    )
    .is_err());
    assert_eq!(fs::read(&outside_file).unwrap(), b"outside");

    let linked_fingerprint = runmat_execution::Digest::sha256(b"linked-study").to_string();
    let linked_digest = linked_fingerprint.trim_start_matches("sha256:");
    symlink(&outside, root.join(format!("sha256_{linked_digest}"))).unwrap();
    assert!(persist_immutable_study_evidence(
        &linked_fingerprint,
        "plan",
        serde_json::json!({"study_id": "linked-study"}),
    )
    .is_err());

    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&outside);
}

#[test]
fn remote_completion_uses_a_new_artifact_key_for_the_numbered_schema() {
    let _guard = analysis_test_guard();
    let identity = AnalysisStudyCompletionIdentity::new("admitted-run").unwrap();
    let study_digest = runmat_execution::Digest::sha256(b"authored-study");
    let step_id = &sample_model().sequence.steps[0].step_id;
    let current = identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        study_digest,
        step_id,
    );
    assert_eq!(
        current,
        identity.local_run_id(
            runmat_analysis_core::PhysicsFamily::Structural,
            study_digest,
            step_id
        )
    );

    let previous_numbered = runmat_execution::Digest::sha256(
        [
            b"runmat-remote-structural-study-completion/v2\0".as_slice(),
            b"analysis_run_artifact/2026-09-001".as_slice(),
            b"\0".as_slice(),
            b"admitted-run".as_slice(),
            study_digest.bytes(),
            step_id.as_str().as_bytes(),
        ]
        .concat(),
    );
    let previous_numbered_run_id = format!(
        "run_remote_{}",
        previous_numbered.to_string().trim_start_matches("sha256:")
    );
    assert_ne!(current, previous_numbered_run_id);

    let retired = runmat_execution::Digest::sha256(
        [
            b"runmat-remote-structural-study-completion/v1\0".as_slice(),
            b"admitted-run".as_slice(),
            study_digest.bytes(),
            step_id.as_str().as_bytes(),
        ]
        .concat(),
    );
    let retired_run_id = format!(
        "run_remote_{}",
        retired.to_string().trim_start_matches("sha256:")
    );
    assert_ne!(current, retired_run_id);

    let root = temp_artifact_root("remote-completion-schema-revision");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("runs")).unwrap();
    fs::write(
        root.join(format!("runs/{retired_run_id}.json")),
        br#"{"schema_version":"analysis_run_artifact/2026-09-validated-field-shells"}"#,
    )
    .unwrap();
    fs::write(
        root.join(format!("runs/{previous_numbered_run_id}.json")),
        br#"{"schema_version":"analysis_run_artifact/2026-09-001"}"#,
    )
    .unwrap();
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let mut run = generated.data;
    run.run_id = current;
    let store = storage::FilesystemAnalysisArtifactStore::new(root.clone());
    storage::AnalysisArtifactStore::persist_run(&store, &run).unwrap();
    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join(format!("runs/{}.json", run.run_id))).unwrap())
            .unwrap();
    assert_eq!(
        persisted["schema_version"],
        "analysis_run_artifact/2026-09-002"
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn filesystem_run_publication_is_create_if_absent_across_store_instances() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("filesystem-run-publication-cas");
    let _ = fs::remove_dir_all(&root);
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let mut first = generated.data;
    first.run_id = "run_remote_filesystem_cas".into();
    let mut conflicting = first.clone();
    conflicting.provenance.solver_host_sync_count += 1;

    storage::AnalysisArtifactStore::persist_run(
        &storage::InMemoryAnalysisArtifactStore::new(),
        &first,
    )
    .expect("the first CAS candidate must be semantically valid");
    storage::AnalysisArtifactStore::persist_run(
        &storage::InMemoryAnalysisArtifactStore::new(),
        &conflicting,
    )
    .expect("the conflicting CAS candidate must also be semantically valid");

    let identical_root = root.join("identical");
    let identical_barrier = Arc::new(std::sync::Barrier::new(2));
    let identical_handles = [first.clone(), first.clone()].map(|run| {
        let root = identical_root.clone();
        let barrier = Arc::clone(&identical_barrier);
        std::thread::spawn(move || {
            let store = storage::FilesystemAnalysisArtifactStore::new(root);
            barrier.wait();
            storage::AnalysisArtifactStore::persist_run(&store, &run)
        })
    });
    for handle in identical_handles {
        handle
            .join()
            .unwrap()
            .expect("concurrent identical publications must converge");
    }

    let conflicting_root = root.join("conflicting");
    let barrier = Arc::new(std::sync::Barrier::new(2));

    let handles = [first.clone(), conflicting.clone()].map(|run| {
        let root = conflicting_root.clone();
        let barrier = Arc::clone(&barrier);
        std::thread::spawn(move || {
            let store = storage::FilesystemAnalysisArtifactStore::new(root);
            barrier.wait();
            storage::AnalysisArtifactStore::persist_run(&store, &run)
        })
    });
    let outcomes = handles.map(|handle| handle.join().unwrap());
    assert_eq!(outcomes.iter().filter(|outcome| outcome.is_ok()).count(), 1);
    assert_eq!(
        outcomes.iter().filter(|outcome| outcome.is_err()).count(),
        1
    );

    let store = storage::FilesystemAnalysisArtifactStore::new(conflicting_root);
    let persisted = storage::AnalysisArtifactStore::load_run(&store, &first.run_id)
        .unwrap()
        .unwrap();
    let winner = if outcomes[0].is_ok() {
        &first
    } else {
        &conflicting
    };
    assert_eq!(&persisted, winner);
    storage::AnalysisArtifactStore::persist_run(&store, winner)
        .expect("an identical retry must be accepted");
    let loser = if outcomes[0].is_err() {
        &first
    } else {
        &conflicting
    };
    assert!(storage::AnalysisArtifactStore::persist_run(&store, loser).is_err());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn filesystem_run_store_rejects_non_token_ids_before_path_resolution() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("filesystem-run-id-token");
    let _ = fs::remove_dir_all(&root);
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let store = storage::FilesystemAnalysisArtifactStore::new(root.clone());

    for unsafe_id in [
        "../escape",
        "/absolute",
        "nested/path",
        "nested\\path",
        ".",
        "run id",
        "rún",
    ] {
        assert!(storage::AnalysisArtifactStore::load_run(&store, unsafe_id).is_err());
        let mut run = generated.data.clone();
        run.run_id = unsafe_id.to_string();
        assert!(storage::AnalysisArtifactStore::persist_run(&store, &run).is_err());
    }
    let mut overlong = generated.data;
    overlong.run_id = "r".repeat(129);
    assert!(storage::AnalysisArtifactStore::persist_run(&store, &overlong).is_err());
    assert!(!root.join("runs").exists());
    assert!(!root.join("escape.json").exists());
    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn filesystem_run_publication_never_steals_a_corrupt_final_target() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("filesystem-run-publication-recovery");
    let _ = fs::remove_dir_all(&root);
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let mut run = generated.data;
    run.run_id = "run_interrupted_publication".into();
    let runs = root.join("runs");
    fs::create_dir_all(&runs).unwrap();
    fs::write(runs.join("run_interrupted_publication.json"), b"{\"run\":").unwrap();

    let store = storage::FilesystemAnalysisArtifactStore::new(root.clone());
    assert!(storage::AnalysisArtifactStore::persist_run(&store, &run).is_err());
    assert!(storage::AnalysisArtifactStore::load_run(&store, &run.run_id).is_err());
    assert_eq!(
        fs::read(runs.join("run_interrupted_publication.json")).unwrap(),
        b"{\"run\":"
    );
    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn filesystem_run_store_rejects_valid_json_with_wrong_envelope_identity() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("filesystem-run-wrong-envelope");
    let _ = fs::remove_dir_all(&root);
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let mut run = generated.data;
    run.run_id = "run_expected_envelope".into();
    let store = storage::FilesystemAnalysisArtifactStore::new(root.clone());
    storage::AnalysisArtifactStore::persist_run(&store, &run).unwrap();
    let path = root.join("runs/run_expected_envelope.json");

    let original = fs::read(&path).unwrap();
    let mut wrong_id: serde_json::Value = serde_json::from_slice(&original).unwrap();
    wrong_id["run"]["run_id"] = serde_json::json!("run_other_envelope");
    fs::write(&path, serde_json::to_vec_pretty(&wrong_id).unwrap()).unwrap();
    assert!(storage::AnalysisArtifactStore::load_run(&store, &run.run_id).is_err());
    assert!(storage::AnalysisArtifactStore::persist_run(&store, &run).is_err());

    let mut wrong_version: serde_json::Value = serde_json::from_slice(&original).unwrap();
    wrong_version["op_version"] = serde_json::json!("fea.run_linear_static/wrong");
    fs::write(&path, serde_json::to_vec_pretty(&wrong_version).unwrap()).unwrap();
    assert!(storage::AnalysisArtifactStore::load_run(&store, &run.run_id).is_err());
    assert!(storage::AnalysisArtifactStore::persist_run(&store, &run).is_err());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn memory_provider_run_publication_is_linearizable() {
    let _guard = analysis_test_guard();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let _provider_guard =
        runmat_filesystem::replace_provider(Arc::new(runmat_filesystem::MemoryFsProvider::new()));
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let mut first = generated.data;
    first.run_id = "run_memory_provider_cas".into();
    let mut second = first.clone();
    second.provenance.solver_host_sync_count += 1;
    storage::AnalysisArtifactStore::persist_run(
        &storage::InMemoryAnalysisArtifactStore::new(),
        &second,
    )
    .unwrap();

    let root = PathBuf::from("/analysis-memory-cas");
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem { root })
        .unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let handles = [first.clone(), second.clone()].map(|run| {
        let barrier = Arc::clone(&barrier);
        std::thread::spawn(move || {
            // Both operations capture the original provider revision before
            // either prepares its private result. Only conditional commit can
            // make the winning result visible on this asynchronous provider.
            futures::executor::block_on(storage::with_artifact_access(true, |store| {
                barrier.wait();
                store.persist_run(&run)
            }))
            .map_err(|error| error.to_string())
            .and_then(|result| result)
        })
    });
    let outcomes = handles.map(|handle| handle.join().unwrap());
    assert_eq!(
        outcomes.iter().filter(|outcome| outcome.is_ok()).count(),
        1,
        "{outcomes:?}"
    );
    assert_eq!(
        outcomes.iter().filter(|outcome| outcome.is_err()).count(),
        1
    );

    let persisted = futures::executor::block_on(storage::with_artifact_access(false, |store| {
        store.load_run(&first.run_id)
    }))
    .unwrap()
    .unwrap()
    .unwrap();
    assert_eq!(persisted, if outcomes[0].is_ok() { first } else { second });
}

#[cfg(unix)]
#[test]
fn filesystem_run_store_rejects_symlink_targets_and_owned_directories() {
    use std::os::unix::fs::symlink;

    let _guard = analysis_test_guard();
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let mut run = generated.data;
    run.run_id = "run_symlink_rejected".into();

    let target_root = temp_artifact_root("filesystem-run-target-symlink");
    let _ = fs::remove_dir_all(&target_root);
    fs::create_dir_all(target_root.join("runs")).unwrap();
    let outside_file = target_root.join("outside.json");
    fs::write(&outside_file, b"outside").unwrap();
    symlink(
        &outside_file,
        target_root.join("runs/run_symlink_rejected.json"),
    )
    .unwrap();
    let store = storage::FilesystemAnalysisArtifactStore::new(target_root.clone());
    assert!(storage::AnalysisArtifactStore::persist_run(&store, &run).is_err());
    assert!(storage::AnalysisArtifactStore::load_run(&store, &run.run_id).is_err());
    assert_eq!(fs::read(&outside_file).unwrap(), b"outside");

    let runs_root = temp_artifact_root("filesystem-runs-symlink");
    let _ = fs::remove_dir_all(&runs_root);
    let outside_runs = temp_artifact_root("filesystem-runs-symlink-outside");
    let _ = fs::remove_dir_all(&outside_runs);
    fs::create_dir_all(&runs_root).unwrap();
    fs::create_dir_all(&outside_runs).unwrap();
    symlink(&outside_runs, runs_root.join("runs")).unwrap();
    let store = storage::FilesystemAnalysisArtifactStore::new(runs_root.clone());
    assert!(storage::AnalysisArtifactStore::persist_run(&store, &run).is_err());

    let root_link_parent = temp_artifact_root("filesystem-root-symlink");
    let _ = fs::remove_dir_all(&root_link_parent);
    fs::create_dir_all(&root_link_parent).unwrap();
    let root_link = root_link_parent.join("artifacts-link");
    symlink(&outside_runs, &root_link).unwrap();
    let store = storage::FilesystemAnalysisArtifactStore::new(root_link);
    assert!(storage::AnalysisArtifactStore::persist_run(&store, &run).is_err());

    let _ = fs::remove_dir_all(&target_root);
    let _ = fs::remove_dir_all(&runs_root);
    let _ = fs::remove_dir_all(&outside_runs);
    let _ = fs::remove_dir_all(&root_link_parent);
}
