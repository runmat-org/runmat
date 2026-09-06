use super::*;

#[test]
fn study_run_evidence_is_immutable_per_run_identity() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("run-addressed-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);

    let study_fingerprint = runmat_execution::Digest::sha256(b"same-study").to_string();
    let first = persist_immutable_study_evidence(
        &study_fingerprint,
        &run_evidence_stage("server-run-a"),
        serde_json::json!({"run_id": "server-run-a"}),
    )
    .unwrap();
    let second = persist_immutable_study_evidence(
        &study_fingerprint,
        &run_evidence_stage("server-run-b"),
        serde_json::json!({"run_id": "server-run-b"}),
    )
    .unwrap();

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
    assert!(persist_immutable_study_evidence(
        &study_fingerprint,
        &run_evidence_stage("server-run-a"),
        serde_json::json!({"run_id": "different-content"}),
    )
    .is_err());
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
fn filesystem_run_publication_is_create_if_absent_across_store_instances() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("filesystem-run-publication-cas");
    let _ = fs::remove_dir_all(&root);
    let generated = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        FormulationRequirement::Experimental,
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
        FormulationRequirement::Experimental,
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
        FormulationRequirement::Experimental,
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
        FormulationRequirement::Experimental,
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
        FormulationRequirement::Experimental,
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

    let barrier = Arc::new(std::sync::Barrier::new(2));
    let handles = [first.clone(), second.clone()].map(|run| {
        let barrier = Arc::clone(&barrier);
        std::thread::spawn(move || {
            let store = storage::FilesystemAnalysisArtifactStore::new(PathBuf::from(
                "/analysis-memory-cas",
            ));
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

    let store =
        storage::FilesystemAnalysisArtifactStore::new(PathBuf::from("/analysis-memory-cas"));
    let persisted = storage::AnalysisArtifactStore::load_run(&store, &first.run_id)
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
        FormulationRequirement::Experimental,
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
