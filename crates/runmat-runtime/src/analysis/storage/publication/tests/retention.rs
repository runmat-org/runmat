use super::*;

struct PolicyRestore(AnalysisArtifactRetentionConfig);
impl PolicyRestore {
    fn set(total: usize, per_kind: usize) -> Self {
        let previous = retention_config().read().unwrap().clone();
        configure_artifact_retention(AnalysisArtifactRetentionConfig {
            max_runs: Some(total),
            max_runs_per_kind: Some(per_kind),
        })
        .unwrap();
        Self(previous)
    }
}
impl Drop for PolicyRestore {
    fn drop(&mut self) {
        configure_artifact_retention(self.0.clone()).unwrap();
    }
}

#[test]
fn memory_retention_removes_entire_root_and_all_children_preserving_retained_fields() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let _policy = PolicyRestore::set(3, 0);
    let run = run();
    let store = InMemoryAnalysisArtifactStore::new();
    let old = store
        .commit_publication(prepare(&run, &["old_a", "old_b"]))
        .unwrap();
    let new = store
        .commit_publication(prepare(&run, &["new_a", "new_b"]))
        .unwrap();
    assert!(store
        .load_publication(old.identity, 1 << 20)
        .unwrap()
        .is_none());
    for id in old.run_ids {
        assert!(store.load_run(&id).unwrap().is_none());
        assert!(store
            .load_assessment_metadata(&id, 1 << 20)
            .unwrap()
            .is_none());
        assert!(!store.publication_runs.read().unwrap().contains_key(&id));
    }
    assert!(store
        .load_publication(new.identity, 1 << 20)
        .unwrap()
        .is_some());
    assert_eq!(store.list_runs().unwrap().len(), 2);
    for id in &new.run_ids {
        assert_eq!(
            store.load_run(id).unwrap().unwrap().run.fields,
            run.run.fields
        );
    }
    assert_eq!(
        store
            .commit_publication(prepare(&run, &["new_a", "new_b"]))
            .unwrap(),
        new
    );
}

#[test]
fn memory_standalone_and_study_writers_share_whole_inventory_retention() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let _policy = PolicyRestore::set(2, 2);
    let run = run();
    let memory = InMemoryAnalysisArtifactStore::new();
    let old = memory
        .commit_publication(prepare(&run, &["original_a", "original_b"]))
        .unwrap();
    let mut standalone = run.clone();
    standalone.run_id = "single".into();
    memory.persist_run(&standalone).unwrap();
    assert!(memory
        .load_publication(old.identity, 1 << 20)
        .unwrap()
        .is_none());
    for id in old.run_ids {
        assert!(memory.load_run(&id).unwrap().is_none());
    }
    assert_eq!(memory.list_runs().unwrap().len(), 1);
    assert_eq!(
        memory.load_run("single").unwrap().unwrap().run.fields,
        run.run.fields
    );
    memory
        .commit_publication(prepare(&run, &["new_a", "new_b"]))
        .unwrap();
    assert!(memory.load_run("single").unwrap().is_none());
    assert_eq!(memory.list_runs().unwrap().len(), 2);
}

#[test]
fn standalone_writers_cannot_adopt_a_committed_study_child_in_either_backend() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let memory = InMemoryAnalysisArtifactStore::new();
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    for store in [&memory as &dyn AnalysisArtifactStore, &disk] {
        let receipt = store
            .commit_publication(prepare(&run, &["reserved_child"]))
            .unwrap();
        let mut standalone = run.clone();
        standalone.run_id = "reserved_child".into();
        assert!(store
            .persist_run(&standalone)
            .unwrap_err()
            .to_string()
            .contains("reserved"));
        assert!(store
            .load_publication(receipt.identity, 1 << 20)
            .unwrap()
            .is_some());
        assert_eq!(
            store
                .load_run("reserved_child")
                .unwrap()
                .unwrap()
                .run
                .fields,
            run.run.fields
        );
    }
}

#[test]
fn incoming_history_that_cannot_fit_rejects_without_evicting_original_publication() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let _policy = PolicyRestore::set(2, 1);
    let run = run();
    let store = InMemoryAnalysisArtifactStore::new();
    let retained = store
        .commit_publication(prepare(&run, &["retained"]))
        .unwrap();
    let too_large = prepare(&run, &["oversized_a", "oversized_b"]);
    let rejected = too_large.descriptor.identity().unwrap();
    assert!(store
        .commit_publication(too_large)
        .unwrap_err()
        .to_string()
        .contains("retention capacity"));
    assert!(store
        .load_publication(retained.identity, 1 << 20)
        .unwrap()
        .is_some());
    assert!(store.load_publication(rejected, 1 << 20).unwrap().is_none());
    assert_eq!(store.list_runs().unwrap().len(), 1);
    assert_eq!(
        store.load_run("retained").unwrap().unwrap().run.fields,
        run.run.fields
    );
}

#[test]
fn disk_retention_replaces_whole_roots_and_standalone_pruning_counts_study_children() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let _policy = PolicyRestore::set(2, 2);
    let run = run();
    let root = std::env::temp_dir().join(format!(
        "runmat-disk-whole-retention-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let disk = FilesystemAnalysisArtifactStore::new(root.clone());
    let old = disk
        .commit_publication(prepare(&run, &["old_a", "old_b"]))
        .unwrap();
    disk.load_run("old_a").unwrap().unwrap(); // Exercise warm-cache eviction too.
    let replacement = disk
        .commit_publication(prepare(&run, &["new_a", "new_b"]))
        .unwrap();
    assert!(disk
        .load_publication(old.identity, 1 << 20)
        .unwrap()
        .is_none());
    for id in &old.run_ids {
        assert!(disk.load_run(id).unwrap().is_none());
    }
    assert_eq!(disk.list_runs().unwrap().len(), 2);
    let cold = FilesystemAnalysisArtifactStore::new(root.clone());
    assert_eq!(
        cold.commit_publication(prepare(&run, &["new_a", "new_b"]))
            .unwrap(),
        replacement
    );
    for id in &replacement.run_ids {
        assert_eq!(
            cold.load_run(id).unwrap().unwrap().run.fields,
            run.run.fields
        );
    }
    let mut standalone = run.clone();
    standalone.run_id = "retained_standalone".into();
    cold.persist_run(&standalone).unwrap();
    assert!(cold
        .load_publication(replacement.identity, 1 << 20)
        .unwrap()
        .is_none());
    for id in replacement.run_ids {
        assert!(cold.load_run(&id).unwrap().is_none());
    }
    assert_eq!(cold.list_runs().unwrap().len(), 1);
    assert_eq!(
        cold.load_run(&standalone.run_id)
            .unwrap()
            .unwrap()
            .run
            .fields,
        run.run.fields
    );
    // A study must also account for a preceding standalone inventory.
    cold.commit_publication(prepare(&run, &["final_a", "final_b"]))
        .unwrap();
    assert!(cold.load_run(&standalone.run_id).unwrap().is_none());
    assert_eq!(cold.list_runs().unwrap().len(), 2);
    drop(cold);
    drop(disk);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn disk_oversized_incoming_group_rejects_before_private_files_or_existing_eviction() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let _policy = PolicyRestore::set(2, 1);
    let run = run();
    let root = std::env::temp_dir().join(format!(
        "runmat-disk-capacity-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let disk = FilesystemAnalysisArtifactStore::new(root.clone());
    let old = disk
        .commit_publication(prepare(&run, &["original"]))
        .unwrap();
    let incoming = prepare(&run, &["too_large_a", "too_large_b"]);
    let incoming_root = incoming.descriptor.identity().unwrap();
    assert!(disk
        .commit_publication(incoming)
        .unwrap_err()
        .to_string()
        .contains("retention capacity"));
    assert!(!filesystem::directory(&disk, incoming_root).exists());
    assert!(disk
        .load_publication(old.identity, 1 << 20)
        .unwrap()
        .is_some());
    assert_eq!(
        disk.load_run("original").unwrap().unwrap().run.fields,
        run.run.fields
    );
    assert_eq!(disk.list_runs().unwrap().len(), 1);
    drop(disk);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn disk_missing_retained_child_rejects_before_incoming_visibility() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let _policy = PolicyRestore::set(2, 0);
    let run = run();
    let root = std::env::temp_dir().join(format!(
        "runmat-disk-missing-child-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let disk = FilesystemAnalysisArtifactStore::new(root.clone());
    let old = disk
        .commit_publication(prepare(&run, &["original_a", "original_b"]))
        .unwrap();
    std::fs::remove_file(filesystem::directory(&disk, old.identity).join("runs/original_a.json"))
        .unwrap();
    let incoming = prepare(&run, &["incoming"]);
    let identity = incoming.descriptor.identity().unwrap();
    assert!(disk
        .commit_publication(incoming)
        .unwrap_err()
        .to_string()
        .contains("record absent"));
    assert!(!filesystem::directory(&disk, identity).exists());
    assert!(disk
        .load_publication(old.identity, 1 << 20)
        .unwrap()
        .is_some());
    assert!(disk.load_run("original_b").unwrap().is_some());
    drop(disk);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn disk_inventory_reads_observe_complete_studies_during_replacement() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let _policy = PolicyRestore::set(3, 0);
    let run = run();
    let root = std::env::temp_dir().join(format!(
        "runmat-disk-concurrent-roots-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let writer = FilesystemAnalysisArtifactStore::new(root.clone());
    writer
        .commit_publication(prepare(&run, &["initial_a", "initial_b"]))
        .unwrap();
    let start = Arc::new(std::sync::Barrier::new(2));
    let reader_start = start.clone();
    let reader_root = root.clone();
    let reader = std::thread::spawn(move || {
        let reader = FilesystemAnalysisArtifactStore::new(reader_root);
        reader_start.wait();
        for _ in 0..40 {
            let inventory = reader.list_runs().unwrap();
            assert_eq!(inventory.len(), 2);
            assert_eq!(
                inventory[0].run_id.rsplit_once('_').unwrap().0,
                inventory[1].run_id.rsplit_once('_').unwrap().0
            );
        }
    });
    start.wait();
    for ordinal in 0..4 {
        writer
            .commit_publication(prepare(
                &run,
                &[&format!("next{ordinal}_a"), &format!("next{ordinal}_b")],
            ))
            .unwrap();
    }
    reader.join().unwrap();
    drop(writer);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn failed_private_attempt_is_reclaimed_without_losing_shared_original_fields() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let aborted = prepare(&run, &["reused_child"]);
    let old_identity = aborted.descriptor.identity().unwrap();
    // Field chunks are streamed during staging; any committed chunk serves the test.
    let digest = aborted
        .chunks
        .keys()
        .copied()
        .chain(aborted.spilled.digests())
        .next()
        .unwrap();
    let original: std::sync::Arc<[u8]> = match aborted.chunks.get(&digest) {
        Some(bytes) => bytes.clone(),
        None => aborted.spilled.read(&digest).unwrap().unwrap().into(),
    };
    std::fs::create_dir_all(disk.objects_directory()).unwrap();
    std::fs::write(disk.object_path(digest), b"invalid preexisting object").unwrap();
    assert!(disk
        .commit_publication(aborted)
        .unwrap_err()
        .to_string()
        .contains("collision"));
    assert!(disk
        .load_publication(old_identity, 1 << 20)
        .unwrap()
        .is_none());
    assert!(disk.load_run("reused_child").unwrap().is_none());
    assert!(filesystem::directory(&disk, old_identity).exists());
    std::fs::write(disk.object_path(digest), original.as_ref()).unwrap();
    let new = disk
        .commit_publication(prepare(&run, &["reused_child", "new_child"]))
        .unwrap();
    assert!(!filesystem::directory(&disk, old_identity).exists());
    for id in new.run_ids {
        assert_eq!(
            disk.load_run(&id).unwrap().unwrap().run.fields,
            run.run.fields
        );
    }
}

#[test]
fn metadata_transaction_does_not_capture_unowned_producer_workspaces() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("producer-workspace");
    std::fs::create_dir(&workspace).unwrap();
    let path = workspace.join("original-worker-field");
    let file = std::fs::File::create(&path).unwrap();
    let size = runmat_filesystem::workspace::MAX_WORKSPACE_PUBLICATION_BYTES as u64 + 1;
    file.set_len(size).unwrap();
    drop(file);
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let receipt = disk
        .commit_publication(prepare(&run, &["original_child"]))
        .unwrap();
    assert_eq!(std::fs::metadata(path).unwrap().len(), size);
    assert!(disk
        .load_publication(receipt.identity, 1 << 20)
        .unwrap()
        .is_some());
    assert_eq!(
        disk.load_run("original_child").unwrap().unwrap().run.fields,
        run.run.fields
    );
}

#[test]
fn malformed_retained_inventory_rejects_before_incoming_visibility_or_eviction() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let _policy = PolicyRestore::set(2, 0);
    let run = run();
    let store = InMemoryAnalysisArtifactStore::new();
    let old = store
        .commit_publication(prepare(&run, &["original_a", "original_b"]))
        .unwrap();
    store.runs.write().unwrap().remove("original_a");
    let new = prepare(&run, &["incoming"]);
    let new_id = new.descriptor.identity().unwrap();
    assert!(store
        .commit_publication(new)
        .unwrap_err()
        .to_string()
        .contains("incomplete child inventory"));
    assert!(store.load_publication(new_id, 1 << 20).unwrap().is_none());
    assert!(store
        .load_publication(old.identity, 1 << 20)
        .unwrap()
        .is_some());
    assert!(store.load_run("original_b").unwrap().is_some());
}

#[test]
fn concurrent_inventory_reads_never_observe_partially_evicted_or_installed_studies() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let _policy = PolicyRestore::set(3, 0);
    let run = run();
    let store = Arc::new(InMemoryAnalysisArtifactStore::new());
    store
        .commit_publication(prepare(&run, &["initial_a", "initial_b"]))
        .unwrap();
    let start = Arc::new(std::sync::Barrier::new(2));
    let reader_store = store.clone();
    let reader_start = start.clone();
    let reader = std::thread::spawn(move || {
        reader_start.wait();
        for _ in 0..200 {
            let inventory = reader_store.list_runs().unwrap();
            assert_eq!(inventory.len(), 2);
            let mut prefixes = inventory
                .iter()
                .map(|run| run.run_id.rsplit_once('_').unwrap().0);
            assert_eq!(prefixes.next(), prefixes.next());
        }
    });
    start.wait();
    for ordinal in 0..8 {
        let a = format!("replacement{ordinal}_a");
        let b = format!("replacement{ordinal}_b");
        store.commit_publication(prepare(&run, &[&a, &b])).unwrap();
    }
    reader.join().unwrap();
}
