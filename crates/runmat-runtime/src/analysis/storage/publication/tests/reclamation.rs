use super::*;

mod resources;

#[test]
fn later_study_write_reclaims_orphans_without_losing_shared_original_fields() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let memory = InMemoryAnalysisArtifactStore::new();
    let mut standalone = run.clone();
    standalone.run_id = "standalone".into();
    for store in [&memory as &dyn AnalysisArtifactStore, &disk] {
        store.persist_run(&standalone).unwrap();
    }
    let orphan = Digest::sha256(b"unreachable original attempt");
    memory.chunks.write().unwrap().insert(
        orphan,
        Arc::from(b"unreachable original attempt".as_slice()),
    );
    std::fs::create_dir_all(disk.objects_directory()).unwrap();
    std::fs::write(disk.object_path(orphan), b"unreachable original attempt").unwrap();
    for store in [&memory as &dyn AnalysisArtifactStore, &disk] {
        let first = store.commit_publication(prepare(&run, &["first"])).unwrap();
        store
            .commit_publication(prepare(&run, &["second"]))
            .unwrap();
        assert!(store
            .load_publication(first.identity, 1 << 20)
            .unwrap()
            .is_some());
        for id in ["first", "second", "standalone"] {
            assert_eq!(
                store.load_run(id).unwrap().unwrap().run.fields,
                run.run.fields
            );
        }
    }
    assert!(!memory.chunks.read().unwrap().contains_key(&orphan));
    assert!(!disk.object_path(orphan).exists());
}

#[test]
fn unknown_legacy_inventory_defers_collection_without_blocking_supported_publication() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    disk.commit_publication(prepare(&run, &["original"]))
        .unwrap();
    let orphan = Digest::sha256(b"orphan");
    std::fs::write(disk.object_path(orphan), b"orphan").unwrap();
    std::fs::write(disk.root.join("runs/unknown.json"), b"{}").unwrap();
    disk.commit_publication(prepare(&run, &["incoming"]))
        .unwrap();
    assert!(disk.object_path(orphan).exists());
    assert_eq!(
        disk.load_run("incoming").unwrap().unwrap().run.fields,
        run.run.fields
    );
    assert_eq!(
        disk.load_run("original").unwrap().unwrap().run.fields,
        run.run.fields
    );
}

// Boundary: an observation attachment written by an older CLI (another schema version, fields
// this reader does not know) is a miss for lookups and defers reclamation; it does not fail the
// next publication with a decode error.
#[test]
fn legacy_observation_attachment_is_a_miss_and_defers_collection() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    disk.commit_publication(prepare(&run, &["original"]))
        .unwrap();
    let publication = std::fs::read_dir(disk.root.join("publications"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.is_dir())
        .unwrap();
    std::fs::write(
        publication.join("observations/original.json"),
        br#"{"schema_version":1,"run_id":"original","publication":{"samples":[{"artifact":"a"}]}}"#,
    )
    .unwrap();
    assert!(disk
        .load_observation_publication("original")
        .unwrap()
        .is_none());
    let orphan = Digest::sha256(b"orphan");
    std::fs::write(disk.object_path(orphan), b"orphan").unwrap();
    disk.commit_publication(prepare(&run, &["incoming"]))
        .unwrap();
    assert!(disk.object_path(orphan).exists());
    assert_eq!(
        disk.load_run("incoming").unwrap().unwrap().run.fields,
        run.run.fields
    );
}

#[test]
fn unowned_object_name_rejects_before_reclaiming_valid_orphan() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let root = tempfile::tempdir().unwrap();
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    std::fs::create_dir(disk.objects_directory()).unwrap();
    let orphan = Digest::sha256(b"orphan");
    std::fs::write(disk.object_path(orphan), b"orphan").unwrap();
    std::fs::write(disk.objects_directory().join("unowned"), b"private data").unwrap();
    assert!(disk
        .commit_publication(prepare(&run(), &["incoming"]))
        .is_err());
    assert!(disk.object_path(orphan).exists());
    assert!(disk.objects_directory().join("unowned").exists());
}

#[test]
fn missing_retained_object_blocks_collection_before_orphan_deletion() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let original = prepare(&run, &["retained"]);
    let required = original
        .chunks
        .keys()
        .copied()
        .chain(original.spilled.digests())
        .next()
        .unwrap();
    disk.commit_publication(original).unwrap();
    std::fs::remove_file(disk.object_path(required)).unwrap();
    let orphan = Digest::sha256(b"orphan");
    std::fs::write(disk.object_path(orphan), b"orphan").unwrap();
    let scope = disk.access(true).unwrap();
    let error = filesystem::reclamation::sweep(&scope, std::iter::empty()).unwrap_err();
    assert!(
        matches!(error, AnalysisPublicationError::Rejected { ref message } if message.contains("absent objects"))
    );
    assert!(disk.object_path(orphan).exists());
}

#[test]
fn standalone_disk_writer_reclaims_orphans_and_preserves_original_study_children() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let study = disk
        .commit_publication(prepare(&run, &["child_a", "child_b"]))
        .unwrap();
    let orphan = Digest::sha256(b"aborted standalone content");
    std::fs::write(disk.object_path(orphan), b"aborted standalone content").unwrap();
    let mut standalone = run.clone();
    standalone.run_id = "standalone".into();
    disk.persist_run(&standalone).unwrap();
    assert!(!disk.object_path(orphan).exists());
    assert!(disk
        .load_publication(study.identity, 1 << 20)
        .unwrap()
        .is_some());
    let cold = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    for id in ["child_a", "child_b", "standalone"] {
        assert_eq!(
            cold.load_run(id).unwrap().unwrap().run.fields,
            run.run.fields
        );
    }
}

#[test]
fn standalone_maintenance_rejects_incomplete_committed_inventory_before_reclaiming() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    disk.commit_publication(prepare(&run, &["child_a", "child_b"]))
        .unwrap();
    std::fs::remove_file(disk.root.join("publication-bindings/child_a.json")).unwrap();
    let orphan = Digest::sha256(b"orphan");
    std::fs::write(disk.object_path(orphan), b"orphan").unwrap();
    let mut standalone = run.clone();
    standalone.run_id = "incoming".into();
    assert!(disk.persist_run(&standalone).is_err());
    assert!(disk.object_path(orphan).exists());
    assert!(!disk.run_path("incoming").unwrap().exists());
    assert_eq!(
        disk.load_run("child_b").unwrap().unwrap().run.fields,
        run.run.fields
    );
}
