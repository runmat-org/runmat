use super::*;
use crate::analysis::{analysis_run_linear_static_with_options, tests, AnalysisRunOptions};
use crate::operations::OperationContext;
use runmat_analysis_fea::ComputeBackend;

fn run() -> AnalysisRunResult {
    analysis_run_linear_static_with_options(
        &tests::sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap()
    .data
}
fn prepare(run: &AnalysisRunResult, ids: &[&str]) -> PreparedAnalysisPublication {
    let definition = Digest::sha256(b"atomic-storage-test-definition");
    let builder = AnalysisPublicationBuilder::new(definition, 128 * 1024 * 1024).unwrap();
    for id in ids {
        let mut child = run.clone();
        child.run_id = (*id).into();
        builder.stage_run(&child).unwrap();
    }
    builder.finish(serde_json::json!({
        "study_definition_digest": definition,
        "public_run": { "steps": ids.iter().map(|id| serde_json::json!({"run_id": id})).collect::<Vec<_>>() }
    })).unwrap()
}

#[test]
fn closed_publication_commits_original_fields_and_cold_root_in_both_stores() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = std::env::temp_dir().join(format!(
        "runmat-atomic-cold-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let disk = FilesystemAnalysisArtifactStore::new(root.clone());
    let memory = InMemoryAnalysisArtifactStore::new();
    let ids = ["atomic_first", "atomic_second"];
    for store in [&memory as &dyn AnalysisArtifactStore, &disk] {
        let prepared = prepare(&run, &ids);
        let identity = prepared.descriptor.identity().unwrap();
        for id in ids {
            assert!(store.load_run(id).unwrap().is_none());
        }
        assert!(store.load_publication(identity, 1 << 20).unwrap().is_none());
        let receipt = store.commit_publication(prepared).unwrap();
        assert_eq!(receipt.run_ids, ids);
        assert!(store.load_publication(identity, 1).is_err());
        for id in ids {
            let loaded = store.load_run(id).unwrap().unwrap();
            assert_eq!(loaded.run.fields, run.run.fields);
            assert_eq!(loaded.provenance, run.provenance);
            assert_eq!(loaded.observation_scalars, run.observation_scalars);
            assert!(store
                .load_assessment_metadata(id, 1 << 20)
                .unwrap()
                .is_some());
            assert!(store.load_assessment_metadata(id, 1).is_err());
        }
        assert_eq!(
            store.commit_publication(prepare(&run, &ids)).unwrap(),
            receipt
        );
    }
    let cold = FilesystemAnalysisArtifactStore::new(root.clone());
    let identity = prepare(&run, &ids).descriptor.identity().unwrap();
    assert_eq!(cold.list_runs().unwrap().len(), 2);
    assert!(cold.load_publication(identity, 1 << 20).unwrap().is_some());
    let manifest_path = filesystem::directory(&cold, identity).join("manifest.json");
    assert!(cold
        .load_publication_manifest_path(&manifest_path, 1 << 20)
        .is_ok());
    assert!(matches!(
        cold.load_publication_manifest_path(&manifest_path, 1),
        Err(AssessmentStoreError::ResourceLimit)
    ));
    assert!(FilesystemAnalysisArtifactStore::new(root.join("other"))
        .load_publication_manifest_path(&manifest_path, 1 << 20)
        .is_err());
    assert_eq!(
        cold.load_run(ids[0]).unwrap().unwrap().run.fields,
        run.run.fields
    );
    // A cached child must still depend on the root visibility marker.
    runmat_filesystem::remove_file(filesystem::directory(&cold, identity).join("commit.json"))
        .unwrap();
    assert!(cold.load_run(ids[0]).unwrap().is_none());
    assert!(cold
        .load_assessment_metadata(ids[0], 1 << 20)
        .unwrap()
        .is_none());
    assert!(cold.list_runs().unwrap().is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn late_filesystem_failure_keeps_every_child_and_root_hidden_and_retryable() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = std::env::temp_dir().join(format!(
        "runmat-atomic-failure-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let store = FilesystemAnalysisArtifactStore::new(root.clone());
    let ids = ["atomic_failure_first", "atomic_failure_second"];
    let prepared = prepare(&run, &ids);
    let identity = prepared.descriptor.identity().unwrap();
    let private = filesystem::directory(&store, identity);
    std::fs::create_dir_all(&private).unwrap();
    // Records are written before the manifest; force that late write to fail.
    std::fs::create_dir(private.join("manifest.json")).unwrap();
    assert!(store.commit_publication(prepared).is_err());
    assert!(private
        .join("runs")
        .join(format!("{}.json", ids[0]))
        .exists());
    for id in ids {
        assert!(store.load_run(id).unwrap().is_none());
        assert!(store
            .load_assessment_metadata(id, 1 << 20)
            .unwrap()
            .is_none());
        assert!(store.load_field_inventory(id).unwrap().is_none());
        assert!(store.load_render_topology(id).unwrap().is_none());
    }
    assert!(store.list_runs().unwrap().is_empty());
    assert!(store.load_publication(identity, 1 << 20).unwrap().is_none());
    let mut stolen = run.clone();
    stolen.run_id = ids[0].into();
    assert!(store.persist_run(&stolen).is_err());
    std::fs::remove_dir(private.join("manifest.json")).unwrap();
    store.commit_publication(prepare(&run, &ids)).unwrap();
    assert_eq!(store.list_runs().unwrap().len(), 2);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn inventory_tampering_and_existing_identity_reject_before_memory_visibility() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let memory = InMemoryAnalysisArtifactStore::new();
    let mut bad = prepare(&run, &["atomic_tampered"]);
    bad.manifest["public_run"]["steps"][0]["run_id"] = serde_json::json!("foreign");
    assert!(memory.commit_publication(bad).is_err());
    assert!(memory.list_runs().unwrap().is_empty());
    let mut existing = run.clone();
    existing.run_id = "atomic_collision".into();
    memory.persist_run(&existing).unwrap();
    assert!(memory
        .commit_publication(prepare(&run, &["atomic_collision", "atomic_unpublished"]))
        .is_err());
    assert!(memory.load_run("atomic_unpublished").unwrap().is_none());
    let builder = AnalysisPublicationBuilder::new(Digest::sha256(b"tiny"), 1).unwrap();
    assert!(builder.stage_run(&run).is_err());
    assert!(builder.store.list_runs().unwrap().is_empty());
}

#[test]
fn cold_record_substitution_and_actual_descriptor_read_budget_are_rejected() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = std::env::temp_dir().join(format!(
        "runmat-atomic-tamper-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let store = FilesystemAnalysisArtifactStore::new(root.clone());
    let receipt = store
        .commit_publication(prepare(&run, &["atomic_substitution"]))
        .unwrap();
    let private = filesystem::directory(&store, receipt.identity);
    let marker = private.join("commit.json");
    let descriptor: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
    let mut padded = serde_json::to_vec(&descriptor).unwrap();
    padded.extend(std::iter::repeat_n(b' ', 4096));
    std::fs::write(&marker, padded).unwrap();
    assert!(store.load_publication(receipt.identity, 4096).is_err());
    assert!(store
        .load_assessment_metadata("atomic_substitution", 4096)
        .is_err());
    let path = private.join("runs/atomic_substitution.json");
    let mut record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    record["metadata"]["provenance"]["solver_host_sync_count"] = serde_json::json!(9999);
    std::fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
    assert!(FilesystemAnalysisArtifactStore::new(root.clone())
        .load_run("atomic_substitution")
        .is_err());
    std::fs::remove_file(&path).unwrap();
    let cold = FilesystemAnalysisArtifactStore::new(root.clone());
    assert!(cold.load_run("atomic_substitution").is_err());
    assert!(cold
        .load_assessment_metadata("atomic_substitution", 1 << 20)
        .is_err());
    assert!(cold.list_runs().is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn competing_publications_cannot_split_a_shared_run_identity() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = std::env::temp_dir().join(format!(
        "runmat-atomic-race-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let disk = FilesystemAnalysisArtifactStore::new(root.clone());
    let memory = InMemoryAnalysisArtifactStore::new();
    for store in [&memory as &dyn AnalysisArtifactStore, &disk] {
        let first = prepare(&run, &["atomic_shared", "atomic_left"]);
        let second = prepare(&run, &["atomic_shared", "atomic_right"]);
        let barrier = std::sync::Barrier::new(2);
        let (left, right) = std::thread::scope(|scope| {
            let left = scope.spawn(|| {
                barrier.wait();
                store.commit_publication(first)
            });
            let right = scope.spawn(|| {
                barrier.wait();
                store.commit_publication(second)
            });
            (left.join().unwrap(), right.join().unwrap())
        });
        assert_ne!(left.is_ok(), right.is_ok());
        let receipt = left.or(right).unwrap();
        assert_eq!(store.list_runs().unwrap().len(), 2);
        for id in ["atomic_shared", "atomic_left", "atomic_right"] {
            assert_eq!(
                store.load_run(id).unwrap().is_some(),
                receipt.run_ids.iter().any(|owned| owned == id)
            );
        }
        assert!(store
            .load_publication(receipt.identity, 1 << 20)
            .unwrap()
            .is_some());
    }
    std::fs::remove_dir_all(root).unwrap();
}

mod retention;

mod resources;
mod run_admission;

mod access;

mod reclamation;

mod evidence;
