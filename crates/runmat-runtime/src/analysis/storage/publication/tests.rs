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
    // Block the second actual record write, not the second inserted key:
    // publication records are a HashMap with no insertion-order contract.
    let mut write_order = prepared.runs.keys();
    let first_child_id = write_order.next().unwrap().clone();
    let blocked_child_id = write_order.next().unwrap().clone();
    let identity = prepared.descriptor.identity().unwrap();
    let private = filesystem::directory(&store, identity);
    std::fs::create_dir_all(&private).unwrap();
    // Immutable content and the manifest commit as one group before child
    // records. Block the second child after that barrier and the first record,
    // keeping the final visibility marker absent rather than corrupting it.
    std::fs::create_dir(private.join("runs")).unwrap();
    let blocked_child = private
        .join("runs")
        .join(format!("{blocked_child_id}.json"));
    std::fs::create_dir(&blocked_child).unwrap();
    assert!(store.commit_publication(prepared).is_err());
    assert!(private
        .join("runs")
        .join(format!("{first_child_id}.json"))
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
    std::fs::remove_dir(blocked_child).unwrap();
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

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: a staged run record's identity is digested from its typed fields without building
//   a JSON tree, and equals the digest of that tree with `created_at` removed, for a linear
//   static run. Frequency records use the same view and are not exercised here.
#[test]
fn record_digest_matches_the_json_tree_digest_without_building_the_tree() {
    let _guard = tests::analysis_test_guard();
    let run = run();
    let prepared = prepare(&run, &["digest_view"]);
    assert!(!prepared.runs.is_empty());
    for record in prepared.runs.values() {
        assert_eq!(
            super::record_digest(record).unwrap(),
            super::tree_record_digest(record).unwrap()
        );
    }
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: publication staging streams field chunks to a private spill area, so a budget
//   smaller than the run's total field payload (but covering its largest field in flight)
//   stages, commits to a filesystem store and reads every field back unchanged. Non-field
//   staged inventories (records, tiles, histories) remain resident and are not streamed.
#[test]
fn staging_streams_field_payload_so_the_budget_bounds_bytes_in_flight() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let mut total_field_bytes = 0u64;
    let mut largest_field_bytes = 0u64;
    crate::analysis::operations::visit_analysis_result_fields(&run, |field| {
        if let AnalysisFieldValues::HostF64(values) = &field.values {
            let bytes = values.len() as u64 * 16;
            total_field_bytes += bytes;
            largest_field_bytes = largest_field_bytes.max(bytes);
        }
    });
    assert!(largest_field_bytes > 0 && total_field_bytes > largest_field_bytes);
    // Find the smallest budget that stages: it must be below the total field payload.
    let stages = |budget: u64| {
        let builder =
            AnalysisPublicationBuilder::new(Digest::sha256(b"streamed-staging"), budget).unwrap();
        builder.stage_run(&run).is_ok()
    };
    let (mut low, mut high) = (1u64, 1u64 << 34);
    while low < high {
        let middle = low + (high - low) / 2;
        if stages(middle) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    // Without streaming every field payload would accumulate in staging.
    let mut cumulative = 0u64;
    crate::analysis::storage::preparation::prepare_run_artifact_with_admission(
        &run,
        "2026-01-01T00:00:00Z".into(),
        &mut |bytes| {
            cumulative += bytes;
            Ok(())
        },
        None,
    )
    .unwrap();
    assert!(
        low + (total_field_bytes - largest_field_bytes) <= cumulative,
        "staging retained more than one field payload at a time: {low} vs {cumulative}"
    );
    let root = std::env::temp_dir().join(format!(
        "runmat-streamed-staging-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let store = FilesystemAnalysisArtifactStore::new(root.clone());
    let definition = Digest::sha256(b"streamed-staging-definition");
    let builder = AnalysisPublicationBuilder::new(definition, 1 << 30).unwrap();
    let mut child = run.clone();
    child.run_id = "streamed_staging".into();
    builder.stage_run(&child).unwrap();
    let prepared = builder
        .finish(serde_json::json!({
            "study_definition_digest": definition,
            "public_run": { "steps": [{"run_id": "streamed_staging"}] }
        }))
        .unwrap();
    assert!(prepared.spilled.digests().next().is_some());
    store.commit_publication(prepared).unwrap();
    let loaded = store.load_run("streamed_staging").unwrap().unwrap();
    assert_eq!(loaded.run.fields, run.run.fields);
    std::fs::remove_dir_all(root).unwrap();
}
