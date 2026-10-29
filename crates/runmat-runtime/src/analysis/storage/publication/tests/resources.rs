use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counted<'a> {
    passes: &'a AtomicUsize,
    value: &'a str,
}

#[test]
fn prepared_and_retained_original_inventory_cannot_each_reuse_the_full_ceiling() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let original = store
        .commit_publication(prepare(&run, &["retained_original"]))
        .unwrap();
    let mut incoming = prepare(&run, &["incoming_original"]);
    incoming.maximum_bytes = super::super::resources::admission_bytes(&incoming).unwrap() + 1024;
    incoming.validate().unwrap();
    let incoming_identity = incoming.descriptor.identity().unwrap();
    assert!(matches!(
        store.commit_publication(incoming).unwrap_err(),
        AnalysisPublicationError::ResourceLimit { .. }
    ));
    assert!(store.load_run("incoming_original").unwrap().is_none());
    assert!(!super::super::filesystem::directory(&store, incoming_identity).exists());
    assert!(store.load_run("retained_original").unwrap().is_some());
    assert!(store
        .load_publication(original.identity, 1 << 20)
        .unwrap()
        .is_some());
}
impl Serialize for Counted<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.passes.fetch_add(1, Ordering::SeqCst);
        serializer.serialize_str(self.value)
    }
}

#[test]
fn manifest_projection_rejects_before_second_serialization_or_materialization() {
    let builder = AnalysisPublicationBuilder::new(Digest::sha256(b"budget-test"), 1024).unwrap();
    let passes = AtomicUsize::new(0);
    let input = "a".repeat(2048);
    assert!(builder
        .serialize_manifest(&Counted {
            passes: &passes,
            value: &input
        })
        .is_err());
    assert_eq!(passes.load(Ordering::SeqCst), 1);
    assert!(builder.store.runs.read().unwrap().is_empty());
    assert!(!*builder.admission.lock().unwrap());
}

#[test]
fn prepared_commit_rederives_phase_budget_before_any_backend_visibility() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = std::env::temp_dir().join(format!(
        "atomic-resource-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let disk = FilesystemAnalysisArtifactStore::new(root.clone());
    let memory = InMemoryAnalysisArtifactStore::new();
    for store in [&memory as &dyn AnalysisArtifactStore, &disk] {
        let mut prepared = prepare(&run, &["resource_original"]);
        let required = super::super::resources::retained(
            &prepared.runs,
            &prepared.chunks,
            &prepared.observations,
            prepared.maximum_bytes,
        )
        .unwrap()
            + super::super::resources::commit_work(
                prepared.descriptor.definition,
                &prepared.manifest,
                &prepared.runs,
                &prepared.observations,
                prepared.maximum_bytes,
            )
            .unwrap()
            .total;
        prepared.maximum_bytes = required - 1;
        assert!(matches!(
            store.commit_publication(prepared).unwrap_err(),
            AnalysisPublicationError::ResourceLimit { .. }
        ));
        assert!(store.load_run("resource_original").unwrap().is_none());
        assert!(store.list_runs().unwrap().is_empty());
    }
    assert!(!root.exists());
}

#[test]
fn manifest_preparation_freezes_original_inventory_before_projecting_copies() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let definition = Digest::sha256(b"inventory-freeze");
    let builder = AnalysisPublicationBuilder::new(definition, 128 << 20).unwrap();
    builder.stage_run(&run).unwrap();
    builder.admit_manifest_copy(&run).unwrap();
    let mut late = run.clone();
    late.run_id = "late_inventory_substitution".into();
    assert!(builder
        .stage_run(&late)
        .unwrap_err()
        .to_string()
        .contains("inventory is frozen"));
    assert_eq!(builder.store.runs.read().unwrap().len(), 1);
    assert!(!builder
        .store
        .runs
        .read()
        .unwrap()
        .contains_key(&late.run_id));
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A 4096-byte preparation fixture accumulates two reservations and rejects a third
//   without changing the retained allowance or staging a run. It does not establish typed-input
//   accounting correctness, measured phase peak, scalable growth or complete-study admission.
#[test]
fn preparation_reservations_accumulate_and_failed_growth_preserves_the_allowance() {
    let input = vec![17u8; 4096];
    let one = super::super::typed_resident(&input, 128 << 20).unwrap() * 2;
    let mut builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"preparation-growth"), one * 2).unwrap();
    builder.admit_preparation(&input).unwrap();
    builder.admit_preparation(&input).unwrap();
    assert_eq!(builder.preparation_resident_bytes, one * 2);
    assert!(matches!(
        builder.admit_preparation(&input),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert_eq!(builder.preparation_resident_bytes, one * 2);
    assert!(builder.store.runs.read().unwrap().is_empty());
}

#[test]
fn original_preparation_stays_charged_at_both_backend_commit_boundaries() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let memory = InMemoryAnalysisArtifactStore::new();
    let disk = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    for store in [&memory as &dyn AnalysisArtifactStore, &disk] {
        let mut prepared = prepare(&run, &["preparation_original"]);
        let output = super::super::resources::admission_bytes(&prepared).unwrap();
        prepared.preparation_resident_bytes = super::super::resident(&run, 128 << 20).unwrap() * 2;
        prepared.maximum_bytes = output + prepared.preparation_resident_bytes - 1;
        assert!(matches!(
            store.commit_publication(prepared),
            Err(AnalysisPublicationError::ResourceLimit { .. })
        ));
        assert!(store.load_run("preparation_original").unwrap().is_none());
        assert!(store.list_runs().unwrap().is_empty());
        // A failed reservation does not poison a later sufficiently admitted attempt.
        store
            .commit_publication(prepare(&run, &["preparation_original"]))
            .unwrap();
        assert_eq!(
            store
                .load_run("preparation_original")
                .unwrap()
                .unwrap()
                .run
                .fields,
            run.run.fields
        );
    }
}

#[test]
fn source_workspace_exhaustion_preserves_preparation_and_allows_an_admitted_retry() {
    use runmat_execution::value::{artifact_object_value_id, ValueRef, ValueRefKind};
    let logical = Digest::sha256(b"source-workspace-original");
    let context = Digest::sha256(b"source-workspace-authority");
    let mut reference = ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: artifact_object_value_id(logical, context, "source-workspace"),
        logical_digest: logical,
        encoded_length: 16384,
        media_type: "application/vnd.runmat.test".into(),
        value_schema: "runmat.test".into(),
        encryption_context: context,
        kind: ValueRefKind::ResultObject,
        authorization_scope: "source-workspace".into(),
        resident_fence: None,
    };
    let demand = reference.encoded_length
        + runmat_canonical_codec::projected_canonical_materialization_resident_bytes_estimate(
            reference.encoded_length,
            1,
        )
        .unwrap();
    let mut builder = AnalysisPublicationBuilder::new(logical, demand).unwrap();
    builder.admit_preparation(&"original metadata").unwrap();
    let original = builder.preparation_resident_bytes;
    assert!(matches!(
        builder.admit_source_objects(&[reference.clone()]),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert_eq!(builder.preparation_resident_bytes, original);
    assert!(builder.store.runs.read().unwrap().is_empty());
    reference.encoded_length = u64::MAX;
    assert!(matches!(
        builder.admit_source_objects(&[reference.clone()]),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert_eq!(builder.preparation_resident_bytes, original);
    reference.encoded_length = 8;
    builder.admit_source_objects(&[reference]).unwrap();
    assert_eq!(builder.preparation_resident_bytes, original);
    assert!(builder.source_workspace_bytes > 0);
    let phase = builder.begin_source_phase().unwrap();
    assert_eq!(
        builder.active_source_workspace().unwrap(),
        builder.source_workspace_bytes
    );
    drop(phase);
    assert_eq!(builder.active_source_workspace().unwrap(), 0);
    assert!(builder.store.runs.read().unwrap().is_empty());
}

#[test]
fn cached_assessment_metadata_cannot_reuse_the_staged_inventory_allowance() {
    let _guard = tests::analysis_test_guard();
    let original = run();
    let mut builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"assessment-phase"), 128 << 20).unwrap();
    builder.stage_run(&original).unwrap();
    let baseline = super::super::resources::retained(
        &builder.store.runs.read().unwrap(),
        &builder.store.chunks.read().unwrap(),
        &builder.store.observations.read().unwrap(),
        builder.maximum_bytes,
    )
    .unwrap();
    builder
        .serialize_manifest(&serde_json::json!({"original": true}))
        .unwrap();
    let metadata = AssessmentRunEvidence::from(&original);
    let copy = resident(
        &(
            &original.run.identity,
            original.numerically_acceptable,
            &original.observation_scalars,
        ),
        builder.maximum_bytes,
    )
    .unwrap();
    let returned = resident(&Some(metadata), builder.maximum_bytes).unwrap();
    builder.maximum_bytes = baseline + copy.max(returned);
    let first = builder
        .load_assessment_metadata_admitted(&original.run_id, 1 << 20, 0)
        .unwrap()
        .unwrap();
    let cached = resident(&Some(&first), 128 << 20).unwrap();
    assert!(matches!(
        builder.load_assessment_metadata_admitted(&original.run_id, 1 << 20, cached),
        Err(AssessmentStoreError::ResourceLimit)
    ));
    assert_eq!(first.identity, original.run.identity);
    assert_eq!(first.observation_scalars, original.observation_scalars);
    builder.maximum_bytes = 128 << 20;
    let retry = builder
        .load_assessment_metadata_admitted(&original.run_id, 1 << 20, cached)
        .unwrap()
        .unwrap();
    assert_eq!(retry.observation_scalars, first.observation_scalars);
    // Staging retains the record; its field chunks were streamed to the spill area.
    assert!(builder
        .store
        .runs
        .read()
        .unwrap()
        .contains_key(&original.run_id));
}

#[test]
fn assessment_phase_requires_frozen_inventory_and_subtracts_original_retention() {
    let _guard = tests::analysis_test_guard();
    let original = run();
    let builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"frozen-assessment-phase"), 128 << 20)
            .unwrap();
    builder.stage_run(&original).unwrap();
    assert!(matches!(
        builder.assessment_workspace_limit(),
        Err(AnalysisPublicationError::Rejected { .. })
    ));
    let retained = super::super::resources::retained(
        &builder.store.runs.read().unwrap(),
        &builder.store.chunks.read().unwrap(),
        &builder.store.observations.read().unwrap(),
        builder.maximum_bytes,
    )
    .unwrap();
    builder
        .serialize_manifest(&serde_json::json!({"original": true}))
        .unwrap();
    let available = builder.assessment_workspace_limit().unwrap();
    assert_eq!(available, builder.maximum_bytes - retained);
    let mut foreign = original.clone();
    foreign.run_id = "late-assessment-child".into();
    assert!(builder.stage_run(&foreign).is_err());
    assert_eq!(builder.assessment_workspace_limit().unwrap(), available);
    assert_eq!(builder.store.runs.read().unwrap().len(), 1);
}
