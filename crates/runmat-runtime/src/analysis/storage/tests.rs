use super::*;

#[test]
fn generated_run_sequence_breaks_timestamp_ties_numerically() {
    assert!(generated_run_id_order("run_1000_10") > generated_run_id_order("run_1000_9"));
}

#[test]
fn scalar_evidence_is_derived_and_authenticated_before_metadata_reads() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let mut run = crate::analysis::analysis_run_linear_static_with_options(
        &crate::analysis::tests::sample_model(),
        crate::analysis::ComputeBackend::Cpu,
        crate::analysis::AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: crate::analysis::PrecisionMode::Fp64,
            preconditioner_mode: crate::analysis::PreconditionerMode::Auto,
            quality_policy: crate::analysis::QualityPolicy::Balanced,
            solver_mesh_artifact_path: None,
        },
        crate::operations::OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    let forged = runmat_analysis_fea::results::assessment::ScalarObservationEvidence {
        observation_id: runmat_analysis_core::ObservationId("forged_peak".into()),
        value: 0.0,
        source_identity: Digest::sha256("substituted observation"),
        sample: None,
        entity: None,
    };
    run.observation_scalars.push(forged.clone());
    let mut artifact = prepare_run_artifact(&run, Utc::now().to_rfc3339())
        .unwrap()
        .persisted;
    assert!(!artifact.metadata.observation_scalars.contains(&forged));
    validate_persisted_run(&artifact, &run.run_id).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let stores: Vec<Box<dyn AnalysisArtifactStore>> = vec![
        Box::new(InMemoryAnalysisArtifactStore::new()),
        Box::new(FilesystemAnalysisArtifactStore::new(
            directory.path().into(),
        )),
    ];
    for store in &stores {
        store.persist_run(&run).unwrap();
        assert!(matches!(
            store.load_assessment_metadata(&run.run_id, 0),
            Err(AssessmentStoreError::ResourceLimit)
        ));
        let metadata = store
            .load_assessment_metadata(&run.run_id, 64 * 1024 * 1024)
            .unwrap()
            .unwrap();
        assert!(!metadata.observation_scalars.contains(&forged));
        assert_eq!(metadata.identity, run.run.identity);
    }
    let mut substituted = artifact.clone();
    substituted.metadata.numerically_acceptable = !substituted.metadata.numerically_acceptable;
    // The typed metadata is the single authenticated copy: any substitution breaks its digest.
    assert!(validate_persisted_run(&substituted, &run.run_id)
        .unwrap_err()
        .contains("content digest"));
    artifact.metadata.observation_scalars.push(forged);
    assert!(validate_persisted_run(&artifact, &run.run_id)
        .unwrap_err()
        .contains("content digest"));
}

// witness: FEA-STRUCTURAL-PUBLIC-001/result_publication_defects @
//   linear_structural/tet4/numerical_kernel; coverage: partial;
//   boundary: a persisted run keeps its content once, as typed metadata (no untyped copy), and
//   materializes back to the original fields; observation chunking is covered by publication tests.
#[test]
fn persisted_run_keeps_typed_metadata_as_its_single_content_copy() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let run = crate::analysis::analysis_run_linear_static_with_options(
        &crate::analysis::tests::sample_model(),
        crate::analysis::ComputeBackend::Cpu,
        crate::analysis::AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: crate::analysis::PrecisionMode::Fp64,
            preconditioner_mode: crate::analysis::PreconditionerMode::Auto,
            quality_policy: crate::analysis::QualityPolicy::Balanced,
            solver_mesh_artifact_path: None,
        },
        crate::operations::OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    let prepared = prepare_run_artifact(&run, Utc::now().to_rfc3339()).unwrap();
    let artifact = &prepared.persisted;
    assert!(artifact.run.is_null());
    let encoded = serde_json::to_value(artifact).unwrap();
    assert!(
        encoded.get("run").is_none(),
        "no untyped run copy is written"
    );
    validate_persisted_run(artifact, &run.run_id).unwrap();
    let restored =
        field_reader::materialize_run(artifact, |digest| Ok(prepared.chunks.get(&digest).cloned()))
            .unwrap();
    assert_eq!(restored.run.fields, run.run.fields);
}
