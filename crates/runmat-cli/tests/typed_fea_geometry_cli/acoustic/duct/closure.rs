use super::*;

pub(super) fn reopen(
    project: &TempDir,
    manifest: &Value,
    retained: &runmat_runtime::analysis::AnalysisStudyStepRunData,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> runmat_analysis_execution::CompletedAcousticStudy {
    let cold = crate::closure::reopen_catalog_objects(project, manifest);
    assert!(manifest["public_run"]["states"]
        .as_array()
        .unwrap()
        .is_empty());
    let runmat_runtime::analysis::AnalysisStudyStepNumericalResultData::AcousticFrequency(
        numerical,
    ) = &retained.numerical_result
    else {
        panic!("exact acoustic frequency step missing");
    };
    assert_eq!(numerical.entries.len(), 1);
    // The run record keeps a reference; the state is persisted once as canonical binary.
    let reference = numerical.entries[0].states[0].clone();
    assert_eq!(
        reference.kind,
        runmat_analysis_core::StateKind::AcousticComplex
    );
    reference.verify_identity().unwrap();
    let state = runmat_analysis_fea::execution::CanonicalSemanticState::from_canonical_bytes(
        &fs::read(project.path().join(&reference.artifact_path)).unwrap(),
    )
    .unwrap();
    assert_eq!(state.identity, reference.identity);
    let acoustic = state.payload.acoustic().unwrap();
    assert_eq!(acoustic.frequency.canonical_value(), 0.2);
    let root: runmat_execution::value::ValueRef =
        serde_json::from_value(manifest["numerical_evidence"]["root"].clone()).unwrap();
    let access = runmat_execution_artifact::ArtifactAccess {
        authorization_scope: root.authorization_scope.clone(),
        encryption_context: root.encryption_context,
    };
    let limits = runmat_execution_artifact::object::ObjectInventoryLimits::default();
    let completed = runmat_analysis_execution::import_acoustic_study_root(
        &cold,
        &root,
        root.logical_digest,
        &access,
        limits,
    )
    .unwrap();
    assert_eq!(completed.entries.len(), 1);
    assert_eq!(completed.entries[0].states[0].state, state);
    let field_request = completed.entries[0]
        .result
        .result
        .native_field_request_from_result()
        .unwrap();
    let definitions =
        runmat_analysis_fea::execution::acoustic_native_field_definitions(&field_request).unwrap();
    assert_eq!(
        acoustic
            .fields
            .fields()
            .iter()
            .map(|field| field.definition())
            .collect::<Vec<_>>(),
        definitions.iter().collect::<Vec<_>>(),
        "cold state preserves the complete canonical acoustic field inventory"
    );
    assert_eq!(
        completed.entries[0].result.result.mesh_digest(),
        mesh.canonical_digest
    );
    assert!(runmat_analysis_execution::import_acoustic_study_root(
        &cold,
        &root,
        runmat_execution::Digest::sha256("foreign-root"),
        &access,
        limits
    )
    .is_err());
    completed
}
