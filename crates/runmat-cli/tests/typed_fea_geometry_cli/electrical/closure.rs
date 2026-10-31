use super::*;

pub(super) fn reopen(
    project: &TempDir,
    manifest: &Value,
    retained: &runmat_runtime::analysis::AnalysisRunResult,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> runmat_runtime::analysis::ElectricalStudyCompletion {
    let catalog = &manifest["numerical_evidence"]["catalog"];
    let cold = crate::closure::reopen_catalog_objects(project, manifest);
    let entries = manifest["public_run"]["states"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    let state = runmat_analysis_fea::execution::CanonicalSemanticState::from_canonical_bytes(
        &fs::read(
            project
                .path()
                .join(entries[0]["artifact_path"].as_str().unwrap()),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        state.kind,
        runmat_analysis_core::StateKind::ElectromagneticStatic
    );
    let payload = state.payload.electrical_static().unwrap();
    assert_eq!(payload.fields.fields().len(), 4);
    for field in payload.fields.fields() {
        assert_eq!(
            field.storage(),
            retained
                .run
                .fields
                .iter()
                .find(|f| f.field_id == field.storage().field_id)
                .unwrap()
        );
    }
    let root: runmat_execution::value::ValueRef =
        serde_json::from_value(manifest["numerical_evidence"]["root"].clone()).unwrap();
    let access = runmat_execution_artifact::ArtifactAccess {
        authorization_scope: root.authorization_scope.clone(),
        encryption_context: root.encryption_context,
    };
    let limits = runmat_execution_artifact::object::ObjectInventoryLimits::default();
    assert!(runmat_analysis_execution::import_electrical_study_root(
        &cold,
        &root,
        runmat_execution::Digest::sha256("foreign-root"),
        &access,
        limits,
    )
    .is_err());
    let completed = runmat_analysis_execution::import_electrical_study_root(
        &cold,
        &root,
        root.logical_digest,
        &access,
        limits,
    )
    .unwrap();
    assert_eq!(completed.states.len(), 1);
    assert_eq!(completed.states[0].state, state);
    assert_eq!(completed.result.result.mesh_digest(), mesh.canonical_digest);
    // The native source converter replays exact task/output/accounting authentication;
    // its cache contains ONLY objects returned by the real producer's public manifest.
    let (completion, source) =
        runmat_execution_runner_native::electrical_study_completion_from_source(completed, cold)
            .unwrap();
    assert_eq!(serde_json::to_value(&completion.catalog).unwrap(), *catalog);
    assert_eq!(completion.root, root);
    for reference in &completion.catalog.objects {
        let bytes =
            runmat_execution::value::PublishedObjectSource::read_verified(&source, reference)
                .unwrap();
        assert_eq!(
            runmat_execution::Digest::sha256(bytes),
            reference.logical_digest
        );
    }
    completion
}
