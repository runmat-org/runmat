use super::*;

pub(super) fn reopen(
    project: &TempDir,
    manifest: &Value,
    retained: &runmat_runtime::analysis::AnalysisStudyStepRunData,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> runmat_analysis_execution::CompletedAcousticStudy {
    let catalog = &manifest["numerical_evidence"]["catalog"];
    let paths = manifest["numerical_evidence"]["object_artifact_paths"]
        .as_array()
        .unwrap();
    assert_eq!(paths.len(), catalog["objects"].as_array().unwrap().len());
    assert!(paths.len() > 3);
    let mut cold = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        project.path().join("cold-closure"),
        256 * 1024 * 1024,
    )
    .unwrap();
    let mut digests = std::collections::BTreeSet::new();
    for path in paths {
        let object: Value =
            serde_json::from_slice(&fs::read(project.path().join(path.as_str().unwrap())).unwrap())
                .unwrap();
        let bytes: Vec<u8> = serde_json::from_value(object["bytes"].clone()).unwrap();
        let reference: runmat_execution::value::ValueRef =
            serde_json::from_value(object["reference"].clone()).unwrap();
        assert_eq!(
            runmat_execution::Digest::sha256(&bytes),
            reference.logical_digest
        );
        assert!(digests.insert(reference.logical_digest));
        let logical = runmat_execution_artifact::LogicalObject::new(
            runmat_execution_artifact::ObjectNamespace::ResultValue,
            format!("cold/{}", reference.logical_digest),
            reference.media_type,
            bytes,
        )
        .unwrap();
        runmat_execution_artifact::cache::CacheExport::write_verified(&mut cold, &logical).unwrap();
    }
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
    assert_eq!(reference.kind, runmat_analysis_core::StateKind::AcousticComplex);
    reference.verify_identity().unwrap();
    let state = runmat_analysis_fea::execution::CanonicalSemanticState::from_canonical_bytes(
        &fs::read(&reference.artifact_path).unwrap(),
    )
    .unwrap();
    assert_eq!(state.identity, reference.identity);
    let acoustic = state.payload.acoustic().unwrap();
    assert_eq!(acoustic.frequency.hertz, 0.2);
    assert_eq!(acoustic.fields.fields().len(), 2);
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
