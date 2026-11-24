use super::*;

/// Test objects are `<digest>:<payload>`; bytes without a `:` do not decode.
fn test_digest(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?;
    text.split_once(':').map(|(digest, _)| digest.to_owned())
}

fn object<'a>(path: &'a Path, bytes: &'a [u8], digest: &'a str) -> persistence::MeshObject<'a> {
    persistence::MeshObject {
        path,
        bytes,
        digest,
        identify: &test_digest,
    }
}

#[test]
fn an_in_memory_result_store_saves_meshes_on_the_host_filesystem() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::new());
    let _provider = runmat_filesystem::replace_provider(provider.clone());
    let path = Path::new("/immutable-mesh.cbor");
    persist_canonical(&object(path, b"d1:mesh", "d1")).unwrap();
    persist_canonical(&object(path, b"d1:mesh", "d1")).unwrap();
    // Same digest, different bytes: an encoder defect, named with both digests.
    let error = persist_canonical(&object(path, b"d1:other", "d1")).unwrap_err();
    assert!(error.contains("canonical digest d1"), "{error}");
    assert!(error.contains("stored sha256"), "{error}");
    assert_eq!(runmat_filesystem::read(path).unwrap(), b"d1:mesh");
    // A stale object (another digest) and a corrupt one are replaced.
    runmat_filesystem::write(path, b"d0:older encoding").unwrap();
    persist_canonical(&object(path, b"d1:mesh", "d1")).unwrap();
    assert_eq!(runmat_filesystem::read(path).unwrap(), b"d1:mesh");
    runmat_filesystem::write(path, b"truncated").unwrap();
    persist_canonical(&object(path, b"d1:mesh", "d1")).unwrap();
    assert_eq!(runmat_filesystem::read(path).unwrap(), b"d1:mesh");
}

#[test]
fn meshes_use_the_result_store_provider_not_the_installed_provider() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let store_provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::new());
    let _store_provider = runmat_filesystem::replace_provider(store_provider.clone());
    let root = PathBuf::from("/.artifacts");
    let previous_config = crate::analysis::replace_fea_runtime(crate::analysis::FeaRuntimeConfig {
        artifact_root: Some(root.clone()),
        study_artifact_root: Some(root.join("studies")),
        ..Default::default()
    })
    .unwrap();
    super::super::storage::configure_artifact_store(
        super::super::storage::AnalysisArtifactStoreConfig::Filesystem { root: root.clone() },
    )
    .unwrap();
    // The run installs a different provider (a pinned input snapshot in
    // Desktop). It already holds a stale mesh under the same name.
    let installed = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::new());
    let _installed = runmat_filesystem::replace_provider(installed.clone());
    let meshes = root.join("meshes");
    let artifact = meshes.join("d1.solver-mesh.cbor");
    installed
        .write_project_path(artifact.clone(), b"d0:stale")
        .unwrap();

    // Outside a result transaction, this store's provider refuses direct
    // access; the error says a transaction is required.
    let refused = persistence::MeshPersistence::capture(&meshes)
        .err()
        .expect("direct access refused");
    assert!(refused.contains("result transaction"), "{refused}");

    futures::executor::block_on(async {
        super::super::storage::with_artifact_access(true, |_| -> Result<(), String> {
            let access = persistence::MeshPersistence::capture(&meshes)?;
            access
                .persist_typed(&object(&artifact, b"d1:mesh", "d1"))
                .map_err(|error| error.to_string())?;
            access
                .persist_typed(&object(&artifact, b"d1:mesh", "d1"))
                .map_err(|error| error.to_string())
        })
        .await
        .unwrap()
        .unwrap();
    });
    assert_eq!(
        store_provider.read_project_path(artifact.clone()).unwrap(),
        b"d1:mesh"
    );
    // The installed provider was neither consulted nor changed.
    assert_eq!(
        installed.read_project_path(artifact.clone()).unwrap(),
        b"d0:stale"
    );
    crate::analysis::replace_fea_runtime(previous_config).unwrap();
}

#[test]
fn a_stale_mesh_in_the_result_store_is_replaced_inside_the_transaction() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::new());
    let _provider = runmat_filesystem::replace_provider(provider.clone());
    let root = PathBuf::from("/.artifacts");
    let previous_config = crate::analysis::replace_fea_runtime(crate::analysis::FeaRuntimeConfig {
        artifact_root: Some(root.clone()),
        study_artifact_root: Some(root.join("studies")),
        ..Default::default()
    })
    .unwrap();
    super::super::storage::configure_artifact_store(
        super::super::storage::AnalysisArtifactStoreConfig::Filesystem { root: root.clone() },
    )
    .unwrap();
    let meshes = root.join("meshes");
    let artifact = meshes.join("d1.solver-mesh.cbor");
    provider
        .write_project_path(artifact.clone(), b"d0:stale from an earlier failed run")
        .unwrap();
    futures::executor::block_on(async {
        // A failed operation discards the replacement with everything else.
        let failed = super::super::storage::with_artifact_access(true, |_| -> Result<(), String> {
            let access = persistence::MeshPersistence::capture(&meshes)?;
            access
                .persist_typed(&object(&artifact, b"d1:mesh", "d1"))
                .map_err(|error| error.to_string())?;
            Err("operation refused".into())
        })
        .await
        .unwrap();
        assert_eq!(failed.unwrap_err(), "operation refused");
        assert_eq!(
            provider.read_project_path(artifact.clone()).unwrap(),
            b"d0:stale from an earlier failed run"
        );
        super::super::storage::with_artifact_access(true, |_| -> Result<(), String> {
            let access = persistence::MeshPersistence::capture(&meshes)?;
            access
                .persist_typed(&object(&artifact, b"d1:mesh", "d1"))
                .map_err(|error| error.to_string())?;
            let error = access
                .persist_typed(&object(&artifact, b"d1:different bytes", "d1"))
                .unwrap_err()
                .to_string();
            assert!(error.contains("canonical digest d1"), "{error}");
            Ok(())
        })
        .await
        .unwrap()
        .unwrap();
    });
    assert_eq!(provider.read_project_path(artifact).unwrap(), b"d1:mesh");
    crate::analysis::replace_fea_runtime(previous_config).unwrap();
}

#[test]
fn an_existing_mesh_read_observes_cancellation_and_keeps_the_original() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("original.cbor");
    let mut original = b"d1:".to_vec();
    original.extend(vec![7_u8; 16385]);
    persist_canonical(&object(&path, &original, "d1")).unwrap();
    {
        let _interrupt = crate::interrupt::replace_interrupt(Some(
            runmat_execution::CancellationToken::from_shared(std::sync::Arc::new(
                std::sync::atomic::AtomicBool::new(true),
            )),
        ));
        assert!(persist_canonical(&object(&path, &original, "d1"))
            .unwrap_err()
            .contains("cancelled"));
    }
    assert_eq!(runmat_filesystem::read(&path).unwrap(), original);
    persist_canonical(&object(&path, &original, "d1")).unwrap();
}

#[test]
fn real_solver_meshes_identify_by_their_recomputed_canonical_digest() {
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    let tet4 = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let tet10 = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let bytes = tet4.canonical_encode().unwrap();
    assert_eq!(
        solver_mesh_digest(&bytes).unwrap(),
        encode_digest(tet4.canonical_digest.bytes())
    );
    assert_ne!(
        solver_mesh_digest(&tet10.canonical_encode().unwrap()).unwrap(),
        encode_digest(tet4.canonical_digest.bytes())
    );
    assert!(solver_mesh_digest(&bytes[..bytes.len() - 1]).is_none());
    // One digest, one encoding: encoding twice gives the same bytes.
    assert_eq!(tet4.canonical_encode().unwrap(), bytes);
}

#[test]
fn borrowed_mesh_encoding_observes_runtime_cancellation_and_preserves_original() {
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    let _guard = crate::analysis::tests::analysis_test_guard();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let original = mesh.canonical_encode().unwrap();
        {
            let _interrupt = crate::interrupt::replace_interrupt(Some(
                runmat_execution::CancellationToken::from_shared(std::sync::Arc::new(
                    std::sync::atomic::AtomicBool::new(true),
                )),
            ));
            assert!(mesh
                .canonical_encode_with_cancellation(&MeshPersistenceCancellation)
                .is_err());
        }
        assert_eq!(
            mesh.canonical_encode_with_cancellation(&MeshPersistenceCancellation)
                .unwrap(),
            original
        );
    }
}

#[test]
fn mesh_and_study_evidence_publish_inside_a_pinned_execution() {
    use runmat_filesystem::pinned::{InputManifest, PinnedFsProvider};
    let _guard = crate::analysis::tests::analysis_test_guard();
    let base = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::new());
    let pinned = PinnedFsProvider::new(
        base.clone(),
        InputManifest {
            snapshot_id: "capture".into(),
            entries: Vec::new(),
        },
    )
    .unwrap();
    let pinned_provider = runmat_filesystem::replace_provider(std::sync::Arc::new(pinned));
    let mesh = Path::new("/fixture.solver-mesh.cbor");
    persist_canonical(&object(mesh, b"d1:mesh", "d1")).unwrap();
    persist_canonical(&object(mesh, b"d1:mesh", "d1")).unwrap();
    assert!(persist_canonical(&object(mesh, b"d1:mesk", "d1"))
        .unwrap_err()
        .contains("canonical digest d1"));
    let evidence = PathBuf::from("/study.json");
    atomic_write_bytes(&evidence, b"first").unwrap();
    atomic_write_bytes(&evidence, b"replacement").unwrap();
    drop(pinned_provider);
    let _base_provider = runmat_filesystem::replace_provider(base);
    assert_eq!(runmat_filesystem::read(mesh).unwrap(), b"d1:mesh");
    assert_eq!(runmat_filesystem::read(&evidence).unwrap(), b"replacement");
    let mut names: Vec<_> = runmat_filesystem::read_dir(Path::new("/"))
        .unwrap()
        .into_iter()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["fixture.solver-mesh.cbor", "study.json"]);
}
