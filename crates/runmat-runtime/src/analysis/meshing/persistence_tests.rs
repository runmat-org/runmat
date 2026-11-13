use super::*;
#[test]
fn immutable_mesh_collision_uses_the_installed_virtual_provider() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::new());
    let _provider = runmat_filesystem::replace_provider(provider.clone());
    let path = Path::new("/immutable-mesh.cbor");
    persist_canonical(path, b"original").unwrap();
    persist_canonical(path, b"original").unwrap();
    assert!(persist_canonical(path, b"different")
        .unwrap_err()
        .contains("collision"));
    assert_eq!(runmat_filesystem::read(path).unwrap(), b"original");

    // A captured publication keeps both mesh objects off the live provider
    // until its enclosing operation succeeds. An ordinary provider above still
    // supports direct writes without claiming coherent workspace access.
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
    let artifact = meshes.join("fixture.solver-mesh.cbor");
    let evidence = meshes.join("fixture.meshing-evidence.cbor");
    futures::executor::block_on(async {
        super::super::storage::with_artifact_access(true, |_| -> Result<(), String> {
            let access = persistence::MeshPersistence::capture(&meshes)?;
            access.persist(&artifact, b"mesh")?;
            access.persist(&evidence, b"evidence")?;
            access.persist(&artifact, b"mesh")?;
            assert!(access
                .persist(&artifact, b"collision")
                .unwrap_err()
                .contains("collision"));
            assert!(provider.read_project_path(artifact.clone()).is_err());
            assert!(provider.read_project_path(evidence.clone()).is_err());
            Ok(())
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            provider.read_project_path(artifact.clone()).unwrap(),
            b"mesh"
        );
        assert_eq!(
            provider.read_project_path(evidence.clone()).unwrap(),
            b"evidence"
        );

        let abandoned = meshes.join("abandoned.solver-mesh.cbor");
        let abandoned_evidence = meshes.join("abandoned.meshing-evidence.cbor");
        let failed = super::super::storage::with_artifact_access(true, |_| -> Result<(), String> {
            let access = persistence::MeshPersistence::capture(&meshes)?;
            access.persist(&abandoned, b"discard mesh")?;
            access.persist(&abandoned_evidence, b"discard evidence")?;
            Err("operation refused".into())
        })
        .await
        .unwrap();
        assert_eq!(failed.unwrap_err(), "operation refused");
        assert!(provider.read_project_path(abandoned).is_err());
        assert!(provider.read_project_path(abandoned_evidence).is_err());
        assert_eq!(provider.read_project_path(artifact).unwrap(), b"mesh");
    });
    crate::analysis::replace_fea_runtime(previous_config).unwrap();
}

#[test]
fn immutable_mesh_collision_comparison_handles_matches_changes_short_and_oversized_files() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("original.cbor");
    let original = vec![7_u8; 8193];
    persist_canonical(&path, &original).unwrap();
    persist_canonical(&path, &original).unwrap();
    assert_eq!(runmat_filesystem::read(&path).unwrap(), original);
    for incoming in [vec![8_u8; 8193], vec![7_u8; 8192], vec![7_u8; 8194]] {
        assert!(persist_canonical(&path, &incoming)
            .unwrap_err()
            .contains("collision"));
        assert_eq!(runmat_filesystem::read(&path).unwrap(), original);
    }
}

#[test]
fn immutable_mesh_collision_read_cancels_without_changing_original_or_later_replay() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("original.cbor");
    let original = vec![7_u8; 16385];
    persist_canonical(&path, &original).unwrap();
    {
        let _interrupt = crate::interrupt::replace_interrupt(Some(
            runmat_execution::CancellationToken::from_shared(std::sync::Arc::new(
                std::sync::atomic::AtomicBool::new(true),
            )),
        ));
        assert!(persist_canonical(&path, &original)
            .unwrap_err()
            .contains("cancelled"));
    }
    assert_eq!(runmat_filesystem::read(&path).unwrap(), original);
    persist_canonical(&path, &original).unwrap();
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
