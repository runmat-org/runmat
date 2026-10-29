use super::*;

#[test]
fn compatible_capture_requires_real_absence_and_never_downgrades_an_existing_sidecar() {
    let root = tempfile::tempdir().unwrap();
    let module = root.path().join(format!(
        "capture_probe.{}",
        runmat_mex::mex_suffix().unwrap()
    ));
    // Capture/policy proof only. The unchanged isolated CLI cases exercise
    // actual compiled modules and the host's ABI/API gate before invocation.
    std::fs::write(&module, b"captured original bytes").unwrap();
    let captured = CapturedOrdinaryMex::capture(&module, UnmanifestedMexPolicy::Isolate).unwrap();
    assert!(matches!(
        captured.evidence(),
        OrdinaryMexEvidence::Compatible { .. }
    ));
    assert!(captured.manifest_bytes().is_none());
    assert_eq!(
        captured
            .evidence()
            .admit_policy(UnmanifestedMexPolicy::Deny)
            .err()
            .unwrap()
            .identifier(),
        Some("RunMat:MEX:UnmanifestedBinaryDenied")
    );
    assert_eq!(
        CapturedOrdinaryMex::capture(&module, UnmanifestedMexPolicy::Deny)
            .err()
            .unwrap()
            .identifier(),
        Some("RunMat:MEX:UnmanifestedBinaryDenied")
    );
    std::fs::write(&module, b"changed source bytes").unwrap();
    let changed = CapturedOrdinaryMex::capture(&module, UnmanifestedMexPolicy::Isolate).unwrap();
    assert_eq!(captured.bytes(), b"captured original bytes");
    assert_ne!(captured.identity(), changed.identity());

    let sidecar = runmat_mex::MexArtifactManifest::path_for_module(&module);
    std::fs::write(&sidecar, b"{}").unwrap();
    assert_eq!(
        CapturedOrdinaryMex::capture(&module, UnmanifestedMexPolicy::Isolate)
            .err()
            .unwrap()
            .identifier(),
        Some("RunMat:MEX:Artifact")
    );
    std::fs::remove_file(&sidecar).unwrap();
    std::fs::create_dir(&sidecar).unwrap();
    assert_eq!(
        CapturedOrdinaryMex::capture(&module, UnmanifestedMexPolicy::Isolate)
            .err()
            .unwrap()
            .identifier(),
        Some("RunMat:MEX:Artifact")
    );
    std::fs::remove_dir(&sidecar).unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(root.path().join("missing-manifest"), &sidecar).unwrap();
        assert_eq!(
            CapturedOrdinaryMex::capture(&module, UnmanifestedMexPolicy::Isolate)
                .err()
                .unwrap()
                .identifier(),
            Some("RunMat:MEX:Artifact")
        );
    }
}
