//! Retained planning facts cannot reuse their bytes for visibility edits/provider work.
use super::*;

#[test]
fn admission_failure_preserves_all_retractions_and_the_new_marker_then_retry_commits() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    std::fs::create_dir(root.path().join("runs")).unwrap();
    let ids: Vec<_> = (0..128)
        .map(|ordinal| format!("original_{ordinal:04}"))
        .collect();
    for id in &ids {
        std::fs::write(store.run_path(id).unwrap(), b"original").unwrap();
    }
    let marker = root
        .path()
        .join("publications")
        .join(digest_hex(Digest::sha256(b"new-root")))
        .join("commit.json");
    std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
    let scope = store.access_with_budget(true, 16 << 20).unwrap();
    // A large admitted planner inventory coexists with this transition. The
    // remaining allowance cannot be reused as if that inventory had been dropped.
    let held = scope.remaining_bytes() - 80_000;
    let plan = EvictionPlan {
        entries: vec![retention::Eviction {
            publication: None,
            runs: ids.clone(),
        }],
        _reservation: scope.reserve_resident(held).unwrap(),
    };
    let error = transition(
        &scope,
        Some((marker.clone(), b"closed-root".to_vec())),
        plan,
    )
    .unwrap_err();
    assert!(
        matches!(error, AnalysisPublicationError::ResourceLimit { .. }),
        "resource admission returned a successful or uncertain commit: {error}"
    );
    assert!(!marker.exists());
    for id in &ids {
        assert_eq!(
            std::fs::read(store.run_path(id).unwrap()).unwrap(),
            b"original"
        );
    }
    drop(scope);
    let scope = store.access_with_budget(true, 16 << 20).unwrap();
    let plan = EvictionPlan {
        entries: vec![retention::Eviction {
            publication: None,
            runs: ids.clone(),
        }],
        _reservation: scope.reserve_resident(0).unwrap(),
    };
    transition(
        &scope,
        Some((marker.clone(), b"closed-root".to_vec())),
        plan,
    )
    .unwrap();
    assert_eq!(std::fs::read(marker).unwrap(), b"closed-root");
    for id in &ids {
        assert!(!store.run_path(id).unwrap().exists());
    }
}
