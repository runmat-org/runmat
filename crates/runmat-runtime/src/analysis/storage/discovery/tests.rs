use super::*;
use crate::analysis::storage::FilesystemAnalysisArtifactStore;
use std::cell::Cell;

struct LiveRecord<'a>(&'a Cell<usize>);
impl Drop for LiveRecord<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

#[test]
fn visitor_drops_previous_decoded_payload_before_loading_next_record() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let runs = root.path().join("runs");
    std::fs::create_dir(&runs).unwrap();
    for ordinal in 0..32 {
        std::fs::write(runs.join(format!("run{ordinal}.json")), b"supported").unwrap();
    }
    std::fs::write(runs.join("unrelated.json"), b"unsupported").unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let scope = store.access(false).unwrap();
    let live = Cell::new(0);
    let visited = Cell::new(0);
    visit_supported_artifacts(
        &scope,
        |bytes, _| {
            assert_eq!(live.get(), 0, "previous physical payload is still retained");
            if bytes != b"supported" {
                return Err("unsupported schema".into());
            }
            live.set(1);
            Ok(LiveRecord(&live))
        },
        |record| {
            assert_eq!(live.get(), 1);
            visited.set(visited.get() + 1);
            drop(record);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(visited.get(), 32);
    assert_eq!(live.get(), 0);
}

#[test]
fn consumer_failure_stops_discovery_and_is_not_treated_as_unsupported_schema() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let runs = root.path().join("runs");
    std::fs::create_dir(&runs).unwrap();
    for id in ["first", "second"] {
        std::fs::write(runs.join(format!("{id}.json")), b"supported").unwrap();
    }
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let scope = store.access(false).unwrap();
    let decoded = Cell::new(0);
    let error = visit_supported_artifacts(
        &scope,
        |_, _| {
            decoded.set(decoded.get() + 1);
            Ok(())
        },
        |_| Err("retention inventory rejected".into()),
    )
    .unwrap_err();
    assert_eq!(error, "retention inventory rejected");
    assert_eq!(decoded.get(), 1);
}

#[test]
fn admission_failure_is_fatal_before_schema_decode_or_consumer_execution() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let runs = root.path().join("runs");
    std::fs::create_dir(&runs).unwrap();
    std::fs::write(runs.join("original.json"), b"supported").unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let scope = store.access(false).unwrap();
    let decoded = Cell::new(0);
    let consumed = Cell::new(0);
    let error = visit_supported_artifacts_admitted(
        &scope,
        |_| {
            Err::<(), _>(AnalysisPublicationError::ResourceLimit {
                message: "resident inventory rejected".into(),
            })
        },
        |_, _| {
            decoded.set(decoded.get() + 1);
            Ok(())
        },
        |_| {
            consumed.set(consumed.get() + 1);
            Ok(())
        },
    )
    .unwrap_err();
    assert!(
        matches!(error, AnalysisPublicationError::ResourceLimit { ref message } if message == "resident inventory rejected")
    );
    assert_eq!(decoded.get(), 0);
    assert_eq!(consumed.get(), 0);
}

#[test]
fn admission_guard_covers_decode_and_consumer_then_releases_before_next_original() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let runs = root.path().join("runs");
    std::fs::create_dir(&runs).unwrap();
    for id in ["first", "second"] {
        std::fs::write(runs.join(format!("{id}.json")), b"supported").unwrap();
    }
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let scope = store.access(false).unwrap();
    let live = Cell::new(0);
    let consumed = Cell::new(0);
    visit_supported_artifacts_admitted(
        &scope,
        |_| {
            assert_eq!(live.get(), 0);
            live.set(1);
            Ok(LiveRecord(&live))
        },
        |_, _| {
            assert_eq!(live.get(), 1);
            Ok(())
        },
        |_| {
            assert_eq!(live.get(), 1);
            consumed.set(consumed.get() + 1);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(consumed.get(), 2);
    assert_eq!(live.get(), 0);
}
