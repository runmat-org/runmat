#![cfg(not(target_arch = "wasm32"))]
use runmat_core::{FeaWorkspaceCapture, FeaWorkspaceCheckRequest, RunMatSession};
use runmat_runtime::analysis::{
    FeaCheckStatus, FeaWorkspaceEdit, FeaWorkspaceFile, LogicalSourcePath,
};

fn path(value: &str) -> LogicalSourcePath {
    LogicalSourcePath::new(value).unwrap()
}
fn snapshot() -> FeaWorkspaceCapture {
    FeaWorkspaceCapture { files:vec![
        FeaWorkspaceFile { path:path("main.fea"), version:"working:1".into(), bytes:b"import model supports from \"./supports.fea\"\nmodel assembly { include model.supports }\n".to_vec() },
        FeaWorkspaceFile { path:path("supports.fea"), version:"working:2".into(), bytes:b"model supports {}\n".to_vec() },
    ] }
}

#[test]
fn candidate_checks_a_complete_module_move_without_changing_the_real_project() {
    let root = tempfile::tempdir().unwrap();
    let real = root.path().join("main.fea");
    std::fs::write(&real, "manually edited invalid source").unwrap();
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let result = futures::executor::block_on(session.check_fea_workspace(FeaWorkspaceCheckRequest {
        snapshot:snapshot(), dependencies:vec![], edits:vec![
            FeaWorkspaceEdit::Rename { from:path("supports.fea"), to:path("shared.fea") },
            FeaWorkspaceEdit::Write { path:path("main.fea"), bytes:b"import model supports from \"./shared.fea\"\nmodel assembly { include model.supports }\n".to_vec() },
        ],
    })).unwrap();
    assert!(result.publication_valid, "{:?}", result.roots);
    assert_eq!(result.roots.len(), 2);
    assert!(result
        .roots
        .iter()
        .all(|root| root.status == FeaCheckStatus::Valid));
    assert_eq!(
        std::fs::read_to_string(real).unwrap(),
        "manually edited invalid source"
    );
    assert!(!root.path().join("shared.fea").exists());
}

#[test]
fn deleted_dependency_and_invalid_consumers_block_publication() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("supports.fea"), "model supports {}\n").unwrap();
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let result =
        futures::executor::block_on(session.check_fea_workspace(FeaWorkspaceCheckRequest {
            snapshot: snapshot(),
            dependencies: vec![],
            edits: vec![FeaWorkspaceEdit::Remove {
                path: path("supports.fea"),
            }],
        }))
        .unwrap();
    assert!(!result.publication_valid);
    assert_ne!(result.roots[0].status, FeaCheckStatus::Valid);
    assert!(!result.roots[0].diagnostics.is_empty());
}

#[test]
fn unrelated_broken_draft_does_not_block_dependency_changes() {
    let mut capture = snapshot();
    capture.files.push(FeaWorkspaceFile {
        path: path("unrelated.fea"),
        version: "draft:1".into(),
        bytes: b"this is a broken manual draft".to_vec(),
    });
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let result =
        futures::executor::block_on(session.check_fea_workspace(FeaWorkspaceCheckRequest {
            snapshot: capture,
            dependencies: vec![],
            edits: vec![FeaWorkspaceEdit::Write {
                path: path("supports.fea"),
                bytes: b"model supports {}\n".to_vec(),
            }],
        }))
        .unwrap();
    assert!(result.publication_valid, "{:?}", result.roots);
    assert_eq!(result.roots.len(), 2);
    assert!(result
        .roots
        .iter()
        .any(|root| root.root_module == path("main.fea")));
    assert!(result
        .roots
        .iter()
        .all(|root| root.root_module != path("unrelated.fea")));
}

#[test]
fn configuration_is_admitted_separately_from_source_edits() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let error =
        futures::executor::block_on(session.check_fea_workspace(FeaWorkspaceCheckRequest {
            snapshot: snapshot(),
            dependencies: vec![],
            edits: vec![FeaWorkspaceEdit::Write {
                path: path("runmat.toml"),
                bytes: vec![],
            }],
        }))
        .unwrap_err();
    assert_eq!(
        error,
        "FEA source changes may only mutate admitted .fea/.m sources"
    );
}

#[test]
fn function_source_changes_check_all_importing_consumers() {
    let mut capture = snapshot();
    for name in ["first", "second"] {
        capture.files.push(FeaWorkspaceFile {
            path: path(&format!("{name}.fea")),
            version: format!("base:{name}"),
            bytes: b"import function pulse from \"./signals.m\"\n".to_vec(),
        });
    }
    capture.files.push(FeaWorkspaceFile {
        path: path("signals.m"),
        version: "base:signals".into(),
        bytes: b"function y = pulse(t)\ny = t;\nend\n".to_vec(),
    });
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let result =
        futures::executor::block_on(session.check_fea_workspace(FeaWorkspaceCheckRequest {
            snapshot: capture,
            dependencies: vec![],
            edits: vec![FeaWorkspaceEdit::Write {
                path: path("signals.m"),
                bytes: b"function y = pulse(t)\ny = 2*t;\nend\n".to_vec(),
            }],
        }))
        .unwrap();
    assert_eq!(
        result
            .roots
            .iter()
            .map(|root| root.root_module.clone())
            .collect::<Vec<_>>(),
        vec![path("first.fea"), path("second.fea")]
    );
    // There is no admitted package/function graph in this session, so checking
    // must report unavailable facts rather than execute or invent the function.
    assert!(!result.publication_valid);
}

#[test]
fn working_check_uses_native_drafts_and_imports_without_writing_the_provider() {
    use runmat_filesystem::{FsProvider, MemoryFsProvider};
    let _guard = runmat_filesystem::provider_override_lock();
    let provider = std::sync::Arc::new(MemoryFsProvider::with_current_dir("/project"));
    futures::executor::block_on(provider.write(
        std::path::Path::new("/project/main.fea"),
        b"broken disk content",
    ))
    .unwrap();
    futures::executor::block_on(provider.write(
        std::path::Path::new("/project/supports.fea"),
        b"model supports {}\n",
    ))
    .unwrap();
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let result = futures::executor::block_on(session.check_fea_working_source(
            runmat_core::FeaWorkingCheckRequest {
                root: path("main.fea"),
                overlays: vec![runmat_core::FeaWorkingSource {
                    path: path("main.fea"),
                    bytes: snapshot().files[0].bytes.clone(),
                }],
            },
        ))
        .unwrap();
        assert_eq!(
            result.status,
            FeaCheckStatus::Valid,
            "{:?}",
            result.diagnostics
        );
        assert_eq!(result.root_module, path("main.fea"));
    });
    assert_eq!(
        futures::executor::block_on(provider.read(std::path::Path::new("/project/main.fea")))
            .unwrap(),
        b"broken disk content"
    );
}

#[test]
fn working_check_rejects_duplicate_overlays_and_root_as_a_source_file() {
    use runmat_filesystem::MemoryFsProvider;
    let _guard = runmat_filesystem::provider_override_lock();
    let provider = std::sync::Arc::new(MemoryFsProvider::with_current_dir("/project"));
    runmat_filesystem::with_provider_override(provider, || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let overlay = runmat_core::FeaWorkingSource {
            path: path("main.fea"),
            bytes: b"model assembly {}\n".to_vec(),
        };
        let error = futures::executor::block_on(session.check_fea_working_source(
            runmat_core::FeaWorkingCheckRequest {
                root: path("main.fea"),
                overlays: vec![overlay.clone(), overlay],
            },
        ))
        .unwrap_err();
        assert!(error.contains("duplicate working editor overlay"));
        let error = futures::executor::block_on(session.check_fea_working_source(
            runmat_core::FeaWorkingCheckRequest {
                root: path("main.fea"),
                overlays: vec![runmat_core::FeaWorkingSource {
                    path: LogicalSourcePath::root(),
                    bytes: vec![],
                }],
            },
        ))
        .unwrap_err();
        assert!(error.contains("must name a file"), "{error}");
    });
}

#[test]
fn checked_candidate_retains_exact_source_for_preview_and_never_reads_live_fallbacks() {
    let _lock = runmat_filesystem::provider_override_lock();
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let result =
        futures::executor::block_on(session.check_fea_workspace(FeaWorkspaceCheckRequest {
            snapshot: snapshot(),
            dependencies: vec![],
            edits: vec![],
        }))
        .unwrap();
    let report = result
        .roots
        .iter()
        .find(|report| report.root_module == path("main.fea"))
        .unwrap();
    let source = session.checked_fea_source(report.revision).unwrap();
    let captured = source
        .captured_sources()
        .expect("candidate preview owns its capture");
    let main = source.physical_path();
    assert_eq!(
        futures::executor::block_on(source.read_source(main)).unwrap(),
        captured[main]
    );
    let missing = source.source_root.join("not-captured.m");
    assert!(futures::executor::block_on(source.read_source(&missing)).is_err());
    assert_eq!(source.check_result(), *report);
    futures::executor::block_on(session.clear_project_handoff()).expect("retire project handoff");
    assert!(session.checked_fea_source(report.revision).is_err());
}

#[test]
fn checked_preview_revision_expires_instead_of_silently_rechecking_current_source() {
    let _lock = runmat_filesystem::provider_override_lock();
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let mut first = None;
    for index in 0..6 {
        let capture = FeaWorkspaceCapture {
            files: vec![FeaWorkspaceFile {
                path: path("main.fea"),
                version: format!("draft:{index}"),
                bytes: format!("model definition_{index} {{}}\n").into_bytes(),
            }],
        };
        let report =
            futures::executor::block_on(session.check_fea_workspace(FeaWorkspaceCheckRequest {
                snapshot: capture,
                dependencies: vec![],
                edits: vec![],
            }))
            .unwrap()
            .roots
            .remove(0);
        if index == 0 {
            first = Some(report.revision);
        }
        assert!(session.checked_fea_source(report.revision).is_ok());
    }
    assert!(session.checked_fea_source(first.unwrap()).is_err());
}

#[test]
fn retained_preview_and_queries_cannot_cross_a_provider_mount_change() {
    let _lock = runmat_filesystem::provider_override_lock();
    let first = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir("/"));
    let second = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir("/"));
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let revision = runmat_filesystem::with_provider_override(first.clone(), || {
        futures::executor::block_on(session.check_fea_workspace(FeaWorkspaceCheckRequest {
            snapshot: snapshot(),
            dependencies: vec![],
            edits: vec![],
        }))
        .unwrap()
        .roots
        .remove(0)
        .revision
    });
    runmat_filesystem::with_provider_override(second, || {
        assert!(session.checked_fea_source(revision).is_err());
    });
    runmat_filesystem::with_provider_override(first, || {
        assert!(session.checked_fea_source(revision).is_ok());
    });
}

#[test]
fn working_check_retains_only_the_requested_root_and_ignores_unrelated_broken_studies() {
    use runmat_filesystem::{FsProvider, MemoryFsProvider};
    let _guard = runmat_filesystem::provider_override_lock();
    let provider = std::sync::Arc::new(MemoryFsProvider::with_current_dir("/project"));
    futures::executor::block_on(provider.write(
        std::path::Path::new("/project/aaa.fea"),
        b"model assembly {}\n",
    ))
    .unwrap();
    for index in 0..8 {
        futures::executor::block_on(provider.write(
            std::path::Path::new(&format!("/project/other{index}.fea")),
            b"model other {}\n",
        ))
        .unwrap();
    }
    futures::executor::block_on(provider.write(
        std::path::Path::new("/project/unrelated.fea"),
        b"broken source",
    ))
    .unwrap();
    runmat_filesystem::with_provider_override(provider, || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let result = futures::executor::block_on(session.check_fea_working_source(
            runmat_core::FeaWorkingCheckRequest {
                root: path("aaa.fea"),
                overlays: vec![],
            },
        ))
        .unwrap();
        assert_eq!(result.status, FeaCheckStatus::Valid);
        let source = session.checked_fea_source(result.revision).unwrap();
        assert_eq!(source.checked().root_module, path("aaa.fea"));
        assert_eq!(
            source.physical_path(),
            std::path::Path::new("/project/aaa.fea")
        );
    });
}
