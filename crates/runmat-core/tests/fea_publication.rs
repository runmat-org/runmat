#![cfg(not(target_arch = "wasm32"))]
use futures::executor::block_on;
use runmat_core::RunMatSession;
use runmat_filesystem::{workspace::WorkspacePublicationOutcome, MemoryFsProvider};
use runmat_runtime::analysis::{FeaWorkspaceEdit, LogicalSourcePath};
use std::sync::Arc;

fn path(value: &str) -> LogicalSourcePath {
    LogicalSourcePath::new(value).unwrap()
}
fn setup() -> Arc<MemoryFsProvider> {
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/project"));
    provider.write_project_path("/project/main.fea", b"import model supports from \"./supports.fea\"\nmodel assembly { include model.supports }\n").unwrap();
    provider
        .write_project_path("/project/supports.fea", b"model supports {}\n")
        .unwrap();
    provider
}

#[test]
fn checked_refactor_is_staged_privately_then_published_as_one_revision() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = setup();
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let prepared = block_on(session.prepare_fea_workspace_publication(vec![
            FeaWorkspaceEdit::Rename { from: path("supports.fea"), to: path("shared.fea") },
            FeaWorkspaceEdit::Write { path: path("main.fea"), bytes: b"import model supports from \"./shared.fea\"\nmodel assembly { include model.supports }\n".to_vec() },
        ])).unwrap();
        assert!(prepared.check().publication_valid);
        assert_eq!(prepared.check().roots.len(), 2);
        assert!(provider.read_project_path("/project/shared.fea").is_err());
        assert_eq!(
            provider.read_project_path("/project/supports.fea").unwrap(),
            b"model supports {}\n"
        );
        let result = block_on(session.publish_fea_workspace(&prepared)).unwrap();
        assert!(matches!(
            result,
            WorkspacePublicationOutcome::Published { .. }
        ));
        assert!(provider.read_project_path("/project/supports.fea").is_err());
        assert_eq!(
            provider.read_project_path("/project/shared.fea").unwrap(),
            b"model supports {}\n"
        );
        assert_eq!(
            block_on(session.publish_fea_workspace(&prepared)).unwrap(),
            result
        );
    });
}

#[test]
fn invalid_candidate_and_intervening_dependency_changes_never_publish() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = setup();
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let invalid =
            block_on(
                session.prepare_fea_workspace_publication(vec![FeaWorkspaceEdit::Remove {
                    path: path("supports.fea"),
                }]),
            )
            .unwrap();
        assert!(!invalid.check().publication_valid);
        assert!(block_on(session.publish_fea_workspace(&invalid)).is_err());
        let valid =
            block_on(
                session.prepare_fea_workspace_publication(vec![FeaWorkspaceEdit::Write {
                    path: path("supports.fea"),
                    bytes: b"model supports {}\n".to_vec(),
                }]),
            )
            .unwrap();
        provider
            .write_project_path("/project/new_consumer.fea", b"model new_consumer {}\n")
            .unwrap();
        assert!(matches!(
            block_on(session.publish_fea_workspace(&valid)).unwrap(),
            WorkspacePublicationOutcome::Conflict { .. }
        ));
        assert_eq!(
            provider.read_project_path("/project/supports.fea").unwrap(),
            b"model supports {}\n"
        );
    });
}

#[test]
fn checked_handle_cannot_be_published_into_a_different_provider() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = setup();
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let prepared =
            block_on(
                session.prepare_fea_workspace_publication(vec![FeaWorkspaceEdit::Write {
                    path: path("supports.fea"),
                    bytes: b"model supports {}\n".to_vec(),
                }]),
            )
            .unwrap();
        let other = setup();
        let _changed = runmat_filesystem::replace_provider(other);
        assert!(block_on(session.publish_fea_workspace(&prepared))
            .unwrap_err()
            .contains("provider changed"));
    });
}

#[test]
fn checked_handle_rejects_a_changed_language_environment() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = setup();
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let prepared =
            block_on(
                session.prepare_fea_workspace_publication(vec![FeaWorkspaceEdit::Write {
                    path: path("supports.fea"),
                    bytes: b"model supports {}\n".to_vec(),
                }]),
            )
            .unwrap();
        let prior = session.compat_mode();
        session.set_compat_mode(if prior == runmat_parser::CompatMode::Strict {
            runmat_parser::CompatMode::RunMat
        } else {
            runmat_parser::CompatMode::Strict
        });
        assert!(block_on(session.publish_fea_workspace(&prepared))
            .unwrap_err()
            .contains("compiler environment"));
        session.set_compat_mode(prior);
        assert!(matches!(
            block_on(session.publish_fea_workspace(&prepared)).unwrap(),
            WorkspacePublicationOutcome::Published { .. }
        ));
    });
}
