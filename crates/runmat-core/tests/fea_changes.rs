#![cfg(not(target_arch = "wasm32"))]
use futures::executor::block_on;
use runmat_core::RunMatSession;
use runmat_fea_source::{FeaChangeSetPublication, FeaChangeSetRequest, FeaSourceChange};
use runmat_filesystem::{FsProvider, MemoryFsProvider, SandboxFsProvider};
use runmat_runtime::analysis::LogicalSourcePath;
use std::{path::Path, sync::Arc};

fn logical(value: &str) -> LogicalSourcePath {
    LogicalSourcePath::new(value.to_owned()).unwrap()
}
fn proposal() -> FeaChangeSetRequest {
    FeaChangeSetRequest { base: None, changes: vec![
        FeaSourceChange::Rename { from: logical("supports.fea"), to: logical("shared.fea") },
        FeaSourceChange::Write { path: logical("main.fea"), source: "import model supports from \"./shared.fea\"\nmodel assembly { include model.supports }\n".into() },
    ] }
}
fn initial(provider: &dyn FsProvider) {
    block_on(provider.write(Path::new("/main.fea"), b"import model supports from \"./supports.fea\"\nmodel assembly { include model.supports }\n")).unwrap();
    block_on(provider.write(Path::new("/supports.fea"), b"model supports {}\n")).unwrap();
}

#[test]
fn workspace_relative_staging_does_not_confuse_a_real_project_directory() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/"));
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        for invalid in ["/bracket.fea", "../outside.fea", "project/../outside.fea"] {
            let request = serde_json::json!({ "base": null, "changes": [{ "type": "write", "path": invalid, "source": "model bracket {}\n" }] });
            assert!(serde_json::from_value::<FeaChangeSetRequest>(request).is_err());
        }
        for path in ["bracket.fea", "project/bracket.fea"] {
            let staged = block_on(session.stage_fea_changes(FeaChangeSetRequest {
                base: None,
                changes: vec![FeaSourceChange::Write {
                    path: logical(path),
                    source: "model bracket {}\n".into(),
                }],
            }))
            .unwrap();
            assert!(staged.publication_valid);
            assert!(provider.read_project_path(format!("/{path}")).is_err());
            assert!(matches!(
                block_on(session.publish_fea_changes(staged.id)).unwrap(),
                FeaChangeSetPublication::Published { .. }
            ));
            assert_eq!(
                provider.read_project_path(format!("/{path}")).unwrap(),
                b"model bracket {}\n"
            );
        }
    });
}

#[test]
fn durable_proposal_rechecks_after_restart_and_recovers_a_lost_rename_acknowledgement() {
    let _lock = runmat_filesystem::provider_override_lock();
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(SandboxFsProvider::new(root.path().to_path_buf()).unwrap());
    initial(provider.as_ref());
    let id = runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let staged = block_on(session.stage_fea_changes(proposal())).unwrap();
        assert!(staged.publication_valid);
        assert!(!root.path().join("shared.fea").exists());
        assert!(root.path().join("supports.fea").exists());
        staged.id
    });
    let restarted = Arc::new(SandboxFsProvider::new(root.path().to_path_buf()).unwrap());
    runmat_filesystem::with_provider_override(restarted.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let preview = block_on(session.inspect_fea_changes(id)).unwrap();
        assert!(preview.publication_valid);
        assert_eq!(preview.id, id);
        assert!(!root.path().join("shared.fea").exists());
        let published = block_on(session.publish_fea_changes(id)).unwrap();
        assert!(matches!(
            published,
            FeaChangeSetPublication::Published { .. }
        ));
        assert!(!root.path().join("supports.fea").exists());
        std::fs::write(root.path().join("main.fea"), b"model later_human_edit {}\n").unwrap();
        let mut recovered_session = RunMatSession::with_options(false, false).unwrap();
        let retry = block_on(recovered_session.publish_fea_changes(id)).unwrap();
        assert_eq!(
            serde_json::to_value(retry).unwrap(),
            serde_json::to_value(published).unwrap()
        );
        assert_eq!(
            std::fs::read(root.path().join("main.fea")).unwrap(),
            b"model later_human_edit {}\n"
        );
    });
}

#[test]
fn invalid_proposal_can_be_repaired_without_publishing_its_intermediate_source() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/"));
    initial(provider.as_ref());
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let bad = block_on(session.stage_fea_changes(FeaChangeSetRequest {
            base: None,
            changes: vec![FeaSourceChange::Remove {
                path: logical("supports.fea"),
            }],
        }))
        .unwrap();
        assert!(!bad.publication_valid);
        assert!(matches!(
            block_on(session.publish_fea_changes(bad.id)).unwrap(),
            FeaChangeSetPublication::Invalid { .. }
        ));
        assert!(provider.read_project_path("/supports.fea").is_ok());
        let mut repair = proposal();
        repair.base = Some(bad.id);
        let repaired = block_on(session.stage_fea_changes(repair)).unwrap();
        assert!(repaired.publication_valid);
        assert_ne!(bad.id, repaired.id);
        assert!(matches!(
            block_on(session.publish_fea_changes(repaired.id)).unwrap(),
            FeaChangeSetPublication::Published { .. }
        ));
        block_on(session.discard_fea_changes(bad.id)).unwrap();
        assert!(block_on(session.publish_fea_changes(bad.id)).is_err());
    });
}

#[test]
fn collaborator_edits_block_publication_and_silent_rebase() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/"));
    initial(provider.as_ref());
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let stage = block_on(session.stage_fea_changes(proposal())).unwrap();
        provider
            .write_project_path("/another.fea", b"model collaborator {}\n")
            .unwrap();
        assert!(matches!(
            block_on(session.publish_fea_changes(stage.id)).unwrap(),
            FeaChangeSetPublication::Conflict { .. }
        ));
        let mut repair = proposal();
        repair.base = Some(stage.id);
        assert!(block_on(session.stage_fea_changes(repair)).is_err());
        assert!(provider.read_project_path("/shared.fea").is_err());
        assert!(provider.read_project_path("/supports.fea").is_ok());
    });
}

#[test]
fn stage_identity_does_not_turn_modified_persisted_data_into_a_compiler_attestation() {
    let _lock = runmat_filesystem::provider_override_lock();
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(SandboxFsProvider::new(root.path().to_path_buf()).unwrap());
    initial(provider.as_ref());
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let stage = block_on(session.stage_fea_changes(proposal())).unwrap();
        let file = root
            .path()
            .join(".artifacts/fea-change-sets")
            .join(format!("{}.json", uuid::Uuid::from_bytes(stage.id.0)));
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
        value["publication"]["edits"][1]["bytes"] =
            serde_json::json!(b"not valid FEA source".to_vec());
        std::fs::write(file, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(matches!(
            block_on(session.publish_fea_changes(stage.id)).unwrap(),
            FeaChangeSetPublication::Invalid { .. }
        ));
        assert!(!root.path().join("shared.fea").exists());
        assert!(root.path().join("supports.fea").exists());
    });
}

#[test]
fn source_review_uses_the_staged_original_after_a_collaborator_changes_the_live_file() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/"));
    initial(provider.as_ref());
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let staged = block_on(session.stage_fea_changes(proposal())).unwrap();
        provider
            .write_project_path("/main.fea", b"model collaborator {}\n")
            .unwrap();
        assert!(block_on(session.inspect_fea_changes(staged.id))
            .unwrap_err()
            .contains("workspace changed"));
        let review = block_on(session.review_fea_changes(
            runmat_fea_source::FeaChangeSetReviewRequest {
                id: staged.id,
                change_index: 1,
                before_offset: 0,
                after_offset: 0,
            },
        ))
        .unwrap();
        assert!(review.before.unwrap().text.contains("supports.fea"));
        assert!(review.after.unwrap().text.contains("shared.fea"));
        assert_eq!(review.change_count, 2);
        assert!(matches!(
            block_on(session.publish_fea_changes(staged.id)).unwrap(),
            FeaChangeSetPublication::Conflict { .. }
        ));
        assert!(block_on(session.review_fea_changes(
            runmat_fea_source::FeaChangeSetReviewRequest {
                id: staged.id,
                change_index: 2,
                before_offset: 0,
                after_offset: 0,
            }
        ))
        .is_err());
    });
}
#[test]
fn source_review_is_bounded_and_preserves_multibyte_character_boundaries() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/"));
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let source = format!("model part {{}}\n// {}\n", "界".repeat(32_000));
        let staged = block_on(session.stage_fea_changes(FeaChangeSetRequest {
            base: None,
            changes: vec![FeaSourceChange::Write {
                path: logical("part.fea"),
                source: source.clone(),
            }],
        }))
        .unwrap();
        let mut offset = 0;
        let mut reconstructed = String::new();
        loop {
            let review = block_on(session.review_fea_changes(
                runmat_fea_source::FeaChangeSetReviewRequest {
                    id: staged.id,
                    change_index: 0,
                    before_offset: 0,
                    after_offset: offset,
                },
            ))
            .unwrap();
            assert!(review.before.is_none());
            let after = review.after.unwrap();
            assert!(after.text.len() <= 32 * 1024);
            assert_eq!(after.total_bytes, source.len() as u32);
            reconstructed.push_str(&after.text);
            match after.next_offset {
                Some(next) => offset = next,
                None => break,
            }
        }
        assert_eq!(reconstructed, source);
    });
}

#[test]
fn legacy_source_records_migrate_only_their_declared_identity_format() {
    let _lock = runmat_filesystem::provider_override_lock();
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(SandboxFsProvider::new(root.path().to_path_buf()).unwrap());
    initial(provider.as_ref());
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let stage = block_on(session.stage_fea_changes(proposal())).unwrap();
        let file = root
            .path()
            .join(".artifacts/fea-change-sets")
            .join(format!("{}.json", uuid::Uuid::from_bytes(stage.id.0)));
        let mut record: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
        record
            .as_object_mut()
            .unwrap()
            .remove("path_format_version");
        let originals = record["original_sources"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(path, bytes)| (format!("project/{path}"), bytes.clone()))
            .collect();
        record["original_sources"] = serde_json::Value::Object(originals);
        std::fs::write(&file, serde_json::to_vec(&record).unwrap()).unwrap();
        let review = block_on(session.review_fea_changes(
            runmat_fea_source::FeaChangeSetReviewRequest {
                id: stage.id,
                change_index: 0,
                before_offset: 0,
                after_offset: 0,
            },
        ))
        .unwrap();
        assert!(review.before.is_some());
        assert!(matches!(
            block_on(session.publish_fea_changes(stage.id)).unwrap(),
            FeaChangeSetPublication::Published { .. }
        ));
        assert!(root.path().join("shared.fea").exists());
    });
}

#[test]
fn inspect_retains_incomplete_draft_and_replaces_it_without_writing_source() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/"));
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let first = block_on(session.stage_fea_changes(FeaChangeSetRequest {
            base: None,
            changes: vec![FeaSourceChange::Write {
                path: logical("bracket.fea"),
                source: "model bracket {\n// Load still unresolved\n".into(),
            }],
        }))
        .unwrap();
        assert!(!first.publication_valid);
        let inspected = block_on(session.inspect_fea_changes(first.id)).unwrap();
        assert!(!inspected.publication_valid);
        assert_eq!(inspected.roots.len(), 1);
        assert!(provider.read_project_path("/bracket.fea").is_err());
        let second = block_on(session.stage_fea_changes(FeaChangeSetRequest {
            base: Some(first.id),
            changes: vec![FeaSourceChange::Write {
                path: logical("bracket.fea"),
                source: "model bracket {}\n".into(),
            }],
        }))
        .unwrap();
        let repaired = block_on(session.inspect_fea_changes(second.id)).unwrap();
        assert!(repaired.publication_valid);
        assert!(provider.read_project_path("/bracket.fea").is_err());
        assert!(
            !block_on(session.inspect_fea_changes(first.id))
                .unwrap()
                .publication_valid
        );
    });
}
