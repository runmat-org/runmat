#![cfg(not(target_arch = "wasm32"))]
use futures::executor::block_on;
use runmat_filesystem::{workspace::*, FsProvider, MemoryFsProvider};
use std::{
    path::Path,
    sync::{Arc, Barrier},
    thread,
};

fn capture(provider: &MemoryFsProvider) -> WorkspaceSnapshot {
    block_on(provider.capture_workspace(WorkspaceCaptureRequest {
        root: "/project".into(),
        excluded_subtrees: vec![".artifacts".into()],
        max_files: 10_000,
        max_bytes: 96 * 1024 * 1024,
    }))
    .unwrap()
}

fn request(provider: &MemoryFsProvider, edits: Vec<WorkspaceEdit>) -> WorkspacePublicationRequest {
    WorkspacePublicationRequest {
        id: WorkspacePublicationId::new(),
        root: "/project".into(),
        excluded_subtrees: vec![".artifacts".into()],
        expected_revision: capture(provider).revision,
        edits,
    }
}

fn write(path: &str, value: &[u8]) -> WorkspaceEdit {
    WorkspaceEdit::Write {
        path: path.into(),
        bytes: value.to_vec(),
    }
}

#[test]
fn complete_refactor_is_visible_together_and_retry_is_idempotent_after_later_edits() {
    let provider = MemoryFsProvider::with_current_dir("/project");
    provider
        .write_project_path("/project/main.fea", b"old import")
        .unwrap();
    provider
        .write_project_path("/project/materials.fea", b"steel")
        .unwrap();
    let input = request(
        &provider,
        vec![
            WorkspaceEdit::Rename {
                from: "materials.fea".into(),
                to: "shared/materials.fea".into(),
            },
            write("main.fea", b"new import"),
        ],
    );
    let first = block_on(provider.publish_workspace(input.clone())).unwrap();
    let WorkspacePublicationOutcome::Published { receipt } = &first else {
        panic!("publication")
    };
    let next = capture(&provider);
    assert_eq!(next.revision, receipt.revision);
    assert_eq!(next.files.len(), 2);
    assert_eq!(
        provider.read_project_path("/project/main.fea").unwrap(),
        b"new import"
    );
    assert!(provider
        .read_project_path("/project/materials.fea")
        .is_err());
    assert_eq!(
        provider
            .read_project_path("/project/shared/materials.fea")
            .unwrap(),
        b"steel"
    );
    provider
        .write_project_path("/project/main.fea", b"human edit after publication")
        .unwrap();
    assert_eq!(
        block_on(provider.publish_workspace(input.clone())).unwrap(),
        first
    );
    assert_eq!(
        provider.read_project_path("/project/main.fea").unwrap(),
        b"human edit after publication"
    );
    let mut altered = input;
    altered.edits.push(write("other.fea", b"other"));
    assert_eq!(
        block_on(provider.publish_workspace(altered)).unwrap(),
        WorkspacePublicationOutcome::IdReused
    );
}

#[test]
fn dependency_edits_new_files_and_rights_invalidate_the_capture() {
    for change in 0..3 {
        let provider = MemoryFsProvider::with_current_dir("/project");
        provider
            .write_project_path("/project/dependency.m", b"pure function")
            .unwrap();
        let input = request(&provider, vec![write("main.fea", b"new study")]);
        match change {
            0 => provider
                .write_project_path("/project/dependency.m", b"changed function")
                .unwrap(),
            1 => provider
                .write_project_path("/project/new_consumer.fea", b"consumer")
                .unwrap(),
            _ => block_on(provider.set_readonly(Path::new("/project/dependency.m"), true)).unwrap(),
        }
        assert!(matches!(
            block_on(provider.publish_workspace(input)).unwrap(),
            WorkspacePublicationOutcome::Conflict { .. }
        ));
        assert!(provider.read_project_path("/project/main.fea").is_err());
    }
}

#[test]
fn invalid_late_edit_cannot_leave_earlier_edits_or_parent_directories() {
    let provider = MemoryFsProvider::with_current_dir("/project");
    provider
        .write_project_path("/project/locked.fea", b"protected")
        .unwrap();
    block_on(provider.set_readonly(Path::new("/project/locked.fea"), true)).unwrap();
    let before = capture(&provider);
    let input = request(
        &provider,
        vec![
            write("new/nested.fea", b"staged"),
            write("locked.fea", b"overwrite"),
        ],
    );
    assert_eq!(
        block_on(provider.publish_workspace(input))
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::PermissionDenied
    );
    assert_eq!(capture(&provider), before);
    assert!(provider.metadata_project_path("/project/new").is_err());
    for path in [
        "../outside.fea",
        "/outside.fea",
        "nested/../outside.fea",
        "",
    ] {
        assert!(block_on(
            provider.publish_workspace(request(&provider, vec![write(path, b"escape")]))
        )
        .is_err());
        assert_eq!(capture(&provider), before);
    }
}

#[test]
fn competing_hosts_have_one_winner_and_readers_never_observe_half_a_publication() {
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/project"));
    provider.write_project_path("/project/a.fea", &[0]).unwrap();
    provider.write_project_path("/project/b.fea", &[0]).unwrap();
    let initial = capture(&provider).revision;
    let barrier = Arc::new(Barrier::new(13));
    let hosts = (1..=12)
        .map(|value| {
            let provider = provider.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                block_on(provider.publish_workspace(WorkspacePublicationRequest {
                    id: WorkspacePublicationId::new(),
                    root: "/project".into(),
                    expected_revision: initial,
                    excluded_subtrees: vec![".artifacts".into()],
                    edits: vec![write("a.fea", &[value]), write("b.fea", &[value])],
                }))
                .unwrap()
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    for _ in 0..100 {
        let snapshot = capture(&provider);
        assert_eq!(snapshot.files[0].bytes, snapshot.files[1].bytes);
    }
    let winners = hosts
        .into_iter()
        .map(|host| host.join().unwrap())
        .filter(|outcome| matches!(outcome, WorkspacePublicationOutcome::Published { .. }))
        .count();
    assert_eq!(winners, 1);
}

#[test]
fn captures_are_bounded_and_immutable_after_provider_writes() {
    let provider = MemoryFsProvider::with_current_dir("/project");
    provider
        .write_project_path("/project/main.fea", b"original")
        .unwrap();
    let before = capture(&provider);
    provider
        .write_project_path("/project/main.fea", b"new")
        .unwrap();
    assert_eq!(before.files[0].bytes, b"original");
    for (files, bytes) in [(0, 100), (10, 1)] {
        assert!(
            block_on(provider.capture_workspace(WorkspaceCaptureRequest {
                root: "/project".into(),
                excluded_subtrees: vec![],
                max_files: files,
                max_bytes: bytes,
            }))
            .is_err()
        );
    }
}

#[test]
fn private_journals_do_not_conflict_with_source_and_cannot_be_mutated_by_the_batch() {
    let provider = MemoryFsProvider::with_current_dir("/project");
    provider
        .write_project_path("/project/main.fea", b"old")
        .unwrap();
    let input = request(&provider, vec![write("main.fea", b"new")]);
    provider
        .write_project_path("/project/.artifacts/agent/journal", b"durable preparation")
        .unwrap();
    assert!(matches!(
        block_on(provider.publish_workspace(input)).unwrap(),
        WorkspacePublicationOutcome::Published { .. }
    ));
    assert!(block_on(provider.publish_workspace(request(
        &provider,
        vec![write(".artifacts/injected.fea", b"hidden")]
    )))
    .is_err());
    assert!(!capture(&provider)
        .files
        .iter()
        .any(|file| file.path.starts_with(".artifacts")));
}
