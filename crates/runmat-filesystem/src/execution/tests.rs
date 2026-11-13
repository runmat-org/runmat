use super::*;
use crate::{FsProvider, MemoryFsProvider, SandboxFsProvider};
use futures::executor::block_on;
use std::{
    path::Path,
    sync::{Arc, Barrier},
};

fn work() -> ExecutionWorkId {
    ExecutionWorkId {
        session_id: SessionId("session".into()),
        turn_id: TurnId("turn".into()),
    }
}
fn acquire(provider: &dyn FsProvider, owner: u8) -> ExecutionLeaseOutcome {
    block_on(provider.transition_execution(
        &work(),
        ExecutionLeaseCommand::Acquire {
            owner: ExecutionOwnerId([owner; 16]),
            duration_ms: 30_000,
        },
    ))
    .unwrap()
}
fn lease(outcome: ExecutionLeaseOutcome) -> ExecutionLease {
    match outcome {
        ExecutionLeaseOutcome::Acquired { lease } => lease,
        other => panic!("unexpected {other:?}"),
    }
}

fn race(open: impl Fn() -> Arc<dyn FsProvider>, workers: usize) {
    let barrier = Arc::new(Barrier::new(workers));
    let handles: Vec<_> = (0..workers)
        .map(|i| {
            let provider = open();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                acquire(provider.as_ref(), i as u8 + 1)
            })
        })
        .collect();
    let outcomes: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(
        outcomes
            .iter()
            .filter(|value| matches!(value, ExecutionLeaseOutcome::Acquired { .. }))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|value| matches!(value, ExecutionLeaseOutcome::Busy { .. }))
            .count(),
        workers - 1
    );
}

#[test]
fn memory_authority_arbitrates_under_shared_provider_lock() {
    let provider = Arc::new(MemoryFsProvider::new());
    race(|| provider.clone(), 16);
}

#[test]
fn native_authority_arbitrates_across_independently_opened_mounts() {
    let root = tempfile::tempdir().unwrap();
    race(
        || Arc::new(SandboxFsProvider::new(root.path().to_path_buf()).unwrap()),
        16,
    );
}

#[test]
fn native_reopen_preserves_fences_and_terminal_ownership() {
    let root = tempfile::tempdir().unwrap();
    let open = || SandboxFsProvider::new(root.path().to_path_buf()).unwrap();
    let first = lease(acquire(&open(), 1));
    assert_eq!(lease(acquire(&open(), 1)), first);
    assert_eq!(
        block_on(open().transition_execution(
            &work(),
            ExecutionLeaseCommand::Release {
                lease: first.clone()
            }
        ))
        .unwrap(),
        ExecutionLeaseOutcome::Released
    );
    let second = lease(acquire(&open(), 2));
    assert!(second.fence.0 > first.fence.0);
    assert_eq!(
        block_on(open().transition_execution(
            &work(),
            ExecutionLeaseCommand::Renew {
                lease: first,
                duration_ms: 30_000
            }
        ))
        .unwrap(),
        ExecutionLeaseOutcome::Lost
    );
    assert_eq!(
        block_on(
            open().transition_execution(&work(), ExecutionLeaseCommand::Complete { lease: second })
        )
        .unwrap(),
        ExecutionLeaseOutcome::Completed
    );
    assert_eq!(acquire(&open(), 3), ExecutionLeaseOutcome::Finished);
}

#[test]
fn native_records_are_reserved_and_invalid_records_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let provider = SandboxFsProvider::new(root.path().to_path_buf()).unwrap();
    let held = lease(acquire(&provider, 1));
    let relative = format!(
        "/.artifacts/workspace-publications/executions/{}.json",
        work_key(&work()).unwrap()
    );
    let error = block_on(provider.write(Path::new(&relative), b"{}")).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    let error = block_on(provider.remove_dir_all(Path::new("/.artifacts"))).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    std::fs::write(root.path().join(relative.trim_start_matches('/')), b"{").unwrap();
    assert!(block_on(provider.transition_execution(
        &work(),
        ExecutionLeaseCommand::Renew {
            lease: held,
            duration_ms: 30_000
        }
    ))
    .is_err());
    assert!(block_on(provider.transition_execution(
        &work(),
        ExecutionLeaseCommand::Acquire {
            owner: ExecutionOwnerId([2; 16]),
            duration_ms: 30_000
        }
    ))
    .is_err());
}

#[test]
fn authority_scope_is_the_project_mount_not_the_filename() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let a = SandboxFsProvider::new(first.path().to_path_buf()).unwrap();
    let b = SandboxFsProvider::new(second.path().to_path_buf()).unwrap();
    let first = lease(acquire(&a, 1));
    let second = lease(acquire(&b, 2));
    assert_eq!(first.fence, second.fence);
    assert_ne!(first.owner, second.owner);
    assert_eq!(
        block_on(a.transition_execution(&work(), ExecutionLeaseCommand::Release { lease: second }))
            .unwrap(),
        ExecutionLeaseOutcome::Lost
    );
}

fn publication(provider: &dyn FsProvider) -> crate::workspace::WorkspacePublicationRequest {
    use crate::workspace::*;
    let captured = block_on(provider.capture_workspace(WorkspaceCaptureRequest {
        root: "/".into(),
        excluded_subtrees: vec![".artifacts".into()],
        max_files: 100,
        max_bytes: 1024,
    }))
    .unwrap();
    WorkspacePublicationRequest {
        id: WorkspacePublicationId::new(),
        root: captured.root,
        excluded_subtrees: captured.excluded_subtrees,
        expected_revision: captured.revision,
        edits: vec![WorkspaceEdit::Write {
            path: "study.fea".into(),
            bytes: b"checked candidate".to_vec(),
        }],
    }
}

fn exercise_fenced_publication(provider: &dyn FsProvider) {
    block_on(provider.write(Path::new("/study.fea"), b"previous")).unwrap();
    let first = lease(acquire(provider, 1));
    let request = publication(provider);
    block_on(provider.transition_execution(
        &work(),
        ExecutionLeaseCommand::Release {
            lease: first.clone(),
        },
    ))
    .unwrap();
    let second = lease(acquire(provider, 2));
    assert_eq!(
        block_on(provider.publish_workspace_fenced(request.clone(), &first)).unwrap(),
        FencedWorkspacePublicationOutcome::LeaseLost
    );
    assert_eq!(
        block_on(provider.read(Path::new("/study.fea"))).unwrap(),
        b"previous"
    );
    let result = block_on(provider.publish_workspace_fenced(request.clone(), &second)).unwrap();
    assert!(matches!(
        result,
        FencedWorkspacePublicationOutcome::Publication {
            outcome: crate::workspace::WorkspacePublicationOutcome::Published { .. }
        }
    ));
    assert_eq!(
        block_on(provider.read(Path::new("/study.fea"))).unwrap(),
        b"checked candidate"
    );
    block_on(provider.transition_execution(
        &work(),
        ExecutionLeaseCommand::Complete {
            lease: second.clone(),
        },
    ))
    .unwrap();
    // A lost acknowledgement can recover the exact receipt after completion;
    // this performs no effect and never permits a new stale-owner publication.
    assert_eq!(
        block_on(provider.publish_workspace_fenced(request, &second)).unwrap(),
        result
    );
    let mut next = publication(provider);
    next.edits = vec![crate::workspace::WorkspaceEdit::Remove {
        path: "study.fea".into(),
    }];
    assert_eq!(
        block_on(provider.publish_workspace_fenced(next, &second)).unwrap(),
        FencedWorkspacePublicationOutcome::LeaseLost
    );
    assert_eq!(
        block_on(provider.read(Path::new("/study.fea"))).unwrap(),
        b"checked candidate"
    );
}

#[test]
fn memory_publication_checks_ownership_inside_the_atomic_effect() {
    exercise_fenced_publication(&MemoryFsProvider::new());
}

#[test]
fn native_publication_checks_ownership_inside_the_durable_effect() {
    let root = tempfile::tempdir().unwrap();
    exercise_fenced_publication(&SandboxFsProvider::new(root.path().to_path_buf()).unwrap());
}

#[test]
fn native_crash_preparation_does_not_override_acknowledged_ownership() {
    let root = tempfile::tempdir().unwrap();
    let open = || SandboxFsProvider::new(root.path().to_path_buf()).unwrap();
    let first = lease(acquire(&open(), 1));
    let pending = root
        .path()
        .join(".artifacts/workspace-publications/executions")
        .join(format!("{}.pending", work_key(&work()).unwrap()));
    std::fs::write(&pending, b"unacknowledged partial preparation").unwrap();
    assert_eq!(lease(acquire(&open(), 1)), first);
    assert!(!pending.exists());
    assert!(matches!(
        acquire(&open(), 2),
        ExecutionLeaseOutcome::Busy { .. }
    ));
}

#[test]
fn record_budget_bounds_new_work_without_blocking_an_existing_owner() {
    let memory = MemoryFsProvider::new();
    let first = lease(acquire(&memory, 1));
    for index in 1..MAX_EXECUTION_RECORDS {
        let work = ExecutionWorkId {
            session_id: SessionId("session".into()),
            turn_id: TurnId(format!("work-{index}")),
        };
        block_on(memory.transition_execution(
            &work,
            ExecutionLeaseCommand::Acquire {
                owner: ExecutionOwnerId([1; 16]),
                duration_ms: 30_000,
            },
        ))
        .unwrap();
    }
    let extra = ExecutionWorkId {
        session_id: SessionId("session".into()),
        turn_id: TurnId("extra".into()),
    };
    assert!(block_on(memory.transition_execution(
        &extra,
        ExecutionLeaseCommand::Acquire {
            owner: ExecutionOwnerId([1; 16]),
            duration_ms: 30_000,
        }
    ))
    .is_err());
    assert_eq!(lease(acquire(&memory, 1)), first);

    let root = tempfile::tempdir().unwrap();
    let native = SandboxFsProvider::new(root.path().to_path_buf()).unwrap();
    let first = lease(acquire(&native, 1));
    let records = root
        .path()
        .join(".artifacts/workspace-publications/executions");
    for index in 1..MAX_EXECUTION_RECORDS {
        // Foreign/corrupt files must also consume the bounded journal budget.
        std::fs::write(records.join(format!("foreign-{index}.json")), b"{}").unwrap();
    }
    assert!(block_on(native.transition_execution(
        &extra,
        ExecutionLeaseCommand::Acquire {
            owner: ExecutionOwnerId([1; 16]),
            duration_ms: 30_000,
        }
    ))
    .is_err());
    assert_eq!(lease(acquire(&native, 1)), first);
}
