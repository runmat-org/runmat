use super::*;

#[test]
fn dropped_pending_move_has_no_rename_effect_and_retires_turn() {
    let _lock = runmat_filesystem::provider_override_lock();
    let filesystem = Arc::new(MemoryFsProvider::with_current_dir("/workspace"));
    filesystem
        .write_project_path("/workspace/source.bin", b"unchanged")
        .expect("source");
    let cancellation = CancellationSource::new();
    let dropped = Rc::new(Cell::new(false));
    let pending = dropped.clone();
    let budget = crate::execution::CpuCooperationBudget::with_test_turn(
        crate::execution::CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        cancellation.token(),
        move || Box::pin(PendingTurn(pending.clone())),
    );
    let mut owner = runmat_filesystem::with_provider_override(filesystem.clone(), || {
        super::super::super::super::authority::TransferFilesystem::capture_with_budget(
            super::super::super::super::authority::TransferOperation::Move,
            budget,
        )
        .expect("capture selected move owner")
    });

    let mut call = Box::pin(super::super::super::single::move_path(
        &mut owner,
        "source.bin",
        "target.bin",
        false,
    ));
    assert!(matches!(
        call.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    drop(call);

    assert!(dropped.get());
    assert_eq!(
        filesystem
            .read_project_path("/workspace/source.bin")
            .unwrap(),
        b"unchanged"
    );
    assert!(filesystem
        .read_project_path("/workspace/target.bin")
        .is_err());
}

#[test]
fn dropped_pending_wildcard_preserves_completed_move_without_replay() {
    let _lock = runmat_filesystem::provider_override_lock();
    let filesystem = Arc::new(MemoryFsProvider::with_current_dir("/workspace"));
    filesystem
        .write_project_path("/workspace/a.log", b"first")
        .expect("first source");
    filesystem
        .write_project_path("/workspace/b.log", b"second")
        .expect("second source");
    filesystem
        .write_project_path("/workspace/moved/.keep", b"")
        .expect("destination");
    let cancellation = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let pending_dropped = Rc::new(Cell::new(false));
    let next_turn = turns.clone();
    let pending = pending_dropped.clone();
    let budget = crate::execution::CpuCooperationBudget::with_test_turn(
        crate::execution::CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        cancellation.token(),
        move || -> crate::context::RuntimeServiceFuture<
            Result<(), crate::execution::ExecutionServiceError>,
        > {
            let turn = next_turn.get();
            next_turn.set(turn + 1);
            // Two planning checkpoints and the first execution checkpoint
            // complete; the second execution checkpoint suspends.
            if turn < 3 {
                Box::pin(async { Ok(()) })
            } else {
                Box::pin(PendingTurn(pending.clone()))
            }
        },
    );
    let mut owner = runmat_filesystem::with_provider_override(filesystem.clone(), || {
        super::super::super::super::authority::TransferFilesystem::capture_with_budget(
            super::super::super::super::authority::TransferOperation::Move,
            budget,
        )
        .expect("capture selected move owner")
    });

    let mut call = Box::pin(super::super::super::wildcard::move_matches(
        &mut owner, "*.log", "moved", false,
    ));
    assert!(matches!(
        call.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    drop(call);

    assert!(pending_dropped.get());
    assert!(filesystem.read_project_path("/workspace/a.log").is_err());
    assert_eq!(
        filesystem
            .read_project_path("/workspace/moved/a.log")
            .unwrap(),
        b"first"
    );
    assert_eq!(
        filesystem.read_project_path("/workspace/b.log").unwrap(),
        b"second"
    );
    assert!(filesystem
        .read_project_path("/workspace/moved/b.log")
        .is_err());
}
