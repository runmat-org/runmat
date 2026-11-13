use super::*;

#[test]
fn revoked_lease_after_first_chunk_prevents_next_provider_effect() {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir("/selected");
    memory
        .write_project_path("/selected/source.bin", &vec![9u8; 32 * 1024])
        .expect("source");
    let writes = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(RevokingProvider {
        memory: memory.clone(),
        valid: Arc::new(AtomicBool::new(true)),
        writes: writes.clone(),
        pending_rename_dropped: None,
    });
    let mut captured = runmat_filesystem::with_provider_override(provider, || {
        super::super::super::TransferFilesystem::capture(
            super::super::super::super::authority::TransferOperation::Copy,
            None,
        )
        .expect("capture lease")
    });
    let outcome = futures::executor::block_on(super::super::super::single::copy(
        &mut captured,
        "source.bin",
        "target.bin",
        false,
    ));
    assert_eq!(outcome.status(), 0.0);
    assert_eq!(writes.load(Ordering::SeqCst), 1);
    assert_eq!(
        memory
            .read_project_path("/selected/target.bin")
            .unwrap()
            .len(),
        16 * 1024
    );
}

#[test]
fn move_reports_resource_lease_retirement_after_provider_rename() {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir("/selected");
    memory
        .write_project_path("/selected/source.bin", b"source")
        .expect("source");
    let provider = Arc::new(RevokingProvider {
        memory: memory.clone(),
        valid: Arc::new(AtomicBool::new(true)),
        writes: Arc::new(AtomicUsize::new(0)),
        pending_rename_dropped: None,
    });
    let outcome = runmat_filesystem::with_provider_override(provider, || {
        futures::executor::block_on(super::super::super::super::movefile::execute::evaluate(
            crate::call::FunctionArgumentView::from_slice(&[
                runmat_value::Value::from("source.bin"),
                runmat_value::Value::from("target.bin"),
            ]),
            None,
        ))
        .expect("lease retirement is a filesystem status result")
    });

    assert_eq!(outcome.status(), 0.0);
    assert!(memory.read_project_path("/selected/source.bin").is_err());
    assert_eq!(
        memory.read_project_path("/selected/target.bin").unwrap(),
        b"source"
    );
}

#[test]
fn selected_move_builtin_drops_pending_provider_without_replay() {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir("/selected");
    memory
        .write_project_path("/selected/source.bin", b"source")
        .expect("source");
    let pending_rename_dropped = Arc::new(AtomicBool::new(false));
    let provider = Arc::new(RevokingProvider {
        memory: memory.clone(),
        valid: Arc::new(AtomicBool::new(true)),
        writes: Arc::new(AtomicUsize::new(0)),
        pending_rename_dropped: Some(pending_rename_dropped.clone()),
    });
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    runmat_filesystem::with_provider_override(provider, || {
        let mut call = Box::pin(super::super::super::super::movefile::movefile_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![
                runmat_value::Value::from("source.bin"),
                runmat_value::Value::from("target.bin"),
            ]),
        ));
        assert!(matches!(
            call.as_mut()
                .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
            Poll::Pending
        ));
        drop(call);
    });

    assert!(pending_rename_dropped.load(Ordering::SeqCst));
    assert_eq!(
        memory.read_project_path("/selected/source.bin").unwrap(),
        b"source"
    );
    assert!(memory.read_project_path("/selected/target.bin").is_err());
}

#[test]
fn selected_move_builtin_cancels_pending_provider_with_typed_error() {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir("/selected");
    memory
        .write_project_path("/selected/source.bin", b"source")
        .expect("source");
    let pending_rename_dropped = Arc::new(AtomicBool::new(false));
    let provider = Arc::new(RevokingProvider {
        memory: memory.clone(),
        valid: Arc::new(AtomicBool::new(true)),
        writes: Arc::new(AtomicUsize::new(0)),
        pending_rename_dropped: Some(pending_rename_dropped.clone()),
    });
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(cancellation.token());

    let error = runmat_filesystem::with_provider_override(provider, || {
        let mut call = Box::pin(super::super::super::super::movefile::movefile_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![
                runmat_value::Value::from("source.bin"),
                runmat_value::Value::from("target.bin"),
            ]),
        ));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
        cancellation.cancel(runmat_execution::CancellationReason::User);
        match call.as_mut().poll(&mut context) {
            Poll::Ready(Err(error)) => error,
            other => panic!("cancelled move must finish with an error, got {other:?}"),
        }
    });

    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(pending_rename_dropped.load(Ordering::SeqCst));
    assert_eq!(
        memory.read_project_path("/selected/source.bin").unwrap(),
        b"source"
    );
    assert!(memory.read_project_path("/selected/target.bin").is_err());
}
