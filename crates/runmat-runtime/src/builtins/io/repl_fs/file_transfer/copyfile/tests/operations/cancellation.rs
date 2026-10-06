use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_filesystem::MemoryFsProvider;
use runmat_value::Value;
use std::rc::Rc;
use std::sync::Arc;

#[test]
fn cancelled_copy_rejects_before_provider_effects_with_typed_cause() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let filesystem = Arc::new(MemoryFsProvider::with_current_dir("/workspace"));
    filesystem
        .write_project_path("/workspace/source.bin", b"unchanged")
        .expect("source");
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(cancellation.token());
    let error = runmat_filesystem::with_provider_override(filesystem.clone(), || {
        futures::executor::block_on(super::super::super::copyfile_builtin(
            Some(runtime),
            vec![Value::from("source.bin"), Value::from("target.bin")],
        ))
        .expect_err("cancelled copy")
    });
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(error.source.as_deref().is_some_and(|cause| matches!(
        cause.downcast_ref::<crate::execution::ExecutionServiceError>(),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    )));
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
