use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_filesystem::MemoryFsProvider;
use runmat_value::Value;
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};

struct PendingTurn(Rc<Cell<bool>>);

impl Future for PendingTurn {
    type Output = Result<(), crate::execution::ExecutionServiceError>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

#[test]
fn cancelled_copy_rejects_before_provider_effects_with_typed_cause() {
    let _lock = runmat_filesystem::provider_override_lock();
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
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("source.bin"),
                Value::from("target.bin"),
            ]),
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

#[test]
fn pending_copy_byte_checkpoint_drops_without_replaying_completed_chunk() {
    let _lock = runmat_filesystem::provider_override_lock();
    let filesystem = Arc::new(MemoryFsProvider::with_current_dir("/workspace"));
    filesystem
        .write_project_path("/workspace/source.bin", &vec![7u8; 32 * 1024])
        .expect("source");
    let cancellation = CancellationSource::new();
    let dropped = Rc::new(Cell::new(false));
    let pending = dropped.clone();
    let budget = crate::execution::CpuCooperationBudget::with_test_turn(
        crate::execution::CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(16 * 1024).unwrap()),
        cancellation.token(),
        move || Box::pin(PendingTurn(pending.clone())),
    );
    let mut owner = runmat_filesystem::with_provider_override(filesystem.clone(), || {
        super::super::super::super::authority::TransferFilesystem::capture_with_budget(
            super::super::super::super::authority::TransferOperation::Copy,
            budget,
        )
        .expect("capture copy owner")
    });

    let mut call = Box::pin(super::super::super::single::copy(
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
            .read_project_path("/workspace/target.bin")
            .expect("completed first chunk"),
        vec![7u8; 16 * 1024]
    );
    assert_eq!(
        filesystem
            .read_project_path("/workspace/source.bin")
            .expect("unchanged source")
            .len(),
        32 * 1024
    );
}
