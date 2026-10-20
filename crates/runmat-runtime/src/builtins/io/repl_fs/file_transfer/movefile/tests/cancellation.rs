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
fn cancelled_move_rejects_before_provider_effects_with_typed_cause() {
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
        futures::executor::block_on(super::super::movefile_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("source.bin"),
                Value::from("target.bin"),
            ]),
        ))
        .expect_err("cancelled move")
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
fn captured_move_owner_survives_global_provider_swap() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
    selected
        .write_project_path("/selected/source.bin", b"selected")
        .expect("selected source");
    let other = Arc::new(MemoryFsProvider::with_current_dir("/other"));
    other
        .write_project_path("/selected/source.bin", b"wrong")
        .expect("other source");
    let mut captured = runmat_filesystem::with_provider_override(selected.clone(), || {
        super::super::super::authority::TransferFilesystem::capture(
            super::super::super::authority::TransferOperation::Move,
            None,
        )
        .expect("capture selected move owner")
    });
    let _replacement = runmat_filesystem::replace_provider(other.clone());

    let outcome = futures::executor::block_on(super::super::single::move_path(
        &mut captured,
        "source.bin",
        "target.bin",
        false,
    ))
    .expect("move through captured provider");

    assert_eq!(outcome.status(), 1.0, "{outcome:?}");
    assert!(selected.read_project_path("/selected/source.bin").is_err());
    assert_eq!(
        selected.read_project_path("/selected/target.bin").unwrap(),
        b"selected"
    );
    assert!(other.read_project_path("/other/target.bin").is_err());
}

mod pending;
