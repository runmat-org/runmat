//! Actual FID builtin futures under an execution-service work-quantum turn.

use crate::builtins::io::filetext::{fopen, fread, fwrite, registry};
use crate::context::{RuntimeContext, RuntimeServiceFuture};
use crate::execution::RootedValueSequence;
use crate::execution::{
    AwaitAction, CpuCooperationPolicy, DeferredCall, ExecutionServiceError,
    RuntimeExecutionServices,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use runmat_value::{Tensor, Value};
use std::cell::Cell;
use std::io::Seek;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};

struct TurnService {
    turns: Rc<Cell<usize>>,
    dropped: Rc<Cell<bool>>,
}

struct PendingTurn(Rc<Cell<bool>>);

impl std::future::Future for PendingTurn {
    type Output = Result<(), ExecutionServiceError>;
    fn poll(self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

impl RuntimeExecutionServices for TurnService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("positive quantum"))
    }
    fn cooperate_cpu(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        self.turns.set(self.turns.get() + 1);
        Box::pin(PendingTurn(self.dropped.clone()))
    }
    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"filetext-whole-call-turn-test"])
    }
    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        Ok(None)
    }
    fn ensure_pool(&self, _: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn close_pool(&self, _: &PoolHandle) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn create_future(&self, _: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn spawn(&self, _: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn inspect_future(
        &self,
        _: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn inspect_task(
        &self,
        _: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn claim_next_task_result(
        &self,
        _: &[runmat_value::SharedTaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn mark_task_result_read(&self, _: &TaskHandle) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn begin_await(&self, _: Value) -> Result<AwaitAction, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn complete_future(
        &self,
        _: crate::execution::FutureCompletionRef<'_>,
        _: Result<RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn cancel(&self, _: &Value, _: CancellationReason) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn drain_scope(&self, _: CancellationReason) {}
}

fn root() -> &'static str {
    "/filetext-turn"
}

struct Fixture {
    provider: Arc<runmat_filesystem::MemoryFsProvider>,
    runtime: RuntimeContext,
    cancellation: CancellationSource,
    turns: Rc<Cell<usize>>,
    dropped: Rc<Cell<bool>>,
}

fn fixture() -> Fixture {
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(root()));
    let turns = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(false));
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(TurnService {
            turns: turns.clone(),
            dropped: dropped.clone(),
        }),
        cancellation.token(),
    );
    Fixture {
        provider,
        runtime,
        cancellation,
        turns,
        dropped,
    }
}

fn poll_once<F: std::future::Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    std::future::Future::poll(future, &mut context)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn actual_fread_cancelled_on_first_charged_byte_restores_checked_out_fid() {
    let _registry = registry::test_guard();
    let _filesystem = runmat_filesystem::provider_override_lock();
    registry::reset_for_tests();
    let Fixture {
        provider,
        runtime,
        cancellation,
        turns,
        dropped,
    } = fixture();
    let path = Path::new(root()).join("read.bin");
    provider
        .write_project_path(&path, &[11, 22, 33])
        .expect("fixture");
    let _selected = runmat_filesystem::replace_provider(provider);
    let opened = futures::executor::block_on(fopen::evaluate_in(
        Some(&runtime),
        &[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ],
    ))
    .expect("open source");
    let fid = opened.as_open().expect("open form").fid as i32;
    let original = registry::shared_handle(fid).expect("registered handle");
    let fid_value = Value::Num(fid as f64);
    let mut call = Box::pin(fread::evaluate_in(Some(&runtime), &fid_value, &[]));
    assert!(poll_once(call.as_mut()).is_pending());
    assert_eq!(turns.get(), 1, "the first actual byte charged one turn");
    assert!(matches!(
        registry::acquire_file_in(fid, Some(&runtime)),
        Err(registry::FileOperationError::Busy)
    ));
    cancellation.cancel(CancellationReason::User);
    let error = futures::executor::block_on(call).expect_err("cancelled read");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(dropped.get(), "pending execution turn was dropped");
    let mut restored = registry::acquire_file_in(fid, Some(&runtime)).expect("same FID restored");
    assert_eq!(restored.file_mut().stream_position().expect("cursor"), 1);
    drop(restored);
    assert!(Arc::ptr_eq(
        &original,
        &registry::shared_handle(fid).expect("handle")
    ));
    futures::executor::block_on(registry::close_async_in(fid, Some(&runtime))).expect("close");
    registry::reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abandoned_actual_fwrite_turn_retains_fid_and_only_first_byte() {
    let _registry = registry::test_guard();
    let _filesystem = runmat_filesystem::provider_override_lock();
    registry::reset_for_tests();
    let Fixture {
        provider,
        runtime,
        cancellation: _cancellation,
        turns,
        dropped,
    } = fixture();
    let path: PathBuf = Path::new(root()).join("write.bin");
    let _selected = runmat_filesystem::replace_provider(provider.clone());
    let opened = futures::executor::block_on(fopen::evaluate_in(
        Some(&runtime),
        &[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("wb"),
        ],
    ))
    .expect("open destination");
    let fid = opened.as_open().expect("open form").fid as i32;
    let original = registry::shared_handle(fid).expect("registered handle");
    let fid_value = Value::Num(fid as f64);
    let data = Value::Tensor(Tensor::new(vec![11.0, 22.0, 33.0], vec![1, 3]).expect("data"));
    let mut call = Box::pin(fwrite::evaluate_in(Some(&runtime), &fid_value, &data, &[]));
    assert!(poll_once(call.as_mut()).is_pending());
    assert_eq!(turns.get(), 1, "the first actual byte charged one turn");
    drop(call);
    assert!(dropped.get(), "abandonment drops the pending turn");
    assert!(Arc::ptr_eq(
        &original,
        &registry::shared_handle(fid).expect("handle")
    ));
    futures::executor::block_on(registry::close_async_in(fid, Some(&runtime))).expect("close");
    assert_eq!(
        provider.read_project_path(&path).expect("committed bytes"),
        [11]
    );
    registry::reset_for_tests();
}
