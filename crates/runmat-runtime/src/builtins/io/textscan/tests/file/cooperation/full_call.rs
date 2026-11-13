use super::*;
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
use std::cell::Cell;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::task::{Context, Poll};

struct OutputTurnService {
    turns: Rc<Cell<usize>>,
    pending_on: Rc<Cell<Option<usize>>>,
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

impl RuntimeExecutionServices for OutputTurnService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("positive quantum"))
    }

    fn cooperate_cpu(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        let turn = self.turns.get() + 1;
        self.turns.set(turn);
        if self.pending_on.get() == Some(turn) {
            Box::pin(PendingTurn(self.dropped.clone()))
        } else {
            Box::pin(async { Ok(()) })
        }
    }

    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"textscan-output-turn-test"])
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dropping_actual_textscan_during_output_restores_cursor_and_exclusivity() {
    use std::future::Future;

    let _guard = registry::test_guard();
    registry::reset_for_tests();
    let path = file_fixture("full-output-abandoned.txt", b"10\n20\n");
    let mut open = OpenOptions::new();
    open.read(true);
    let file = block_on(open.open_async(&path)).expect("open file");
    let turns = Rc::new(Cell::new(0));
    let pending_on = Rc::new(Cell::new(None));
    let dropped = Rc::new(Cell::new(false));
    let source = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(OutputTurnService {
            turns: turns.clone(),
            pending_on: pending_on.clone(),
            dropped: dropped.clone(),
        }),
        source.token(),
    );
    let fid = registry::register_file_in(
        RegisteredFile {
            path: path.to_path_buf(),
            permission: "r".to_string(),
            access: registry::FileAccess::READ,
            machinefmt: "native".to_string(),
            encoding: "UTF-8".to_string(),
            handle: Arc::new(StdMutex::new(Some(file))),
        },
        Some(&runtime),
        None,
    );

    // Calibrate the execution adapter at the parser boundary. The next
    // positive work charge in the full call belongs to output construction.
    let format = parse_format_spec("%f").expect("format");
    let options = TextscanOptions::default();
    let mut cooperation = TextscanCooperation::new(Some(&runtime));
    let parsed = block_on(input::parse_registered_file_in(
        Some(&runtime),
        fid,
        &format,
        Some(1),
        &options,
        &mut cooperation,
    ))
    .expect("parse without publication");
    assert_eq!(parsed.columns.len(), 1);
    drop(parsed);
    let parser_turns = turns.get();
    assert!(parser_turns > 0);
    turns.set(0);
    pending_on.set(Some(parser_turns + 1));

    let mut call = Box::pin(textscan_builtin(
        Some(runtime.clone()),
        Value::Num(fid as f64),
        Value::from("%f"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ));
    let mut cx = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(
        Future::poll(call.as_mut(), &mut cx),
        Poll::Pending
    ));
    assert_eq!(
        turns.get(),
        parser_turns + 1,
        "suspended in output construction"
    );
    assert!(matches!(
        registry::acquire_file_in(fid, Some(&runtime)),
        Err(registry::FileOperationError::Busy)
    ));
    assert_eq!(
        block_on(registry::close_async_in(fid, Some(&runtime)))
            .expect_err("pending output blocks close")
            .kind(),
        std::io::ErrorKind::WouldBlock
    );
    drop(call);
    pending_on.set(None);
    assert!(
        dropped.get(),
        "abandonment drops the execution-service turn"
    );

    let mut restored = registry::acquire_file_in(fid, Some(&runtime)).expect("same owner restored");
    assert_eq!(
        std::io::Seek::stream_position(restored.file_mut()).expect("cursor"),
        0
    );
    drop(restored);
    let output = block_on(textscan_builtin(
        Some(runtime.clone()),
        Value::Num(fid as f64),
        Value::from("%f"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .expect("subsequent invocation succeeds");
    assert_eq!(numeric_column(&output, 0), vec![10.0]);
    block_on(registry::close_async_in(fid, Some(&runtime)))
        .expect("close")
        .expect("entry");
    registry::reset_for_tests();
}
