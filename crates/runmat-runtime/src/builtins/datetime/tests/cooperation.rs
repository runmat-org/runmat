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
use std::error::Error;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

struct PendingTurn {
    dropped: Rc<Cell<bool>>,
    ready: Option<Rc<Cell<bool>>>,
}
impl Future for PendingTurn {
    type Output = Result<(), ExecutionServiceError>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        if self.ready.as_ref().is_some_and(|ready| ready.get()) {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }
}
impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.dropped.set(true);
    }
}

struct PendingDateService {
    turns: Rc<Cell<usize>>,
    dropped: Rc<Cell<bool>>,
    quantum: u64,
    ready: Option<Rc<Cell<bool>>>,
}
impl RuntimeExecutionServices for PendingDateService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(self.quantum).unwrap())
    }
    fn cooperate_cpu(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        self.turns.set(self.turns.get() + 1);
        Box::pin(PendingTurn {
            dropped: self.dropped.clone(),
            ready: self.ready.clone(),
        })
    }
    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"datetime-cellstr-test"])
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
fn public_datetime_cellstr_parse_turn_cancels_or_drops_without_publication() {
    for cancel in [false, true] {
        let cells = (0..256)
            .map(|_| Value::CharArray(CharArray::new_row("2024-04-09")))
            .collect();
        let input = Value::Cell(runmat_value::CellArray::new(cells, 256, 1).unwrap());
        let source = CancellationSource::new();
        let turns = Rc::new(Cell::new(0));
        let dropped = Rc::new(Cell::new(false));
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(PendingDateService {
                turns: turns.clone(),
                dropped: dropped.clone(),
                ready: None,
                // Preflight and owned gather each charge 256 cells; the first
                // decoded character reaches the parse-phase service turn.
                quantum: 513,
            }),
            source.token(),
        );
        let mut call = Box::pin(super::super::constructor_builtin::datetime_builtin_source(
            Some(runtime),
            vec![input, Value::from("InputFormat"), Value::from("yyyy-MM-dd")],
        ));
        let waker = futures::task::noop_waker();
        assert!(matches!(
            call.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        assert_eq!(turns.get(), 1);
        if cancel {
            source.cancel(CancellationReason::User);
            let error = match call.as_mut().poll(&mut Context::from_waker(&waker)) {
                Poll::Ready(Err(error)) => error,
                _ => panic!("cancelled datetime constructor must finish"),
            };
            assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
            assert!(matches!(
                error
                    .source()
                    .and_then(|source| source.downcast_ref::<ExecutionServiceError>()),
                Some(ExecutionServiceError::Cancelled)
            ));
        }
        drop(call);
        assert!(
            dropped.get(),
            "pending service turn must retire on cancellation/drop"
        );
        let fresh = futures::executor::block_on(
            super::super::constructor_builtin::datetime_builtin_source(
                None,
                vec![Value::CharArray(CharArray::new_row("2024-04-09"))],
            ),
        )
        .expect("subsequent invocation remains usable");
        assert_eq!(
            serial_tensor_for_object(&as_datetime(fresh))
                .unwrap()
                .shape(),
            [1, 1]
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn canonical_dispatcher_passes_invocation_cooperation_to_datetime() {
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(false));
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(PendingDateService {
            turns: turns.clone(),
            dropped: dropped.clone(),
            ready: None,
            quantum: 1,
        }),
        source.token(),
    );
    let args = [Value::Cell(
        runmat_value::CellArray::new(
            vec![Value::CharArray(CharArray::new_row("2024-04-09"))],
            1,
            1,
        )
        .unwrap(),
    )];
    let mut call = Box::pin(crate::dispatcher::call_builtin_async_in_context(
        &runtime, "datetime", &args,
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(turns.get(), 1);
    source.cancel(CancellationReason::User);
    let error = match call.as_mut().poll(&mut Context::from_waker(&waker)) {
        Poll::Ready(Err(error)) => error,
        _ => panic!("canonical datetime cancellation must complete"),
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    drop(call);
    assert!(dropped.get());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn one_large_cellstr_character_vector_suspends_during_bounded_decode() {
    let text = format!("  2024-04-09{}", " ".repeat(16_384));
    let input = Value::Cell(
        runmat_value::CellArray::new(vec![Value::CharArray(CharArray::new_row(&text))], 1, 1)
            .unwrap(),
    );
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(false));
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(PendingDateService {
            turns: turns.clone(),
            dropped: dropped.clone(),
            ready: None,
            quantum: 2048,
        }),
        source.token(),
    );
    let mut call = Box::pin(super::super::constructor_builtin::datetime_builtin_source(
        Some(runtime),
        vec![
            input.clone(),
            Value::from("InputFormat"),
            Value::from("yyyy-MM-dd"),
        ],
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(
        turns.get(),
        1,
        "second 1024-character decode chunk suspends"
    );
    source.cancel(CancellationReason::User);
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Ready(Err(_))
    ));
    drop(call);
    assert!(dropped.get());

    let result =
        futures::executor::block_on(super::super::constructor_builtin::datetime_builtin_source(
            None,
            vec![input, Value::from("InputFormat"), Value::from("yyyy-MM-dd")],
        ))
        .expect("cooperative whitespace scanning preserves the original date grammar");
    let object = as_datetime(result);
    assert_eq!(serial_tensor_for_object(&object).unwrap().shape(), [1, 1]);
    assert_eq!(
        serial_tensor_for_object(&object).unwrap().materialize_f64(),
        vec![serial_for_date(2024, 4, 9)]
    );
}

struct ComponentParallel {
    finite: Cell<bool>,
    ledger: crate::context::host_effect::ManagedHostEffectLedger,
}
impl crate::context::RuntimeParallelService for ComponentParallel {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
        let snapshot = self.ledger.snapshot();
        crate::context::RuntimeParallelResources {
            memory_available_bytes: self.finite.get().then_some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(
        &self,
    ) -> Option<crate::context::host_effect::ManagedHostEffectLedger> {
        Some(self.ledger.clone())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn stored_second_pending_work_releases_exact_output_on_cancel_rotation_or_upgrade() {
    #[derive(Clone, Copy)]
    enum Stop {
        Cancel,
        Rotate,
        Upgrade,
    }
    for stop in [Stop::Cancel, Stop::Rotate, Stop::Upgrade] {
        let ledger = crate::context::host_effect::ManagedHostEffectLedger::new(1_048_576);
        let parallel = Rc::new(ComponentParallel {
            finite: Cell::new(!matches!(stop, Stop::Upgrade)),
            ledger: ledger.clone(),
        });
        let source = CancellationSource::new();
        let ready = Rc::new(Cell::new(false));
        let turns = Rc::new(Cell::new(0));
        let dropped = Rc::new(Cell::new(false));
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(PendingDateService {
                turns: turns.clone(),
                dropped: dropped.clone(),
                quantum: 1,
                ready: Some(ready.clone()),
            }),
            source.token(),
        )
        .with_service_ports(
            crate::context::RuntimeServicePorts::default().with_parallel(parallel.clone()),
        );
        let input = super::super::datetime_object_from_serials(
            None,
            vec![719_529.0; 2],
            vec![1, 2],
            "yyyy-MM-dd",
        )
        .unwrap();
        let mut call = Box::pin(super::super::second_builtin::second_builtin(
            Some(runtime),
            input,
        ));
        let waker = futures::task::noop_waker();
        assert!(matches!(
            call.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        assert_eq!(turns.get(), 1);
        if !matches!(stop, Stop::Upgrade) {
            assert!(ledger.snapshot().live_claimed_bytes > 0);
        }
        match stop {
            Stop::Cancel => {
                source.cancel(CancellationReason::User);
            }
            Stop::Rotate => ledger.revoke(),
            Stop::Upgrade => parallel.finite.set(true),
        }
        ready.set(true);
        let Poll::Ready(Err(error)) = call.as_mut().poll(&mut Context::from_waker(&waker)) else {
            panic!("authority loss cannot publish a component")
        };
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        match stop {
            Stop::Cancel => assert!(matches!(
                error
                    .source()
                    .and_then(|e| e.downcast_ref::<ExecutionServiceError>()),
                Some(ExecutionServiceError::Cancelled)
            )),
            Stop::Rotate | Stop::Upgrade => assert!(matches!(
                error
                    .source()
                    .and_then(|e| e.downcast_ref::<ExecutionServiceError>()),
                Some(ExecutionServiceError::Infrastructure(_))
            )),
        }
        drop(call);
        assert!(dropped.get());
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }
}
