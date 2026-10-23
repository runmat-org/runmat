//! Whole-builtin cooperation and publication boundaries.

use crate::context::{
    RuntimeContext, RuntimeHttpService, RuntimeServiceFuture, RuntimeServicePorts,
};
use crate::execution::RootedValueSequence;
use crate::execution::{
    AwaitAction, CpuCooperationPolicy, DeferredCall, ExecutionServiceError,
    RuntimeExecutionServices,
};
use crate::http::{HttpRequest, HttpResponse, TransportError};
use futures::task::noop_waker;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use runmat_value::{StringArray, Value};
use std::cell::Cell;
use std::future::Future;
use std::io::Write;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};

struct TurnService {
    pending_at: usize,
    turns: Rc<Cell<usize>>,
    dropped: Rc<Cell<usize>>,
}

struct PendingTurn(Rc<Cell<usize>>);

impl Future for PendingTurn {
    type Output = Result<(), ExecutionServiceError>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

impl RuntimeExecutionServices for TurnService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("positive quantum"))
    }

    fn cooperate_cpu(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        let turn = self.turns.get() + 1;
        self.turns.set(turn);
        if turn == self.pending_at {
            Box::pin(PendingTurn(self.dropped.clone()))
        } else {
            Box::pin(async { Ok(()) })
        }
    }

    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"gzip-whole-builtin-cooperation-test"])
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

struct Fixture {
    provider: Arc<runmat_filesystem::MemoryFsProvider>,
    runtime: RuntimeContext,
    cancellation: CancellationSource,
    turns: Rc<Cell<usize>>,
    dropped: Rc<Cell<usize>>,
}

fn fixture(pending_at: usize) -> Fixture {
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/workspace",
    ));
    let cancellation = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(0));
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(TurnService {
            pending_at,
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

fn gzip_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(bytes).expect("encode fixture");
    encoder.finish().expect("finish fixture")
}

struct ReadyHttpService {
    body: Vec<u8>,
    sends: Rc<Cell<usize>>,
}

impl RuntimeHttpService for ReadyHttpService {
    fn send(
        &self,
        _: RuntimeContext,
        _: HttpRequest,
    ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
        self.sends.set(self.sends.get() + 1);
        let body = self.body.clone();
        Box::pin(async move {
            Ok(HttpResponse {
                headers: Vec::new(),
                body,
            })
        })
    }
}

fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    Future::poll(future, &mut context)
}

fn call<'a>(
    runtime: &'a RuntimeContext,
    name: &'a str,
    args: &'a [Value],
) -> impl Future<Output = crate::BuiltinResult<crate::execution::RootedValue>> + 'a {
    crate::call_builtin_async_with_outputs_in_context(runtime, name, args, 1)
}

#[derive(Clone, Copy)]
enum ArchiveOperation {
    Gzip,
    Gunzip,
}

impl ArchiveOperation {
    fn name(self) -> &'static str {
        match self {
            Self::Gzip => "gzip",
            Self::Gunzip => "gunzip",
        }
    }

    fn source_path(self) -> &'static str {
        match self {
            Self::Gzip => "/workspace/source.bin",
            Self::Gunzip => "/workspace/source.bin.gz",
        }
    }

    fn output_path(self) -> &'static str {
        match self {
            Self::Gzip => "/workspace/out/source.bin.gz",
            Self::Gunzip => "/workspace/out/source.bin",
        }
    }

    fn source_bytes(self) -> Vec<u8> {
        let bytes = (0..32_769)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        match self {
            Self::Gzip => bytes,
            Self::Gunzip => gzip_bytes(&bytes),
        }
    }
}

#[test]
fn selected_builtin_cancel_or_drop_retires_pending_turn_before_publication() {
    let _lock = runmat_filesystem::provider_override_lock();
    for operation in [ArchiveOperation::Gzip, ArchiveOperation::Gunzip] {
        for cancel in [false, true] {
            let fixture = fixture(1);
            fixture
                .provider
                .write_project_path(operation.source_path(), &operation.source_bytes())
                .expect("source fixture");
            let _provider = runmat_filesystem::replace_provider(fixture.provider.clone());
            let args = [
                Value::String(operation.source_path().into()),
                Value::String("out".into()),
            ];
            let mut future = Box::pin(call(&fixture.runtime, operation.name(), &args));
            assert!(poll_once(future.as_mut()).is_pending());
            assert_eq!(fixture.turns.get(), 1, "{} selected turn", operation.name());
            assert!(fixture
                .provider
                .metadata_project_path("/workspace/out")
                .is_err());
            assert!(fixture
                .provider
                .metadata_project_path(operation.output_path())
                .is_err());

            if cancel {
                fixture.cancellation.cancel(CancellationReason::User);
                let Poll::Ready(Err(error)) = poll_once(future.as_mut()) else {
                    panic!("{} cancellation must finish", operation.name());
                };
                assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
                assert_eq!(
                    error.failure_origin,
                    crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation
                );
            }
            drop(future);
            assert_eq!(
                fixture.dropped.get(),
                1,
                "{} turn retired",
                operation.name()
            );
            assert!(fixture
                .provider
                .metadata_project_path("/workspace/out")
                .is_err());
            assert!(fixture
                .provider
                .metadata_project_path(operation.output_path())
                .is_err());
        }
    }
}

#[test]
fn selected_url_gunzip_cancel_or_drop_retires_codec_turn_before_publication() {
    let _lock = runmat_filesystem::provider_override_lock();
    for cancel in [false, true] {
        let fixture = fixture(1);
        let sends = Rc::new(Cell::new(0));
        let runtime =
            fixture
                .runtime
                .clone()
                .with_service_ports(RuntimeServicePorts::default().with_http(Rc::new(
                    ReadyHttpService {
                        body: gzip_bytes(&vec![7; 32_769]),
                        sends: sends.clone(),
                    },
                )));
        let _provider = runmat_filesystem::replace_provider(fixture.provider.clone());
        let args = [
            Value::String("https://example.invalid/archive.bin.gz".into()),
            Value::String("out".into()),
        ];
        let mut future = Box::pin(call(&runtime, "gunzip", &args));
        assert!(poll_once(future.as_mut()).is_pending());
        assert_eq!(sends.get(), 1, "selected HTTP response fetched once");
        assert_eq!(fixture.turns.get(), 1, "selected codec budget reached");
        assert!(fixture
            .provider
            .metadata_project_path("/workspace/out")
            .is_err());
        assert!(fixture
            .provider
            .metadata_project_path("/workspace/out/archive.bin")
            .is_err());

        if cancel {
            fixture.cancellation.cancel(CancellationReason::User);
            let Poll::Ready(Err(error)) = poll_once(future.as_mut()) else {
                panic!("URL gunzip cancellation must finish");
            };
            assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
            assert_eq!(
                error.failure_origin,
                crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation
            );
        }
        drop(future);
        assert_eq!(fixture.dropped.get(), 1, "selected turn retired");
        assert!(fixture
            .provider
            .metadata_project_path("/workspace/out")
            .is_err());
        assert!(fixture
            .provider
            .metadata_project_path("/workspace/out/archive.bin")
            .is_err());
    }
}

#[test]
fn multi_input_keeps_prior_publication_when_later_item_is_abandoned() {
    let _lock = runmat_filesystem::provider_override_lock();
    for operation in [ArchiveOperation::Gzip, ArchiveOperation::Gunzip] {
        // gzip charges encode+finish for the first member; gunzip completes the
        // first member in one decoder step. Suspend on the next member.
        let pending_at = match operation {
            ArchiveOperation::Gzip => 3,
            ArchiveOperation::Gunzip => 2,
        };
        let fixture = fixture(pending_at);
        let (first_source, second_source, first_output, second_output) = match operation {
            ArchiveOperation::Gzip => (
                "/workspace/first.bin",
                "/workspace/second.bin",
                "/workspace/out/first.bin.gz",
                "/workspace/out/second.bin.gz",
            ),
            ArchiveOperation::Gunzip => (
                "/workspace/first.bin.gz",
                "/workspace/second.bin.gz",
                "/workspace/out/first.bin",
                "/workspace/out/second.bin",
            ),
        };
        for (path, byte) in [(first_source, b'a'), (second_source, b'b')] {
            let bytes = match operation {
                ArchiveOperation::Gzip => vec![byte],
                ArchiveOperation::Gunzip => gzip_bytes(&[byte]),
            };
            fixture
                .provider
                .write_project_path(path, &bytes)
                .expect("source fixture");
        }
        let _provider = runmat_filesystem::replace_provider(fixture.provider.clone());
        let sources = StringArray::new(
            vec![first_source.to_string(), second_source.to_string()],
            vec![2, 1],
        )
        .expect("source list");
        let args = [Value::StringArray(sources), Value::String("out".into())];
        let mut future = Box::pin(call(&fixture.runtime, operation.name(), &args));
        assert!(poll_once(future.as_mut()).is_pending());
        assert_eq!(fixture.turns.get(), pending_at);
        assert!(fixture.provider.metadata_project_path(first_output).is_ok());
        assert!(fixture
            .provider
            .metadata_project_path(second_output)
            .is_err());
        drop(future);
        assert_eq!(fixture.dropped.get(), 1);
        assert!(fixture.provider.metadata_project_path(first_output).is_ok());
        assert!(fixture
            .provider
            .metadata_project_path(second_output)
            .is_err());
    }
}
