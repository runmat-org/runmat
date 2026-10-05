use super::*;
use crate::context::{
    RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts, RuntimeTcpBind, RuntimeTcpByteOrder,
    RuntimeTcpClient, RuntimeTcpClientStatus, RuntimeTcpCloseScope, RuntimeTcpConnect,
    RuntimeTcpIssuer, RuntimeTcpReadMode, RuntimeTcpReadResult, RuntimeTcpServer,
    RuntimeTcpService,
};
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::{ForeignRef, ForeignResourceKey, ObjectInstance};
use std::cell::Cell;
use std::error::Error;
use std::rc::Rc;
use std::time::Duration;

struct CountedWriteService {
    accepted: Option<usize>,
    calls: Rc<Cell<usize>>,
    pending_dropped: Rc<Cell<bool>>,
}

struct PendingDrop(Rc<Cell<bool>>);

impl Drop for PendingDrop {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

impl RuntimeTcpService for CountedWriteService {
    fn describe_client(
        &self,
        _: RuntimeTcpIssuer,
        _: ForeignRef,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClientStatus, TcpTransportError>> {
        Box::pin(async {
            Ok(RuntimeTcpClientStatus {
                connected: true,
                bytes_available: 0,
                byte_order: RuntimeTcpByteOrder::Little,
                timeout: None,
            })
        })
    }

    fn write(
        &self,
        _: RuntimeTcpIssuer,
        _: ForeignRef,
        _: Vec<u8>,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        let calls = self.calls.clone();
        let accepted = self.accepted;
        let pending_dropped = self.pending_dropped.clone();
        Box::pin(async move {
            calls.set(calls.get() + 1);
            match accepted {
                Some(accepted) => Ok(accepted),
                None => {
                    let _guard = PendingDrop(pending_dropped);
                    std::future::pending::<()>().await;
                    unreachable!("pending test write was cancelled before completion")
                }
            }
        })
    }

    fn connect(
        &self,
        _: RuntimeTcpIssuer,
        _: RuntimeTcpConnect,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        panic!("write-count test must not connect")
    }
    fn bind(
        &self,
        _: RuntimeTcpIssuer,
        _: RuntimeTcpBind,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpServer, TcpTransportError>> {
        panic!("write-count test must not bind")
    }
    fn accept(
        &self,
        _: RuntimeTcpIssuer,
        _: ForeignRef,
        _: Option<Duration>,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        panic!("write-count test must not accept")
    }
    fn read(
        &self,
        _: RuntimeTcpIssuer,
        _: ForeignRef,
        _: RuntimeTcpReadMode,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpReadResult, TcpTransportError>> {
        panic!("write-count test must not read")
    }
    fn close(
        &self,
        _: RuntimeTcpIssuer,
        _: ForeignRef,
    ) -> RuntimeServiceFuture<Result<bool, TcpTransportError>> {
        panic!("write-count test must not close")
    }
    fn close_scope(
        &self,
        _: RuntimeTcpIssuer,
        _: RuntimeTcpCloseScope,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        panic!("write-count test must not close scope")
    }
}

fn receiver() -> Value {
    let authority = ForeignRef::detached(
        ForeignResourceKey {
            host_identity: "test.tcp.count".into(),
            handle: 1,
            generation: 1,
        },
        ForeignTypeIdentity {
            family: "runmat.tcp".into(),
            name: "tcpclient".into(),
            version: 1,
        },
        ForeignOwnership::Shared,
        ForeignAffinity::OriginProcess,
        ForeignLifetime::Persistent,
    );
    Value::Object(
        ObjectInstance::new(runmat_builtins::TCPCLIENT_CLASS).with_tcp_authority(authority),
    )
}

#[test]
fn short_and_oversized_service_successes_never_publish_element_count() {
    for (accepted, expected_progress) in [
        (7, TcpWriteProgress::PrefixSent(7)),
        (9, TcpWriteProgress::Indeterminate),
    ] {
        let service = Rc::new(CountedWriteService {
            accepted: Some(accepted),
            calls: Rc::new(Cell::new(0)),
            pending_dropped: Rc::new(Cell::new(false)),
        });
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(RuntimeServicePorts::default().with_tcp(service.clone()));
        let error = futures::executor::block_on(write_builtin(
            Some(runtime),
            receiver(),
            Value::Num(1.0),
            vec![],
        ))
        .expect_err("invalid accepted byte count must not publish one element");
        assert_eq!(error.identifier(), WRITE_ERROR_INTERNAL.identifier);
        assert_eq!(service.calls.get(), 1, "no replay or retry");
        let source = Error::source(&error)
            .and_then(|source| source.downcast_ref::<TcpTransportError>())
            .expect("typed transport cause");
        assert_eq!(source.write_progress(), Some(expected_progress));
    }
}

#[test]
fn pending_write_cancellation_wakes_and_preserves_indeterminate_progress() {
    use futures::task::{waker_ref, ArcWake};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::future::Future;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::task::{Context, Poll};

    struct Wakes(AtomicUsize);
    impl ArcWake for Wakes {
        fn wake_by_ref(waker: &Arc<Self>) {
            waker.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let service = Rc::new(CountedWriteService {
        accepted: None,
        calls: Rc::new(Cell::new(0)),
        pending_dropped: Rc::new(Cell::new(false)),
    });
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_tcp(service.clone()))
        .with_cancellation_token(cancellation.token());
    let mut call = Box::pin(write_builtin(
        Some(runtime),
        receiver(),
        Value::Num(1.0),
        vec![],
    ));
    let wakes = Arc::new(Wakes(AtomicUsize::new(0)));
    let waker = waker_ref(&wakes);
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(service.calls.get(), 1);
    cancellation.cancel(CancellationReason::User);
    assert!(wakes.0.load(Ordering::SeqCst) > 0);
    let Poll::Ready(Err(error)) = call.as_mut().poll(&mut Context::from_waker(&waker)) else {
        panic!("pending TCP write did not settle on cancellation");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    let source = Error::source(&error)
        .and_then(|source| source.downcast_ref::<TcpTransportError>())
        .expect("typed transport disposition");
    assert_eq!(
        source.write_progress(),
        Some(TcpWriteProgress::Indeterminate)
    );
    assert!(service.pending_dropped.get());
    assert_eq!(service.calls.get(), 1, "cancelled write is never replayed");
}
