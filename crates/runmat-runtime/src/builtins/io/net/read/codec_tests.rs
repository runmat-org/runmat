use super::*;
use crate::builtins::io::net::test_support::cpu_turn::CodecTurnService;
use crate::context::{
    RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts, RuntimeTcpBind, RuntimeTcpClient,
    RuntimeTcpClientStatus, RuntimeTcpCloseScope, RuntimeTcpConnect, RuntimeTcpInvocation,
    RuntimeTcpIssuer, RuntimeTcpReadPublication, RuntimeTcpReadResult, RuntimeTcpServer,
    RuntimeTcpService,
};
use crate::tcp::TcpTransportError;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::{ForeignRef, ForeignResourceKey, ObjectInstance};
use std::cell::Cell;
use std::future::Future;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

struct ReadReceipt(Arc<AtomicUsize>);

impl RuntimeTcpReadPublication for ReadReceipt {
    fn rollback(self: Box<Self>, bytes: Vec<u8>) {
        self.0.store(bytes.len(), Ordering::SeqCst);
    }
}

struct CountedReadService {
    calls: Rc<Cell<usize>>,
    restored: Arc<AtomicUsize>,
}

impl RuntimeTcpService for CountedReadService {
    fn register_session_cleanup(&self, _: &RuntimeContext) -> Result<(), TcpTransportError> {
        Ok(())
    }
    fn read(
        &self,
        _: RuntimeTcpInvocation,
        _: ForeignRef,
        _: RuntimeTcpReadMode,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpReadResult, TcpTransportError>> {
        let calls = self.calls.clone();
        let restored = self.restored.clone();
        Box::pin(async move {
            calls.set(calls.get() + 1);
            Ok(RuntimeTcpReadResult::new(
                vec![7; 8192],
                false,
                RuntimeTcpByteOrder::Little,
                Box::new(ReadReceipt(restored)),
            ))
        })
    }
    fn describe_client(
        &self,
        _: RuntimeTcpInvocation,
        _: ForeignRef,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClientStatus, TcpTransportError>> {
        panic!("read codec witness does not describe")
    }
    fn connect(
        &self,
        _: RuntimeTcpInvocation,
        _: RuntimeTcpConnect,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        panic!("read codec witness does not connect")
    }
    fn bind(
        &self,
        _: RuntimeTcpInvocation,
        _: RuntimeTcpBind,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpServer, TcpTransportError>> {
        panic!("read codec witness does not bind")
    }
    fn accept(
        &self,
        _: RuntimeTcpInvocation,
        _: ForeignRef,
        _: Option<Duration>,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        panic!("read codec witness does not accept")
    }
    fn write(
        &self,
        _: RuntimeTcpInvocation,
        _: ForeignRef,
        _: Vec<u8>,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        panic!("read codec witness does not write")
    }
    fn close(
        &self,
        _: RuntimeTcpIssuer,
        _: ForeignRef,
    ) -> RuntimeServiceFuture<Result<bool, TcpTransportError>> {
        panic!("read codec witness does not close")
    }
    fn close_scope(
        &self,
        _: RuntimeTcpIssuer,
        _: RuntimeTcpCloseScope,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        panic!("read codec witness does not close scope")
    }
}

fn receiver() -> Value {
    let authority = ForeignRef::detached(
        ForeignResourceKey {
            host_identity: "test.tcp.read.codec".into(),
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
fn full_read_codec_cancellation_restores_unpublished_bytes_once() {
    let service = Rc::new(CountedReadService {
        calls: Rc::new(Cell::new(0)),
        restored: Arc::new(AtomicUsize::new(0)),
    });
    let turns = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(false));
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(CodecTurnService {
        turns: turns.clone(),
        dropped: dropped.clone(),
        pending_on: 1,
    }))
    .with_service_ports(RuntimeServicePorts::default().with_tcp(service.clone()))
    .with_cancellation_token(cancellation.token());
    let mut call = Box::pin(read_builtin(
        Some(runtime),
        receiver(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(8192.0)]),
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(turns.get(), 1, "first CPU turn occurs during output decode");
    assert_eq!(service.calls.get(), 1);
    assert_eq!(
        service.restored.load(Ordering::SeqCst),
        0,
        "receipt stays unpublished while pending"
    );
    cancellation.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = call.as_mut().poll(&mut Context::from_waker(&waker)) else {
        panic!("read codec turn did not settle on cancellation");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(dropped.get());
    assert_eq!(service.restored.load(Ordering::SeqCst), 8192);
    assert_eq!(service.calls.get(), 1, "read was not replayed");
}
