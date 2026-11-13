use super::*;
use crate::builtins::close::close_builtin;
use crate::context::{
    RuntimeServiceFuture, RuntimeServicePorts, RuntimeTcpBind, RuntimeTcpClient,
    RuntimeTcpClientStatus, RuntimeTcpConnect, RuntimeTcpInvocation, RuntimeTcpReadMode,
    RuntimeTcpReadResult, RuntimeTcpServer, RuntimeTcpService,
};
use crate::tcp::TcpTransportError;
use runmat_builtins::{CLOSE_NETWORK_SELECTOR_EXTENSION, CLOSE_TCP_CELL_EXTENSION};
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::{CellArray, CharArray, ForeignResourceKey, ObjectInstance};
use std::cell::Cell;
use std::rc::Rc;
use std::time::Duration;

#[derive(Default)]
struct CloseCalls {
    endpoint: Cell<usize>,
    scope: Cell<usize>,
}

impl RuntimeTcpService for CloseCalls {
    fn register_session_cleanup(&self, _: &RuntimeContext) -> Result<(), TcpTransportError> {
        Ok(())
    }

    fn close(
        &self,
        _: RuntimeTcpIssuer,
        _: ForeignRef,
    ) -> RuntimeServiceFuture<Result<bool, TcpTransportError>> {
        self.endpoint.set(self.endpoint.get() + 1);
        Box::pin(async { Ok(true) })
    }

    fn close_scope(
        &self,
        _: RuntimeTcpIssuer,
        _: RuntimeTcpCloseScope,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        self.scope.set(self.scope.get() + 1);
        Box::pin(async { Ok(1) })
    }

    fn describe_client(
        &self,
        _: RuntimeTcpInvocation,
        _: ForeignRef,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClientStatus, TcpTransportError>> {
        panic!("close admission must not describe a client")
    }
    fn connect(
        &self,
        _: RuntimeTcpInvocation,
        _: RuntimeTcpConnect,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        panic!("close admission must not connect")
    }
    fn bind(
        &self,
        _: RuntimeTcpInvocation,
        _: RuntimeTcpBind,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpServer, TcpTransportError>> {
        panic!("close admission must not bind")
    }
    fn accept(
        &self,
        _: RuntimeTcpInvocation,
        _: ForeignRef,
        _: Option<Duration>,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        panic!("close admission must not accept")
    }
    fn read(
        &self,
        _: RuntimeTcpInvocation,
        _: ForeignRef,
        _: RuntimeTcpReadMode,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpReadResult, TcpTransportError>> {
        panic!("close admission must not read")
    }
    fn write(
        &self,
        _: RuntimeTcpInvocation,
        _: ForeignRef,
        _: Vec<u8>,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        panic!("close admission must not write")
    }
}

fn fixture() -> (RuntimeContext, Rc<CloseCalls>) {
    let service = Rc::new(CloseCalls::default());
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_tcp(service.clone()));
    (runtime, service)
}

fn client() -> Value {
    let authority = ForeignRef::detached(
        ForeignResourceKey {
            host_identity: "test.close.admission".into(),
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
    Value::Object(ObjectInstance::new(TCPCLIENT_CLASS).with_tcp_authority(authority))
}

fn char_selector(shape: Vec<usize>) -> Value {
    Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("clients".chars().collect(), shape)
            .expect("selector shape"),
    )
}

#[test]
fn paged_selector_rejects_without_tcp_effect_but_trailing_singleton_admits() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let (runtime, calls) = fixture();
    let paged = char_selector(vec![1, 1, 7]);
    assert!(futures::executor::block_on(close_if_network_targets(
        Some(&runtime),
        crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(&paged))
    ))
    .expect("paged text is not a TCP selector")
    .is_none());
    assert!(futures::executor::block_on(close_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![paged])
    ))
    .is_err());
    assert_eq!(calls.scope.get(), 0);
    assert_eq!(calls.endpoint.get(), 0);

    let trailing_singleton = char_selector(vec![1, 7, 1]);
    assert_eq!(
        futures::executor::block_on(close_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![trailing_singleton])
        ))
        .expect("scalar character selector"),
        1.0
    );
    assert_eq!(calls.scope.get(), 1);
}

#[test]
fn pinned_mode_rejects_selector_and_tcp_cell_before_service_but_keeps_direct_receiver() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let (runtime, calls) = fixture();
    let selector = futures::executor::block_on(close_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("clients")]),
    ))
    .expect_err("TCP scope selector is RunMat-only");
    assert_eq!(
        selector.identifier(),
        CLOSE_NETWORK_SELECTOR_EXTENSION.error_identifier
    );
    let cell = Value::Cell(CellArray::new(vec![client()], 1, 1).expect("cell"));
    let container = futures::executor::block_on(close_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![cell]),
    ))
    .expect_err("TCP cell is RunMat-only");
    assert_eq!(
        container.identifier(),
        CLOSE_TCP_CELL_EXTENSION.error_identifier
    );
    assert_eq!(calls.scope.get(), 0);
    assert_eq!(calls.endpoint.get(), 0);

    assert_eq!(
        futures::executor::block_on(close_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![client()])
        ))
        .expect("direct TCP receiver remains admitted"),
        1.0
    );
    assert_eq!(calls.endpoint.get(), 1);
}

#[test]
fn nested_tcp_cell_is_runmat_only_and_validates_before_partial_close() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let (runtime, calls) = fixture();
    let valid = Value::Cell(
        CellArray::new(vec![Value::from("clients"), client()], 1, 2).expect("valid targets"),
    );
    assert_eq!(
        futures::executor::block_on(close_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![valid])
        ))
        .expect("RunMat TCP cell"),
        1.0
    );
    assert_eq!(calls.scope.get(), 1);
    assert_eq!(calls.endpoint.get(), 1);

    let invalid = Value::Cell(
        CellArray::new(
            vec![
                Value::from("clients"),
                Value::Object(ObjectInstance::new(TCPSERVER_CLASS)),
            ],
            1,
            2,
        )
        .expect("invalid second target"),
    );
    let error = futures::executor::block_on(close_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![invalid]),
    ))
    .expect_err("invalid later target rejects before TCP effect");
    assert_eq!(error.identifier(), Some("RunMat:close:InvalidHandle"));
    assert_eq!(calls.scope.get(), 1);
    assert_eq!(calls.endpoint.get(), 1);
}
