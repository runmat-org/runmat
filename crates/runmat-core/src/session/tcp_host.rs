//! Lazy Core composition of the session-owned TCP service on the shared host
//! reactor. Constructing a session does not start networking or a new driver.

#![cfg(not(target_arch = "wasm32"))]

use runmat_runtime::context::{
    RuntimeServiceFuture, RuntimeTcpBind, RuntimeTcpClient, RuntimeTcpClientStatus,
    RuntimeTcpCloseScope, RuntimeTcpConnect, RuntimeTcpIssuer, RuntimeTcpReadMode,
    RuntimeTcpReadResult, RuntimeTcpServer, RuntimeTcpService,
};
use runmat_runtime::tcp::{NativeTcpService, TcpTransportError, TcpTransportErrorKind};
use runmat_value::ForeignRef;
use std::cell::OnceCell;
use std::rc::Rc;
use std::time::Duration;

#[derive(Default)]
struct LazyNativeTcpService {
    service: Rc<OnceCell<NativeTcpService>>,
}

impl LazyNativeTcpService {
    fn call<T: 'static>(
        &self,
        call: impl FnOnce(&NativeTcpService) -> RuntimeServiceFuture<Result<T, TcpTransportError>>
            + 'static,
    ) -> RuntimeServiceFuture<Result<T, TcpTransportError>> {
        let service = Rc::clone(&self.service);
        Box::pin(async move {
            if service.get().is_none() {
                let driver = super::host_driver::host_driver().map_err(|source| {
                    TcpTransportError::with_source(
                        TcpTransportErrorKind::Unsupported,
                        "native TCP host driver is unavailable",
                        source,
                    )
                })?;
                // One local Rc owner; there is no await or callback between
                // this check and publication. Failed driver construction is
                // returned and remains retryable, never cached as a dummy.
                let _ = service.set(NativeTcpService::new(driver.handle().clone()));
            }
            let native = service
                .get()
                .expect("TCP service initialized on this local owner");
            call(native).await
        })
    }
}

impl RuntimeTcpService for LazyNativeTcpService {
    fn describe_client(
        &self,
        issuer: RuntimeTcpIssuer,
        client: ForeignRef,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClientStatus, TcpTransportError>> {
        self.call(move |native| native.describe_client(issuer, client))
    }

    fn connect(
        &self,
        issuer: RuntimeTcpIssuer,
        request: RuntimeTcpConnect,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        self.call(move |native| native.connect(issuer, request))
    }

    fn bind(
        &self,
        issuer: RuntimeTcpIssuer,
        request: RuntimeTcpBind,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpServer, TcpTransportError>> {
        self.call(move |native| native.bind(issuer, request))
    }

    fn accept(
        &self,
        issuer: RuntimeTcpIssuer,
        server: ForeignRef,
        timeout: Option<Duration>,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        self.call(move |native| native.accept(issuer, server, timeout))
    }

    fn read(
        &self,
        issuer: RuntimeTcpIssuer,
        client: ForeignRef,
        mode: RuntimeTcpReadMode,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpReadResult, TcpTransportError>> {
        self.call(move |native| native.read(issuer, client, mode))
    }

    fn write(
        &self,
        issuer: RuntimeTcpIssuer,
        client: ForeignRef,
        bytes: Vec<u8>,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        self.call(move |native| native.write(issuer, client, bytes))
    }

    fn close(
        &self,
        issuer: RuntimeTcpIssuer,
        endpoint: ForeignRef,
    ) -> RuntimeServiceFuture<Result<bool, TcpTransportError>> {
        self.call(move |native| native.close(issuer, endpoint))
    }

    fn close_scope(
        &self,
        issuer: RuntimeTcpIssuer,
        scope: RuntimeTcpCloseScope,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        self.call(move |native| native.close_scope(issuer, scope))
    }
}

pub(super) fn default_tcp_service() -> Rc<dyn RuntimeTcpService> {
    Rc::new(LazyNativeTcpService::default())
}
