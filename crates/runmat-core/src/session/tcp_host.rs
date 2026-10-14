//! Lazy Core composition of the session-owned TCP service on the shared host
//! reactor. Constructing a session does not start networking or a new driver.

#![cfg(not(target_arch = "wasm32"))]

use runmat_runtime::context::{
    RuntimeServiceFuture, RuntimeTcpBind, RuntimeTcpClient, RuntimeTcpClientStatus,
    RuntimeTcpCloseScope, RuntimeTcpConnect, RuntimeTcpInvocation, RuntimeTcpIssuer,
    RuntimeTcpReadMode, RuntimeTcpReadResult, RuntimeTcpServer, RuntimeTcpService,
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
    fn get_or_initialize(
        service: &OnceCell<NativeTcpService>,
    ) -> Result<&NativeTcpService, TcpTransportError> {
        if service.get().is_none() {
            let driver = super::host_driver::host_driver().map_err(|source| {
                TcpTransportError::with_source(
                    TcpTransportErrorKind::Unsupported,
                    "native TCP host driver is unavailable",
                    source,
                )
            })?;
            // The cell has one local owner. Failed initialization remains
            // retryable; no await or callback separates check and publication.
            let _ = service.set(NativeTcpService::new(driver.handle().clone()));
        }
        Ok(service
            .get()
            .expect("TCP service initialized on this local owner"))
    }

    fn call<T: 'static>(
        &self,
        call: impl FnOnce(&NativeTcpService) -> RuntimeServiceFuture<Result<T, TcpTransportError>>
            + 'static,
    ) -> RuntimeServiceFuture<Result<T, TcpTransportError>> {
        let service = Rc::clone(&self.service);
        Box::pin(async move {
            let native = Self::get_or_initialize(&service)?;
            call(native).await
        })
    }
}

impl RuntimeTcpService for LazyNativeTcpService {
    fn register_session_cleanup(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), TcpTransportError> {
        // Registration happens at TCP invocation admission, not session
        // construction. Use the same native owner as subsequent socket calls.
        Self::get_or_initialize(&self.service)?.register_session_cleanup(runtime)
    }

    fn describe_client(
        &self,
        invocation: RuntimeTcpInvocation,
        client: ForeignRef,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClientStatus, TcpTransportError>> {
        self.call(move |native| native.describe_client(invocation, client))
    }

    fn connect(
        &self,
        invocation: RuntimeTcpInvocation,
        request: RuntimeTcpConnect,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        self.call(move |native| native.connect(invocation, request))
    }

    fn bind(
        &self,
        invocation: RuntimeTcpInvocation,
        request: RuntimeTcpBind,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpServer, TcpTransportError>> {
        self.call(move |native| native.bind(invocation, request))
    }

    fn accept(
        &self,
        invocation: RuntimeTcpInvocation,
        server: ForeignRef,
        timeout: Option<Duration>,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        self.call(move |native| native.accept(invocation, server, timeout))
    }

    fn read(
        &self,
        invocation: RuntimeTcpInvocation,
        client: ForeignRef,
        mode: RuntimeTcpReadMode,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpReadResult, TcpTransportError>> {
        self.call(move |native| native.read(invocation, client, mode))
    }

    fn write(
        &self,
        invocation: RuntimeTcpInvocation,
        client: ForeignRef,
        bytes: Vec<u8>,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        self.call(move |native| native.write(invocation, client, bytes))
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

#[cfg(test)]
mod tests;
