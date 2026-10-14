//! Native TCP owner on the host's reusable Tokio driver.

mod operation;
mod registry;
mod unread;
#[cfg(test)]
mod tests;

use crate::context::{
    RuntimeServiceFuture, RuntimeTcpBind, RuntimeTcpClient, RuntimeTcpClientStatus,
    RuntimeTcpCloseScope, RuntimeTcpConnect, RuntimeTcpInvocation, RuntimeTcpIssuer,
    RuntimeTcpReadMode, RuntimeTcpReadResult, RuntimeTcpServer, RuntimeTcpService,
};
use crate::tcp::{TcpTransportError, TcpTransportErrorKind};
use runmat_execution::profiling::CommandProfilingReceipt;
use runmat_value::ForeignRef;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use registry::Registry;

pub struct NativeTcpService {
    driver: tokio::runtime::Handle,
    registry: Arc<std::sync::Mutex<Registry>>,
}

impl Drop for NativeTcpService {
    fn drop(&mut self) {
        registry::shutdown_all(&self.registry);
    }
}

impl NativeTcpService {
    pub fn new(driver: tokio::runtime::Handle) -> Self {
        Self {
            driver,
            registry: Arc::new(std::sync::Mutex::new(Registry::default())),
        }
    }

    fn on_driver<T: Send + 'static>(
        &self,
        work: impl Future<Output = Result<T, TcpTransportError>> + Send + 'static,
        profiling: Option<CommandProfilingReceipt>,
    ) -> RuntimeServiceFuture<Result<T, TcpTransportError>> {
        let driver = self.driver.clone();
        Box::pin(async move {
            let mut task = AbortOnDrop(crate::execution::native_task::spawn(
                &driver, work, profiling,
            ));
            (&mut task.0).await.map_err(|source| {
                TcpTransportError::with_source(
                    TcpTransportErrorKind::Io,
                    "TCP host task did not complete",
                    source,
                )
            })?
        })
    }
}

struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl RuntimeTcpService for NativeTcpService {
    fn register_session_cleanup(
        &self,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<(), TcpTransportError> {
        registry::register_session_cleanup(&self.registry, runtime)
    }

    fn describe_client(
        &self,
        invocation: RuntimeTcpInvocation,
        client: ForeignRef,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClientStatus, TcpTransportError>> {
        let (issuer, profiling) = invocation.into_parts();
        let registry = Arc::clone(&self.registry);
        self.on_driver(
            operation::describe_client(registry, issuer, client),
            profiling,
        )
    }

    fn connect(
        &self,
        invocation: RuntimeTcpInvocation,
        request: RuntimeTcpConnect,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        let (issuer, profiling) = invocation.into_parts();
        let registry = Arc::clone(&self.registry);
        self.on_driver(operation::connect(registry, issuer, request), profiling)
    }

    fn bind(
        &self,
        invocation: RuntimeTcpInvocation,
        request: RuntimeTcpBind,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpServer, TcpTransportError>> {
        let (issuer, profiling) = invocation.into_parts();
        let registry = Arc::clone(&self.registry);
        self.on_driver(operation::bind(registry, issuer, request), profiling)
    }

    fn accept(
        &self,
        invocation: RuntimeTcpInvocation,
        server: ForeignRef,
        timeout: Option<Duration>,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, TcpTransportError>> {
        let (issuer, profiling) = invocation.into_parts();
        let registry = Arc::clone(&self.registry);
        self.on_driver(
            operation::accept(registry, issuer, server, timeout),
            profiling,
        )
    }

    fn read(
        &self,
        invocation: RuntimeTcpInvocation,
        client: ForeignRef,
        mode: RuntimeTcpReadMode,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpReadResult, TcpTransportError>> {
        let (issuer, profiling) = invocation.into_parts();
        let registry = Arc::clone(&self.registry);
        self.on_driver(operation::read(registry, issuer, client, mode), profiling)
    }

    fn write(
        &self,
        invocation: RuntimeTcpInvocation,
        client: ForeignRef,
        bytes: Vec<u8>,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        let (issuer, profiling) = invocation.into_parts();
        let registry = Arc::clone(&self.registry);
        self.on_driver(operation::write(registry, issuer, client, bytes), profiling)
    }

    fn close(
        &self,
        issuer: RuntimeTcpIssuer,
        endpoint: ForeignRef,
    ) -> RuntimeServiceFuture<Result<bool, TcpTransportError>> {
        let registry = Arc::clone(&self.registry);
        Box::pin(async move { registry::close(&registry, &issuer, &endpoint) })
    }

    fn close_scope(
        &self,
        issuer: RuntimeTcpIssuer,
        scope: RuntimeTcpCloseScope,
    ) -> RuntimeServiceFuture<Result<usize, TcpTransportError>> {
        let registry = Arc::clone(&self.registry);
        Box::pin(async move { registry::close_scope(&registry, &issuer, scope) })
    }
}
