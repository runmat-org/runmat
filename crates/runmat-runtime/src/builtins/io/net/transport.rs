//! One platform boundary for the native TCP transport used by this family.

use crate::{
    context::{RuntimeContext, RuntimeTcpService},
    execution::ExecutionServiceError,
    BuiltinResult,
};
use std::future::Future;
use std::rc::Rc;

/// Whether cancellation occurred before the operation was polled or while
/// externally pending work owned by this invocation was in flight.
pub(super) struct TcpAwaitFailure {
    pub(super) cause: ExecutionServiceError,
    pub(super) launched: bool,
}

impl TcpAwaitFailure {
    pub(super) fn into_runtime_error(self, builtin: &'static str) -> crate::RuntimeError {
        if self.cause == ExecutionServiceError::Cancelled {
            crate::build_runtime_error(format!("{builtin}: execution was cancelled"))
                .with_builtin(builtin)
                .with_identifier("RunMat:Cancelled")
                .with_source(self.cause)
                .build()
        } else {
            self.cause.into_runtime_error()
        }
    }
}

/// The execution service owns the cancellation wake and CPU turn. A future
/// cancelled before the initial checkpoint is never polled; dropping an
/// in-flight future invokes the transport's abort/rollback owner.
pub(super) async fn await_service<T>(
    runtime: Option<&RuntimeContext>,
    future: impl Future<Output = T>,
) -> Result<T, TcpAwaitFailure> {
    let Some(runtime) = runtime else {
        return Ok(future.await);
    };
    let mut budget = runtime.cpu_cooperation();
    budget
        .checkpoint(0)
        .await
        .map_err(|cause| TcpAwaitFailure {
            cause,
            launched: false,
        })?;
    budget
        .await_cancellable(future)
        .await
        .map_err(|cause| TcpAwaitFailure {
            cause,
            launched: true,
        })
}

/// Resolve the installed session authority before an operand download or a
/// socket effect. A browser target retains its typed unsupported diagnostic.
pub(super) fn service(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
) -> BuiltinResult<Rc<dyn RuntimeTcpService>> {
    require_native_tcp(builtin)?;
    runtime
        .ok_or_else(|| {
            crate::context::RuntimeCapabilityError::new(
                crate::context::RuntimeCapability::Tcp,
                builtin,
            )
        })
        .map_err(crate::context::RuntimeCapabilityError::into_runtime_error)?
        .service_ports()
        .require_tcp(builtin)
        .cloned()
        .map_err(crate::context::RuntimeCapabilityError::into_runtime_error)
}

pub(super) fn require_native_tcp(builtin: &'static str) -> BuiltinResult<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = builtin;
        Ok(())
    }
    #[cfg(target_arch = "wasm32")]
    {
        let error = &runmat_builtins::NET_ERROR_UNSUPPORTED_TARGET;
        Err(
            crate::build_runtime_error(format!("{builtin}: {}", error.message))
                .with_builtin(builtin)
                .with_identifier(error.identifier.expect("declared TCP target identifier"))
                .build(),
        )
    }
}
