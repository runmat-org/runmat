//! One native invocation's exact provider and thread-affine context lifetime.
use runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind;
use runmat_accelerate_api::{AccelProvider, NativeDeviceContextGuard};
use std::rc::Rc;

use super::exports::ExportScope;

pub(crate) struct NativeInvocationContext {
    provider: &'static dyn AccelProvider,
    guard: Option<NativeDeviceContextGuard>,
    exports: Rc<ExportScope>,
}

impl NativeInvocationContext {
    pub(crate) fn enter(provider: &'static dyn AccelProvider) -> Result<Self, String> {
        let guard = catch_allocation_cleanup_unwind(|| {
            provider
                .enter_native_device_context()
                .map_err(|error| error.to_string())
        })
        .map_err(|_| "CUDA context entry panicked".to_string())??;
        Ok(Self {
            provider,
            guard: Some(guard),
            exports: ExportScope::enter(provider),
        })
    }

    pub(crate) fn leave(mut self) -> Result<(), String> {
        self.cleanup()
    }

    fn cleanup(&mut self) -> Result<(), String> {
        let Some(guard) = self.guard.take() else {
            return Ok(());
        };
        self.exports.stop_exports();
        // Keep the context current through synchronization, including failure.
        // Neither step consults the mutable ambient provider selection.
        let synchronize = catch_allocation_cleanup_unwind(|| {
            self.provider
                .synchronize_native_device()
                .map_err(|error| error.to_string())
        })
        .map_err(|_| "provider synchronization panicked".to_string())
        .and_then(|result| result);
        let leave =
            catch_allocation_cleanup_unwind(|| guard.leave().map_err(|error| error.to_string()))
                .map_err(|_| "context exit panicked".to_string())
                .and_then(|result| result);
        self.exports.finish();
        match (synchronize, leave) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(synchronize), Err(leave)) => Err(format!(
                "CUDA synchronization failed: {synchronize}; context exit failed: {leave}"
            )),
            (Err(synchronize), Ok(())) => {
                Err(format!("CUDA synchronization failed: {synchronize}"))
            }
            (Ok(()), Err(leave)) => Err(format!("CUDA context exit failed: {leave}")),
        }
    }
}

impl Drop for NativeInvocationContext {
    fn drop(&mut self) {
        // Explicit ABI leave reports errors to MexCallState. Abandonment is a
        // best-effort safety net, with the same sync-before-pop ordering.
        let _ = catch_allocation_cleanup_unwind(|| self.cleanup());
    }
}
