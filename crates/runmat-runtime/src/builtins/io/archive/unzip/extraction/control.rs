use super::*;
use runmat_execution::cancellation::CancellationToken;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Only the caller owns this lease. Dropping the invocation stops a detached
/// worker at its next extraction boundary, even without token cancellation.
#[cfg(not(target_arch = "wasm32"))]
pub(super) struct Abandonment(Option<Arc<AtomicBool>>);

#[cfg(not(target_arch = "wasm32"))]
impl Abandonment {
    pub(super) fn new() -> Self {
        Self(Some(Arc::new(AtomicBool::new(false))))
    }
    pub(super) fn worker_flag(&self) -> Arc<AtomicBool> {
        self.0.as_ref().expect("active extraction lease").clone()
    }
    pub(super) fn disarm(mut self) {
        self.0.take();
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Abandonment {
    fn drop(&mut self) {
        if let Some(flag) = &self.0 {
            flag.store(true, Ordering::Release);
        }
    }
}

pub(super) struct ExtractionControl {
    cancellation: Option<CancellationToken>,
    #[cfg(not(target_arch = "wasm32"))]
    abandoned: Arc<AtomicBool>,
    #[cfg(target_arch = "wasm32")]
    budget: Option<crate::execution::CpuCooperationBudget>,
}

impl ExtractionControl {
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn native(
        cancellation: Option<CancellationToken>,
        abandoned: Arc<AtomicBool>,
    ) -> Self {
        Self {
            cancellation,
            abandoned,
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn browser(runtime: Option<&crate::context::RuntimeContext>) -> Self {
        Self {
            cancellation: runtime.map(crate::context::RuntimeContext::cancellation),
            budget: runtime.map(crate::context::RuntimeContext::cpu_cooperation),
        }
    }

    pub(super) async fn completed(&mut self, work: u64) -> BuiltinResult<()> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(cancelled());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.abandoned.load(Ordering::Acquire) {
            return Err(cancelled());
        }
        #[cfg(target_arch = "wasm32")]
        if let Some(budget) = &mut self.budget {
            budget.checkpoint(work).await.map_err(|error| match error {
                crate::execution::ExecutionServiceError::Cancelled => cancelled(),
                other => other.into_runtime_error(),
            })?;
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = work;
        Ok(())
    }
}

fn cancelled() -> RuntimeError {
    crate::runtime_error::semantic_error("RunMat:Cancelled", "unzip invocation was cancelled")
}
