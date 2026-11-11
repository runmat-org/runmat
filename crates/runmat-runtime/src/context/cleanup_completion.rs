//! Scoped value-retirement completion. Host shutdown keeps its global drain.
use std::future::Future;
use std::rc::Rc;
use std::task::Poll;

use super::RuntimeContext;
use crate::builtins::introspection::on_cleanup::lifecycle::{
    dispatch,
    registry::{CleanupCompletionId, CleanupRegistry},
};
use crate::RuntimeError;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

/// Inline lazy authority: ordinary numeric execution retains no runtime clone,
/// registry alias or cohort ID. Existing registry entries own callback storage.
pub struct RuntimeFunctionCleanup {
    runtime: Option<RuntimeContext>,
    registry: Option<Rc<CleanupRegistry>>,
    completion: Option<CleanupCompletionId>,
}

impl RuntimeContext {
    pub fn has_cleanup_candidates(&self) -> bool {
        self.cleanup_registry()
            .is_some_and(|registry| registry.has_active())
    }

    /// The synchronous producer brackets its actual value/root mutation.
    pub fn begin_value_retirement(&self) -> Result<RuntimeFunctionCleanup, RuntimeError> {
        let mut cleanup = self.function_cleanup_scope()?;
        // A native site can suspend before it mutates a value. Select its
        // completion now, but queue work only after the synchronous site has
        // actually retired roots; an early exit then cannot strand a receipt.
        cleanup.selected_context(self)?;
        Ok(cleanup)
    }

    pub fn function_cleanup_scope(&self) -> Result<RuntimeFunctionCleanup, RuntimeError> {
        Ok(RuntimeFunctionCleanup {
            runtime: None,
            registry: None,
            completion: None,
        })
    }
}

impl RuntimeFunctionCleanup {
    /// Select one completion before any root release can queue a callback.
    /// A numeric site with no active cleanup target retains no context or ID.
    pub fn selected_context<'a>(
        &'a mut self,
        runtime: &'a RuntimeContext,
    ) -> Result<&'a RuntimeContext, RuntimeError> {
        if self.runtime.is_none() {
            if let Some(registry) = runtime
                .cleanup_registry()
                .filter(|registry| registry.has_active())
            {
                let completion = registry.issue_completion()?;
                self.runtime = Some(runtime.clone().with_cleanup_completion(Some(completion)));
                self.registry = Some(registry);
                self.completion = Some(completion);
            }
        }
        Ok(self.runtime.as_ref().unwrap_or(runtime))
    }

    fn before_retirement(&mut self, runtime: &RuntimeContext) -> Result<(), RuntimeError> {
        self.selected_context(runtime)?;
        let selected = self.runtime.as_ref().unwrap_or(runtime);
        if let (Some(registry), Some(completion)) = (&self.registry, self.completion) {
            dispatch::schedule_completion(selected, registry, &[], completion)?;
        }
        Ok(())
    }

    pub fn admit_retired_values(&mut self, runtime: &RuntimeContext) -> Result<bool, RuntimeError> {
        // Check after mutation too: the instruction may have created its first
        // cleanup target while staying wholly ready.
        self.selected_context(runtime)?;
        let selected = self.runtime.as_ref().unwrap_or(runtime);
        let (Some(registry), Some(completion)) = (&self.registry, self.completion) else {
            return Ok(false);
        };
        dispatch::schedule_completion(selected, registry, &[], completion)?;
        Ok(registry.has_completion_work(completion))
    }

    pub async fn finish(self) -> Result<(), RuntimeError> {
        let (Some(runtime), Some(registry), Some(completion)) =
            (self.runtime, self.registry, self.completion)
        else {
            return Ok(());
        };
        dispatch::drain_completion(&runtime, &registry, completion)
            .await
            .map_err(no_replay)
    }

    /// Bracket actual callee polls, not a receipt waiting on another driver.
    /// The future is destroyed before final admission, retaining its rooted
    /// successor outputs. No executor task interleaves within a synchronous poll.
    pub async fn run<F, T>(
        self,
        runtime: &RuntimeContext,
        future: F,
    ) -> Result<(T, Result<(), RuntimeError>), RuntimeError>
    where
        F: Future<Output = T>,
    {
        self.run_with_post_poll(runtime, future, |_| Ok(())).await
    }

    /// The post-poll hook may publish a frame after its instruction has
    /// changed slots. It receives this token's exact completion context before
    /// the publication can trigger cleanup reachability observers.
    pub async fn run_with_post_poll<F, T, H>(
        mut self,
        runtime: &RuntimeContext,
        future: F,
        mut post_poll: H,
    ) -> Result<(T, Result<(), RuntimeError>), RuntimeError>
    where
        F: Future<Output = T>,
        H: FnMut(&RuntimeContext) -> Result<(), RuntimeError>,
    {
        let result = {
            // Declared before the future so its final destructor still runs
            // under the selected completion, without a guard across Pending.
            let selected_drop_guard;
            let mut future = std::pin::pin!(Some(future));
            let result = futures::future::poll_fn(|context| {
                if let Err(error) = self.before_retirement(runtime) {
                    return Poll::Ready(Err(error));
                }
                let result = {
                    // The callee's existing scope remains authoritative until
                    // this token actually owns work. The empty path adds no
                    // RuntimeContext clone or TLS scope activation.
                    let _selected = self.runtime.as_ref().map(RuntimeContext::enter);
                    future
                        .as_mut()
                        .as_pin_mut()
                        .expect("cleanup future remains present until completion")
                        .poll(context)
                };
                let publication = if result.is_ready() {
                    match self.selected_context(runtime) {
                        Ok(_) => {
                            let _selected = self.runtime.as_ref().map(RuntimeContext::enter);
                            // Pin::set destroys the completed future in place,
                            // including !Unpin captures, before publication
                            // inspects surviving owners. Numeric calls keep
                            // their clone-free empty completion path.
                            future.as_mut().set(None);
                            post_poll(self.runtime.as_ref().unwrap_or(runtime))
                        }
                        Err(error) => {
                            let _original = runtime.enter();
                            future.as_mut().set(None);
                            Err(error)
                        }
                    }
                } else {
                    Ok(())
                };
                let admission = self.admit_retired_values(runtime).map(|_| ());
                let admission = crate::runtime_error::complete_with_cleanup(publication, admission);
                match result {
                    Poll::Ready(value) => Poll::Ready(Ok((value, admission))),
                    Poll::Pending => match admission {
                        Ok(()) => Poll::Pending,
                        Err(error) => Poll::Ready(Err(error)),
                    },
                }
            })
            .await;
            selected_drop_guard = future
                .as_ref()
                .get_ref()
                .is_some()
                .then(|| self.runtime.as_ref().unwrap_or(runtime).enter());
            let _ = &selected_drop_guard;
            result
        };
        let retirement = self.admit_retired_values(runtime).map(|_| ());
        let drained = self.finish().await;
        let cleanup =
            crate::runtime_error::complete_with_cleanup(retirement, drained).map_err(no_replay);
        match result {
            Ok((value, admission)) => Ok((
                value,
                crate::runtime_error::complete_with_cleanup(admission, cleanup).map_err(no_replay),
            )),
            Err(error) => {
                crate::runtime_error::complete_with_cleanup(Err(error), cleanup).map_err(no_replay)
            }
        }
    }
}

fn no_replay(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
