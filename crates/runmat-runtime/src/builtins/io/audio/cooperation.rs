//! Audio-owned work accounting; the runtime execution service owns yield policy.

use crate::context::host_effect::{
    stale_resource, HostEffectResourceLivenessLease, HostEffectStageReservation,
};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use std::future::Future;
use std::sync::Arc;

pub(super) struct AudioCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    builtin: &'static str,
    cancelled: bool,
    resource_liveness_lease: Option<Arc<dyn HostEffectResourceLivenessLease>>,
    stage_reservation: Option<Arc<dyn HostEffectStageReservation>>,
}

impl AudioCooperation {
    pub(super) fn new(runtime: Option<&RuntimeContext>, builtin: &'static str) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
            builtin,
            cancelled: false,
            resource_liveness_lease: None,
            stage_reservation: None,
        }
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget, builtin: &'static str) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".into()),
            builtin,
            cancelled: false,
            resource_liveness_lease: None,
            stage_reservation: None,
        }
    }

    /// True for a typed cancellation from either the invocation token or its
    /// execution service; no diagnostic-string inference is needed by feedback.
    pub(super) fn was_cancelled(&self) -> bool {
        self.cancelled
    }

    pub(super) fn install_resource_liveness_lease(
        &mut self,
        lease: Arc<dyn HostEffectResourceLivenessLease>,
    ) {
        self.resource_liveness_lease = Some(lease);
    }

    pub(super) fn install_stage_reservation(
        &mut self,
        reservation: Arc<dyn HostEffectStageReservation>,
    ) {
        self.stage_reservation = Some(reservation);
    }

    fn check_resource_liveness(&self) -> BuiltinResult<()> {
        if self
            .resource_liveness_lease
            .as_ref()
            .is_some_and(|permit| !permit.is_current())
            || self
                .stage_reservation
                .as_ref()
                .is_some_and(|permit| !permit.is_current())
        {
            Err(stale_resource())
        } else {
            Ok(())
        }
    }

    /// Suspend on provider I/O without losing the invocation's cancellation
    /// wakeup. The caller retains the original I/O error for its own descriptor.
    pub(super) async fn await_io<T, E, F>(&mut self, future: F) -> BuiltinResult<Result<T, E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        self.check_resource_liveness()?;
        let Some(budget) = self.budget.as_ref() else {
            let value = future.await;
            if value.is_ok() {
                self.check_resource_liveness()?;
            }
            return Ok(value);
        };
        let outcome = budget.await_cancellable(future).await;
        let value = outcome.map_err(|error| self.runtime_error(error))?;
        if value.is_ok() {
            self.check_resource_liveness()?;
        }
        Ok(value)
    }

    pub(super) async fn completed(&mut self, units: usize) -> BuiltinResult<()> {
        self.check_resource_liveness()?;
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        let outcome = budget.checkpoint(units as u64).await;
        outcome.map_err(|error| self.runtime_error(error))?;
        self.check_resource_liveness()
    }

    fn runtime_error(&mut self, error: ExecutionServiceError) -> RuntimeError {
        match error {
            ExecutionServiceError::Cancelled => {
                self.cancelled = true;
                build_runtime_error(format!("{}: execution was cancelled", self.builtin))
                    .with_identifier(format!(
                        "{}:Cancelled",
                        self.namespace.as_deref().unwrap_or("RunMat")
                    ))
                    .with_builtin(self.builtin)
                    .build()
            }
            other => other.into_runtime_error(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::CpuCooperationPolicy;
    use futures::executor::block_on;
    use runmat_execution::cancellation::CancellationSource;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Revocable(Arc<AtomicBool>);
    impl HostEffectResourceLivenessLease for Revocable {
        fn is_current(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
    }

    #[test]
    fn resource_revocation_during_await_rejects_before_audio_publication() {
        let valid = Arc::new(AtomicBool::new(true));
        let mut cooperation = AudioCooperation::new(None, "audioread");
        cooperation.install_resource_liveness_lease(Arc::new(Revocable(valid.clone())));
        let error = block_on(cooperation.await_io(async {
            valid.store(false, Ordering::SeqCst);
            Ok::<_, std::io::Error>(7u8)
        }))
        .expect_err("revoked authority");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:StaleResourceAuthority")
        );
    }

    #[test]
    fn provider_error_remains_available_when_lease_revokes_during_await() {
        let valid = Arc::new(AtomicBool::new(true));
        let mut cooperation = AudioCooperation::new(None, "audioinfo");
        cooperation.install_resource_liveness_lease(Arc::new(Revocable(valid.clone())));
        let error = block_on(cooperation.await_io(async {
            valid.store(false, Ordering::SeqCst);
            Err::<(), _>(std::io::Error::other("opened provider failed"))
        }))
        .expect("provider result retains its original cause")
        .expect_err("provider error");
        assert_eq!(error.to_string(), "opened provider failed");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn service_cancelled_is_typed_even_when_invocation_token_is_not_marked() {
        let source = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            || Box::pin(async { Err(ExecutionServiceError::Cancelled) }),
        );
        let mut cooperation = AudioCooperation::with_budget(budget, "audioread");
        let error = block_on(cooperation.completed(1)).expect_err("service cancellation");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert!(cooperation.was_cancelled());
        assert!(!source.token().is_cancelled());
    }
}
