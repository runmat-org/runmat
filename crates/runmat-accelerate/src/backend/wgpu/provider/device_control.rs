//! Queue completion and actual per-session physical allocation reset.
use super::*;
use runmat_accelerate_api::observation::WgpuObservationProgressError;
use runmat_accelerate_api::{DeviceAllocationScope, ProviderDeviceControl, ProviderGpuDeviceFacts};
use runmat_execution::resource::{ExactShared, RetainedHostAllocationClaim};
use std::task::{Poll, Waker};

#[derive(Default)]
struct CompletionState {
    result: Option<std::result::Result<(), WgpuObservationProgressError>>,
    waker: Option<Waker>,
}
struct Completion {
    state: Mutex<CompletionState>,
}
#[derive(Clone)]
pub(super) struct FundedCompletion {
    state: ExactShared<Completion>,
    // Field order keeps the precharged allocation live until state destruction.
    _claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}
impl FundedCompletion {
    pub(super) fn new(claim: Option<Arc<dyn RetainedHostAllocationClaim>>) -> Result<Self> {
        Ok(Self {
            state: ExactShared::try_new(Completion {
                state: Mutex::new(CompletionState::default()),
            })
            .map_err(|_| anyhow!("WGPU device completion allocation failed"))?,
            _claim: claim,
        })
    }
    pub(super) fn host_bytes() -> Option<u64> {
        ExactShared::<Completion>::requested_bytes()
            .checked_add(std::mem::size_of::<FundedCompletion>())?
            .try_into()
            .ok()
    }
    pub(super) fn complete(&self, result: std::result::Result<(), WgpuObservationProgressError>) {
        let wake = {
            let mut state = self
                .state
                .get()
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.result = Some(result);
            state.waker.take()
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }
    pub(super) async fn wait(&self) -> std::result::Result<(), WgpuObservationProgressError> {
        std::future::poll_fn(|cx| {
            let replacement = cx.waker().clone();
            let (ready, prior) = {
                let mut state = self
                    .state
                    .get()
                    .state
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                if let Some(result) = state.result {
                    (Some(result), state.waker.take())
                } else {
                    (None, state.waker.replace(replacement))
                }
            };
            drop(prior);
            if let Some(result) = ready {
                Poll::Ready(result)
            } else {
                Poll::Pending
            }
        })
        .await
    }
}
impl WgpuProvider {
    async fn control_device(
        &self,
        scope: Option<DeviceAllocationScope>,
        claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.device_health.loss_reason().is_none() && !self.allocations.is_lost(),
            "WGPU device is unavailable"
        );
        let completion = FundedCompletion::new(claim)?;
        let callback = completion.clone();
        #[cfg(target_arch = "wasm32")]
        self.access
            .on_submitted_work_done(move || callback.complete(Ok(())))?;
        #[cfg(not(target_arch = "wasm32"))]
        self.observation_progress
            .as_ref()
            .ok_or_else(|| anyhow!("WGPU device completion progress is unavailable"))?
            .complete_queue_tail(Box::new(move |result| callback.complete(result)))
            .map_err(anyhow::Error::from)?;
        completion.wait().await.map_err(anyhow::Error::from)?;
        anyhow::ensure!(
            self.device_health.loss_reason().is_none() && !self.allocations.is_lost(),
            "WGPU device was lost while waiting for completion"
        );
        if let Some(scope) = scope {
            self.allocations.reset_scope(scope, &self.access).await?;
        }
        Ok(())
    }
}
impl ProviderDeviceControl for WgpuProvider {
    fn control_completion_tail_bytes(&self) -> Option<u64> {
        // The bounded progress channel/job slot is provider-owned backing
        // present before the request; only these request allocations can
        // outlive readiness. Do not charge a second copy of shared state.
        FundedCompletion::host_bytes()
    }
    fn reset_host_bytes(&self, scope: DeviceAllocationScope) -> Option<u64> {
        self.control_host_bytes()?
            .checked_add(self.allocations.reset_selection_bytes(scope)?)
    }
    fn has_device_scope(&self, scope: DeviceAllocationScope) -> Option<bool> {
        self.allocations.has_scope(scope).ok()
    }
    fn gpu_device_facts(&self) -> ProviderGpuDeviceFacts<'_> {
        let available = self.device_health.loss_reason().is_none() && !self.allocations.is_lost();
        #[cfg(not(target_arch = "wasm32"))]
        let available = available && !self.access.is_emission_fenced();
        ProviderGpuDeviceFacts {
            name: &self.adapter_info.name,
            available,
            total_memory_bytes: None,
            available_memory_bytes: None,
            supports_double: Some(self.enabled_features.contains(wgpu::Features::SHADER_F64)),
        }
    }
    fn control_host_bytes(&self) -> Option<u64> {
        // Completion retains the callback through physical exit, including
        // cancellation. Quote that backing once, together with this future.
        FundedCompletion::host_bytes()?.checked_add(
            u64::try_from(std::mem::size_of_val(&self.control_device(None, None))).ok()?,
        )
    }
    fn wait_for_device(
        &self,
        claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> Option<AccelProviderFuture<'_, ()>> {
        Some(Box::pin(self.control_device(None, claim)))
    }
    fn reset_device_scope(
        &self,
        scope: DeviceAllocationScope,
        claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> Option<AccelProviderFuture<'_, ()>> {
        Some(Box::pin(self.control_device(Some(scope), claim)))
    }
}
