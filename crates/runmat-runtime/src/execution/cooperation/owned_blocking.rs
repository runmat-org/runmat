//! Control admission for the existing bounded owned CPU lane.
use super::{CpuCooperationBudget, ExecutionServiceError};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPolicy, HostAllocationReceipt},
    RuntimeContext,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::execution::blocking_cpu::OwnedCpuJob;
use crate::execution::blocking_cpu::RuntimeBlockingCpuService;

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "owned_blocking/browser_tests.rs"]
mod browser_tests;

/// Values transferred to the physical owned CPU lane. Native jobs cross a
/// thread boundary; the browser lane executes inline on its original worker.
#[cfg(not(target_arch = "wasm32"))]
pub trait OwnedCpuPayload: Send + 'static {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + 'static> OwnedCpuPayload for T {}

/// Browser payloads remain local; this does not grant native thread transport.
#[cfg(target_arch = "wasm32")]
pub trait OwnedCpuPayload: 'static {}
#[cfg(target_arch = "wasm32")]
impl<T: 'static> OwnedCpuPayload for T {}

impl CpuCooperationBudget {
    /// Ordinary embedding adapter. Missing context is not a finite grant.
    pub async fn run_owned_blocking<T, G, F>(
        &self,
        service: Option<&dyn RuntimeBlockingCpuService>,
        retained: G,
        work: F,
    ) -> Result<T, ExecutionServiceError>
    where
        T: OwnedCpuPayload,
        G: OwnedCpuPayload,
        F: FnOnce() -> T + OwnedCpuPayload,
    {
        let runtime = crate::context::legacy::active();
        if HostAllocationReceipt::policy(runtime.as_ref())
            .map_err(ExecutionServiceError::HostAllocation)?
            != HostAllocationPolicy::Unbounded
        {
            return Err(ExecutionServiceError::HostAllocation(
                HostAllocationError::MissingManagedOwner,
            ));
        }
        self.run_selected(runtime.as_ref(), service, retained, work)
            .await
    }

    /// Retain both numerical and control storage through physical kernel exit.
    /// Cancellation retires this await, never an accepted worker's ownership.
    pub async fn run_owned_blocking_in_context<T, G, F>(
        &self,
        runtime: &RuntimeContext,
        service: Option<&dyn RuntimeBlockingCpuService>,
        retained: G,
        work: F,
    ) -> Result<T, ExecutionServiceError>
    where
        T: OwnedCpuPayload,
        G: OwnedCpuPayload,
        F: FnOnce() -> T + OwnedCpuPayload,
    {
        self.run_selected(Some(runtime), service, retained, work)
            .await
    }

    async fn run_selected<T, G, F>(
        &self,
        runtime: Option<&RuntimeContext>,
        service: Option<&dyn RuntimeBlockingCpuService>,
        retained: G,
        work: F,
    ) -> Result<T, ExecutionServiceError>
    where
        T: OwnedCpuPayload,
        G: OwnedCpuPayload,
        F: FnOnce() -> T + OwnedCpuPayload,
    {
        if self.cancellation.is_cancelled() {
            return Err(ExecutionServiceError::Cancelled);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let service = service.ok_or_else(|| {
                ExecutionServiceError::Infrastructure(
                    "owned blocking CPU service was not installed by this host".into(),
                )
            })?;
            let (send, receive) =
                crate::execution::blocking_cpu::completion::channel(runtime, retained)?;
            let job = OwnedCpuJob::prepare_kernel(runtime, send, self.cancellation.clone(), work)?;
            self.await_cancellable(service.submit(runtime, job)?)
                .await??;
            self.await_cancellable(receive).await?
        }
        #[cfg(target_arch = "wasm32")]
        {
            // Same-thread portable kernels have no job/channel heap boundary.
            let _ = service;
            let owner = HostAllocationReceipt::capture_owner(runtime)
                .map_err(ExecutionServiceError::HostAllocation)?;
            if let Some(owner) = &owner {
                owner
                    .check(runtime.expect("captured managed CPU owner"))
                    .map_err(ExecutionServiceError::HostAllocation)?;
            }
            let _retained = retained;
            let result = work();
            if let Some(owner) = &owner {
                owner
                    .check(runtime.expect("captured managed CPU owner"))
                    .map_err(ExecutionServiceError::HostAllocation)?;
            }
            if self.cancellation.is_cancelled() {
                return Err(ExecutionServiceError::Cancelled);
            }
            Ok(result)
        }
    }
}
