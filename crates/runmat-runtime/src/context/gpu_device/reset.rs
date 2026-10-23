//! The existing host supervisor, not the awaiting caller, owns reset to its
//! terminal disposition. Reset is ordered retirement, not a rollback promise.
use super::{GpuDeviceSelection, RuntimeContext};
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::{local_task_admission::LocalTaskCommandOrigin, RuntimeLocalTaskLease};
use crate::runtime_error::RuntimeErrorOwner;
use crate::{BuiltinResult, RuntimeError};
use runmat_accelerate_api::{AccelProvider, DeviceAllocationScope};
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::{cell::RefCell, sync::Arc};

#[derive(Default)]
pub(crate) struct GpuDeviceResetState {
    lease: RefCell<Option<RuntimeLocalTaskLease>>,
    failure: RefCell<Option<RuntimeErrorOwner>>,
}
impl std::fmt::Debug for GpuDeviceResetState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDeviceResetState")
            .field("failed", &self.failure.borrow().is_some())
            .finish_non_exhaustive()
    }
}
fn error(message: impl Into<String>) -> RuntimeError {
    crate::runtime_error::semantic_error("gpuDevice:Lifecycle", message)
}
fn same(left: &'static dyn AccelProvider, right: &'static dyn AccelProvider) -> bool {
    runmat_accelerate_api::same_provider_instance(left, right)
}
fn include_control_quote(
    non_tail: &mut u64,
    tails: &mut u64,
    bytes: u64,
    tail: u64,
) -> BuiltinResult<()> {
    let body = bytes
        .checked_sub(tail)
        .ok_or_else(|| error("GPU control quote omits its completion tail"))?;
    let all_tails = tails
        .checked_add(tail)
        .ok_or_else(|| error("GPU completion-tail geometry overflow"))?;
    *non_tail = (*non_tail).max(body);
    *tails = all_tails;
    Ok(())
}
fn relevant(
    provider: &'static dyn AccelProvider,
    target: &'static dyn AccelProvider,
    previous: Option<&'static dyn AccelProvider>,
    scope: DeviceAllocationScope,
) -> BuiltinResult<bool> {
    let required = same(provider, target) || previous.is_some_and(|old| same(provider, old));
    match provider.gpu_device_control() {
        Some(control) => control
            .has_device_scope(scope)
            .map(|owned| owned || required)
            .ok_or_else(|| {
                error("GPU provider cannot determine its session allocation disposition")
            }),
        None if required => Err(error(
            "Selected GPU provider does not support device lifecycle control",
        )),
        // Backends without this capability do not issue session+epoch WGPU
        // allocations. They cannot be selected/reset as GPUDevice objects.
        None => Ok(false),
    }
}

struct ResetGuard {
    runtime: RuntimeContext,
    target: &'static dyn AccelProvider,
    complete: bool,
    failure_header_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}
fn retain_failure(
    cause: RuntimeError,
    claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
) -> RuntimeErrorOwner {
    match claim {
        None => RuntimeErrorOwner::new(cause),
        Some(claim) => RuntimeErrorOwner::try_new_exact(cause, |_| {
            Ok::<_, std::convert::Infallible>(runmat_async::RetainedDiagnosticAllocation::Owned(
                Box::new(claim),
            ))
        })
        .unwrap_or_else(|_| {
            std::alloc::handle_alloc_error(std::alloc::Layout::new::<RuntimeError>())
        }),
    }
}
impl Drop for ResetGuard {
    fn drop(&mut self) {
        if !self.complete
            && matches!(self.runtime.gpu_device_selection(),
            GpuDeviceSelection::Resetting(active) if same(active, self.target))
        {
            self.runtime
                .state()
                .gpu_device_selection
                .set(GpuDeviceSelection::ResetFailed(self.target));
            // Backend callback/record owners retain their original physical
            // claims. Shutdown must never restore selection or advance epoch.
            if self
                .runtime
                .state()
                .gpu_device_reset
                .failure
                .borrow()
                .is_none()
            {
                *self.runtime.state().gpu_device_reset.failure.borrow_mut() = Some(retain_failure(
                    error(
                        "GPU reset task stopped before terminal retirement; session remains fenced",
                    ),
                    self.failure_header_claim.take(),
                ));
            }
        }
    }
}

async fn execute(
    runtime: RuntimeContext,
    target: &'static dyn AccelProvider,
    previous: Option<&'static dyn AccelProvider>,
    scope: DeviceAllocationScope,
    deselect: bool,
    claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    mut guard: ResetGuard,
) {
    let result = async {
        let mut after = None;
        while let Some((id, provider)) = runmat_accelerate_api::next_registered_provider(after) {
            after = Some(id);
            if !relevant(provider, target, previous, scope)? {
                continue;
            }
            if !runmat_accelerate_api::device_control_provider_is_authorized(provider) {
                return Err(error(
                    "GPU reset owner no longer has an execution device lease",
                ));
            }
            let control = provider
                .gpu_device_control()
                .ok_or_else(|| error("GPU reset capability disappeared"))?;
            let future = control
                .reset_device_scope(scope, claim.clone())
                .ok_or_else(|| error("GPU provider does not implement physical session reset"))?;
            future.await.map_err(|cause| {
                crate::runtime_error::with_provider_execution_source(
                    error("GPU reset failed after ordered owner retirement"),
                    runmat_accelerate_api::ProviderExecutionError::from(cause).after_effect(),
                )
            })?;
        }
        if !runtime.finish_gpu_device_reset(target, scope, deselect) {
            return Err(error(
                "GPU reset session boundary changed before completion",
            ));
        }
        Ok(())
    }
    .await;
    match result {
        Ok(()) => {
            guard.complete = true;
            runtime.state().gpu_device_reset.failure.borrow_mut().take();
        }
        Err(cause) => {
            let previous = runtime.state().gpu_device_reset.failure.borrow_mut().take();
            let cause = if let Some(previous) = previous {
                // A retry is subsequent physical cleanup, not permission to
                // discard the original failed disposition/source owner.
                crate::runtime_error::complete_with_cleanup::<()>(
                    Err(RuntimeError::from_shared_original(previous)),
                    Err(cause),
                )
                .unwrap_err()
            } else {
                cause
            };
            *runtime.state().gpu_device_reset.failure.borrow_mut() =
                Some(retain_failure(cause, guard.failure_header_claim.take()));
        }
    }
}

impl RuntimeContext {
    #[cfg(test)]
    pub(crate) fn gpu_device_reset_lease_for_test(&self) -> Option<RuntimeLocalTaskLease> {
        self.state().gpu_device_reset.lease.borrow().clone()
    }
    pub(crate) async fn reset_gpu_device(
        &self,
        target: &'static dyn AccelProvider,
        deselect: bool,
    ) -> BuiltinResult<()> {
        if let GpuDeviceSelection::ResetFailed(failed) = self.gpu_device_selection() {
            if !same(failed, target) {
                return Err(error(
                    "Retry must retain the original failed GPU reset target",
                ));
            }
        }
        if matches!(
            self.gpu_device_selection(),
            GpuDeviceSelection::Resetting(_)
        ) {
            return Err(error("GPU session reset is already in progress"));
        }
        let scope = DeviceAllocationScope::new(
            self.gpu_device_session_key(),
            self.state().gpu_device_epoch.get(),
        )
        .ok_or_else(|| error("GPU session scope is unavailable"))?;
        scope
            .successor()
            .ok_or_else(|| error("GPU session epoch exhausted"))?;
        let previous = match self.gpu_device_selection() {
            GpuDeviceSelection::Selected(provider) | GpuDeviceSelection::ResetFailed(provider) => {
                Some(provider)
            }
            _ => None,
        };
        let mut after = None;
        let (mut non_tail, mut tails) = (0u64, 0u64);
        let mut found = false;
        while let Some((id, provider)) = runmat_accelerate_api::next_registered_provider(after) {
            after = Some(id);
            if !relevant(provider, target, previous, scope)? {
                continue;
            }
            found |= same(provider, target);
            if !runmat_accelerate_api::device_control_provider_is_authorized(provider) {
                return Err(error(
                    "GPU session owner lacks the required execution device lease",
                ));
            }
            let bytes = provider
                .gpu_device_control()
                .and_then(|control| control.reset_host_bytes(scope))
                .ok_or_else(|| error("GPU provider cannot admit physical lifecycle control"))?;
            let tail = provider
                .gpu_device_control()
                .and_then(|control| control.control_completion_tail_bytes())
                .ok_or_else(|| error("GPU provider completion-tail ownership is not admitted"))?;
            include_control_quote(&mut non_tail, &mut tails, bytes, tail)?;
        }
        if !found {
            return Err(error(
                "GPU target is not a registered physical device owner",
            ));
        }
        // Prior native worker callbacks may still be dropping their captures
        // after readiness. Every exact tail can overlap the current non-tail;
        // selection/future metadata is sequential and is not multiplied.
        let bytes = non_tail
            .checked_add(tails)
            .ok_or_else(|| error("GPU reset control geometry overflow"))?;
        let allocation = |cause| {
            host_allocation_runtime_error(
                Some(self),
                "gpuDevice:HostAllocation",
                "GPU reset control admission failed",
                cause,
            )
        };
        let header_bytes = RuntimeErrorOwner::requested_header_bytes()
            .checked_add(std::mem::size_of::<Arc<dyn RetainedHostAllocationClaim>>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| error("GPU reset diagnostic header geometry overflow"))?;
        let retained = HostAllocationPlan::retained_metadata_bytes(bytes)
            .map_err(allocation)?
            .checked_add(
                HostAllocationPlan::retained_metadata_bytes(header_bytes).map_err(allocation)?,
            )
            .ok_or_else(|| error("GPU reset retained metadata geometry overflow"))?;
        let receipt = HostAllocationReceipt::admit(
            Some(self),
            HostAllocationPlan::new(0, retained).map_err(allocation)?,
        )
        .map_err(allocation)?;
        let claim = receipt
            .as_ref()
            .map(|receipt| receipt.claim_retained_metadata(self, bytes))
            .transpose()
            .map_err(allocation)?;
        let failure_header_claim = receipt
            .as_ref()
            .map(|receipt| receipt.claim_retained_metadata(self, header_bytes))
            .transpose()
            .map_err(allocation)?;
        // Guard is captured by the factory too: host rejection/shutdown cannot
        // strand Resetting, even if the task has never received its first poll.
        let guard = ResetGuard {
            runtime: self.clone(),
            target,
            complete: false,
            failure_header_claim,
        };
        let prepared = self.prepare_poll_scope()?.prepare_future::<_>()?;
        let future = prepared.try_scope_with(|| {
            execute(
                self.clone(),
                target,
                previous,
                scope,
                deselect,
                claim,
                guard,
            )
        })?;
        self.begin_gpu_device_reset(target)
            .ok_or_else(|| error("GPU reset admission changed"))?;
        let lease = self.spawn_local_with_command_origin(
            "GPU device reset",
            LocalTaskCommandOrigin::CurrentSelected,
            move |_| Box::pin(future),
        )?;
        *self.state().gpu_device_reset.lease.borrow_mut() = Some(lease.clone());
        lease.join().await;
        if let Some(cause) = self.state().gpu_device_reset.failure.borrow().as_ref() {
            return Err(RuntimeError::from_shared_original(cause.clone()));
        }
        if matches!(
            self.gpu_device_selection(),
            GpuDeviceSelection::Resetting(_) | GpuDeviceSelection::ResetFailed(_)
        ) {
            return Err(error("GPU reset did not reach physical completion"));
        }
        self.state().gpu_device_reset.lease.borrow_mut().take();
        Ok(())
    }
}

#[cfg(test)]
mod quote_tests {
    use super::*;
    #[test]
    fn sequential_control_preflight_keeps_all_completion_tails() {
        let (mut body, mut tails) = (0, 0);
        include_control_quote(&mut body, &mut tails, 100, 30).unwrap();
        include_control_quote(&mut body, &mut tails, 80, 20).unwrap();
        assert_eq!(body.checked_add(tails), Some(120));
        assert!(include_control_quote(&mut body, &mut tails, 1, 2).is_err());
        let prior = (body, tails);
        assert!(include_control_quote(&mut body, &mut tails, u64::MAX, u64::MAX).is_err());
        assert_eq!(
            (body, tails),
            prior,
            "invalid quote cannot publish partial geometry"
        );
    }
}
