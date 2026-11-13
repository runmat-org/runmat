//! Session/epoch provenance and optional real-device control. No provider
//! registry or execution authority is inferred from device names.
use crate::{AccelProvider, AccelProviderFuture, AcceleratorThreadBound};
use once_cell::sync::OnceCell;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceAllocationScope {
    session: u64,
    epoch: u64,
}
impl DeviceAllocationScope {
    pub fn new(session: u64, epoch: u64) -> Option<Self> {
        (session != 0 && epoch != 0).then_some(Self { session, epoch })
    }
    pub fn session(self) -> u64 {
        self.session
    }
    pub fn epoch(self) -> u64 {
        self.epoch
    }
    pub fn successor(self) -> Option<Self> {
        Some(Self {
            session: self.session,
            epoch: self.epoch.checked_add(1)?,
        })
    }
    /// Reset is a session retirement boundary, including earlier epochs left
    /// in this provider by a previous selection, never a peer-session wildcard.
    pub fn covers(self, allocation: Self) -> bool {
        self.session == allocation.session && allocation.epoch <= self.epoch
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceAllocationAdmission {
    Unscoped,
    Scoped(DeviceAllocationScope),
    Fenced,
}
type ScopeResolver = fn() -> DeviceAllocationAdmission;
static SCOPE_RESOLVER: OnceCell<ScopeResolver> = OnceCell::new();
pub fn register_device_allocation_scope_resolver(resolve: ScopeResolver) {
    if let Some(existing) = SCOPE_RESOLVER.get() {
        assert!(
            std::ptr::fn_addr_eq(*existing, resolve),
            "device scope resolver must be unique"
        );
    } else if SCOPE_RESOLVER.set(resolve).is_err() {
        register_device_allocation_scope_resolver(resolve);
    }
}
pub fn current_device_allocation_scope() -> Option<DeviceAllocationScope> {
    match SCOPE_RESOLVER.get().map(|resolve| resolve()) {
        Some(DeviceAllocationAdmission::Scoped(scope)) => Some(scope),
        _ => None,
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaleDeviceAllocationScope;
impl std::fmt::Display for StaleDeviceAllocationScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GPU allocation belongs to another session or a retired device epoch")
    }
}
impl std::error::Error for StaleDeviceAllocationScope {}
/// Compare the creation stamp, not the eventual registration task's new epoch.
pub fn validate_device_allocation_scope(
    scope: Option<DeviceAllocationScope>,
) -> Result<(), StaleDeviceAllocationScope> {
    let admission = SCOPE_RESOLVER
        .get()
        .map_or(DeviceAllocationAdmission::Unscoped, |resolve| resolve());
    if scope_matches_admission(scope, admission) {
        Ok(())
    } else {
        Err(StaleDeviceAllocationScope)
    }
}
fn scope_matches_admission(
    scope: Option<DeviceAllocationScope>,
    admission: DeviceAllocationAdmission,
) -> bool {
    match admission {
        DeviceAllocationAdmission::Fenced => false,
        DeviceAllocationAdmission::Unscoped => scope.is_none(),
        DeviceAllocationAdmission::Scoped(current) => scope == Some(current),
    }
}
#[derive(Debug)]
pub struct ProviderGpuDeviceFacts<'a> {
    pub name: &'a str,
    pub available: bool,
    /// Actual backend measurements only, never an allocation limit.
    pub total_memory_bytes: Option<u64>,
    pub available_memory_bytes: Option<u64>,
    pub supports_double: Option<bool>,
}
pub trait ProviderDeviceControl: AcceleratorThreadBound {
    fn gpu_device_facts(&self) -> ProviderGpuDeviceFacts<'_>;
    /// Query the actual allocation/pool owner, including failed and unpublished
    /// records. Unsupported owners must not report a fabricated empty scope.
    fn has_device_scope(&self, _scope: DeviceAllocationScope) -> Option<bool> {
        None
    }
    /// Physical completion state and boxed control future; claim stays with
    /// callback/future owners after cancellation, not an ambient TLS budget.
    fn control_host_bytes(&self) -> Option<u64> {
        None
    }
    /// Explicit reset capability and complete peak quote, including the
    /// allocation owner's exact retirement-ID selection. None is unsupported.
    fn reset_host_bytes(&self, _scope: DeviceAllocationScope) -> Option<u64> {
        None
    }
    /// Request backing which can remain live after signaling readiness and
    /// returning the control future. None means overlap cannot be admitted.
    fn control_completion_tail_bytes(&self) -> Option<u64> {
        None
    }
    fn wait_for_device(
        &self,
        _claim: Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
    ) -> Option<AccelProviderFuture<'_, ()>> {
        None
    }
    /// None explicitly means unsupported; never a successful no-op reset.
    fn reset_device_scope(
        &self,
        _scope: DeviceAllocationScope,
        _claim: Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
    ) -> Option<AccelProviderFuture<'_, ()>> {
        None
    }
}
pub fn device_control_provider_is_authorized(provider: &dyn AccelProvider) -> bool {
    !matches!(
        super::EXECUTION_PROVIDER_AUTHORIZER.get().map(|authorize| {
            authorize(crate::ExecutionProviderAdmission::DeviceControl(provider))
        }),
        Some(crate::ExecutionProviderAuthorization::Denied)
    )
}
#[cfg(feature = "wgpu")]
pub(crate) fn device_access_is_authorized(access: &crate::WgpuDeviceAccess) -> bool {
    !matches!(
        super::EXECUTION_PROVIDER_AUTHORIZER.get().map(|authorize| {
            authorize(crate::ExecutionProviderAdmission::DeviceAccess(access))
        }),
        Some(crate::ExecutionProviderAuthorization::Denied)
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scope_epoch_is_exact_and_never_wraps() {
        assert!(DeviceAllocationScope::new(0, 1).is_none());
        let first = DeviceAllocationScope::new(7, 1).unwrap();
        assert_eq!(first.successor().unwrap().session(), 7);
        assert_ne!(first, first.successor().unwrap());
        assert_ne!(first, DeviceAllocationScope::new(8, 1).unwrap());
        assert!(first.successor().unwrap().covers(first));
        assert!(!first.covers(first.successor().unwrap()));
        assert!(!first.covers(DeviceAllocationScope::new(8, 1).unwrap()));
        assert!(DeviceAllocationScope::new(7, u64::MAX)
            .unwrap()
            .successor()
            .is_none());
    }
    #[test]
    fn fenced_missing_stamp_is_not_standalone_and_old_publication_is_rejected() {
        let old = DeviceAllocationScope::new(7, 1).unwrap();
        assert!(scope_matches_admission(
            None,
            DeviceAllocationAdmission::Unscoped
        ));
        assert!(!scope_matches_admission(
            None,
            DeviceAllocationAdmission::Fenced
        ));
        assert!(!scope_matches_admission(
            Some(old),
            DeviceAllocationAdmission::Fenced
        ));
        assert!(!scope_matches_admission(
            Some(old),
            DeviceAllocationAdmission::Scoped(old.successor().unwrap())
        ));
    }
}
