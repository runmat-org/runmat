//! Task-local, source-route physical-owner audit. This is neither an execution
//! grant nor a cost estimate. A selected original VM region can learn only if
//! every observed provider selection and physical WGPU submission used its
//! bound owner; unknown or other-device work suppresses feedback.

use crate::AccelProvider;
#[cfg(feature = "wgpu")]
use crate::WgpuDeviceAccess;
use runmat_execution::resource::ExactShared;
use std::cell::{Cell, RefCell};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct RouteState {
    submissions: u64,
    conflicting_owner: bool,
}

struct AuditBacking {
    provider: &'static dyn AccelProvider,
    #[cfg(feature = "wgpu")]
    access: WgpuDeviceAccess,
    state: Mutex<RouteState>,
    _backing_claim: Option<Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
}

/// One exact-layout shared physical audit, not an execution authority.
#[derive(Clone)]
pub struct SourceRegionRouteAudit(ExactShared<AuditBacking>);

#[derive(Debug)]
pub struct SourceRegionAuditAllocationError;
impl std::fmt::Display for SourceRegionAuditAllocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("source region audit allocation failed")
    }
}
impl std::error::Error for SourceRegionAuditAllocationError {}

impl SourceRegionRouteAudit {
    #[cfg(feature = "wgpu")]
    pub fn try_new(
        provider: &'static dyn AccelProvider,
        access: WgpuDeviceAccess,
        backing_claim: Option<Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
    ) -> Result<Self, SourceRegionAuditAllocationError> {
        ExactShared::try_new(AuditBacking {
            provider,
            access,
            state: Mutex::new(RouteState::default()),
            _backing_claim: backing_claim,
        })
        .map(Self)
        .map_err(|_| SourceRegionAuditAllocationError)
    }

    pub fn backing_bytes() -> Option<u64> {
        u64::try_from(ExactShared::<AuditBacking>::requested_bytes()).ok()
    }

    pub fn exact_single_provider(&self) -> bool {
        let state = self
            .0
            .get()
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.submissions > 0 && !state.conflicting_owner
    }

    fn conflict(&self) {
        self.0
            .get()
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .conflicting_owner = true;
    }

    fn selected(&self, provider: &dyn AccelProvider) {
        if !crate::same_provider_instance(self.0.get().provider, provider) {
            self.0
                .get()
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .conflicting_owner = true;
        }
    }

    #[cfg(feature = "wgpu")]
    fn submitted(&self, access: &WgpuDeviceAccess) {
        let mut state = self
            .0
            .get()
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !self.0.get().access.same_device(access) {
            state.conflicting_owner = true;
        } else if let Some(next) = state.submissions.checked_add(1) {
            state.submissions = next;
        } else {
            state.conflicting_owner = true;
        }
    }
}

pub type SourceRegionRouteAuditSlot = Arc<Mutex<Option<SourceRegionRouteAudit>>>;

thread_local! {
    static ACTIVE: RefCell<Option<SourceRegionRouteAudit>> = const { RefCell::new(None) };
    static INHERITED: RefCell<Option<SourceRegionRouteAudit>> = const { RefCell::new(None) };
    static POLL_OWNER: Cell<bool> = const { Cell::new(false) };
}

struct Restore(
    Option<SourceRegionRouteAudit>,
    Option<SourceRegionRouteAudit>,
    bool,
);
impl Drop for Restore {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            active.replace(self.0.take());
        });
        INHERITED.with(|inherited| {
            inherited.replace(self.1.take());
        });
        POLL_OWNER.with(|owner| owner.set(self.2));
    }
}

fn with_active<T>(audit: Option<SourceRegionRouteAudit>, f: impl FnOnce() -> T) -> T {
    let previous = ACTIVE.with(|active| active.replace(audit));
    let inherited = INHERITED.with(|inherited| inherited.replace(previous.clone()));
    let _restore = Restore(
        previous,
        inherited,
        POLL_OWNER.with(|owner| owner.replace(true)),
    );
    f()
}

/// A selected region can start or end inside the same interpreter poll. Update
/// that poll immediately; its owning wrapper restores the prior task-local
/// state on Pending, Ready, or unwind.
pub fn set_source_region_for_current_poll(audit: Option<SourceRegionRouteAudit>) {
    POLL_OWNER.with(|owner| {
        if owner.get() {
            ACTIVE.with(|active| {
                let previous = active.replace(
                    audit.or_else(|| INHERITED.with(|inherited| inherited.borrow().clone())),
                );
                if let (Some(previous), Some(current)) =
                    (previous.as_ref(), active.borrow().as_ref())
                {
                    if !previous.0.ptr_eq(&current.0) {
                        previous.conflict();
                    }
                }
            });
        }
    });
}

pub fn record_source_region_provider_selection(provider: &dyn AccelProvider) {
    ACTIVE.with(|active| {
        if let Some(audit) = active.borrow().as_ref() {
            audit.selected(provider);
        }
    });
}

#[cfg(feature = "wgpu")]
pub fn record_source_region_wgpu_submission(access: &WgpuDeviceAccess) {
    ACTIVE.with(|active| {
        if let Some(audit) = active.borrow().as_ref() {
            audit.submitted(access);
        }
    });
}

/// Any device effect outside a canonical, access-bound submission lacks the
/// exact queue identity needed for one-tail completion evidence. This is
/// conservative even for transfers that might use the selected queue.
pub fn record_source_region_unattributed_device_effect() {
    ACTIVE.with(|active| {
        if let Some(audit) = active.borrow().as_ref() {
            audit
                .0
                .get()
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .conflicting_owner = true;
        }
    });
}

/// Poll the VM's already boxed loop under its active audit. The VM owns pin
/// projection, so this boundary adds no second future allocation.
pub fn with_source_region_poll<T>(
    slot: Option<&SourceRegionRouteAuditSlot>,
    poll: impl FnOnce() -> T,
) -> T {
    // A nested VM frame still belongs to the outer selected region until its
    // exact source span ends. An empty child slot must not hide its provider
    // selections or submissions from that region's one-owner audit.
    let outer = ACTIVE.with(|active| active.borrow().clone());
    let selected = slot.and_then(|slot| match slot.lock() {
        Ok(slot) => slot.clone(),
        Err(_) => {
            if let Some(outer) = &outer {
                outer.conflict();
            }
            None
        }
    });
    with_source_region_audit_from(outer, selected, poll)
}

/// Scope the same physical audit around a native machine entry or the poll
/// of its existing hostcall future. No execution/completion right is issued.
pub fn with_source_region_audit<T>(
    selected: Option<SourceRegionRouteAudit>,
    poll: impl FnOnce() -> T,
) -> T {
    let outer = ACTIVE.with(|active| active.borrow().clone());
    with_source_region_audit_from(outer, selected, poll)
}

fn with_source_region_audit_from<T>(
    outer: Option<SourceRegionRouteAudit>,
    selected: Option<SourceRegionRouteAudit>,
    poll: impl FnOnce() -> T,
) -> T {
    let active = match (outer, selected) {
        (Some(outer), Some(child)) if !outer.0.ptr_eq(&child.0) => {
            // Distinct nested route ownership cannot be described by the
            // outer route's single-provider queue tail.
            outer.conflict();
            Some(child)
        }
        (Some(outer), None) => Some(outer),
        (_, child) => child,
    };
    with_active(active, poll)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backing_quote_is_the_exact_shared_allocation_layout() {
        assert_eq!(
            SourceRegionRouteAudit::backing_bytes(),
            u64::try_from(ExactShared::<AuditBacking>::requested_bytes()).ok()
        );
        assert!(
            ExactShared::<AuditBacking>::requested_bytes() > std::mem::size_of::<AuditBacking>()
        );
    }

    // This oracle deliberately has no WGPU cfg: CPU-only Runtime/native users
    // need the same allocation-free poll boundary, not backend authority.
    #[test]
    fn empty_source_scope_is_backend_agnostic_and_restores_poll_owner() {
        assert!(!POLL_OWNER.with(Cell::get));
        with_source_region_audit(None, || {
            assert!(POLL_OWNER.with(Cell::get));
            set_source_region_for_current_poll(None);
            with_source_region_poll(None, || {
                assert!(POLL_OWNER.with(Cell::get));
            });
            assert!(POLL_OWNER.with(Cell::get));
        });
        assert!(!POLL_OWNER.with(Cell::get));
        assert!(ACTIVE.with(|active| active.borrow().is_none()));
    }
}
