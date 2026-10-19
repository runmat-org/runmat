//! Task-local, source-route physical-owner audit. This is neither an execution
//! grant nor a cost estimate. A selected original VM region can learn only if
//! every observed provider selection and physical WGPU submission used its
//! bound owner; unknown or other-device work suppresses feedback.

use crate::{AccelProvider, WgpuDeviceAccess};
use std::cell::{Cell, RefCell};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct RouteState {
    submissions: u64,
    conflicting_owner: bool,
}

pub struct VmRegionRouteAudit {
    provider: &'static dyn AccelProvider,
    access: WgpuDeviceAccess,
    state: Mutex<RouteState>,
}

impl VmRegionRouteAudit {
    pub fn new(provider: &'static dyn AccelProvider, access: WgpuDeviceAccess) -> Arc<Self> {
        Arc::new(Self {
            provider,
            access,
            state: Mutex::new(RouteState::default()),
        })
    }

    pub fn exact_single_provider(&self) -> bool {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.submissions > 0 && !state.conflicting_owner
    }

    fn conflict(&self) {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .conflicting_owner = true;
    }

    fn selected(&self, provider: &dyn AccelProvider) {
        if !crate::same_provider_instance(self.provider, provider) {
            self.state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .conflicting_owner = true;
        }
    }

    fn submitted(&self, access: &WgpuDeviceAccess) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !self.access.same_device(access) {
            state.conflicting_owner = true;
        } else if let Some(next) = state.submissions.checked_add(1) {
            state.submissions = next;
        } else {
            state.conflicting_owner = true;
        }
    }
}

pub type VmRegionRouteAuditSlot = Arc<Mutex<Option<Arc<VmRegionRouteAudit>>>>;

thread_local! {
    static ACTIVE: RefCell<Option<Arc<VmRegionRouteAudit>>> = const { RefCell::new(None) };
    static INHERITED: RefCell<Option<Arc<VmRegionRouteAudit>>> = const { RefCell::new(None) };
    static POLL_OWNER: Cell<bool> = const { Cell::new(false) };
}

struct Restore(
    Option<Arc<VmRegionRouteAudit>>,
    Option<Arc<VmRegionRouteAudit>>,
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

fn with_active<T>(audit: Option<Arc<VmRegionRouteAudit>>, f: impl FnOnce() -> T) -> T {
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
pub fn set_vm_region_for_current_poll(audit: Option<Arc<VmRegionRouteAudit>>) {
    POLL_OWNER.with(|owner| {
        if owner.get() {
            ACTIVE.with(|active| {
                let previous = active.replace(
                    audit.or_else(|| INHERITED.with(|inherited| inherited.borrow().clone())),
                );
                if let (Some(previous), Some(current)) =
                    (previous.as_ref(), active.borrow().as_ref())
                {
                    if !Arc::ptr_eq(previous, current) {
                        previous.conflict();
                    }
                }
            });
        }
    });
}

pub fn record_vm_region_provider_selection(provider: &dyn AccelProvider) {
    ACTIVE.with(|active| {
        if let Some(audit) = active.borrow().as_ref() {
            audit.selected(provider);
        }
    });
}

pub fn record_vm_region_wgpu_submission(access: &WgpuDeviceAccess) {
    ACTIVE.with(|active| {
        if let Some(audit) = active.borrow().as_ref() {
            audit.submitted(access);
        }
    });
}

/// Any device effect outside a canonical, access-bound submission lacks the
/// exact queue identity needed for one-tail completion evidence. This is
/// conservative even for transfers that might use the selected queue.
pub fn record_vm_region_unattributed_device_effect() {
    ACTIVE.with(|active| {
        if let Some(audit) = active.borrow().as_ref() {
            audit
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .conflicting_owner = true;
        }
    });
}

/// Poll the VM's already boxed loop under its active audit. The VM owns pin
/// projection, so this boundary adds no second future allocation.
pub fn with_vm_region_poll<T>(
    slot: Option<&VmRegionRouteAuditSlot>,
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
    let active = match (outer, selected) {
        (Some(outer), Some(child)) if !Arc::ptr_eq(&outer, &child) => {
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
