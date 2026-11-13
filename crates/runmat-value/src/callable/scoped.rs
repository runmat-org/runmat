//! Process-local callable capability. No callback body or borrowed state lives here.
use super::CallableInstanceId;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Debug)]
struct Identity {
    instance: CallableInstanceId,
    live: AtomicBool,
}

/// A clone retains the exact capability, not a source name or executable ID.
/// The instance ID is only an index: an invoker must also match the original
/// identity allocation and its live scope before invoking any user code.
#[derive(Clone, Debug)]
pub struct ScopedFunctionHandle(Arc<Identity>);

/// Sole lifetime owner. This value must not be published to user code.
#[derive(Debug)]
pub struct ScopedFunctionRevoker(Arc<Identity>);

impl ScopedFunctionHandle {
    pub fn issue() -> (Self, ScopedFunctionRevoker) {
        let identity = Arc::new(Identity {
            instance: CallableInstanceId::mint(),
            live: AtomicBool::new(true),
        });
        (Self(Arc::clone(&identity)), ScopedFunctionRevoker(identity))
    }
    pub fn instance_id(&self) -> CallableInstanceId {
        self.0.instance
    }
    pub fn is_live(&self) -> bool {
        self.0.live.load(Ordering::Acquire)
    }
    pub fn same_capability(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl PartialEq for ScopedFunctionHandle {
    fn eq(&self, other: &Self) -> bool {
        self.same_capability(other)
    }
}
impl Eq for ScopedFunctionHandle {}
impl ScopedFunctionRevoker {
    pub fn revoke(&self) {
        self.0.live.store(false, Ordering::Release);
    }
}
impl Drop for ScopedFunctionRevoker {
    fn drop(&mut self) {
        self.revoke();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_original_lease_controls_cloned_capability_lifetime() {
        let (handle, lease) = ScopedFunctionHandle::issue();
        let retained = handle.clone();
        let (other, _other_lease) = ScopedFunctionHandle::issue();
        assert!(retained.same_capability(&handle));
        assert_ne!(retained, other);
        drop(handle);
        assert!(retained.is_live());
        drop(lease);
        assert!(!retained.is_live());
    }
}
