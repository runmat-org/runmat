use std::sync::Arc;

/// Shared, pointer-stable ownership for an authoritative host allocation.
///
/// The payload type owns its allocation and defines its element layout. This
/// wrapper supplies the lifetime and copy-on-write policy used by ordinary
/// values and in-process foreign-runtime leases. It intentionally exposes no
/// representation details through a stable ABI.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HostBuffer<T: Clone> {
    payload: Arc<T>,
}

impl<T: Clone> HostBuffer<T> {
    pub(crate) fn new(payload: T) -> Self {
        Self {
            payload: Arc::new(payload),
        }
    }

    pub(crate) fn get(&self) -> &T {
        self.payload.as_ref()
    }

    pub(crate) fn make_mut(&mut self) -> &mut T {
        Arc::make_mut(&mut self.payload)
    }

    pub(crate) fn is_shared(&self) -> bool {
        Arc::strong_count(&self.payload) > 1
    }

    pub(crate) fn make_unique(&mut self) {
        let _ = self.make_mut();
    }

    pub(crate) fn shares_allocation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.payload, &other.payload)
    }

    pub(crate) fn into_inner(self) -> T {
        Arc::try_unwrap(self.payload).unwrap_or_else(|payload| (*payload).clone())
    }
}
