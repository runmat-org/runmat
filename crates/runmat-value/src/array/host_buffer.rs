use std::sync::Arc;

mod exact;
use exact::ExactShared;

/// Shared, pointer-stable ownership for an authoritative host allocation.
///
/// The payload type owns its allocation and defines its element layout. This
/// wrapper supplies the lifetime and copy-on-write policy used by ordinary
/// values and in-process foreign-runtime leases. It intentionally exposes no
/// representation details through a stable ABI.
#[derive(Debug)]
pub(crate) struct HostBuffer<T: Clone> {
    payload: Owner<T>,
}

#[derive(Debug)]
enum Owner<T: Clone> {
    Ordinary(Arc<T>),
    Exact(ExactShared<T>),
}

impl<T: Clone> Clone for HostBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            payload: match &self.payload {
                Owner::Ordinary(value) => Owner::Ordinary(value.clone()),
                Owner::Exact(value) => Owner::Exact(value.clone()),
            },
        }
    }
}

impl<T: Clone + PartialEq> PartialEq for HostBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<T: Clone> HostBuffer<T> {
    pub(crate) fn new(payload: T) -> Self {
        Self {
            payload: Owner::Ordinary(Arc::new(payload)),
        }
    }

    pub(crate) const fn exact_metadata_bytes() -> usize {
        ExactShared::<T>::requested_bytes()
    }

    /// The exact producer precharges `exact_metadata_bytes()` before this
    /// known-layout allocation. Its payload owns the producer's claim.
    pub(crate) fn new_exact(payload: T) -> Self {
        Self {
            payload: Owner::Exact(ExactShared::new(payload)),
        }
    }

    pub(crate) fn try_new_exact(payload: T) -> Result<Self, T> {
        ExactShared::try_new(payload).map(|owner| Self {
            payload: Owner::Exact(owner),
        })
    }

    pub(crate) fn get(&self) -> &T {
        match &self.payload {
            Owner::Ordinary(value) => value.as_ref(),
            Owner::Exact(value) => value.get(),
        }
    }

    pub(crate) fn make_mut(&mut self) -> &mut T {
        if matches!(&self.payload, Owner::Exact(value) if !value.is_unique()) {
            self.payload = Owner::Ordinary(Arc::new(self.get().clone()));
        }
        match &mut self.payload {
            Owner::Ordinary(value) => Arc::make_mut(value),
            Owner::Exact(value) => value.get_mut_unique(),
        }
    }

    pub(crate) fn is_shared(&self) -> bool {
        match &self.payload {
            Owner::Ordinary(value) => Arc::strong_count(value) > 1,
            Owner::Exact(value) => !value.is_unique(),
        }
    }

    pub(crate) fn make_unique(&mut self) {
        let _ = self.make_mut();
    }

    pub(crate) fn shares_allocation_with(&self, other: &Self) -> bool {
        match (&self.payload, &other.payload) {
            (Owner::Ordinary(left), Owner::Ordinary(right)) => Arc::ptr_eq(left, right),
            (Owner::Exact(left), Owner::Exact(right)) => left.ptr_eq(right),
            _ => false,
        }
    }

    pub(crate) fn into_inner(self) -> T {
        match self.payload {
            Owner::Ordinary(value) => {
                Arc::try_unwrap(value).unwrap_or_else(|payload| (*payload).clone())
            }
            Owner::Exact(value) => value.into_inner(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_owner_shares_pointer_then_detaches_on_mutation() {
        let mut first = HostBuffer::new_exact(vec![1_u64, 2]);
        let second = first.clone();
        assert!(first.shares_allocation_with(&second));
        assert!(first.is_shared());
        first.make_mut()[0] = 9;
        assert!(!first.shares_allocation_with(&second));
        assert_eq!(first.get(), &[9, 2]);
        assert_eq!(second.get(), &[1, 2]);
        assert_eq!(second.into_inner(), vec![1, 2]);
    }
}
