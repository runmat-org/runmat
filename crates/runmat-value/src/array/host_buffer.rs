use runmat_execution::resource::ExactShared;

/// Shared, pointer-stable ownership for an authoritative host allocation.
///
/// The payload type owns its allocation and defines its element layout. This
/// wrapper supplies the lifetime and copy-on-write policy used by ordinary
/// values and in-process foreign-runtime leases. It intentionally exposes no
/// representation details through a stable ABI.
#[derive(Debug)]
pub(crate) struct HostBuffer<T: Clone> {
    payload: ExactShared<T>,
}

impl<T: Clone> Clone for HostBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            payload: self.payload.clone(),
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
            payload: ExactShared::new(payload),
        }
    }

    pub(crate) const fn exact_metadata_bytes() -> usize {
        ExactShared::<T>::requested_bytes()
    }

    /// The exact producer precharges `exact_metadata_bytes()` before this
    /// known-layout allocation. Its payload owns the producer's claim.
    pub(crate) fn new_exact(payload: T) -> Self {
        Self::new(payload)
    }

    pub(crate) fn try_new_exact(payload: T) -> Result<Self, T> {
        ExactShared::try_new(payload).map(|payload| Self { payload })
    }

    pub(crate) fn get(&self) -> &T {
        self.payload.get()
    }

    /// Borrow an exclusively owned payload without copying or allocating.
    pub(crate) fn try_get_mut_unique(&mut self) -> Option<&mut T> {
        if self.payload.is_unique() {
            Some(self.payload.get_mut_unique())
        } else {
            None
        }
    }

    pub(crate) fn make_mut(&mut self) -> &mut T {
        if !self.payload.is_unique() {
            self.payload = ExactShared::new(self.get().clone());
        }
        self.payload.get_mut_unique()
    }

    pub(crate) fn is_shared(&self) -> bool {
        !self.payload.is_unique()
    }

    pub(crate) fn make_unique(&mut self) {
        let _ = self.make_mut();
    }

    pub(crate) fn shares_allocation_with(&self, other: &Self) -> bool {
        self.payload.ptr_eq(&other.payload)
    }

    pub(crate) fn into_inner(self) -> T {
        self.payload.into_inner()
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
        assert!(first.try_get_mut_unique().is_none());
        first.make_mut()[0] = 9;
        assert!(first.try_get_mut_unique().is_some());
        assert!(!first.shares_allocation_with(&second));
        assert_eq!(first.get(), &[9, 2]);
        assert_eq!(second.get(), &[1, 2]);
        assert_eq!(second.into_inner(), vec![1, 2]);
    }

    #[test]
    fn ordinary_owner_uses_the_same_copy_on_write_contract() {
        let first = HostBuffer::new(vec![2_u64, 3]);
        let mut second = first.clone();
        assert!(first.shares_allocation_with(&second));
        second.make_mut().push(5);
        assert!(!first.shares_allocation_with(&second));
        assert_eq!(first.into_inner(), vec![2, 3]);
        assert_eq!(second.into_inner(), vec![2, 3, 5]);
    }
}
