use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Opaque handle token for a RunMat GC allocation.
///
/// `GcHandle` is intentionally not `Send` or `Sync`; a handle does not prove that
/// the pointed-to object can be accessed safely from another thread.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<runmat_gc_api::GcHandle>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<runmat_gc_api::GcHandle>();
/// ```
///
/// ```compile_fail
/// fn deref(handle: runmat_gc_api::GcHandle) {
///     let _ = *handle;
/// }
/// ```
///
/// ```compile_fail
/// use std::ops::Deref;
///
/// fn deref_method(handle: runmat_gc_api::GcHandle) {
///     let _ = handle.deref();
/// }
/// ```
///
/// ```compile_fail
/// fn mutable_reference(mut handle: runmat_gc_api::GcHandle) {
///     let _: &mut _ = &mut *handle;
/// }
/// ```
#[derive(Copy, Clone)]
pub struct GcHandle {
    raw: NonNull<()>,
    epoch: usize,
    _not_send_sync: PhantomData<*const ()>,
}

impl GcHandle {
    /// # Safety
    ///
    /// `raw` must point to a live RunMat GC allocation. This is an unchecked
    /// bridge for `runmat-gc`; callers must not fabricate handles from ordinary
    /// Rust allocations or stale addresses.
    #[doc(hidden)]
    pub unsafe fn from_ptr_unchecked(raw: NonNull<()>) -> Self {
        Self {
            raw,
            epoch: 0,
            _not_send_sync: PhantomData,
        }
    }

    /// # Safety
    ///
    /// `raw` and `epoch` must identify a currently live RunMat GC allocation.
    /// This is an unchecked bridge for `runmat-gc`; callers must not fabricate
    /// handles from ordinary Rust allocations or stale allocation epochs.
    #[doc(hidden)]
    pub unsafe fn from_parts_unchecked(raw: NonNull<()>, epoch: usize) -> Self {
        Self {
            raw,
            epoch,
            _not_send_sync: PhantomData,
        }
    }

    pub fn addr(&self) -> usize {
        self.raw.as_ptr() as usize
    }

    #[doc(hidden)]
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// # Safety
    ///
    /// The returned pointer may only be dereferenced by code that can prove the
    /// handle is still owned by the RunMat GC heap and that aliasing rules are
    /// upheld for the intended access.
    #[doc(hidden)]
    pub unsafe fn as_ptr_unchecked(&self) -> NonNull<()> {
        self.raw
    }
}

impl PartialEq for GcHandle {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw && self.epoch == other.epoch
    }
}

impl Eq for GcHandle {}

impl Hash for GcHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
        self.epoch.hash(state)
    }
}

impl fmt::Debug for GcHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GcHandle({:p}@{})", self.raw.as_ptr(), self.epoch)
    }
}

impl fmt::Display for GcHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:p}", self.raw.as_ptr())
    }
}

#[cfg(test)]
mod tests {
    use super::GcHandle;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::ptr::NonNull;

    #[test]
    fn equality_is_pointer_identity_not_pointee_value() {
        let raw_a = Box::into_raw(Box::new(7_u64));
        let raw_b = Box::into_raw(Box::new(7_u64));
        let ptr_a = unsafe {
            GcHandle::from_ptr_unchecked(NonNull::new(raw_a.cast()).expect("non-null test pointer"))
        };
        let ptr_b = unsafe {
            GcHandle::from_ptr_unchecked(NonNull::new(raw_b.cast()).expect("non-null test pointer"))
        };

        assert_eq!(ptr_a, ptr_a);
        assert_ne!(ptr_a, ptr_b);

        unsafe {
            drop(Box::from_raw(raw_a));
            drop(Box::from_raw(raw_b));
        }
    }

    #[test]
    fn hash_includes_pointer_and_epoch_identity() {
        let raw_a = Box::into_raw(Box::new(7_u64));
        let raw_b = Box::into_raw(Box::new(7_u64));
        let ptr_a = unsafe {
            GcHandle::from_parts_unchecked(
                NonNull::new(raw_a.cast()).expect("non-null test pointer"),
                1,
            )
        };
        let ptr_a_next_epoch = unsafe {
            GcHandle::from_parts_unchecked(
                NonNull::new(raw_a.cast()).expect("non-null test pointer"),
                2,
            )
        };
        let ptr_b = unsafe {
            GcHandle::from_parts_unchecked(
                NonNull::new(raw_b.cast()).expect("non-null test pointer"),
                1,
            )
        };

        let mut ptr_hash = DefaultHasher::new();
        ptr_a.hash(&mut ptr_hash);
        let mut same_hash = DefaultHasher::new();
        ptr_a.hash(&mut same_hash);
        let mut next_epoch_hash = DefaultHasher::new();
        ptr_a_next_epoch.hash(&mut next_epoch_hash);

        assert_eq!(ptr_hash.finish(), same_hash.finish());
        assert_ne!(ptr_hash.finish(), next_epoch_hash.finish());
        assert_ne!(ptr_a, ptr_a_next_epoch);
        assert_ne!(ptr_a, ptr_b);

        unsafe {
            drop(Box::from_raw(raw_a));
            drop(Box::from_raw(raw_b));
        }
    }

    #[test]
    fn debug_and_display_do_not_dereference_pointee() {
        let raw = Box::into_raw(Box::new(7_u64));
        let ptr = unsafe {
            GcHandle::from_ptr_unchecked(NonNull::new(raw.cast()).expect("non-null test pointer"))
        };

        assert!(format!("{ptr:?}").starts_with("GcHandle(0x"));
        assert!(ptr.to_string().starts_with("0x"));

        unsafe {
            drop(Box::from_raw(raw));
        }
    }
}
