use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroUsize;

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
    raw: NonZeroUsize,
    _not_send_sync: PhantomData<*const ()>,
}

impl GcHandle {
    /// # Safety
    ///
    /// `raw` must be the non-zero address of a live RunMat GC allocation. This is
    /// an unchecked bridge for `runmat-gc`; callers must not fabricate handles
    /// from ordinary Rust allocations or stale addresses.
    #[doc(hidden)]
    pub unsafe fn from_addr_unchecked(raw: NonZeroUsize) -> Self {
        Self {
            raw,
            _not_send_sync: PhantomData,
        }
    }

    pub fn addr(&self) -> usize {
        self.raw.get()
    }
}

impl PartialEq for GcHandle {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for GcHandle {}

impl Hash for GcHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state)
    }
}

impl fmt::Debug for GcHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GcHandle({:p})", self.raw.get() as *const ())
    }
}

impl fmt::Display for GcHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:p}", self.raw.get() as *const ())
    }
}

#[cfg(test)]
mod tests {
    use super::GcHandle;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    #[test]
    fn equality_is_pointer_identity_not_pointee_value() {
        let raw_a = Box::into_raw(Box::new(7_u64));
        let raw_b = Box::into_raw(Box::new(7_u64));
        let ptr_a = unsafe {
            GcHandle::from_addr_unchecked(
                std::num::NonZeroUsize::new(raw_a as usize).expect("non-null test pointer"),
            )
        };
        let ptr_b = unsafe {
            GcHandle::from_addr_unchecked(
                std::num::NonZeroUsize::new(raw_b as usize).expect("non-null test pointer"),
            )
        };

        assert_eq!(ptr_a, ptr_a);
        assert_ne!(ptr_a, ptr_b);

        unsafe {
            drop(Box::from_raw(raw_a));
            drop(Box::from_raw(raw_b));
        }
    }

    #[test]
    fn hash_is_pointer_identity() {
        let raw_a = Box::into_raw(Box::new(7_u64));
        let raw_b = Box::into_raw(Box::new(7_u64));
        let ptr_a = unsafe {
            GcHandle::from_addr_unchecked(
                std::num::NonZeroUsize::new(raw_a as usize).expect("non-null test pointer"),
            )
        };
        let ptr_b = unsafe {
            GcHandle::from_addr_unchecked(
                std::num::NonZeroUsize::new(raw_b as usize).expect("non-null test pointer"),
            )
        };

        let mut ptr_hash = DefaultHasher::new();
        ptr_a.hash(&mut ptr_hash);
        let mut raw_hash = DefaultHasher::new();
        raw_a.hash(&mut raw_hash);

        assert_eq!(ptr_hash.finish(), raw_hash.finish());
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
            GcHandle::from_addr_unchecked(
                std::num::NonZeroUsize::new(raw as usize).expect("non-null test pointer"),
            )
        };

        assert!(format!("{ptr:?}").starts_with("GcHandle(0x"));
        assert!(ptr.to_string().starts_with("0x"));

        unsafe {
            drop(Box::from_raw(raw));
        }
    }
}
