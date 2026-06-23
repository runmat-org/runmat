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
/// assert_send::<runmat_gc_api::GcHandle<u8>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<runmat_gc_api::GcHandle<u8>>();
/// ```
#[derive(Copy, Clone)]
pub struct GcHandle<T> {
    raw: NonZeroUsize,
    _not_send_sync: PhantomData<*const T>,
}

impl<T> GcHandle<T> {
    /// # Safety
    ///
    /// - `ptr` must be non-null and correctly aligned for `T`.
    /// - `ptr` must point to a valid instance of `T` allocated by RunMat's GC and
    ///   remain alive for the duration of all uses of the returned `GcHandle`.
    /// - The caller is responsible for upholding aliasing and lifetime invariants
    ///   when accessing the pointer through raw pointer APIs.
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        let raw = NonZeroUsize::new(ptr as usize).expect("GC handle pointer must be non-null");
        Self {
            raw,
            _not_send_sync: PhantomData,
        }
    }

    pub fn addr(&self) -> usize {
        self.raw.get()
    }

    /// # Safety
    ///
    /// The returned raw pointer may become invalid if the underlying object is
    /// collected by the GC. The caller must ensure the object is kept alive
    /// (e.g., via rooting) for any dereference, and must respect aliasing and
    /// lifetime rules when using the pointer.
    pub unsafe fn as_raw(&self) -> *const T {
        self.raw.get() as *const T
    }
}

impl<T> PartialEq for GcHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<T> Eq for GcHandle<T> {}

impl<T> Hash for GcHandle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state)
    }
}

impl<T> fmt::Debug for GcHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GcHandle({:p})", self.raw.get() as *const T)
    }
}

impl<T> fmt::Display for GcHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:p}", self.raw.get() as *const T)
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
        let ptr_a = unsafe { GcHandle::from_raw(raw_a) };
        let ptr_b = unsafe { GcHandle::from_raw(raw_b) };

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
        let ptr_a = unsafe { GcHandle::from_raw(raw_a) };
        let ptr_b = unsafe { GcHandle::from_raw(raw_b) };

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
        let ptr = unsafe { GcHandle::from_raw(raw) };

        assert!(format!("{ptr:?}").starts_with("GcHandle(0x"));
        assert!(ptr.to_string().starts_with("0x"));

        unsafe {
            drop(Box::from_raw(raw));
        }
    }
}
