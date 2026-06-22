use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

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
    ptr: *const T,
    _phantom: PhantomData<T>,
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
        Self {
            ptr,
            _phantom: PhantomData,
        }
    }

    pub fn null() -> Self {
        Self {
            ptr: std::ptr::null(),
            _phantom: PhantomData,
        }
    }

    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    pub fn addr(&self) -> usize {
        self.ptr as usize
    }

    /// # Safety
    ///
    /// The returned raw pointer may become invalid if the underlying object is
    /// collected by the GC. The caller must ensure the object is kept alive
    /// (e.g., via rooting) for any dereference, and must respect aliasing and
    /// lifetime rules when using the pointer.
    pub unsafe fn as_raw(&self) -> *const T {
        self.ptr
    }
}

impl<T> PartialEq for GcHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T> Eq for GcHandle<T> {}

impl<T> Hash for GcHandle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ptr.hash(state)
    }
}

impl<T> fmt::Debug for GcHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "GcHandle(null)")
        } else {
            write!(f, "GcHandle({:p})", self.ptr)
        }
    }
}

impl<T> fmt::Display for GcHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "null")
        } else {
            write!(f, "{:p}", self.ptr)
        }
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
        let ptr = GcHandle::<u64>::null();

        assert_eq!(format!("{ptr:?}"), "GcHandle(null)");
        assert_eq!(ptr.to_string(), "null");
    }
}
