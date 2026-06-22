#![forbid(unsafe_op_in_unsafe_fn)]

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

#[derive(Copy, Clone)]
pub struct GcPtr<T> {
    ptr: *const T,
    _phantom: PhantomData<T>,
}

impl<T> GcPtr<T> {
    /// # Safety
    ///
    /// - `ptr` must be non-null and correctly aligned for `T`.
    /// - `ptr` must point to a valid instance of `T` allocated by RunMat's GC and
    ///   remain alive for the duration of all uses of the returned `GcPtr`.
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
    /// # Safety
    ///
    /// The returned raw pointer may become invalid if the underlying object is
    /// collected by the GC. The caller must ensure the object is kept alive
    /// (e.g., via rooting) for any dereference, and must respect aliasing and
    /// lifetime rules when using the pointer.
    pub unsafe fn as_raw(&self) -> *const T {
        self.ptr
    }
    /// # Safety
    ///
    /// Returns a mutable raw pointer to the underlying object. The caller must
    /// ensure exclusive access when mutating through this pointer, that the
    /// object outlives all uses, and that aliasing/lifetime invariants are
    /// respected. Mutating a collected or shared object is undefined behavior.
    pub unsafe fn as_raw_mut(&self) -> *mut T {
        self.ptr as *mut T
    }
}

impl<T> PartialEq for GcPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}
impl<T> Eq for GcPtr<T> {}
impl<T> Hash for GcPtr<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ptr.hash(state)
    }
}
impl<T> fmt::Debug for GcPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "GcPtr(null)")
        } else {
            write!(f, "GcPtr({:p})", self.ptr)
        }
    }
}
impl<T> fmt::Display for GcPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "null")
        } else {
            write!(f, "{:p}", self.ptr)
        }
    }
}

unsafe impl<T: Send> Send for GcPtr<T> {}
unsafe impl<T: Sync> Sync for GcPtr<T> {}

#[cfg(test)]
mod tests {
    use super::GcPtr;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    #[test]
    fn equality_is_pointer_identity_not_pointee_value() {
        let raw_a = Box::into_raw(Box::new(7_u64));
        let raw_b = Box::into_raw(Box::new(7_u64));
        let ptr_a = unsafe { GcPtr::from_raw(raw_a) };
        let ptr_b = unsafe { GcPtr::from_raw(raw_b) };

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
        let ptr_a = unsafe { GcPtr::from_raw(raw_a) };
        let ptr_b = unsafe { GcPtr::from_raw(raw_b) };

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
        let ptr = GcPtr::<u64>::null();

        assert_eq!(format!("{ptr:?}"), "GcPtr(null)");
        assert_eq!(ptr.to_string(), "null");
    }
}
