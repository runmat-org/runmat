//! Garbage collected smart pointers
//!
//! Provides GcPtr<T>, a smart pointer type that works with the garbage collector.
//! Supports optional pointer compression for memory efficiency.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

#[cfg(feature = "pointer-compression")]
use crate::compression::CompressedPtr;

/// A garbage collected pointer to a value of type T
///
/// This is the primary interface for working with GC-managed objects.
/// It provides automatic memory management while maintaining Rust's
/// safety guarantees.
pub struct GcPtr<T> {
    #[cfg(not(feature = "pointer-compression"))]
    ptr: *const T,

    #[cfg(feature = "pointer-compression")]
    ptr: CompressedPtr,

    _phantom: PhantomData<T>,
}

impl<T> GcPtr<T> {
    /// Create a new GcPtr from a raw pointer
    ///
    /// # Safety
    ///
    /// The pointer must be valid and point to GC-managed memory
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        Self {
            #[cfg(not(feature = "pointer-compression"))]
            ptr,

            #[cfg(feature = "pointer-compression")]
            ptr: CompressedPtr::compress(ptr as *const u8),

            _phantom: PhantomData,
        }
    }

    /// Get the raw pointer value
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid as long as the GC hasn't
    /// moved or collected the object
    pub unsafe fn as_raw(&self) -> *const T {
        #[cfg(not(feature = "pointer-compression"))]
        {
            self.ptr
        }

        #[cfg(feature = "pointer-compression")]
        {
            self.ptr.decompress() as *const T
        }
    }

    /// Get a mutable raw pointer
    ///
    /// # Safety
    ///
    /// Same safety requirements as as_raw(), plus the caller must ensure
    /// exclusive access to the object
    pub unsafe fn as_raw_mut(&self) -> *mut T {
        self.as_raw() as *mut T
    }

    /// Check if this pointer is null
    pub fn is_null(&self) -> bool {
        #[cfg(not(feature = "pointer-compression"))]
        {
            self.ptr.is_null()
        }

        #[cfg(feature = "pointer-compression")]
        {
            self.ptr.is_null()
        }
    }

    /// Create a null GcPtr
    pub fn null() -> Self {
        Self {
            #[cfg(not(feature = "pointer-compression"))]
            ptr: std::ptr::null(),

            #[cfg(feature = "pointer-compression")]
            ptr: CompressedPtr::null(),

            _phantom: PhantomData,
        }
    }

    /// Cast to a different type
    ///
    /// # Safety
    ///
    /// The caller must ensure the cast is valid
    pub unsafe fn cast<U>(&self) -> GcPtr<U> {
        GcPtr {
            #[cfg(not(feature = "pointer-compression"))]
            ptr: self.ptr as *const U,

            #[cfg(feature = "pointer-compression")]
            ptr: self.ptr,

            _phantom: PhantomData,
        }
    }
}

impl<T> Deref for GcPtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.as_raw() }
    }
}

impl<T> DerefMut for GcPtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.as_raw_mut() }
    }
}

impl<T> Clone for GcPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for GcPtr<T> {}

impl<T: PartialEq> PartialEq for GcPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.is_null() && other.is_null() {
            true
        } else if self.is_null() || other.is_null() {
            false
        } else {
            **self == **other
        }
    }
}

impl<T: Eq> Eq for GcPtr<T> {}

impl<T: Hash> Hash for GcPtr<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if !self.is_null() {
            (**self).hash(state);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for GcPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "GcPtr(null)")
        } else {
            write!(f, "GcPtr({:?})", **self)
        }
    }
}

impl<T: fmt::Display> fmt::Display for GcPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "null")
        } else {
            write!(f, "{}", **self)
        }
    }
}

unsafe impl<T: Send> Send for GcPtr<T> {}
unsafe impl<T: Sync> Sync for GcPtr<T> {}

/// A weak reference to a GC-managed object
///
/// WeakGcPtr doesn't prevent the object from being collected,
/// making it useful for breaking reference cycles.
pub struct WeakGcPtr<T> {
    #[cfg(not(feature = "pointer-compression"))]
    ptr: *const T,

    #[cfg(feature = "pointer-compression")]
    ptr: CompressedPtr,

    _phantom: PhantomData<T>,
}

impl<T> WeakGcPtr<T> {
    /// Create a weak reference from a strong GcPtr
    pub fn from_gc_ptr(ptr: &GcPtr<T>) -> Self {
        Self {
            #[cfg(not(feature = "pointer-compression"))]
            ptr: ptr.ptr,

            #[cfg(feature = "pointer-compression")]
            ptr: ptr.ptr,

            _phantom: PhantomData,
        }
    }

    /// Try to upgrade to a strong GcPtr
    ///
    /// Returns None if the object has been collected
    pub fn upgrade(&self) -> Option<GcPtr<T>> {
        // Check if object is still alive by consulting the GC
        // This is a simplified version - real implementation would
        // check the object's mark bit or generation
        unsafe {
            #[cfg(not(feature = "pointer-compression"))]
            let raw_ptr = self.ptr;

            #[cfg(feature = "pointer-compression")]
            let raw_ptr = self.ptr.decompress() as *const T;

            if raw_ptr.is_null() {
                None
            } else {
                // In production, would check GC mark bits/generation validity
                Some(GcPtr::from_raw(raw_ptr))
            }
        }
    }

    /// Check if the weak reference is expired
    pub fn is_expired(&self) -> bool {
        self.upgrade().is_none()
    }
}

impl<T> Clone for WeakGcPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for WeakGcPtr<T> {}

/// Extension trait for easy GC pointer creation
pub trait IntoGcPtr<T> {
    fn into_gc_ptr(self) -> crate::Result<GcPtr<T>>;
}

// Removed runmat_builtins coupling to avoid dependency cycle

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::Value;

    #[test]
    fn test_gc_ptr_null() {
        let ptr: GcPtr<Value> = GcPtr::null();
        assert!(ptr.is_null());
    }

    #[test]
    fn test_gc_ptr_clone() {
        let ptr: GcPtr<Value> = GcPtr::null();
        let ptr2 = ptr;
        assert_eq!(ptr.is_null(), ptr2.is_null());
    }

    #[test]
    fn test_weak_gc_ptr() {
        let ptr: GcPtr<Value> = GcPtr::null();
        let weak = WeakGcPtr::from_gc_ptr(&ptr);

        // Null pointer should always be expired
        assert!(weak.is_expired());
    }
}
