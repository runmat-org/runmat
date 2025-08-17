#![forbid(unsafe_op_in_unsafe_fn)]

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

#[derive(Copy, Clone)]
pub struct GcPtr<T> {
    ptr: *const T,
    _phantom: PhantomData<T>,
}

impl<T> GcPtr<T> {
    pub unsafe fn from_raw(ptr: *const T) -> Self { Self { ptr, _phantom: PhantomData } }
    pub fn null() -> Self { Self { ptr: std::ptr::null(), _phantom: PhantomData } }
    pub fn is_null(&self) -> bool { self.ptr.is_null() }
    pub unsafe fn as_raw(&self) -> *const T { self.ptr }
    pub unsafe fn as_raw_mut(&self) -> *mut T { self.ptr as *mut T }
}

impl<T> Deref for GcPtr<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target { unsafe { &*self.ptr } }
}
impl<T> DerefMut for GcPtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { unsafe { &mut *(self.ptr as *mut T) } }
}

impl<T: PartialEq> PartialEq for GcPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.is_null() && other.is_null() { true }
        else if self.is_null() || other.is_null() { false }
        else { **self == **other }
    }
}
impl<T: Eq> Eq for GcPtr<T> {}
impl<T: Hash> Hash for GcPtr<T> { fn hash<H: Hasher>(&self, state: &mut H) { if !self.is_null() { (**self).hash(state) } } }
impl<T: fmt::Debug> fmt::Debug for GcPtr<T> { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { if self.is_null() { write!(f, "GcPtr(null)") } else { write!(f, "GcPtr({:?})", **self) } } }
impl<T: fmt::Display> fmt::Display for GcPtr<T> { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { if self.is_null() { write!(f, "null") } else { write!(f, "{}", **self) } } }

unsafe impl<T: Send> Send for GcPtr<T> {}
unsafe impl<T: Sync> Sync for GcPtr<T> {}


