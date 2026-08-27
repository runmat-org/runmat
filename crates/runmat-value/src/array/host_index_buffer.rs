use std::ffi::c_void;
use std::ops::{Deref, DerefMut};

use super::host_buffer::HostBuffer;
use super::{record_host_copy, HostCopyReason};

/// Pointer-stable copy-on-write storage for native-width host indices.
///
/// This is the canonical owner for CSC column pointers and row indices on a
/// host whose foreign sparse ABI uses `usize`-width indices. Adapters with a
/// different index width must perform and account for an explicit conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct HostIndexBuffer {
    storage: HostBuffer<Vec<usize>>,
}

impl HostIndexBuffer {
    pub fn new(values: Vec<usize>) -> Self {
        Self {
            storage: HostBuffer::new(values),
        }
    }

    pub fn len(&self) -> usize {
        self.storage.get().len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.get().is_empty()
    }

    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        self.storage.shares_allocation_with(&other.storage)
    }

    pub fn make_unique(&mut self) {
        self.detach_for_write();
        self.storage.make_unique();
    }

    pub fn resize(&mut self, len: usize, value: usize) {
        self.detach_for_write();
        self.storage.make_mut().resize(len, value);
    }

    pub fn into_vec(self) -> Vec<usize> {
        if self.storage.is_shared() {
            record_host_copy(HostCopyReason::OwnedMaterialization, self.byte_len());
        }
        self.storage.into_inner()
    }

    /// # Safety
    ///
    /// The pointer is invocation-scoped and logically read-only. No Rust
    /// reference into the allocation may be used during foreign execution.
    pub unsafe fn foreign_data_pointer(&self) -> *mut c_void {
        self.storage.get().as_ptr().cast_mut().cast()
    }

    /// # Safety
    ///
    /// The pointer must not outlive this buffer, and no overlapping Rust
    /// reference may be used while foreign code writes through it.
    pub unsafe fn foreign_data_pointer_mut(&mut self) -> *mut c_void {
        self.detach_for_write();
        self.storage.make_mut().as_mut_ptr().cast()
    }

    fn byte_len(&self) -> usize {
        self.len().saturating_mul(std::mem::size_of::<usize>())
    }

    fn detach_for_write(&mut self) {
        if self.storage.is_shared() {
            record_host_copy(HostCopyReason::CopyOnWriteMutation, self.byte_len());
        }
    }
}

impl From<Vec<usize>> for HostIndexBuffer {
    fn from(values: Vec<usize>) -> Self {
        Self::new(values)
    }
}

impl PartialEq<Vec<usize>> for HostIndexBuffer {
    fn eq(&self, other: &Vec<usize>) -> bool {
        self.as_ref() == other.as_slice()
    }
}

impl PartialEq<HostIndexBuffer> for Vec<usize> {
    fn eq(&self, other: &HostIndexBuffer) -> bool {
        self.as_slice() == other.as_ref()
    }
}

impl PartialEq<[usize]> for HostIndexBuffer {
    fn eq(&self, other: &[usize]) -> bool {
        self.as_ref() == other
    }
}

impl AsRef<[usize]> for HostIndexBuffer {
    fn as_ref(&self) -> &[usize] {
        self.storage.get()
    }
}

impl Deref for HostIndexBuffer {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        self.storage.get()
    }
}

impl DerefMut for HostIndexBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.detach_for_write();
        self.storage.make_mut()
    }
}

impl<'a> IntoIterator for &'a HostIndexBuffer {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
