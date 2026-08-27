use std::ffi::c_void;
use std::ops::{Deref, DerefMut};

use super::host_allocation::HostData;
use super::host_buffer::HostBuffer;
use super::{record_host_copy, AdoptedHostAllocation, HostCopyReason};

/// Pointer-stable copy-on-write storage for dense logical arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct HostLogicalBuffer {
    storage: HostBuffer<HostData<u8>>,
}

impl HostLogicalBuffer {
    pub fn new(mut values: Vec<u8>) -> Self {
        values
            .iter_mut()
            .for_each(|value| *value = u8::from(*value != 0));
        Self {
            storage: HostBuffer::new(HostData::from_vec(values)),
        }
    }

    pub fn try_adopt(
        allocation: AdoptedHostAllocation,
        len: usize,
    ) -> Result<Self, (AdoptedHostAllocation, String)> {
        if allocation.provenance().byte_length < len {
            return Err((
                allocation,
                "logical allocation is smaller than the requested payload".into(),
            ));
        }
        // SAFETY: provenance proves at least `len` readable bytes.
        let input = unsafe { std::slice::from_raw_parts(allocation.pointer().as_ptr(), len) };
        if input.iter().any(|value| *value > 1) {
            return Err((
                allocation,
                "logical allocation contains noncanonical bytes".into(),
            ));
        }
        HostData::try_adopt(allocation, len).map(|storage| Self {
            storage: HostBuffer::new(storage),
        })
    }

    pub fn len(&self) -> usize {
        self.storage.get().as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.get().as_slice().is_empty()
    }

    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        self.storage.shares_allocation_with(&other.storage)
    }

    pub fn make_unique(&mut self) {
        if self.storage.is_shared() {
            record_host_copy(HostCopyReason::CopyOnWriteMutation, self.len());
        }
        self.storage.make_unique();
    }

    pub fn resize(&mut self, len: usize, value: u8) {
        if self.storage.is_shared() || self.storage.get().is_adopted() {
            record_host_copy(HostCopyReason::CopyOnWriteMutation, self.len());
        }
        self.storage
            .make_mut()
            .ensure_rust_vec()
            .resize(len, u8::from(value != 0));
    }

    pub fn as_slice(&self) -> &[u8] {
        self
    }

    pub fn drain<R>(&mut self, range: R) -> std::vec::Drain<'_, u8>
    where
        R: std::ops::RangeBounds<usize>,
    {
        if self.storage.is_shared() || self.storage.get().is_adopted() {
            record_host_copy(HostCopyReason::CopyOnWriteMutation, self.len());
        }
        self.storage.make_mut().ensure_rust_vec().drain(range)
    }

    pub fn into_vec(self) -> Vec<u8> {
        if self.storage.is_shared() || self.storage.get().is_adopted() {
            record_host_copy(HostCopyReason::OwnedMaterialization, self.len());
        }
        self.storage.into_inner().into_vec()
    }

    /// # Safety
    ///
    /// The pointer is invocation-scoped and logically read-only. No Rust
    /// reference into the allocation may be used during foreign execution.
    pub unsafe fn foreign_data_pointer(&self) -> *mut c_void {
        self.storage.get().as_slice().as_ptr().cast_mut().cast()
    }

    /// # Safety
    ///
    /// The pointer must not outlive this buffer, and no overlapping Rust
    /// reference may be used while foreign code writes through it.
    pub unsafe fn foreign_data_pointer_mut(&mut self) -> *mut c_void {
        if self.storage.is_shared() {
            record_host_copy(HostCopyReason::CopyOnWriteMutation, self.len());
        }
        self.storage.make_mut().as_mut_slice().as_mut_ptr().cast()
    }
}

impl From<Vec<u8>> for HostLogicalBuffer {
    fn from(values: Vec<u8>) -> Self {
        Self::new(values)
    }
}

impl PartialEq<Vec<u8>> for HostLogicalBuffer {
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<HostLogicalBuffer> for Vec<u8> {
    fn eq(&self, other: &HostLogicalBuffer) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<&[u8]> for HostLogicalBuffer {
    fn eq(&self, other: &&[u8]) -> bool {
        self.as_slice() == *other
    }
}

impl PartialEq<[u8]> for HostLogicalBuffer {
    fn eq(&self, other: &[u8]) -> bool {
        self.as_slice() == other
    }
}

impl Deref for HostLogicalBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.storage.get().as_slice()
    }
}

impl DerefMut for HostLogicalBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.storage.is_shared() {
            record_host_copy(HostCopyReason::CopyOnWriteMutation, self.len());
        }
        self.storage.make_mut().as_mut_slice()
    }
}

impl IntoIterator for HostLogicalBuffer {
    type Item = u8;
    type IntoIter = std::vec::IntoIter<u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a> IntoIterator for &'a HostLogicalBuffer {
    type Item = &'a u8;
    type IntoIter = std::slice::Iter<'a, u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
