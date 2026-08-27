use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;

/// Metadata captured by the allocator that created an adoptable host block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostAllocationProvenance {
    pub allocator: String,
    pub byte_length: usize,
    pub capacity_bytes: usize,
    pub alignment: usize,
}

/// Releases an allocation whose ownership has moved into a RunMat value.
pub trait HostAllocationRelease: fmt::Debug + Send + Sync {
    /// # Safety
    ///
    /// `pointer` is the exact live allocation registered with this release
    /// authority and is released exactly once.
    unsafe fn release(&self, pointer: NonNull<u8>);
}

/// A host allocation with proven origin, layout, capacity, and release authority.
pub struct AdoptedHostAllocation {
    pointer: NonNull<u8>,
    provenance: HostAllocationProvenance,
    release: Arc<dyn HostAllocationRelease>,
}

impl AdoptedHostAllocation {
    /// # Safety
    ///
    /// `pointer` must identify a live allocation described exactly by
    /// `provenance`. `release` must accept that allocation and remain valid on
    /// whichever thread drops the resulting RunMat value. No other owner may
    /// release or resize the allocation after this call.
    pub unsafe fn new(
        pointer: NonNull<u8>,
        provenance: HostAllocationProvenance,
        release: Arc<dyn HostAllocationRelease>,
    ) -> Self {
        Self {
            pointer,
            provenance,
            release,
        }
    }

    pub fn pointer(&self) -> NonNull<u8> {
        self.pointer
    }

    pub fn provenance(&self) -> &HostAllocationProvenance {
        &self.provenance
    }
}

impl fmt::Debug for AdoptedHostAllocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdoptedHostAllocation")
            .field("pointer", &self.pointer)
            .field("provenance", &self.provenance)
            .finish_non_exhaustive()
    }
}

impl Drop for AdoptedHostAllocation {
    fn drop(&mut self) {
        // SAFETY: construction transfers the exact allocation to this owner,
        // and Drop runs once after every typed view has gone away.
        unsafe { self.release.release(self.pointer) };
    }
}

// The constructor requires a thread-safe release authority and documents that
// the allocation may be dropped on any thread. Access remains governed by
// ordinary Rust borrowing through `HostData<T>`.
unsafe impl Send for AdoptedHostAllocation {}
unsafe impl Sync for AdoptedHostAllocation {}

#[derive(Debug)]
pub(crate) enum HostData<T> {
    Rust(Vec<T>),
    Adopted(AdoptedHostData<T>),
}

#[derive(Debug)]
pub(crate) struct AdoptedHostData<T> {
    allocation: AdoptedHostAllocation,
    len: usize,
    marker: PhantomData<T>,
}

impl<T> HostData<T> {
    pub(crate) fn from_vec(values: Vec<T>) -> Self {
        Self::Rust(values)
    }

    pub(crate) fn try_adopt(
        allocation: AdoptedHostAllocation,
        len: usize,
    ) -> Result<Self, (AdoptedHostAllocation, String)> {
        let Some(required_bytes) = len.checked_mul(std::mem::size_of::<T>()) else {
            return Err((
                allocation,
                "typed host-buffer byte length overflowed".into(),
            ));
        };
        let byte_length = allocation.provenance().byte_length;
        let capacity_bytes = allocation.provenance().capacity_bytes;
        let recorded_alignment = allocation.provenance().alignment;
        let expected_alignment = std::mem::align_of::<T>();
        if byte_length < required_bytes {
            return Err((
                allocation,
                format!(
                    "host allocation contains {} bytes but {required_bytes} are required",
                    byte_length
                ),
            ));
        }
        if capacity_bytes < byte_length {
            return Err((
                allocation,
                "host allocation capacity is smaller than its byte length".into(),
            ));
        }
        if recorded_alignment < expected_alignment
            || !recorded_alignment.is_power_of_two()
            || !(allocation.pointer().as_ptr() as usize).is_multiple_of(expected_alignment)
        {
            return Err((
                allocation,
                format!("host allocation is not aligned for {expected_alignment}-byte elements"),
            ));
        }
        Ok(Self::Adopted(AdoptedHostData {
            allocation,
            len,
            marker: PhantomData,
        }))
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Self::Rust(values) => values,
            Self::Adopted(values) => values.as_slice(),
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Rust(values) => values,
            Self::Adopted(values) => {
                // SAFETY: `&mut self` supplies exclusive access and adoption
                // validated the complete typed layout.
                unsafe {
                    std::slice::from_raw_parts_mut(
                        values.allocation.pointer().as_ptr().cast::<T>(),
                        values.len,
                    )
                }
            }
        }
    }

    pub(crate) fn is_adopted(&self) -> bool {
        matches!(self, Self::Adopted(_))
    }
}

impl<T: Clone> HostData<T> {
    pub(crate) fn ensure_rust_vec(&mut self) -> &mut Vec<T> {
        if let Self::Adopted(values) = self {
            *self = Self::Rust(values.as_slice().to_vec());
        }
        let Self::Rust(values) = self else {
            unreachable!("adopted host data was materialized")
        };
        values
    }

    pub(crate) fn into_vec(self) -> Vec<T> {
        match self {
            Self::Rust(values) => values,
            Self::Adopted(values) => values.as_slice().to_vec(),
        }
    }
}

impl<T> AdoptedHostData<T> {
    fn as_slice(&self) -> &[T] {
        // SAFETY: this type is only built by `HostData::try_adopt`.
        unsafe {
            std::slice::from_raw_parts(self.allocation.pointer().as_ptr().cast::<T>(), self.len)
        }
    }
}

impl<T: Clone> Clone for HostData<T> {
    fn clone(&self) -> Self {
        Self::Rust(self.as_slice().to_vec())
    }
}

impl<T: PartialEq> PartialEq for HostData<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T> Deref for HostData<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for HostData<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct TestRelease {
        layout: Layout,
        releases: Arc<AtomicUsize>,
    }

    impl HostAllocationRelease for TestRelease {
        unsafe fn release(&self, pointer: NonNull<u8>) {
            self.releases.fetch_add(1, Ordering::SeqCst);
            // SAFETY: the test allocates this pointer with the same layout.
            unsafe { std::alloc::dealloc(pointer.as_ptr(), self.layout) };
        }
    }

    #[test]
    fn adopted_data_keeps_pointer_identity_and_releases_once() {
        let layout = Layout::array::<f64>(2).unwrap();
        // SAFETY: the layout is valid and checked for allocation failure.
        let pointer = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
        // SAFETY: the allocation contains two aligned f64 slots.
        unsafe {
            pointer.cast::<f64>().as_ptr().write(3.0);
            pointer.cast::<f64>().as_ptr().add(1).write(4.0);
        }
        let releases = Arc::new(AtomicUsize::new(0));
        let allocation = unsafe {
            AdoptedHostAllocation::new(
                pointer,
                HostAllocationProvenance {
                    allocator: "test".into(),
                    byte_length: layout.size(),
                    capacity_bytes: layout.size(),
                    alignment: layout.align(),
                },
                Arc::new(TestRelease {
                    layout,
                    releases: releases.clone(),
                }),
            )
        };
        let data = HostData::<f64>::try_adopt(allocation, 2).unwrap();
        assert_eq!(data.as_ptr().cast::<u8>(), pointer.as_ptr());
        assert_eq!(data.as_slice(), &[3.0, 4.0]);
        drop(data);
        assert_eq!(releases.load(Ordering::SeqCst), 1);
    }
}
