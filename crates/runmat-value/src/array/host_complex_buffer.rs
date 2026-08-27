use std::ffi::c_void;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

use super::host_allocation::HostData;
use super::host_buffer::HostBuffer;
use super::{record_host_copy, AdoptedHostAllocation, ComplexElement, HostCopyReason};

/// Pointer-stable copy-on-write storage for interleaved complex elements.
///
/// `ComplexElement<T>` supplies the stable element layout. This owner supplies
/// allocation lifetime and RunMat value semantics; foreign callers receive an
/// invocation-scoped pointer, never ownership of a Rust collection.
#[derive(Debug, Clone, PartialEq)]
pub struct HostComplexBuffer<T: Clone + Copy> {
    storage: HostBuffer<HostData<ComplexElement<T>>>,
}

impl<T: Clone + Copy> HostComplexBuffer<T> {
    pub fn from_elements(values: Vec<ComplexElement<T>>) -> Self {
        Self {
            storage: HostBuffer::new(HostData::from_vec(values)),
        }
    }

    pub fn try_adopt(
        allocation: AdoptedHostAllocation,
        len: usize,
    ) -> Result<Self, (AdoptedHostAllocation, String)> {
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
            record_host_copy(HostCopyReason::CopyOnWriteMutation, self.byte_len());
        }
        self.storage.make_unique();
    }

    pub fn resize(&mut self, len: usize, value: ComplexElement<T>) {
        if self.storage.is_shared() || self.storage.get().is_adopted() {
            record_host_copy(HostCopyReason::CopyOnWriteMutation, self.byte_len());
        }
        self.storage.make_mut().ensure_rust_vec().resize(len, value);
    }

    pub fn into_elements(self) -> Vec<ComplexElement<T>> {
        if self.storage.is_shared() || self.storage.get().is_adopted() {
            record_host_copy(HostCopyReason::OwnedMaterialization, self.byte_len());
        }
        self.storage.into_inner().into_vec()
    }

    fn byte_len(&self) -> usize {
        self.len()
            .checked_mul(std::mem::size_of::<ComplexElement<T>>())
            .expect("allocated complex buffer byte length fits usize")
    }

    /// Returns a logically read-only pointer for one synchronous foreign call.
    ///
    /// # Safety
    ///
    /// The pointer must not outlive this owner or its invocation lease. No Rust
    /// reference into the allocation may be used while foreign code executes.
    pub unsafe fn foreign_data_pointer(&self) -> *mut c_void {
        self.storage.get().as_slice().as_ptr().cast_mut().cast()
    }

    /// Returns a writable pointer after applying copy-on-write semantics.
    ///
    /// # Safety
    ///
    /// The pointer must remain invocation-scoped and must not overlap a live
    /// Rust reference.
    pub unsafe fn foreign_data_pointer_mut(&mut self) -> *mut c_void {
        if self.storage.is_shared() {
            record_host_copy(HostCopyReason::CopyOnWriteMutation, self.byte_len());
        }
        self.storage.make_mut().as_mut_slice().as_mut_ptr().cast()
    }
}

impl<T: Clone + Copy> From<Vec<ComplexElement<T>>> for HostComplexBuffer<T> {
    fn from(values: Vec<ComplexElement<T>>) -> Self {
        Self::from_elements(values)
    }
}

impl<T: Clone + Copy> From<Vec<(T, T)>> for HostComplexBuffer<T> {
    fn from(values: Vec<(T, T)>) -> Self {
        values.into_iter().collect()
    }
}

impl<T: Clone + Copy> FromIterator<ComplexElement<T>> for HostComplexBuffer<T> {
    fn from_iter<I: IntoIterator<Item = ComplexElement<T>>>(iter: I) -> Self {
        Self::from_elements(iter.into_iter().collect())
    }
}

impl<T: Clone + Copy> FromIterator<(T, T)> for HostComplexBuffer<T> {
    fn from_iter<I: IntoIterator<Item = (T, T)>>(iter: I) -> Self {
        Self::from_elements(iter.into_iter().map(ComplexElement::from).collect())
    }
}

impl<T: Clone + Copy> Deref for HostComplexBuffer<T> {
    type Target = [ComplexElement<T>];

    fn deref(&self) -> &Self::Target {
        self.storage.get().as_slice()
    }
}

impl<T: Clone + Copy> DerefMut for HostComplexBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.storage.is_shared() {
            record_host_copy(HostCopyReason::CopyOnWriteMutation, self.byte_len());
        }
        self.storage.make_mut().as_mut_slice()
    }
}

impl<T: Clone + Copy> IntoIterator for HostComplexBuffer<T> {
    type Item = (T, T);
    type IntoIter =
        std::iter::Map<std::vec::IntoIter<ComplexElement<T>>, fn(ComplexElement<T>) -> (T, T)>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_elements().into_iter().map(Into::<(T, T)>::into)
    }
}

impl<'a, T: Clone + Copy> IntoIterator for &'a HostComplexBuffer<T> {
    type Item = &'a ComplexElement<T>;
    type IntoIter = std::slice::Iter<'a, ComplexElement<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Clone + Copy> IntoIterator for &'a mut HostComplexBuffer<T> {
    type Item = &'a mut ComplexElement<T>;
    type IntoIter = std::slice::IterMut<'a, ComplexElement<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_shares_until_mutation() {
        let original: HostComplexBuffer<f64> = vec![(1.0, -1.0), (2.0, -2.0)].into();
        let mut changed = original.clone();
        assert!(original.shares_allocation_with(&changed));

        changed[0] = ComplexElement(7.0, -7.0);

        assert!(!original.shares_allocation_with(&changed));
        assert_eq!(original[0], ComplexElement(1.0, -1.0));
        assert_eq!(changed[0], ComplexElement(7.0, -7.0));
    }

    #[test]
    fn foreign_pointer_uses_the_authoritative_allocation() {
        let values: HostComplexBuffer<f32> = vec![(1.0, 2.0), (3.0, 4.0)].into();
        // SAFETY: the pointer is only compared while `values` remains alive.
        let foreign = unsafe { values.foreign_data_pointer() };
        assert_eq!(foreign, values.as_ptr().cast_mut().cast());
        assert_eq!(
            std::mem::size_of::<ComplexElement<f32>>(),
            2 * std::mem::size_of::<f32>()
        );
    }
}
