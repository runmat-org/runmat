//! Independently claimed, shared column-major host shape metadata.
//!
//! Numeric payload and shape have distinct physical owners. Cloning shares
//! this immutable shape without another allocation or quota claim.

use std::alloc::{alloc, Layout};
use std::ops::Deref;
use std::sync::OnceLock;

use runmat_execution::resource::ExactShared;

use super::exact_host_numeric::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};

#[derive(Debug)]
struct ShapePayload {
    dimensions: Vec<usize>,
    // The claim drops after dimensions. ExactShared drops its header first.
    _retained_claim: OnceLock<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug)]
pub struct SharedShape(ExactShared<ShapePayload>);

/// Incrementally initializes one already-admitted final shape backing.
/// An incomplete builder never exposes a shape to array constructors.
#[derive(Debug)]
pub struct SharedShapeBuilder {
    shape: SharedShape,
    rank: usize,
}

impl SharedShapeBuilder {
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    pub fn expected_len(&self) -> usize {
        self.rank
    }

    pub fn try_push(&mut self, dimension: usize) -> Result<(), usize> {
        let dimensions = &mut self.shape.0.get_mut_unique().dimensions;
        if dimensions.len() == self.rank {
            return Err(dimension);
        }
        dimensions.push(dimension);
        Ok(())
    }

    pub fn try_finish(self) -> Result<SharedShape, Self> {
        if self.len() == self.rank {
            Ok(self.shape)
        } else {
            Err(self)
        }
    }
}

/// One canonical constructor input for a fresh ordinary shape or an
/// independently owned shape transferred from another value.
pub enum HostShapeInput {
    Owned(Vec<usize>),
    Shared(SharedShape),
}

impl From<Vec<usize>> for HostShapeInput {
    fn from(shape: Vec<usize>) -> Self {
        Self::Owned(shape)
    }
}

impl From<SharedShape> for HostShapeInput {
    fn from(shape: SharedShape) -> Self {
        Self::Shared(shape)
    }
}

impl SharedShape {
    pub fn shared_storage_identity(&self) -> crate::exact_array_storage::SharedStorageIdentity {
        crate::exact_array_storage::SharedStorageIdentity::of(self.0.get())
    }
    /// Reserves the original backing and header before either allocation.
    pub fn begin_exact(
        rank: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<SharedShapeBuilder, ExactHostNumericAllocationError> {
        let claim = permit.reserve_exact_claim(Self::requested_bytes_for_rank(rank)?)?;
        let dimensions = super::exact_vec::exact_capacity_vec(rank)?;
        let payload = ShapePayload {
            dimensions,
            _retained_claim: OnceLock::from(claim),
        };
        let shape = ExactShared::try_new(payload)
            .map(Self)
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
        Ok(SharedShapeBuilder { shape, rank })
    }
    /// Physical shape bytes, excluding the permit's claim-control charge.
    pub fn requested_bytes_for_rank(rank: usize) -> Result<usize, ExactHostNumericAllocationError> {
        Self::requested_bytes_for_capacity(rank)
    }

    /// Claim acquisition precedes both backing and owner allocations.
    pub fn new_admitted_from_slice(
        shape: &[usize],
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let requested = Self::requested_bytes_for_rank(shape.len())?;
        let claim = permit.reserve_exact_claim(requested)?;
        let dimensions = exact_dimensions(shape)?;
        let payload = ShapePayload {
            dimensions,
            _retained_claim: OnceLock::from(claim),
        };
        ExactShared::try_new(payload)
            .map(Self)
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }

    /// Move an existing Rust shape backing into the canonical shared owner.
    /// Its claim covers the actual capacity and the ExactShared header.
    pub(crate) fn new_admitted_from_owned(
        dimensions: Vec<usize>,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let requested = Self::requested_bytes_for_capacity(dimensions.capacity())?;
        let claim = permit.reserve_exact_claim(requested)?;
        let payload = ShapePayload {
            dimensions,
            _retained_claim: OnceLock::from(claim),
        };
        ExactShared::try_new(payload)
            .map(Self)
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }

    pub(crate) fn requested_bytes_for_capacity(
        capacity: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        Layout::array::<usize>(capacity)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
            .size()
            .checked_add(ExactShared::<ShapePayload>::requested_bytes())
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    /// Only callers without finite output admission may create this owner.
    pub fn new_unmanaged(shape: Vec<usize>) -> Self {
        Self(ExactShared::new(ShapePayload {
            dimensions: shape,
            _retained_claim: OnceLock::new(),
        }))
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.0.get().dimensions.capacity()
    }

    pub fn is_admitted(&self) -> bool {
        self.0.get()._retained_claim.get().is_some()
    }

    pub fn unclaimed_shared_bytes(&self) -> Result<Option<usize>, ExactHostNumericAllocationError> {
        if self.is_admitted() {
            Ok(None)
        } else {
            Self::requested_bytes_for_capacity(self.0.get().dimensions.capacity()).map(Some)
        }
    }

    /// Attach exactly one claim to the original shared physical owner. All
    /// aliases observe promotion and retain the claim through final drop.
    pub fn try_promote_shared_claim(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactHostNumericAllocationError> {
        if self.is_admitted() {
            return Ok(());
        }
        let bytes = Self::requested_bytes_for_capacity(self.0.get().dimensions.capacity())?;
        let claim = match permit.reserve_exact_claim(bytes) {
            Ok(claim) => claim,
            Err(_) if self.is_admitted() => return Ok(()),
            Err(source) => return Err(source),
        };
        let _ = self.0.get()._retained_claim.set(claim);
        Ok(())
    }

    #[cfg(test)]
    fn shares_allocation_with(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}

fn exact_dimensions(shape: &[usize]) -> Result<Vec<usize>, ExactHostNumericAllocationError> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }
    let layout = Layout::array::<usize>(shape.len())
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
    // SAFETY: a nonnull block has exactly Vec<usize>'s layout; all elements
    // are initialized from the borrowed source before publication.
    let pointer = unsafe { alloc(layout) }.cast::<usize>();
    if pointer.is_null() {
        return Err(ExactHostNumericAllocationError::AllocationFailed);
    }
    unsafe {
        pointer.copy_from_nonoverlapping(shape.as_ptr(), shape.len());
        Ok(Vec::from_raw_parts(pointer, shape.len(), shape.len()))
    }
}

impl Deref for SharedShape {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0.get().dimensions
    }
}

impl<'a> IntoIterator for &'a SharedShape {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

/// Dimension iteration retains only the immutable shape owner, not the
/// originating tensor's payload. Consuming a shape never materializes a Vec.
#[derive(Clone, Debug)]
pub struct SharedShapeIntoIter {
    shape: SharedShape,
    indices: std::ops::Range<usize>,
}

impl Iterator for SharedShapeIntoIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|index| self.shape[index])
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.indices.nth(n).map(|index| self.shape[index])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }

    fn count(self) -> usize {
        self.indices.len()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl DoubleEndedIterator for SharedShapeIntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.indices.next_back().map(|index| self.shape[index])
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.indices.nth_back(n).map(|index| self.shape[index])
    }
}

impl ExactSizeIterator for SharedShapeIntoIter {}
impl std::iter::FusedIterator for SharedShapeIntoIter {}

impl IntoIterator for SharedShape {
    type Item = usize;
    type IntoIter = SharedShapeIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        let indices = 0..self.len();
        Self::IntoIter {
            shape: self,
            indices,
        }
    }
}

impl From<Vec<usize>> for SharedShape {
    fn from(shape: Vec<usize>) -> Self {
        Self::new_unmanaged(shape)
    }
}

impl PartialEq for SharedShape {
    fn eq(&self, other: &Self) -> bool {
        self.0.get().dimensions == other.0.get().dimensions
    }
}

impl PartialEq<Vec<usize>> for SharedShape {
    fn eq(&self, other: &Vec<usize>) -> bool {
        self.0.get().dimensions == *other
    }
}

impl PartialEq<SharedShape> for Vec<usize> {
    fn eq(&self, other: &SharedShape) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<[usize]> for SharedShape {
    fn eq(&self, other: &[usize]) -> bool {
        self.0.get().dimensions.as_slice() == other
    }
}

impl PartialEq<&[usize]> for SharedShape {
    fn eq(&self, other: &&[usize]) -> bool {
        self.as_slice() == *other
    }
}

impl<const N: usize> PartialEq<[usize; N]> for SharedShape {
    fn eq(&self, other: &[usize; N]) -> bool {
        self.0.get().dimensions.as_slice() == other
    }
}

#[cfg(test)]
mod tests {
    use super::SharedShape;
    use crate::array::exact_host_numeric::{
        ExactHostAllocationPermit, ExactHostAllocationReservation, ExactHostNumericAllocationError,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct TestClaim {
        live: Arc<AtomicUsize>,
        bytes: usize,
    }

    impl Drop for TestClaim {
        fn drop(&mut self) {
            self.live.fetch_sub(self.bytes, Ordering::SeqCst);
        }
    }

    impl ExactHostAllocationReservation for TestClaim {}

    #[derive(Debug)]
    struct TestPermit(Arc<AtomicUsize>);

    impl ExactHostAllocationPermit for TestPermit {
        fn reserve_exact(
            &self,
            bytes: usize,
        ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError>
        {
            self.0.fetch_add(bytes, Ordering::SeqCst);
            Ok(Arc::new(TestClaim {
                live: self.0.clone(),
                bytes,
            }))
        }
    }

    #[test]
    fn clone_shares_shape_and_replacement_leaves_other_dimensions_unchanged() {
        let mut first = SharedShape::new_unmanaged(vec![2, 3]);
        let second = first.clone();
        assert!(first.shares_allocation_with(&second));
        assert_eq!(&*second, &[2, 3]);
        assert!((&second).into_iter().copied().eq([2, 3]));
        assert!(second.clone().into_iter().rev().eq([3, 2]));
        first = SharedShape::new_unmanaged(vec![3, 2]);
        assert!(!first.shares_allocation_with(&second));
        assert_eq!(&*first, &[3, 2]);
        assert_eq!(&*second, &[2, 3]);
    }

    #[test]
    fn independently_claimed_shape_survives_source_retirement_and_releases_once() {
        let live = Arc::new(AtomicUsize::new(0));
        let permit = TestPermit(live.clone());
        let expected = SharedShape::requested_bytes_for_rank(2).unwrap();
        let mut staged = SharedShape::begin_exact(2, &permit).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), expected);
        staged.try_push(4).unwrap();
        let mut staged = staged.try_finish().unwrap_err();
        assert_eq!(staged.expected_len(), 2);
        staged.try_push(7).unwrap();
        assert_eq!(staged.try_push(9), Err(9));
        let staged = staged.try_finish().unwrap();
        assert_eq!(&*staged, &[4, 7]);
        assert_eq!(staged.capacity(), 2);
        drop(staged);
        assert_eq!(live.load(Ordering::SeqCst), 0);
        let source = SharedShape::new_admitted_from_slice(&[4, 7], &permit).unwrap();
        let alias = source.clone();
        assert_eq!(live.load(Ordering::SeqCst), expected);
        drop(source);
        assert_eq!(live.load(Ordering::SeqCst), expected);
        assert_eq!(&*alias, &[4, 7]);
        let mut dimensions = alias.into_iter();
        assert_eq!(dimensions.next(), Some(4));
        assert_eq!(dimensions.next_back(), Some(7));
        assert_eq!(dimensions.next(), None);
        assert_eq!(live.load(Ordering::SeqCst), expected);
        drop(dimensions);
        assert_eq!(live.load(Ordering::SeqCst), 0);

        let mut dimensions = Vec::with_capacity(8);
        dimensions.extend([4, 7]);
        let pointer = dimensions.as_ptr();
        let expected = SharedShape::requested_bytes_for_capacity(dimensions.capacity()).unwrap();
        let source = SharedShape::new_unmanaged(dimensions);
        let alias = source.clone();
        assert_eq!(source.unclaimed_shared_bytes().unwrap(), Some(expected));
        source.try_promote_shared_claim(&permit).unwrap();
        assert!(alias.is_admitted());
        assert_eq!(alias.as_ptr(), pointer);
        alias.try_promote_shared_claim(&permit).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), expected);
        drop(source);
        assert_eq!(live.load(Ordering::SeqCst), expected);
        drop(alias);
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn all_dense_constructors_reuse_one_independent_shape_owner() {
        use crate::array::{ComplexTensor, LogicalArray, Tensor};

        let shape = SharedShape::new_unmanaged(vec![2, 1]);
        let real = Tensor::new(vec![1.0, 2.0], shape.clone()).unwrap();
        let complex = ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0)], shape.clone()).unwrap();
        let logical = LogicalArray::new(vec![1, 0], shape.clone()).unwrap();
        assert!(shape.shares_allocation_with(&real.shared_shape()));
        assert!(shape.shares_allocation_with(&complex.shared_shape()));
        assert!(shape.shares_allocation_with(&logical.shared_shape()));
    }
}
