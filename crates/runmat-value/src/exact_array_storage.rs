//! Shared ownership of exact collection backing, including VM output lists.
//!
//! Children retain their own allocation and GC-root policies. Admitting this
//! collection does not certify a clone or materialization of its children.

use std::alloc::Layout;
use std::fmt;
use std::ops::Deref;
use std::sync::OnceLock;

use runmat_execution::resource::ExactShared;

mod value_projection;
pub use value_projection::{RetainedArrayValue, RetainedArrayValueError};
mod builder;
pub use builder::ExactArrayStorageBuilder;

use crate::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError, Value,
};

#[derive(Debug)]
struct ExactArrayPayload<T> {
    values: Vec<T>,
    // The vector and shared header must retire before their claim refunds.
    claim: OnceLock<ExactHostAllocationClaim>,
}

fn claim_cell(claim: Option<ExactHostAllocationClaim>) -> OnceLock<ExactHostAllocationClaim> {
    let cell = OnceLock::new();
    if let Some(claim) = claim {
        let _ = cell.set(claim);
    }
    cell
}

impl<T> ExactArrayPayload<T> {
    fn has_claim(&self) -> bool {
        self.claim.get().is_some()
    }

    fn into_parts(self) -> (Vec<T>, Option<ExactHostAllocationClaim>) {
        (self.values, self.claim.into_inner())
    }
}

/// The canonical exact collection backing. Ordinary unbounded construction
/// retains the producer's Vec without a header. Read-mostly ordinary
/// aggregates may explicitly opt into shared backing; exactly admitted
/// storage always shares its physical owner without cloning children.
#[derive(Debug)]
enum ExactArrayStorageInner<T> {
    Empty,
    EmptyExact,
    OwnedUnbounded(Vec<T>),
    SharedOrdinary {
        payload: ExactShared<ExactArrayPayload<T>>,
        len: usize,
    },
    SharedExact {
        payload: ExactShared<ExactArrayPayload<T>>,
        len: usize,
    },
}

#[derive(Debug)]
pub struct ExactArrayStorage<T>(ExactArrayStorageInner<T>);

/// Physical owner identity for a traversal retaining that owner. It conveys
/// no source, callable or allocation authority and must not outlive the owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SharedStorageIdentity(usize);
impl SharedStorageIdentity {
    pub(crate) fn of<T>(value: &T) -> Self {
        Self(value as *const T as usize)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactArrayMutationError {
    AdmittedStorage,
}

impl fmt::Display for ExactArrayMutationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("admitted storage requires an admitted destination")
    }
}

impl std::error::Error for ExactArrayMutationError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactArrayClaimPromotionError {
    UnsharedBacking,
    Allocation(ExactHostNumericAllocationError),
}

impl fmt::Display for ExactArrayClaimPromotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsharedBacking => formatter.write_str("claim promotion requires shared backing"),
            Self::Allocation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ExactArrayClaimPromotionError {}

/// The VM's established Value-specialized exact output collection.
pub type OutputListStorage = ExactArrayStorage<Value>;
pub type OutputListStorageBuilder = ExactArrayStorageBuilder<Value>;

#[derive(Debug)]
pub enum ExactArrayInitializationError<E> {
    Allocation(ExactHostNumericAllocationError),
    Element(E),
}

pub type OutputListInitializationError<E> = ExactArrayInitializationError<E>;

/// Failed move-in adoption. A reserved claim stays with the original Vec, so
/// physical backing retires before its claim refunds even on header failure.
#[derive(Debug)]
pub struct ExactArrayAdoptionError<T> {
    values: Vec<T>,
    claim: Option<ExactHostAllocationClaim>,
    source: ExactHostNumericAllocationError,
}

impl<T> ExactArrayAdoptionError<T> {
    pub fn source(&self) -> &ExactHostNumericAllocationError {
        &self.source
    }

    /// Only a refusal preceding reservation may return a bare Vec. A
    /// claim-bearing failure must retire the entire carrier intact.
    pub fn try_into_unclaimed_values(self) -> Result<Vec<T>, Self> {
        if self.claim.is_some() {
            return Err(self);
        }
        Ok(self.values)
    }
}

impl<E: fmt::Display> fmt::Display for ExactArrayInitializationError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocation(error) => write!(formatter, "exact-array allocation failed: {error}"),
            Self::Element(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ExactArrayInitializationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            Self::Element(error) => Some(error),
        }
    }
}

impl<T> ExactArrayStorage<T> {
    pub const fn empty() -> Self {
        Self(ExactArrayStorageInner::Empty)
    }

    /// Adopt an ordinary owned vector under an explicitly unbounded policy.
    /// Existing child allocations move into the owner without cloning.
    pub fn unbounded(values: Vec<T>) -> Self {
        Self(ExactArrayStorageInner::OwnedUnbounded(values))
    }

    /// Read-mostly ordinary aggregate backing. Clones share the original Vec
    /// and children in O(1); mutation requires uniqueness or explicit COW.
    pub fn shared_unbounded(values: Vec<T>) -> Self {
        let len = values.len();
        Self(ExactArrayStorageInner::SharedOrdinary {
            payload: ExactShared::new(ExactArrayPayload {
                values,
                claim: OnceLock::new(),
            }),
            len,
        })
    }

    /// Requested bytes for the exact-capacity vector and its actual shared
    /// control allocation. Child allocations require independent authority.
    pub fn requested_bytes(count: usize) -> Result<usize, ExactHostNumericAllocationError> {
        if count == 0 {
            return Ok(0);
        }
        Self::requested_bytes_for_capacity(count)
    }

    /// Physical vector capacity, not logical length, funds move-in adoption.
    /// The caller remains responsible for independently owned children.
    pub fn requested_bytes_for_capacity(
        capacity: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        if capacity == 0 {
            return Ok(0);
        }
        if std::mem::size_of::<T>() == 0 {
            return Err(ExactHostNumericAllocationError::LayoutOverflow);
        }
        Layout::array::<T>(capacity)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
            .size()
            .checked_add(ExactShared::<ExactArrayPayload<T>>::requested_bytes())
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    fn shared_physical_bytes(capacity: usize) -> Result<usize, ExactHostNumericAllocationError> {
        Layout::array::<T>(capacity)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
            .size()
            .checked_add(ExactShared::<ExactArrayPayload<T>>::requested_bytes())
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    /// Bytes not yet claimed by this *physical* shared owner. Quoting is
    /// read-only; another alias may attach the one claim before preparation.
    pub fn unclaimed_shared_bytes(&self) -> Result<Option<usize>, ExactArrayClaimPromotionError> {
        match &self.0 {
            ExactArrayStorageInner::Empty | ExactArrayStorageInner::EmptyExact => Ok(None),
            ExactArrayStorageInner::OwnedUnbounded(_) => {
                Err(ExactArrayClaimPromotionError::UnsharedBacking)
            }
            ExactArrayStorageInner::SharedOrdinary { payload, .. }
            | ExactArrayStorageInner::SharedExact { payload, .. } => {
                if payload.get().has_claim() {
                    Ok(None)
                } else {
                    Self::shared_physical_bytes(payload.get().values.capacity())
                        .map(Some)
                        .map_err(ExactArrayClaimPromotionError::Allocation)
                }
            }
        }
    }

    /// Attach one exact claim to the original immutable shared backing. All
    /// aliases observe the same claim; a failed reservation leaves the owner
    /// unchanged, and the final alias retires backing before refunding it.
    pub fn try_promote_shared_claim(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactArrayClaimPromotionError> {
        match &self.0 {
            ExactArrayStorageInner::Empty | ExactArrayStorageInner::EmptyExact => Ok(()),
            ExactArrayStorageInner::OwnedUnbounded(_) => {
                Err(ExactArrayClaimPromotionError::UnsharedBacking)
            }
            ExactArrayStorageInner::SharedOrdinary { payload, .. }
            | ExactArrayStorageInner::SharedExact { payload, .. } => {
                if payload.get().has_claim() {
                    return Ok(());
                }
                let bytes = Self::shared_physical_bytes(payload.get().values.capacity())
                    .map_err(ExactArrayClaimPromotionError::Allocation)?;
                // Never call an external permit while holding an owner lock.
                // A concurrent alias may win promotion; its claim is then the
                // one physical authority and our redundant reservation drops.
                let claim = match permit.reserve_exact_claim(bytes) {
                    Ok(claim) => claim,
                    Err(_error) if payload.get().has_claim() => return Ok(()),
                    Err(error) => return Err(ExactArrayClaimPromotionError::Allocation(error)),
                };
                let _ = payload.get().claim.set(claim);
                Ok(())
            }
        }
    }

    /// Adopt an already allocated vector without copying its backing. The
    /// failed adoption retains the original Vec and any claim in one opaque
    /// carrier; the exact claim retires after the vector and shared header.
    pub fn try_adopt_exact(
        values: Vec<T>,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactArrayAdoptionError<T>> {
        // SAFETY: the global allocator supplies a fresh block with the exact
        // requested layout, as required by ExactShared::try_new_with.
        unsafe {
            Self::try_adopt_exact_with(values, permit, |layout| {
                // SAFETY: alloc returns null or a fresh block of this layout.
                std::alloc::alloc(layout)
            })
        }
    }

    /// # Safety
    /// A non-null allocator result must be a fresh block from the global
    /// allocator with exactly the requested layout and alignment.
    unsafe fn try_adopt_exact_with(
        values: Vec<T>,
        permit: &dyn ExactHostAllocationPermit,
        allocate: impl FnOnce(Layout) -> *mut u8,
    ) -> Result<Self, ExactArrayAdoptionError<T>> {
        if values.capacity() == 0 {
            return Ok(Self(ExactArrayStorageInner::EmptyExact));
        }
        let requested = match Self::requested_bytes_for_capacity(values.capacity()) {
            Ok(requested) => requested,
            Err(source) => {
                return Err(ExactArrayAdoptionError {
                    values,
                    claim: None,
                    source,
                })
            }
        };
        let claim = match permit.reserve_exact_claim(requested) {
            Ok(claim) => claim,
            Err(source) => {
                return Err(ExactArrayAdoptionError {
                    values,
                    claim: None,
                    source,
                })
            }
        };
        let len = values.len();
        // SAFETY: upheld by this private function's allocator contract.
        let payload = unsafe {
            ExactShared::try_new_with(
                ExactArrayPayload {
                    values,
                    claim: claim_cell(Some(claim)),
                },
                allocate,
            )
        }
        .map_err(|payload| {
            let (values, claim) = payload.into_parts();
            ExactArrayAdoptionError {
                values,
                claim,
                source: ExactHostNumericAllocationError::AllocationFailed,
            }
        })?;
        Ok(Self(ExactArrayStorageInner::SharedExact { payload, len }))
    }

    /// Reserve both collection allocations before constructing any element.
    /// Each element moves into its final slot; the producer is responsible
    /// for independent child allocation/alias and root admission.
    pub fn try_initialize<E>(
        count: usize,
        permit: &dyn ExactHostAllocationPermit,
        mut initialize: impl FnMut(usize) -> Result<T, E>,
    ) -> Result<Self, ExactArrayInitializationError<E>> {
        use ExactArrayInitializationError::{Allocation, Element};

        let mut builder = Self::try_begin(count, permit).map_err(Allocation)?;
        for index in 0..count {
            let value = initialize(index).map_err(Element)?;
            builder
                .try_push(value)
                .map_err(|_| Allocation(ExactHostNumericAllocationError::LayoutOverflow))?;
        }
        builder
            .try_finish()
            .map_err(|_| Allocation(ExactHostNumericAllocationError::LayoutOverflow))
    }

    /// Reserve the final exact collection backing before producing children.
    /// The staged owner can be held across cooperative decode checkpoints.
    pub fn try_begin(
        count: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<ExactArrayStorageBuilder<T>, ExactHostNumericAllocationError> {
        ExactArrayStorageBuilder::try_new(count, permit)
    }

    pub fn as_slice(&self) -> &[T] {
        match &self.0 {
            ExactArrayStorageInner::Empty | ExactArrayStorageInner::EmptyExact => &[],
            ExactArrayStorageInner::OwnedUnbounded(values) => values,
            ExactArrayStorageInner::SharedOrdinary { payload, len }
            | ExactArrayStorageInner::SharedExact { payload, len } => &payload.get().values[..*len],
        }
    }

    pub fn shared_storage_identity(&self) -> Option<SharedStorageIdentity> {
        match &self.0 {
            ExactArrayStorageInner::SharedOrdinary { payload, .. }
            | ExactArrayStorageInner::SharedExact { payload, .. } => {
                Some(SharedStorageIdentity::of(payload.get()))
            }
            _ => None,
        }
    }

    pub fn has_retained_claim(&self) -> bool {
        match &self.0 {
            ExactArrayStorageInner::SharedOrdinary { payload, .. }
            | ExactArrayStorageInner::SharedExact { payload, .. } => payload.get().has_claim(),
            _ => false,
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(
            &self.0,
            ExactArrayStorageInner::EmptyExact | ExactArrayStorageInner::SharedExact { .. }
        ) || self.has_retained_claim()
    }

    /// Whether cloning this collection retains the same immutable physical
    /// backing rather than cloning its elements. Callers preparing a finite
    /// aggregate alias must check this before cloning opaque child arrays.
    pub fn is_shallow_clone(&self) -> bool {
        !matches!(&self.0, ExactArrayStorageInner::OwnedUnbounded(_))
    }

    /// Whether this backing currently has no shared aliases. This is an
    /// observation for preparing an ownership transfer, not an allocation
    /// grant; consuming operations still validate uniqueness themselves.
    pub fn is_unique(&self) -> bool {
        match &self.0 {
            ExactArrayStorageInner::Empty
            | ExactArrayStorageInner::EmptyExact
            | ExactArrayStorageInner::OwnedUnbounded(_) => true,
            ExactArrayStorageInner::SharedOrdinary { payload, .. }
            | ExactArrayStorageInner::SharedExact { payload, .. } => payload.is_unique(),
        }
    }

    /// Preserve ordinary growing collection behavior without permitting a
    /// finite exact backing to reallocate outside its selected operation.
    pub fn try_push_unbounded(&mut self, value: T) -> Result<(), T> {
        match &mut self.0 {
            ExactArrayStorageInner::Empty => {
                self.0 = Self::unbounded(vec![value]).0;
                Ok(())
            }
            ExactArrayStorageInner::EmptyExact => Err(value),
            ExactArrayStorageInner::OwnedUnbounded(values) => {
                values.push(value);
                Ok(())
            }
            ExactArrayStorageInner::SharedOrdinary { payload, len }
                if payload.is_unique() && !payload.get().has_claim() =>
            {
                let values = &mut payload.get_mut_unique().values;
                values.truncate(*len);
                values.push(value);
                *len += 1;
                Ok(())
            }
            ExactArrayStorageInner::SharedOrdinary { .. } => Err(value),
            ExactArrayStorageInner::SharedExact { .. } => Err(value),
        }
    }

    /// An explicit copy-on-write boundary for ordinary unbounded consumers.
    /// Exact storage, including exact empty storage, never takes this path.
    /// Callers selecting finite Managed work must prepare a new exact output
    /// instead of invoking this ordinary operation.
    pub fn make_unique_unbounded(&mut self) -> Result<(), ExactArrayMutationError>
    where
        T: Clone,
    {
        match &mut self.0 {
            ExactArrayStorageInner::Empty | ExactArrayStorageInner::OwnedUnbounded(_) => Ok(()),
            ExactArrayStorageInner::EmptyExact | ExactArrayStorageInner::SharedExact { .. } => {
                Err(ExactArrayMutationError::AdmittedStorage)
            }
            ExactArrayStorageInner::SharedOrdinary { payload, len } => {
                if payload.get().has_claim() {
                    return Err(ExactArrayMutationError::AdmittedStorage);
                }
                if !payload.is_unique() {
                    let values = payload.get().values[..*len].to_vec();
                    *payload = ExactShared::new(ExactArrayPayload {
                        values,
                        claim: OnceLock::new(),
                    });
                }
                Ok(())
            }
        }
    }

    /// Mutate the existing physical backing only when no shared alias can
    /// observe it. This preserves an exact vector/header claim across a
    /// same-cardinality transformation without creating another owner.
    pub fn try_as_mut_slice_unique(&mut self) -> Option<&mut [T]> {
        match &mut self.0 {
            ExactArrayStorageInner::Empty | ExactArrayStorageInner::EmptyExact => Some(&mut []),
            ExactArrayStorageInner::OwnedUnbounded(values) => Some(values),
            ExactArrayStorageInner::SharedOrdinary { payload, len }
            | ExactArrayStorageInner::SharedExact { payload, len } => {
                if !payload.is_unique() {
                    return None;
                }
                Some(&mut payload.get_mut_unique().values[..*len])
            }
        }
    }

    /// Physical vector capacity owned by this backing, including any exact
    /// move-in slack. This does not include independently owned children.
    pub fn capacity(&self) -> usize {
        match &self.0 {
            ExactArrayStorageInner::Empty | ExactArrayStorageInner::EmptyExact => 0,
            ExactArrayStorageInner::OwnedUnbounded(values) => values.capacity(),
            ExactArrayStorageInner::SharedOrdinary { payload, .. }
            | ExactArrayStorageInner::SharedExact { payload, .. } => {
                payload.get().values.capacity()
            }
        }
    }

    /// Extract unique storage with its claim intact. A shared collection is
    /// returned unchanged rather than allocating an unadmitted copy.
    pub fn try_into_owned(self) -> Result<OwnedExactArrayValues<T>, Self> {
        match self.0 {
            ExactArrayStorageInner::Empty => Ok(OwnedExactArrayValues {
                values: Vec::new(),
                claim: None,
                exact: false,
            }),
            ExactArrayStorageInner::EmptyExact => Ok(OwnedExactArrayValues {
                values: Vec::new(),
                claim: None,
                exact: true,
            }),
            ExactArrayStorageInner::OwnedUnbounded(values) => Ok(OwnedExactArrayValues {
                values,
                claim: None,
                exact: false,
            }),
            ExactArrayStorageInner::SharedOrdinary { payload, len } => match payload.try_unwrap() {
                Ok(payload) => {
                    let (mut values, claim) = payload.into_parts();
                    values.truncate(len);
                    Ok(OwnedExactArrayValues {
                        values,
                        exact: claim.is_some(),
                        claim,
                    })
                }
                Err(payload) => Err(Self(ExactArrayStorageInner::SharedOrdinary {
                    payload,
                    len,
                })),
            },
            ExactArrayStorageInner::SharedExact { payload, len } => match payload.try_unwrap() {
                Ok(payload) => {
                    let (mut values, claim) = payload.into_parts();
                    values.truncate(len);
                    Ok(OwnedExactArrayValues {
                        values,
                        claim,
                        exact: true,
                    })
                }
                Err(payload) => Err(Self(ExactArrayStorageInner::SharedExact { payload, len })),
            },
        }
    }

    pub fn truncate(&mut self, len: usize) {
        match &mut self.0 {
            ExactArrayStorageInner::Empty | ExactArrayStorageInner::EmptyExact => {}
            ExactArrayStorageInner::OwnedUnbounded(values) => values.truncate(len),
            ExactArrayStorageInner::SharedOrdinary {
                len: visible_len, ..
            }
            | ExactArrayStorageInner::SharedExact {
                len: visible_len, ..
            } => *visible_len = (*visible_len).min(len),
        }
    }

    pub fn try_into_first(self) -> Result<Option<T>, Self> {
        self.try_into_owned()
            .map(|values| values.into_iter().next())
    }
}

/// Character payloads may be read by value without cloning a child owner.
/// A shared exact alias stays live until iteration ends; a unique owner moves
/// its original vector and retained claim into the iterator.
pub enum ExactUtf16CodeUnitIntoIter {
    Owned(ExactArrayIntoIter<crate::Utf16CodeUnit>),
    Shared {
        owner: ExactArrayStorage<crate::Utf16CodeUnit>,
        index: usize,
    },
}

impl ExactArrayStorage<crate::Utf16CodeUnit> {
    pub fn into_code_unit_iter(self) -> ExactUtf16CodeUnitIntoIter {
        match self.try_into_owned() {
            Ok(owned) => ExactUtf16CodeUnitIntoIter::Owned(owned.into_iter()),
            Err(owner) => ExactUtf16CodeUnitIntoIter::Shared { owner, index: 0 },
        }
    }
}

impl Iterator for ExactUtf16CodeUnitIntoIter {
    type Item = crate::Utf16CodeUnit;

    fn next(&mut self) -> Option<crate::Utf16CodeUnit> {
        match self {
            Self::Owned(values) => values.next(),
            Self::Shared { owner, index } => {
                let character = owner.get(*index).copied()?;
                *index += 1;
                Some(character)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = match self {
            Self::Owned(values) => values.len(),
            Self::Shared { owner, index } => owner.len() - *index,
        };
        (count, Some(count))
    }
}

impl ExactSizeIterator for ExactUtf16CodeUnitIntoIter {}
impl std::iter::FusedIterator for ExactUtf16CodeUnitIntoIter {}

impl<T: Clone> Clone for ExactArrayStorage<T> {
    fn clone(&self) -> Self {
        match &self.0 {
            ExactArrayStorageInner::Empty => Self::empty(),
            ExactArrayStorageInner::EmptyExact => Self(ExactArrayStorageInner::EmptyExact),
            ExactArrayStorageInner::OwnedUnbounded(values) => Self::unbounded(values.clone()),
            ExactArrayStorageInner::SharedOrdinary { payload, len } => {
                Self(ExactArrayStorageInner::SharedOrdinary {
                    payload: payload.clone(),
                    len: *len,
                })
            }
            ExactArrayStorageInner::SharedExact { payload, len } => {
                Self(ExactArrayStorageInner::SharedExact {
                    payload: payload.clone(),
                    len: *len,
                })
            }
        }
    }
}

impl<T> Default for ExactArrayStorage<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> From<Vec<T>> for ExactArrayStorage<T> {
    fn from(values: Vec<T>) -> Self {
        Self::unbounded(values)
    }
}

impl<T> Deref for ExactArrayStorage<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: PartialEq> PartialEq for ExactArrayStorage<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<U>, U> PartialEq<Vec<U>> for ExactArrayStorage<T> {
    fn eq(&self, other: &Vec<U>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<[U; N]> for ExactArrayStorage<T> {
    fn eq(&self, other: &[U; N]) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<U>, U> PartialEq<ExactArrayStorage<U>> for &[T] {
    fn eq(&self, other: &ExactArrayStorage<U>) -> bool {
        *self == other.as_slice()
    }
}

impl<'a, T> IntoIterator for &'a ExactArrayStorage<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

/// A unique output vector whose physical claim follows its backing through
/// consumption. No bare Vec escape can discard that claim prematurely.
#[derive(Debug)]
pub struct OwnedExactArrayValues<T> {
    values: Vec<T>,
    claim: Option<ExactHostAllocationClaim>,
    exact: bool,
}

pub type OwnedOutputListValues = OwnedExactArrayValues<Value>;

impl<T> OwnedExactArrayValues<T> {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.values
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Remove an element without changing the physical capacity or claim.
    pub fn try_remove(&mut self, index: usize) -> Option<T> {
        (index < self.values.len()).then(|| self.values.remove(index))
    }

    pub fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }

    /// Append only within the already owned physical capacity. Growth needs
    /// a separately admitted destination; failure returns the input intact.
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.values.len() == self.values.capacity() {
            return Err(value);
        }
        self.values.push(value);
        Ok(())
    }

    /// Republish the same vector without copying its elements or dropping
    /// its claim. Extraction retired the previous shared header; an exact
    /// claim still reserves that header's layout for this replacement.
    /// Header allocation failure returns the complete original owner.
    pub fn try_into_storage(self) -> Result<ExactArrayStorage<T>, Self> {
        let Self {
            values,
            claim,
            exact,
        } = self;
        if values.capacity() == 0 {
            return Ok(ExactArrayStorage(if exact {
                ExactArrayStorageInner::EmptyExact
            } else {
                ExactArrayStorageInner::Empty
            }));
        }
        let len = values.len();
        match ExactShared::try_new(ExactArrayPayload {
            values,
            claim: claim_cell(claim),
        }) {
            Ok(payload) => Ok(ExactArrayStorage(if exact {
                ExactArrayStorageInner::SharedExact { payload, len }
            } else {
                ExactArrayStorageInner::SharedOrdinary { payload, len }
            })),
            Err(payload) => {
                let (values, claim) = payload.into_parts();
                Err(Self {
                    values,
                    claim,
                    exact,
                })
            }
        }
    }

    /// Legacy mutation is available only to ordinary unbounded storage. A
    /// claimed backing cannot shed its owner or fund a replacement Vec.
    pub fn try_into_unbounded_vec(self) -> Result<Vec<T>, Self> {
        if self.exact {
            return Err(self);
        }
        Ok(self.values)
    }
}

impl<T> IntoIterator for OwnedExactArrayValues<T> {
    type Item = T;
    type IntoIter = ExactArrayIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        ExactArrayIntoIter {
            values: self.values.into_iter(),
            _claim: self.claim,
        }
    }
}

impl<T> AsRef<[T]> for OwnedExactArrayValues<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T> AsMut<[T]> for OwnedExactArrayValues<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// The original vector backing is freed before its retained claim refunds,
/// including early iterator drop and partial consumption.
#[derive(Debug)]
pub struct ExactArrayIntoIter<T> {
    values: std::vec::IntoIter<T>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl<T> ExactArrayIntoIter<T> {
    /// Borrow the remaining already-owned physical slots without copying.
    pub fn as_slice(&self) -> &[T] {
        self.values.as_slice()
    }
}

pub type OutputListIntoIter = ExactArrayIntoIter<Value>;

impl<T> Iterator for ExactArrayIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.values.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}

impl<T> DoubleEndedIterator for ExactArrayIntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.values.next_back()
    }
}

impl<T> ExactSizeIterator for ExactArrayIntoIter<T> {}
impl<T> std::iter::FusedIterator for ExactArrayIntoIter<T> {}

#[cfg(test)]
mod tests;
