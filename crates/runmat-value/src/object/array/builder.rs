//! One admitted ObjectArray generation. The provenance lookup table is
//! construction-only exact storage; final aliases retain one immutable list.
use super::*;
use crate::{ExactArrayStorageBuilder, ExactHostAllocationPermit, ExactHostNumericAllocationError};
use std::{collections::hash_map::RandomState, hash::BuildHasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectArrayQuote {
    /// Actual construction peak, including the temporary fixed identity index.
    pub requested_bytes: usize,
    pub retained_claims: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectArrayBuildError {
    InvalidShape,
    ShapeCountMismatch,
    Incomplete,
    InvalidElement,
    InvalidProvenance,
    Capacity,
    Allocation(ExactHostNumericAllocationError),
}
impl From<ExactHostNumericAllocationError> for ObjectArrayBuildError {
    fn from(source: ExactHostNumericAllocationError) -> Self {
        Self::Allocation(source)
    }
}
impl fmt::Display for ObjectArrayBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                f.write_str("object array shape must contain at least two dimensions")
            }
            Self::ShapeCountMismatch => {
                f.write_str("object array shape differs from its element count")
            }
            Self::Incomplete => f.write_str("object array construction is incomplete"),
            Self::InvalidElement => {
                f.write_str("object array element has a different class or storage kind")
            }
            Self::InvalidProvenance => {
                f.write_str("object array selected product has a different class")
            }
            Self::Capacity => {
                f.write_str("object array construction exceeded its reserved capacity")
            }
            Self::Allocation(source) => source.fmt(f),
        }
    }
}
impl std::error::Error for ObjectArrayBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(source) => Some(source),
            _ => None,
        }
    }
}
pub struct ObjectArrayElementError {
    kind: ObjectArrayBuildError,
    message: String,
    value: Value,
}
impl ObjectArrayElementError {
    pub fn into_parts(self) -> (ObjectArrayBuildError, String, Value) {
        (self.kind, self.message, self.value)
    }
}
/// No mutable published view or implicit COW. All elements and empty-only
/// selected products are validated before their slots become visible.
pub struct ObjectArrayBuilder {
    data: ExactArrayStorageBuilder<Value>,
    shape: SharedShape,
    class_name: SharedClassIdentity,
    provenance: ExactArrayStorageBuilder<Rc<SelectedObjectClassFacts>>,
    index: ExactArrayStorageBuilder<usize>,
    hasher: RandomState,
    storage_kind: Option<admission::StorageKind>,
    unselected: bool,
}
impl ObjectArrayBuilder {
    pub fn quote(
        class: &SharedClassIdentity,
        count: usize,
        shape: &[usize],
        provenance_capacity: usize,
    ) -> Result<ObjectArrayQuote, ObjectArrayBuildError> {
        let mut quote = Self::quote_storage(count, shape, provenance_capacity)?;
        if !class.has_retained_claim() {
            quote.requested_bytes = quote
                .requested_bytes
                .checked_add(class.requested_bytes()?)
                .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
            quote.retained_claims += 1;
        }
        Ok(quote)
    }
    /// Storage quote excluding an independently prepared/retained class owner.
    pub fn quote_storage(
        count: usize,
        shape: &[usize],
        provenance_capacity: usize,
    ) -> Result<ObjectArrayQuote, ObjectArrayBuildError> {
        validate_shape(count, shape)?;
        if count != 0 && provenance_capacity != 0 {
            return Err(ObjectArrayBuildError::InvalidProvenance);
        }
        let index = index_capacity(provenance_capacity)?;
        let requested_bytes = OutputListStorage::requested_bytes(count)?
            .checked_add(SharedShape::requested_bytes_for_rank(shape.len())?)
            .and_then(|bytes| {
                bytes.checked_add(
                    ExactArrayStorage::<Rc<SelectedObjectClassFacts>>::requested_bytes(
                        provenance_capacity,
                    )
                    .ok()?,
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(ExactArrayStorage::<usize>::requested_bytes(index).ok()?)
            })
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Ok(ObjectArrayQuote {
            requested_bytes,
            retained_claims: 1
                + usize::from(count != 0)
                + 2 * usize::from(provenance_capacity != 0),
        })
    }
    pub fn try_new(
        class_name: SharedClassIdentity,
        count: usize,
        shape: &[usize],
        provenance_capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ObjectArrayBuildError> {
        Self::quote(&class_name, count, shape, provenance_capacity)?;
        let shape = SharedShape::new_admitted_from_slice(shape, permit)?;
        Self::try_new_retaining_shape(class_name, count, shape, provenance_capacity, permit)
    }
    pub fn quote_storage_retaining_shape(
        count: usize,
        shape: &SharedShape,
        provenance_capacity: usize,
    ) -> Result<ObjectArrayQuote, ObjectArrayBuildError> {
        let mut quote = Self::quote_storage(count, shape, provenance_capacity)?;
        quote.requested_bytes -= SharedShape::requested_bytes_for_rank(shape.len())?;
        quote.retained_claims -= 1;
        if let Some(bytes) = shape.unclaimed_shared_bytes()? {
            quote.requested_bytes = quote
                .requested_bytes
                .checked_add(bytes)
                .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
            quote.retained_claims += 1;
        }
        Ok(quote)
    }
    pub fn quote_retaining_shape(
        class: &SharedClassIdentity,
        count: usize,
        shape: &SharedShape,
        provenance_capacity: usize,
    ) -> Result<ObjectArrayQuote, ObjectArrayBuildError> {
        let mut quote = Self::quote_storage_retaining_shape(count, shape, provenance_capacity)?;
        if !class.has_retained_claim() {
            quote.requested_bytes = quote
                .requested_bytes
                .checked_add(class.requested_bytes()?)
                .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
            quote.retained_claims += 1;
        }
        Ok(quote)
    }
    /// Retain the one canonical shape generation, promoting it in place only
    /// if its original allocation has not yet acquired a claim.
    pub fn try_new_retaining_shape(
        class_name: SharedClassIdentity,
        count: usize,
        shape: SharedShape,
        provenance_capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ObjectArrayBuildError> {
        Self::quote_retaining_shape(&class_name, count, &shape, provenance_capacity)?;
        shape.try_promote_shared_claim(permit)?;
        let class_name = class_name
            .try_admit_exact(permit)
            .map_err(|e| ObjectArrayBuildError::Allocation(e.source()))?;
        let data = OutputListStorage::try_begin(count, permit)?;
        let provenance = ExactArrayStorage::try_begin(provenance_capacity, permit)?;
        let capacity = index_capacity(provenance_capacity)?;
        let mut index = ExactArrayStorage::try_begin(capacity, permit)?;
        for _ in 0..capacity {
            index
                .try_push(0)
                .map_err(|_| ObjectArrayBuildError::Capacity)?;
        }
        Ok(Self {
            data,
            shape,
            class_name,
            provenance,
            index,
            hasher: RandomState::new(),
            storage_kind: None,
            unselected: false,
        })
    }
    pub fn try_push(&mut self, value: Value) -> Result<(), ObjectArrayElementError> {
        if self.data.initialized().len() == self.data.expected_len() {
            return Err(ObjectArrayElementError {
                kind: ObjectArrayBuildError::Capacity,
                message: "object array construction exceeded its reserved capacity".into(),
                value,
            });
        }
        let mut storage_kind = self.storage_kind;
        if let Err(message) =
            admission::validate_value(&value, self.class_name.as_identity(), &mut storage_kind)
        {
            return Err(ObjectArrayElementError {
                kind: ObjectArrayBuildError::InvalidElement,
                message,
                value,
            });
        }
        self.storage_kind = storage_kind;
        self.data
            .try_push(value)
            .map_err(|value| ObjectArrayElementError {
                kind: ObjectArrayBuildError::Capacity,
                message: "object array construction exceeded its reserved capacity".into(),
                value,
            })
    }
    pub fn try_push_selected_provenance(
        &mut self,
        facts: Rc<SelectedObjectClassFacts>,
    ) -> Result<bool, ObjectArrayBuildError> {
        if facts.class() != self.class_name.as_identity() {
            return Err(ObjectArrayBuildError::InvalidProvenance);
        }
        if self.data.expected_len() != 0 {
            return Ok(false);
        }
        let capacity = self.index.initialized().len();
        if capacity == 0 {
            return Err(ObjectArrayBuildError::Capacity);
        }
        let mask = capacity - 1;
        let mut bucket = self.hasher.hash_one(Rc::as_ptr(&facts)) as usize & mask;
        for _ in 0..capacity {
            let ordinal = self.index.initialized()[bucket];
            if ordinal == 0 {
                self.provenance
                    .try_push(facts)
                    .map_err(|_| ObjectArrayBuildError::Capacity)?;
                self.index.initialized_mut()[bucket] = self.provenance.initialized().len();
                return Ok(true);
            }
            if Rc::ptr_eq(&self.provenance.initialized()[ordinal - 1], &facts) {
                return Ok(false);
            }
            bucket = bucket.wrapping_add(1) & mask;
        }
        Err(ObjectArrayBuildError::Capacity)
    }
    pub fn with_unselected_provenance(mut self) -> Self {
        if self.data.expected_len() == 0 {
            self.unselected = true;
        }
        self
    }
    pub fn finish(self) -> Result<ObjectArray, ObjectArrayBuildError> {
        let data = self
            .data
            .try_finish()
            .map_err(|_| ObjectArrayBuildError::Incomplete)?;
        Ok(ObjectArray {
            data,
            shape: self.shape,
            class_name: self.class_name,
            selected_provenance: self.provenance.finish_initialized(),
            unselected_provenance: self.unselected,
        })
    }
}
fn index_capacity(count: usize) -> Result<usize, ExactHostNumericAllocationError> {
    if count == 0 {
        return Ok(0);
    }
    count
        .checked_mul(2)
        .and_then(usize::checked_next_power_of_two)
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
}
fn validate_shape(count: usize, shape: &[usize]) -> Result<(), ObjectArrayBuildError> {
    if shape.len() < 2 {
        return Err(ObjectArrayBuildError::InvalidShape);
    }
    let expected = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
    if expected != count {
        return Err(ObjectArrayBuildError::ShapeCountMismatch);
    }
    Ok(())
}
