//! Physical vectors with heterogeneous component dimensions. The layout is kept
//! once beside numeric storage; translations and rotations are never assigned a
//! single invented vector unit.
use super::{DynamicDimension, Quantity, UnitError};

pub trait QuantityLayout {
    fn component_count(&self) -> usize;
    fn component_descriptor(&self, index: usize) -> Option<DynamicDimension>;
}
impl<L: QuantityLayout + ?Sized> QuantityLayout for &L {
    fn component_count(&self) -> usize {
        (**self).component_count()
    }
    fn component_descriptor(&self, index: usize) -> Option<DynamicDimension> {
        (**self).component_descriptor(index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantityField<L, S> {
    layout: L,
    storage: S,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantityFieldError {
    ShapeMismatch,
    MissingComponentDescriptor,
    InvalidValue(UnitError),
}
impl<L: QuantityLayout, S: AsRef<[f64]>> QuantityField<L, S> {
    pub fn try_new(layout: L, storage: S) -> Result<Self, QuantityFieldError> {
        let values = storage.as_ref();
        if values.len() != layout.component_count() {
            return Err(QuantityFieldError::ShapeMismatch);
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err(QuantityFieldError::InvalidValue(UnitError::NonFiniteValue));
        }
        for index in 0..values.len() {
            if layout.component_descriptor(index).is_none() {
                return Err(QuantityFieldError::MissingComponentDescriptor);
            }
        }
        Ok(Self { layout, storage })
    }
    /// Consumes a checked field at an explicitly reviewed numerical/storage
    /// boundary without allocating a second copy of its buffer.
    pub fn into_canonical_storage(self) -> S {
        self.storage
    }
    pub fn layout(&self) -> &L {
        &self.layout
    }
    pub fn len(&self) -> usize {
        self.storage.as_ref().len()
    }
    pub fn is_empty(&self) -> bool {
        self.storage.as_ref().is_empty()
    }
    pub fn component(&self, index: usize) -> Option<Quantity<DynamicDimension>> {
        Quantity::try_from_descriptor(
            self.layout.component_descriptor(index)?,
            *self.storage.as_ref().get(index)?,
        )
        .ok()
    }
    /// Scoped extraction for an explicitly reviewed numerical algorithm. The
    /// typed layout stays attached to this field before and after the closure.
    pub fn with_canonical_storage<T>(&self, numerical: impl FnOnce(&[f64]) -> T) -> T {
        numerical(self.storage.as_ref())
    }
}
