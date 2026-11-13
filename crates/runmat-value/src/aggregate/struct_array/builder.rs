//! Explicit exact destination for a canonical structure-array generation.
use super::StructArray;
use crate::{
    ExactHostAllocationPermit, ExactHostNumericAllocationError, ExactOrderedMap,
    ExactOrderedMapBuilder, MapInsertError, OutputListStorage, SharedShape, Utf8Text,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructArrayQuote {
    pub requested_bytes: usize,
    pub retained_claims: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructArrayConstructionError {
    InvalidShape,
    UnadmittedShape,
    Allocation(ExactHostNumericAllocationError),
}
impl From<ExactHostNumericAllocationError> for StructArrayConstructionError {
    fn from(source: ExactHostNumericAllocationError) -> Self {
        Self::Allocation(source)
    }
}
impl std::fmt::Display for StructArrayConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape => f.write_str("invalid nonscalar structure-array shape"),
            Self::UnadmittedShape => f.write_str("structure-array shape has no retained claim"),
            Self::Allocation(source) => source.fmt(f),
        }
    }
}
impl std::error::Error for StructArrayConstructionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(source) => Some(source),
            _ => None,
        }
    }
}
pub(super) fn checked_length(shape: &[usize]) -> Result<usize, StructArrayConstructionError> {
    if shape.len() < 2 {
        return Err(StructArrayConstructionError::InvalidShape);
    }
    let length = shape
        .iter()
        .try_fold(1usize, |length, extent| length.checked_mul(*extent))
        .ok_or(StructArrayConstructionError::Allocation(
            ExactHostNumericAllocationError::LayoutOverflow,
        ))?;
    if length == 1 {
        return Err(StructArrayConstructionError::InvalidShape);
    }
    Ok(length)
}

/// The caller quotes keys separately. Child payloads retain their own owners;
/// this destination funds only map/index, value slots and shape metadata.
pub struct StructArrayBuilder {
    fields: ExactOrderedMapBuilder<OutputListStorage>,
    shape: SharedShape,
    length: usize,
}

/// Ordinary-policy construction retaining already-owned canonical columns.
/// This mutable producer is never a published StructArray facade.
pub struct UnboundedStructArrayBuilder {
    fields: crate::UnboundedOrderedMapBuilder<OutputListStorage>,
    shape: SharedShape,
    length: usize,
}
impl UnboundedStructArrayBuilder {
    pub fn new(shape: SharedShape) -> Result<Self, StructArrayConstructionError> {
        Self::with_capacity(shape, 0)
    }
    pub fn with_capacity(
        shape: SharedShape,
        field_count: usize,
    ) -> Result<Self, StructArrayConstructionError> {
        let length = checked_length(&shape)?;
        Ok(Self {
            fields: crate::UnboundedOrderedMapBuilder::with_capacity(field_count),
            shape,
            length,
        })
    }
    pub fn insert(
        &mut self,
        key: String,
        values: OutputListStorage,
    ) -> Result<Option<OutputListStorage>, StructArrayFieldError> {
        if values.len() != self.length {
            return Err(StructArrayFieldError::InvalidColumn {
                key: key.into(),
                values,
            });
        }
        Ok(self.fields.insert(key, values))
    }
    pub fn finish(self) -> StructArray {
        StructArray::from_shared_columns(self.fields.finish(), self.shape)
    }
}

#[derive(Debug)]
pub enum StructArrayFieldError {
    InvalidColumn {
        key: Utf8Text,
        values: OutputListStorage,
    },
    Map(MapInsertError<OutputListStorage>),
}
impl std::fmt::Display for StructArrayFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidColumn { .. } => {
                f.write_str("structure-array column shape or backing is invalid")
            }
            Self::Map(source) => write!(
                f,
                "structure-array column insertion failed: {:?}",
                source.kind
            ),
        }
    }
}
impl std::error::Error for StructArrayFieldError {}

impl StructArrayBuilder {
    pub fn quote(
        field_count: usize,
        shape: &[usize],
    ) -> Result<StructArrayQuote, StructArrayConstructionError> {
        let length = checked_length(shape)?;
        let map = ExactOrderedMap::<OutputListStorage>::quote(field_count)?;
        let columns = OutputListStorage::requested_bytes(length)?
            .checked_mul(field_count)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let requested_bytes = map
            .requested_bytes
            .checked_add(columns)
            .and_then(|bytes| {
                bytes.checked_add(SharedShape::requested_bytes_for_rank(shape.len()).ok()?)
            })
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let retained_claims = map
            .retained_claims
            .checked_add(
                usize::from(length != 0)
                    .checked_mul(field_count)
                    .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?,
            )
            .and_then(|claims| claims.checked_add(1))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Ok(StructArrayQuote {
            requested_bytes,
            retained_claims,
        })
    }
    pub fn try_new(
        field_count: usize,
        shape: &[usize],
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, StructArrayConstructionError> {
        let length = checked_length(shape)?;
        let fields = ExactOrderedMapBuilder::try_new(field_count, permit)?;
        let shape = SharedShape::new_admitted_from_slice(shape, permit)?;
        Ok(Self {
            fields,
            shape,
            length,
        })
    }
    /// Retain a separately admitted existing shape without reallocating it.
    pub fn try_new_retaining_shape(
        field_count: usize,
        shape: SharedShape,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, StructArrayConstructionError> {
        let length = checked_length(&shape)?;
        if !shape.is_admitted() {
            return Err(StructArrayConstructionError::UnadmittedShape);
        }
        Ok(Self {
            fields: ExactOrderedMapBuilder::try_new(field_count, permit)?,
            shape,
            length,
        })
    }
    pub fn try_insert_prepared(
        &mut self,
        key: Utf8Text,
        values: OutputListStorage,
    ) -> Result<Option<OutputListStorage>, StructArrayFieldError> {
        if values.len() != self.length || (values.capacity() != 0 && !values.is_exact()) {
            return Err(StructArrayFieldError::InvalidColumn { key, values });
        }
        self.fields
            .try_insert_prepared(key, values)
            .map_err(StructArrayFieldError::Map)
    }
    pub fn finish(self) -> StructArray {
        StructArray::from_shared_columns(self.fields.finish(), self.shape)
    }
}
