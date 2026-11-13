//! Move-only producers of one canonical immutable scalar structure.
use crate::{
    ExactHostAllocationPermit, ExactHostNumericAllocationError, ExactOrderedMapBuilder,
    MapInsertError, StructValue, UnboundedOrderedMapBuilder, Utf8Text, Value,
};

#[derive(Debug)]
pub struct StructValueBuilder {
    fields: ExactOrderedMapBuilder<Value>,
}

#[derive(Debug)]
pub enum StructFieldInsertError {
    KeyAllocation {
        source: ExactHostNumericAllocationError,
        value: Value,
    },
    Map(MapInsertError<Value>),
}
impl std::fmt::Display for StructFieldInsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyAllocation { source, .. } => {
                write!(f, "structure key allocation failed: {source}")
            }
            Self::Map(error) => write!(f, "structure insertion failed: {:?}", error.kind),
        }
    }
}
impl std::error::Error for StructFieldInsertError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::KeyAllocation { source, .. } => Some(source),
            Self::Map(_) => None,
        }
    }
}
impl StructValueBuilder {
    pub fn try_new(
        capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        Ok(Self {
            fields: ExactOrderedMapBuilder::try_new(capacity, permit)?,
        })
    }
    pub fn len(&self) -> usize {
        self.fields.len()
    }
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
    pub fn try_insert_prepared(
        &mut self,
        key: Utf8Text,
        value: Value,
    ) -> Result<Option<Value>, MapInsertError<Value>> {
        self.fields.try_insert_prepared(key, value)
    }
    pub fn try_insert(
        &mut self,
        name: &str,
        value: Value,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Option<Value>, StructFieldInsertError> {
        let key = match Utf8Text::try_copy_str_exact_with(name, permit, |_| {
            Ok::<(), std::convert::Infallible>(())
        }) {
            Ok(key) => key,
            Err(crate::ExactArrayInitializationError::Allocation(source)) => {
                return Err(StructFieldInsertError::KeyAllocation { source, value })
            }
            Err(crate::ExactArrayInitializationError::Element(never)) => match never {},
        };
        self.try_insert_prepared(key, value)
            .map_err(StructFieldInsertError::Map)
    }
    pub fn remove(&mut self, name: &str) -> Option<Value> {
        self.fields.remove(name)
    }
    pub fn finish(self) -> StructValue {
        StructValue::from_owned_fields(self.fields.finish())
    }
}

/// Only explicitly selected ordinary/unbounded producers use this builder.
/// Its mutable fields are unpublished state, never a StructValue facade.
#[derive(Debug)]
pub struct UnboundedStructValueBuilder {
    pub fields: UnboundedOrderedMapBuilder<Value>,
}
impl UnboundedStructValueBuilder {
    /// Explicit ordinary-policy replacement generation. A shared source may
    /// clone children; callers under finite policy must stage an admitted
    /// replacement instead of selecting this constructor.
    pub fn from_structure(source: StructValue) -> Self {
        let mut builder = Self::with_capacity(source.fields.len());
        for (name, value) in source.fields.into_entries_unbounded() {
            builder.insert_text(name, value);
        }
        builder
    }
    pub fn new() -> Self {
        Self::with_capacity(0)
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            fields: UnboundedOrderedMapBuilder::with_capacity(capacity),
        }
    }
    pub fn len(&self) -> usize {
        self.fields.len()
    }
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }
    pub fn insert(&mut self, name: impl Into<String>, value: Value) -> Option<Value> {
        self.fields.insert(name.into(), value)
    }
    pub fn insert_text(&mut self, name: Utf8Text, value: Value) -> Option<Value> {
        self.fields.insert_text(name, value)
    }
    pub fn remove(&mut self, name: &str) -> Option<Value> {
        self.fields.remove(name)
    }
    pub fn finish(self) -> StructValue {
        StructValue::from_owned_fields(self.fields.finish())
    }
}
impl Default for UnboundedStructValueBuilder {
    fn default() -> Self {
        Self::new()
    }
}
