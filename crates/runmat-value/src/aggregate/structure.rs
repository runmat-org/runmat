use crate::{ExactOrderedMap, Value};

mod builder;
pub use builder::{StructFieldInsertError, StructValueBuilder, UnboundedStructValueBuilder};

/// Scalar fields share their original immutable entry/index generation.
#[derive(Debug, Clone, PartialEq)]
pub struct StructValue {
    pub fields: ExactOrderedMap<Value>,
}

impl StructValue {
    pub fn new() -> Self {
        Self {
            fields: ExactOrderedMap::empty(),
        }
    }
    pub(crate) fn from_owned_fields(fields: ExactOrderedMap<Value>) -> Self {
        Self { fields }
    }
    pub fn field_names(&self) -> impl ExactSizeIterator<Item = &str> + Clone {
        self.fields.keys()
    }
    /// Explicit ordinary-policy construction with one final publication.
    pub fn from_fields_unbounded(fields: impl IntoIterator<Item = (String, Value)>) -> Self {
        let fields = fields.into_iter();
        let mut builder = UnboundedStructValueBuilder::with_capacity(fields.size_hint().0);
        for (name, value) in fields {
            builder.insert(name, value);
        }
        builder.finish()
    }
    pub fn try_from_fields_unbounded<E>(
        fields: impl IntoIterator<Item = Result<(String, Value), E>>,
    ) -> Result<Self, E> {
        let fields = fields.into_iter();
        let mut builder = UnboundedStructValueBuilder::with_capacity(fields.size_hint().0);
        for field in fields {
            let (name, value) = field?;
            builder.insert(name, value);
        }
        Ok(builder.finish())
    }
    /// Ordinary-policy traversal only. Aliases may copy children here;
    /// finite consumers use fields.try_into_entries or admitted construction.
    pub fn into_fields_unbounded(self) -> impl ExactSizeIterator<Item = (String, Value)> {
        self.fields
            .into_entries_unbounded()
            .map(|(name, value)| (name.into_string_unbounded(), value))
    }
    /// Existing-field update cannot grow or copy a generation implicitly.
    pub fn try_replace_unique(&mut self, name: &str, value: Value) -> Result<Value, Value> {
        self.fields.try_replace_unique(name, value)
    }
}
impl Default for StructValue {
    fn default() -> Self {
        Self::new()
    }
}
