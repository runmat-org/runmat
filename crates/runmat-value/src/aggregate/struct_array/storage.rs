//! Canonical immutable column ownership and explicit ordinary staging.
use super::{total_len, StructArray};
use crate::{ExactOrderedMap, OutputListStorage, SharedShape, UnboundedOrderedMapBuilder, Value};

impl StructArray {
    /// Publish already-owned immutable metadata without copying columns or
    /// shape. This validates representation, never grants finite authority.
    pub fn try_from_owned_columns(
        fields: ExactOrderedMap<OutputListStorage>,
        shape: SharedShape,
    ) -> Result<Self, super::StructArrayConstructionError> {
        let length = super::builder::checked_length(&shape)?;
        if fields
            .values()
            .any(|values| values.len() != length || !values.is_shallow_clone())
        {
            return Err(super::StructArrayConstructionError::InvalidShape);
        }
        Ok(Self { fields, shape })
    }
    pub(crate) fn publish_columns_unbounded(
        fields: impl IntoIterator<Item = (String, Vec<Value>)>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let length = total_len(&shape)?;
        let fields = fields.into_iter();
        let mut builder = UnboundedOrderedMapBuilder::with_capacity(fields.size_hint().0);
        for (name, values) in fields {
            if values.len() != length {
                return Err("structure array fields must contain one value per element".into());
            }
            let values = if values.capacity() == 0 {
                OutputListStorage::empty()
            } else {
                OutputListStorage::shared_unbounded(values)
            };
            builder.insert(name, values);
        }
        Ok(Self {
            fields: builder.finish(),
            shape: SharedShape::new_unmanaged(shape),
        })
    }

    pub(crate) fn into_columns_unbounded(
        self,
    ) -> impl ExactSizeIterator<Item = (String, Vec<Value>)> {
        self.fields.into_entries_unbounded().map(|(name, values)| {
            let values = match values.try_into_owned() {
                Ok(values) => match values.try_into_unbounded_vec() {
                    Ok(values) => values,
                    Err(values) => values.as_slice().to_vec(),
                },
                Err(values) => values.as_slice().to_vec(),
            };
            (name.into_string_unbounded(), values)
        })
    }

    /// Existing slots only: no allocation, no key/column structural edits,
    /// and no implicit COW. Shared columns require an admitted replacement.
    pub fn try_for_each_value_mut_unique<E>(
        &mut self,
        mut visitor: impl FnMut(&mut Value) -> Result<(), E>,
    ) -> Option<Result<(), E>> {
        if !self.fields.is_unique() || self.fields.values().any(|column| !column.is_unique()) {
            return None;
        }
        for column in self
            .fields
            .try_values_mut_unique()
            .expect("unique columns map")
        {
            for value in column.try_as_mut_slice_unique().expect("unique column") {
                if let Err(error) = visitor(value) {
                    return Some(Err(error));
                }
            }
        }
        Some(Ok(()))
    }

    /// Existing column slots only. Both canonical owners must be unique;
    /// this never grows backing, changes keys, or performs implicit COW.
    pub fn try_column_mut_unique(&mut self, name: &str) -> Option<&mut [Value]> {
        self.fields
            .try_get_mut_unique(name)?
            .try_as_mut_slice_unique()
    }

    pub(crate) fn from_shared_columns(
        fields: ExactOrderedMap<OutputListStorage>,
        shape: SharedShape,
    ) -> Self {
        Self { fields, shape }
    }
}
