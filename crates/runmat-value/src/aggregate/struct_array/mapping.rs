use super::StructArray;
use crate::Value;
use std::future::Future;

impl StructArray {
    pub fn for_each_value(&self, mut visitor: impl FnMut(&Value)) {
        self.fields.values().flatten().for_each(&mut visitor);
    }

    pub fn for_each_value_mut(&mut self, mut visitor: impl FnMut(&mut Value)) {
        self.fields.values_mut().flatten().for_each(&mut visitor);
    }

    pub fn any_value(&self, mut predicate: impl FnMut(&Value) -> bool) -> bool {
        self.fields.values().flatten().any(&mut predicate)
    }

    /// Visits every value with its field name and column-major linear index.
    pub fn try_for_each_indexed_value_mut<E>(
        &mut self,
        mut visitor: impl FnMut(&str, usize, &mut Value) -> Result<(), E>,
    ) -> Result<(), E> {
        for (name, values) in &mut self.fields {
            for (index, value) in values.iter_mut().enumerate() {
                visitor(name, index, value)?;
            }
        }
        Ok(())
    }

    /// Applies a fallible owned transformation to every field value while
    /// preserving array shape and the exact ordered field schema.
    pub fn try_map_values<E>(
        self,
        mut transform: impl FnMut(Value) -> Result<Value, E>,
    ) -> Result<Self, E> {
        let fields = self
            .fields
            .into_iter()
            .map(|(name, values)| {
                values
                    .into_iter()
                    .map(&mut transform)
                    .collect::<Result<Vec<_>, _>>()
                    .map(|values| (name, values))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            fields,
            shape: self.shape,
        })
    }

    pub async fn try_map_values_async<E, F, Fut>(self, mut transform: F) -> Result<Self, E>
    where
        F: FnMut(Value) -> Fut,
        Fut: Future<Output = Result<Value, E>>,
    {
        let mut fields = indexmap::IndexMap::with_capacity(self.fields.len());
        for (name, values) in self.fields {
            let mut mapped = Vec::with_capacity(values.len());
            for value in values {
                mapped.push(transform(value).await?);
            }
            fields.insert(name, mapped);
        }
        Ok(Self {
            fields,
            shape: self.shape,
        })
    }
}
