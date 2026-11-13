use super::StructArray;
use crate::Value;
use std::future::Future;

impl StructArray {
    pub(crate) fn trace_field_values(&self, index: usize) -> Option<&[Value]> {
        self.fields
            .get_index(index)
            .map(|(_, values)| values.as_slice())
    }
    pub fn for_each_value(&self, mut visitor: impl FnMut(&Value)) {
        self.fields
            .values()
            .flat_map(|values| values.iter())
            .for_each(&mut visitor);
    }
    pub fn any_value(&self, mut predicate: impl FnMut(&Value) -> bool) -> bool {
        self.fields
            .values()
            .flat_map(|values| values.iter())
            .any(&mut predicate)
    }
    /// Explicit ordinary replacement; published aliases are never mutated.
    pub fn for_each_value_mut_unbounded(&mut self, mut visitor: impl FnMut(&mut Value)) {
        let shape = self.shape.to_vec();
        let mut fields = self.clone().into_columns_unbounded().collect::<Vec<_>>();
        for (_, values) in &mut fields {
            values.iter_mut().for_each(&mut visitor);
        }
        *self = Self::publish_columns_unbounded(fields, shape).expect("validated original schema");
    }
    pub fn try_for_each_indexed_value_mut_unbounded<E>(
        &mut self,
        mut visitor: impl FnMut(&str, usize, &mut Value) -> Result<(), E>,
    ) -> Result<(), E> {
        let shape = self.shape.to_vec();
        let mut fields = self.clone().into_columns_unbounded().collect::<Vec<_>>();
        for (name, values) in &mut fields {
            for (index, value) in values.iter_mut().enumerate() {
                visitor(name, index, value)?;
            }
        }
        *self = Self::publish_columns_unbounded(fields, shape).expect("validated original schema");
        Ok(())
    }
    pub fn try_map_values_unbounded<E>(
        self,
        mut transform: impl FnMut(Value) -> Result<Value, E>,
    ) -> Result<Self, E> {
        let shape = self.shape.to_vec();
        let fields = self
            .into_columns_unbounded()
            .map(|(name, values)| {
                values
                    .into_iter()
                    .map(&mut transform)
                    .collect::<Result<Vec<_>, _>>()
                    .map(|values| (name, values))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::publish_columns_unbounded(fields, shape)
            .expect("transformation preserves field lengths"))
    }
    pub async fn try_map_values_async_unbounded<E, F, Fut>(
        self,
        mut transform: F,
    ) -> Result<Self, E>
    where
        F: FnMut(Value) -> Fut,
        Fut: Future<Output = Result<Value, E>>,
    {
        let shape = self.shape.to_vec();
        let mut fields = Vec::with_capacity(self.fields.len());
        for (name, values) in self.into_columns_unbounded() {
            let mut mapped = Vec::with_capacity(values.len());
            for value in values {
                mapped.push(transform(value).await?);
            }
            fields.push((name, mapped));
        }
        Ok(Self::publish_columns_unbounded(fields, shape)
            .expect("transformation preserves field lengths"))
    }
}
