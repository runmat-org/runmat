use super::StructArray;
use crate::{UnboundedOrderedMapBuilder, Value};
use std::collections::HashSet;

impl StructArray {
    pub fn insert_field_unbounded(
        &mut self,
        name: String,
        values: Vec<Value>,
    ) -> Result<(), String> {
        if self.fields.contains_key(&name) {
            return Err(format!("structure array field '{name}' already exists"));
        }
        if values.len() != self.len() {
            return Err(format!(
                "structure array field has {} values but {} were required",
                values.len(),
                self.len()
            ));
        }
        let shape = self.shape.to_vec();
        let mut fields = self
            .clone()
            .into_columns_unbounded()
            .collect::<UnboundedOrderedMapBuilder<_>>();
        fields.insert(name, values);
        *self = Self::publish_columns_unbounded(fields, shape)?;
        Ok(())
    }
    pub fn remove_field_unbounded(&mut self, name: &str) -> Option<Vec<Value>> {
        if !self.fields.contains_key(name) {
            return None;
        }
        let shape = self.shape.to_vec();
        let mut fields = self
            .clone()
            .into_columns_unbounded()
            .collect::<UnboundedOrderedMapBuilder<_>>();
        let removed = fields.remove(name);
        *self = Self::publish_columns_unbounded(fields, shape)
            .expect("removal preserves field lengths");
        removed
    }
    pub fn replace_field_values_unbounded(
        &mut self,
        name: &str,
        values: Vec<Value>,
        mut before_replace: impl FnMut(&Value, &Value),
    ) -> Result<(), String> {
        if values.len() != self.len() {
            return Err("structure array field assignment count does not match array".into());
        }
        let column = self
            .fields
            .get(name)
            .ok_or_else(|| format!("structure array field '{name}' does not exist"))?;
        for (old, new) in column.iter().zip(&values) {
            before_replace(old, new);
        }
        let shape = self.shape.to_vec();
        let mut fields = self
            .clone()
            .into_columns_unbounded()
            .collect::<UnboundedOrderedMapBuilder<_>>();
        fields.insert(name.to_owned(), values);
        *self = Self::publish_columns_unbounded(fields, shape)?;
        Ok(())
    }
    pub fn reorder_fields_unbounded(&mut self, order: &[String]) -> Result<(), String> {
        let unique = order.iter().map(String::as_str).collect::<HashSet<_>>();
        if order.len() != self.fields.len()
            || unique.len() != order.len()
            || order.iter().any(|name| !self.fields.contains_key(name))
        {
            return Err("structure array field order must contain its exact field set".into());
        }
        let shape = self.shape.to_vec();
        let mut fields = self
            .clone()
            .into_columns_unbounded()
            .collect::<UnboundedOrderedMapBuilder<_>>();
        let reordered = order
            .iter()
            .map(|name| {
                (
                    name.clone(),
                    fields.remove(name).expect("validated field set"),
                )
            })
            .collect::<Vec<_>>();
        *self = Self::publish_columns_unbounded(reordered, shape)?;
        Ok(())
    }
}
