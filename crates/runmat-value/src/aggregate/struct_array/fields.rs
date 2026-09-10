use super::StructArray;
use crate::Value;
use indexmap::IndexMap;
use std::collections::HashSet;

impl StructArray {
    pub fn insert_field(&mut self, name: String, values: Vec<Value>) -> Result<(), String> {
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
        self.fields.insert(name, values);
        Ok(())
    }

    pub fn remove_field(&mut self, name: &str) -> Option<Vec<Value>> {
        self.fields.shift_remove(name)
    }

    pub fn replace_field_values(
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
            .get_mut(name)
            .ok_or_else(|| format!("structure array field '{name}' does not exist"))?;
        for (old, new) in column.iter().zip(&values) {
            before_replace(old, new);
        }
        *column = values;
        Ok(())
    }

    pub fn reorder_fields(&mut self, order: &[String]) -> Result<(), String> {
        let unique = order.iter().collect::<HashSet<_>>();
        if order.len() != self.fields.len()
            || unique.len() != order.len()
            || order.iter().any(|name| !self.fields.contains_key(name))
        {
            return Err("structure array field order must contain its exact field set".into());
        }
        let mut reordered = IndexMap::with_capacity(order.len());
        for name in order {
            let values = self
                .fields
                .shift_remove(name)
                .ok_or_else(|| "structure array field order is invalid".to_string())?;
            reordered.insert(name.clone(), values);
        }
        self.fields = reordered;
        Ok(())
    }
}
