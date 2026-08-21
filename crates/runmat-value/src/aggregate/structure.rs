use super::*;
use indexmap::IndexMap;

#[derive(Debug, Clone, PartialEq)]
pub struct StructValue {
    pub fields: IndexMap<String, Value>,
}

impl StructValue {
    pub fn new() -> Self {
        Self {
            fields: IndexMap::new(),
        }
    }

    /// Insert a field, preserving insertion order when the name is new.
    pub fn insert(&mut self, name: impl Into<String>, value: Value) -> Option<Value> {
        self.fields.insert(name.into(), value)
    }

    /// Remove a field while preserving the relative order of remaining fields.
    pub fn remove(&mut self, name: &str) -> Option<Value> {
        self.fields.shift_remove(name)
    }

    /// Returns an iterator over field names in their stored order.
    pub fn field_names(&self) -> impl Iterator<Item = &String> {
        self.fields.keys()
    }
}

impl Default for StructValue {
    fn default() -> Self {
        Self::new()
    }
}
