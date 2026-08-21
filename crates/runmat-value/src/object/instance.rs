use super::DynamicPropertyDef;
use crate::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectInstance {
    pub class_name: String,
    pub properties: HashMap<String, Value>,
    pub dynamic_properties: Option<Box<HashMap<String, DynamicPropertyDef>>>,
}

impl ObjectInstance {
    pub fn new(class_name: String) -> Self {
        Self {
            class_name,
            properties: HashMap::new(),
            dynamic_properties: None,
        }
    }

    pub fn is_class(&self, name: &str) -> bool {
        self.class_name == name
    }

    pub fn dynamic_property(&self, name: &str) -> Option<&DynamicPropertyDef> {
        self.dynamic_properties
            .as_ref()
            .and_then(|properties| properties.get(name))
    }

    pub fn dynamic_property_mut(&mut self, name: &str) -> Option<&mut DynamicPropertyDef> {
        self.dynamic_properties
            .as_mut()
            .and_then(|properties| properties.get_mut(name))
    }

    pub fn has_dynamic_property(&self, name: &str) -> bool {
        self.dynamic_property(name).is_some()
    }

    pub fn insert_dynamic_property(
        &mut self,
        name: String,
        property: DynamicPropertyDef,
    ) -> Option<DynamicPropertyDef> {
        self.dynamic_properties
            .get_or_insert_with(|| Box::new(HashMap::new()))
            .insert(name, property)
    }

    pub fn remove_dynamic_property(&mut self, name: &str) -> Option<DynamicPropertyDef> {
        let properties = self.dynamic_properties.as_mut()?;
        let removed = properties.remove(name);
        if properties.is_empty() {
            self.dynamic_properties = None;
        }
        removed
    }

    pub fn dynamic_property_names(&self) -> Vec<String> {
        self.dynamic_properties
            .as_ref()
            .map(|properties| properties.keys().cloned().collect())
            .unwrap_or_default()
    }
}
