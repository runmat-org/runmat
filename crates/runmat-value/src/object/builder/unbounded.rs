use super::super::{
    DynamicPropertyDef, NativeObjectPayload, ObjectInstance, ObjectPayload,
    SelectedObjectClassFacts, SharedObjectPayload,
};
use crate::SharedClassIdentity;
use crate::{UnboundedOrderedMapBuilder, Utf8Text, Value};

#[derive(Debug)]
pub struct UnboundedObjectBuilder {
    class_name: SharedClassIdentity,
    properties: UnboundedOrderedMapBuilder<Value>,
    dynamic_properties: UnboundedOrderedMapBuilder<DynamicPropertyDef>,
    selected_class: Option<std::rc::Rc<SelectedObjectClassFacts>>,
    native_payload: Option<NativeObjectPayload>,
}
impl UnboundedObjectBuilder {
    pub fn new(class_name: impl Into<SharedClassIdentity>) -> Self {
        Self::with_property_capacity(class_name, 0)
    }
    pub fn with_property_capacity(
        class_name: impl Into<SharedClassIdentity>,
        property_capacity: usize,
    ) -> Self {
        Self {
            class_name: class_name.into(),
            properties: UnboundedOrderedMapBuilder::with_capacity(property_capacity),
            dynamic_properties: UnboundedOrderedMapBuilder::with_capacity(0),
            selected_class: None,
            native_payload: None,
        }
    }
    pub fn insert(&mut self, name: String, value: Value) -> Option<Value> {
        self.properties.insert(name, value)
    }
    pub fn property(&self, name: &str) -> Option<&Value> {
        self.properties.get(name)
    }
    pub fn property_mut(&mut self, name: &str) -> Option<&mut Value> {
        self.properties.get_mut(name)
    }
    pub fn insert_text(&mut self, name: Utf8Text, value: Value) -> Option<Value> {
        self.properties.insert_text(name, value)
    }
    pub fn remove(&mut self, name: &str) -> Option<Value> {
        self.properties.remove(name)
    }
    pub fn insert_dynamic(
        &mut self,
        name: String,
        definition: DynamicPropertyDef,
    ) -> Option<DynamicPropertyDef> {
        self.dynamic_properties.insert(name, definition)
    }
    pub fn dynamic_property_mut(&mut self, name: &str) -> Option<&mut DynamicPropertyDef> {
        self.dynamic_properties.get_mut(name)
    }
    pub fn remove_dynamic(&mut self, name: &str) -> Option<DynamicPropertyDef> {
        self.dynamic_properties.remove(name)
    }
    pub fn replace_memmap_receiver(&mut self, value: super::super::MemmapReceiverState) {
        self.native_payload = Some(NativeObjectPayload::memmap_receiver_unbounded(value));
    }
    pub fn replace_numeric_factor_data(&mut self, value: super::super::NumericFactorData) {
        self.native_payload = Some(NativeObjectPayload::numeric_factor_unbounded(value));
    }
    /// Consume an explicitly unbounded object generation as a property
    /// sequence. Class and object-only metadata are retired with the builder;
    /// child values move exactly once into the caller's replacement owner.
    pub fn into_properties_unbounded(self) -> impl ExactSizeIterator<Item = (String, Value)> {
        self.properties
            .finish()
            .into_entries_unbounded()
            .map(|(name, value)| (name.into_string_unbounded(), value))
    }
    pub fn finish(self) -> ObjectInstance {
        ObjectInstance::from_payload(SharedObjectPayload::unbounded(ObjectPayload {
            class_name: self.class_name,
            properties: self.properties.finish(),
            dynamic_properties: self.dynamic_properties.finish(),
            selected_class: self.selected_class,
            native_payload: self.native_payload,
            payload_claim: None,
        }))
    }
    pub(in crate::object) fn from_payload(payload: ObjectPayload) -> Self {
        debug_assert!(
            !payload.class_name.has_retained_claim()
                && payload.payload_claim.is_none()
                && payload.properties.is_unique()
                && payload.dynamic_properties.is_unique(),
            "caller must retain the original object until unbounded edit eligibility is proven"
        );
        let ObjectPayload {
            class_name,
            properties,
            dynamic_properties,
            selected_class,
            native_payload,
            payload_claim: None,
        } = payload
        else {
            unreachable!("unbounded edit eligibility was checked")
        };
        let properties = properties
            .try_into_entries()
            .expect("unique property map was checked");
        let dynamic_properties = dynamic_properties
            .try_into_entries()
            .expect("unique metadata map was checked");
        let mut property_builder = UnboundedOrderedMapBuilder::with_capacity(properties.len());
        for (key, value) in properties {
            property_builder.insert_text(key, value);
        }
        let mut dynamic_builder =
            UnboundedOrderedMapBuilder::with_capacity(dynamic_properties.len());
        for (key, value) in dynamic_properties {
            dynamic_builder.insert_text(key, value);
        }
        Self {
            class_name,
            properties: property_builder,
            dynamic_properties: dynamic_builder,
            selected_class,
            native_payload,
        }
    }

    pub(in crate::object) fn copy_from_payload(payload: &ObjectPayload) -> Self {
        let mut properties = UnboundedOrderedMapBuilder::with_capacity(payload.properties.len());
        for entry in payload.properties.entries() {
            properties.insert_text(entry.key().clone(), entry.value().clone());
        }
        let mut dynamic_properties =
            UnboundedOrderedMapBuilder::with_capacity(payload.dynamic_properties.len());
        for entry in payload.dynamic_properties.entries() {
            dynamic_properties.insert_text(entry.key().clone(), entry.value().clone());
        }
        Self {
            class_name: payload.class_name.clone(),
            properties,
            dynamic_properties,
            selected_class: payload.selected_class.clone(),
            native_payload: payload
                .native_payload
                .as_ref()
                .map(NativeObjectPayload::clone_unbounded),
        }
    }
}
