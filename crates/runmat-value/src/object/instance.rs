use super::{
    DynamicPropertyDef, ObjectPayload, SelectedObjectClassFacts, SharedObjectPayload,
    UnboundedObjectBuilder,
};
use crate::{ExactHostAllocationPermit, ExactHostNumericAllocationError, ExactOrderedMap, Value};
use runmat_types::ClassIdentity;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct ObjectInstance {
    payload: SharedObjectPayload,
}

impl PartialEq for ObjectInstance {
    fn eq(&self, other: &Self) -> bool {
        self.payload == other.payload
    }
}

impl ObjectInstance {
    pub fn new(class_name: impl Into<crate::SharedClassIdentity>) -> Self {
        Self::from_unbounded_parts(
            class_name.into(),
            ExactOrderedMap::empty(),
            ExactOrderedMap::empty(),
        )
    }

    pub(super) fn from_unbounded_parts(
        class_name: crate::SharedClassIdentity,
        properties: ExactOrderedMap<Value>,
        dynamic_properties: ExactOrderedMap<DynamicPropertyDef>,
    ) -> Self {
        Self::from_payload(SharedObjectPayload::unbounded(ObjectPayload {
            class_name,
            properties,
            dynamic_properties,
            selected_class: None,
            native_payload: None,
            payload_claim: None,
        }))
    }

    pub(super) fn from_payload(payload: SharedObjectPayload) -> Self {
        Self { payload }
    }

    pub(super) fn generation(&self) -> &ObjectPayload {
        self.payload.get()
    }

    pub(super) fn generation_mut_unique(&mut self) -> Option<&mut ObjectPayload> {
        self.payload.get_mut_unique()
    }

    pub fn class_name(&self) -> &ClassIdentity {
        self.generation().class_name.as_identity()
    }

    pub fn class_name_owner(&self) -> &crate::SharedClassIdentity {
        &self.generation().class_name
    }

    pub fn properties(&self) -> &ExactOrderedMap<Value> {
        &self.generation().properties
    }

    pub fn property(&self, name: &str) -> Option<&Value> {
        self.properties().get(name)
    }

    pub fn property_count(&self) -> usize {
        self.generation().properties.len()
    }

    pub fn dynamic_property_count(&self) -> usize {
        self.generation().dynamic_properties.len()
    }

    /// Physical metadata still needing claims before this generation can be
    /// copied into an independently admitted Object payload. Child Values are
    /// already canonical owners and are not included here.
    pub fn unclaimed_metadata_quote(
        &self,
    ) -> Result<super::ExactObjectQuote, ExactHostNumericAllocationError> {
        let generation = self.generation();
        let mut requested_bytes = 0usize;
        let mut retained_claims = 0usize;
        let mut add = |bytes: usize| -> Result<(), ExactHostNumericAllocationError> {
            requested_bytes = requested_bytes
                .checked_add(bytes)
                .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
            if bytes != 0 {
                retained_claims = retained_claims
                    .checked_add(1)
                    .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
            }
            Ok(())
        };
        if !generation.class_name.has_retained_claim() {
            add(generation.class_name.requested_bytes()?)?;
        }
        for entry in generation.properties.entries() {
            if !entry.key().has_retained_claim() {
                add(crate::Utf8Text::requested_bytes_for_capacity(
                    entry.key().capacity(),
                )?)?;
            }
        }
        for entry in generation.dynamic_properties.entries() {
            if !entry.key().has_retained_claim() {
                add(crate::Utf8Text::requested_bytes_for_capacity(
                    entry.key().capacity(),
                )?)?;
            }
            let definition = entry.value();
            if !definition.name.has_retained_claim() {
                add(crate::Utf8Text::requested_bytes_for_capacity(
                    definition.name.capacity(),
                )?)?;
            }
            if !definition.defining_class.has_retained_claim() {
                add(definition.defining_class.requested_bytes()?)?;
            }
            if !definition.description.has_retained_claim() {
                add(crate::Utf8Text::requested_bytes_for_capacity(
                    definition.description.capacity(),
                )?)?;
            }
        }
        let (native_bytes, native_claims) = self.native_unclaimed_quote()?;
        requested_bytes = requested_bytes
            .checked_add(native_bytes)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        retained_claims = retained_claims
            .checked_add(native_claims)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Ok(super::ExactObjectQuote {
            requested_bytes,
            retained_claims,
        })
    }

    /// Promote original immutable metadata owners in place. Aliases observe
    /// the same claims; no class/key/description bytes are copied.
    pub fn try_admit_metadata_exact(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactHostNumericAllocationError> {
        let generation = self.generation();
        generation
            .class_name
            .clone()
            .try_admit_exact(permit)
            .map_err(|error| error.source())?;
        for entry in generation.properties.entries() {
            entry
                .key()
                .clone()
                .try_admit_exact(permit)
                .map_err(|error| error.source())?;
        }
        for entry in generation.dynamic_properties.entries() {
            entry
                .key()
                .clone()
                .try_admit_exact(permit)
                .map_err(|error| error.source())?;
            entry
                .value()
                .clone()
                .try_admit_exact(permit)
                .map_err(|error| error.source())?;
        }
        self.try_admit_native_exact(permit)?;
        Ok(())
    }

    pub fn shares_generation_with(&self, other: &Self) -> bool {
        self.payload.shares_allocation_with(&other.payload)
    }

    pub fn permits_unbounded_edit(&self) -> bool {
        self.generation().can_copy_unbounded()
    }

    pub(crate) fn permits_exact_move_edit(&self) -> bool {
        self.payload.is_unique()
            && self.generation().properties.is_unique()
            && self.generation().dynamic_properties.is_unique()
    }

    pub(in crate::object) fn into_payload_unique(self) -> Result<ObjectPayload, Self> {
        self.payload.try_unwrap().map_err(Self::from_payload)
    }

    /// Begin an explicit structural edit without copying children. Shared or
    /// admitted generations are returned unchanged; their replacement needs
    /// a separately admitted destination.
    pub fn try_into_unbounded_builder(self) -> Result<UnboundedObjectBuilder, Self> {
        if !self.generation().can_begin_unbounded_edit() {
            return Err(self);
        }
        match self.payload.try_unwrap() {
            Ok(payload) => Ok(UnboundedObjectBuilder::from_payload(payload)),
            Err(payload) => Err(Self::from_payload(payload)),
        }
    }

    /// Explicit ordinary-policy edit. Shared ordinary generations are copied
    /// into a new unpublished generation; admitted generations are refused.
    pub fn into_unbounded_builder(self) -> Result<UnboundedObjectBuilder, Self> {
        if !self.generation().can_copy_unbounded() {
            return Err(self);
        }
        if self.generation().can_begin_unbounded_edit() {
            return match self.payload.try_unwrap() {
                Ok(payload) => Ok(UnboundedObjectBuilder::from_payload(payload)),
                Err(payload) => Ok(UnboundedObjectBuilder::copy_from_payload(payload.get())),
            };
        }
        let builder = UnboundedObjectBuilder::copy_from_payload(self.generation());
        Ok(builder)
    }

    pub fn dynamic_property_definitions(
        &self,
    ) -> impl ExactSizeIterator<Item = (&str, &DynamicPropertyDef)> + DoubleEndedIterator {
        self.generation().dynamic_properties.iter()
    }

    pub fn with_selected_class(mut self, selected: Rc<SelectedObjectClassFacts>) -> Option<Self> {
        self.try_attach_selected_class(selected).then_some(self)
    }

    pub fn try_attach_selected_class(&mut self, selected: Rc<SelectedObjectClassFacts>) -> bool {
        let Some(payload) = self.generation_mut_unique() else {
            return false;
        };
        if payload.payload_claim.is_some()
            || payload.class_name.has_retained_claim()
            || payload.class_name != *selected.class()
            || payload.selected_class.is_some()
            || payload.native_payload.is_some()
        {
            return false;
        }
        payload.selected_class = Some(selected);
        true
    }

    pub fn selected_class(&self) -> Option<&Rc<SelectedObjectClassFacts>> {
        self.generation().selected_class.as_ref()
    }

    pub(crate) fn trace_dynamic_property_values(
        &self,
    ) -> impl ExactSizeIterator<Item = &DynamicPropertyDef> + DoubleEndedIterator {
        self.generation().dynamic_properties.values()
    }
    pub fn is_class(&self, identity: runmat_types::StaticClassIdentity) -> bool {
        self.class_name().is(identity)
    }
    pub fn dynamic_property(&self, name: &str) -> Option<&DynamicPropertyDef> {
        self.generation().dynamic_properties.get(name)
    }
    pub fn dynamic_property_names(
        &self,
    ) -> impl ExactSizeIterator<Item = &str> + DoubleEndedIterator {
        self.generation().dynamic_properties.keys()
    }
    pub fn has_dynamic_property(&self, name: &str) -> bool {
        self.dynamic_property(name).is_some()
    }

    pub fn try_replace_property_unique(
        &mut self,
        name: &str,
        value: Value,
    ) -> Result<Value, Value> {
        let Some(payload) = self.generation_mut_unique() else {
            return Err(value);
        };
        payload.properties.try_replace_unique(name, value)
    }

    /// Borrow an existing slot only when both original payload and map are
    /// already unique. This never copies backing, changes claims, or grows a map.
    pub fn try_property_mut_unique(&mut self, name: &str) -> Option<&mut Value> {
        self.generation_mut_unique()?
            .properties
            .try_get_mut_unique(name)
    }

    /// Explicit ordinary-policy property edit. This may copy a shared
    /// ordinary generation and is therefore never a Managed fallback.
    pub fn replace_property_unbounded(&mut self, name: String, value: Value) -> Option<Value> {
        let value = match self.try_replace_property_unique(&name, value) {
            Ok(old) => return Some(old),
            Err(value) => value,
        };
        assert!(
            self.generation().can_copy_unbounded(),
            "unbounded property replacement requires an ordinary object generation"
        );
        let mut replacement = UnboundedObjectBuilder::copy_from_payload(self.generation());
        let old = replacement.insert(name, value);
        *self = replacement.finish();
        old
    }
}
