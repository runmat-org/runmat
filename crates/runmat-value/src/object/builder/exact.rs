use super::super::{
    DynamicPropertyDef, NativeObjectPayload, ObjectInstance, ObjectPayload,
    SelectedObjectClassFacts, SharedObjectPayload,
};
use crate::SharedClassIdentity;
use crate::{
    exact_capacity_vec, ExactHostAllocationClaim, ExactHostAllocationPermit,
    ExactHostNumericAllocationError, ExactOrderedMapBuilder, MapInsertError, Utf8Text, Value,
};
use runmat_types::ClassIdentity;

#[derive(Debug)]
pub struct PreparedObjectClassName {
    identity: SharedClassIdentity,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectClassPreparationError {
    Invalid(runmat_types::InvalidClassIdentity),
    Allocation(ExactHostNumericAllocationError),
}
impl std::fmt::Display for ObjectClassPreparationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(error) => error.fmt(formatter),
            Self::Allocation(error) => error.fmt(formatter),
        }
    }
}
impl std::error::Error for ObjectClassPreparationError {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactObjectQuote {
    pub requested_bytes: usize,
    pub retained_claims: usize,
}
impl PreparedObjectClassName {
    pub fn into_identity(self) -> SharedClassIdentity {
        self.identity
    }

    pub fn prepare(
        name: &str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ObjectClassPreparationError> {
        let bytes = SharedClassIdentity::requested_bytes_for_capacity(name.len())
            .map_err(ObjectClassPreparationError::Allocation)?;
        let claim = permit
            .reserve_exact_claim(bytes)
            .map_err(ObjectClassPreparationError::Allocation)?;
        let mut bytes =
            exact_capacity_vec(name.len()).map_err(ObjectClassPreparationError::Allocation)?;
        bytes.extend_from_slice(name.as_bytes());
        let spelling = String::from_utf8(bytes).expect("class spelling originated as UTF-8");
        let identity =
            ClassIdentity::new(spelling).map_err(ObjectClassPreparationError::Invalid)?;
        let identity = SharedClassIdentity::try_new_claimed(identity, claim)
            .map_err(ObjectClassPreparationError::Allocation)?;
        Ok(Self { identity })
    }
}
#[derive(Debug)]
pub struct ExactObjectBuilder {
    class_name: SharedClassIdentity,
    properties: ExactOrderedMapBuilder<Value>,
    dynamic_properties: ExactOrderedMapBuilder<DynamicPropertyDef>,
    selected_class: Option<std::rc::Rc<SelectedObjectClassFacts>>,
    native_payload: Option<NativeObjectPayload>,
    payload_claim: ExactHostAllocationClaim,
}

/// Caller-owned source for an exact replacement generation. Admission errors
/// leave the complete original object here, so no child/native owner is lost
/// and the error result stays small.
#[derive(Debug)]
pub struct ObjectEditInput {
    object: Option<ObjectInstance>,
}

impl ObjectEditInput {
    pub fn new(object: ObjectInstance) -> Self {
        Self {
            object: Some(object),
        }
    }
    pub fn object(&self) -> Option<&ObjectInstance> {
        self.object.as_ref()
    }
    pub fn into_object(self) -> Option<ObjectInstance> {
        self.object
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectEditPreparationError {
    Consumed,
    SharedGeneration,
    SourceNotAdmitted,
    CapacityInsufficient,
    Allocation(ExactHostNumericAllocationError),
}

impl std::fmt::Display for ObjectEditPreparationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Consumed => formatter.write_str("object edit source was already consumed"),
            Self::SharedGeneration => formatter.write_str("object generation is shared"),
            Self::SourceNotAdmitted => {
                formatter.write_str("object generation contains unadmitted metadata")
            }
            Self::CapacityInsufficient => {
                formatter.write_str("object replacement capacity is smaller than its source")
            }
            Self::Allocation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ObjectEditPreparationError {}
impl ExactObjectBuilder {
    pub fn numeric_factor_native_requested_bytes() -> usize {
        NativeObjectPayload::numeric_factor_requested_bytes()
    }

    pub const fn numeric_factor_native_retained_claim_count() -> usize {
        1
    }
    pub fn containers_map_native_requested_bytes(
        value: &super::super::ContainersMapStorage,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        NativeObjectPayload::containers_map_requested_bytes(value)
    }

    pub const fn containers_map_native_retained_claim_count() -> usize {
        1
    }

    pub fn memmap_receiver_native_requested_bytes(
        value: &super::super::MemmapReceiverState,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        NativeObjectPayload::memmap_receiver_requested_bytes(value)
    }

    pub const fn memmap_receiver_native_retained_claim_count() -> usize {
        1
    }
    pub const fn payload_requested_bytes() -> usize {
        SharedObjectPayload::requested_bytes()
    }
    pub fn quote(
        class_name: &str,
        property_capacity: usize,
    ) -> Result<ExactObjectQuote, ExactHostNumericAllocationError> {
        Self::quote_with_dynamic_capacity(class_name, property_capacity, 0)
    }
    /// Container/header quote only. Dynamic definitions supplied to this
    /// builder must already own exact name/class/description generations.
    pub fn quote_with_dynamic_capacity(
        class_name: &str,
        property_capacity: usize,
        dynamic_property_capacity: usize,
    ) -> Result<ExactObjectQuote, ExactHostNumericAllocationError> {
        let properties = crate::ExactOrderedMap::<Value>::quote(property_capacity)?;
        let dynamic_properties =
            crate::ExactOrderedMap::<DynamicPropertyDef>::quote(dynamic_property_capacity)?;
        let requested_bytes = Self::payload_requested_bytes()
            .checked_add(SharedClassIdentity::requested_bytes_for_capacity(
                class_name.len(),
            )?)
            .and_then(|bytes| bytes.checked_add(properties.requested_bytes))
            .and_then(|bytes| bytes.checked_add(dynamic_properties.requested_bytes))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Ok(ExactObjectQuote {
            requested_bytes,
            retained_claims: 2 + properties.retained_claims + dynamic_properties.retained_claims,
        })
    }
    pub fn try_new(
        class_name: PreparedObjectClassName,
        property_capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        Self::try_new_with_dynamic_capacity(class_name, property_capacity, 0, permit)
    }
    pub fn try_new_with_dynamic_capacity(
        class_name: PreparedObjectClassName,
        property_capacity: usize,
        dynamic_property_capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let payload_claim = permit.reserve_exact_claim(Self::payload_requested_bytes())?;
        let properties = ExactOrderedMapBuilder::try_new(property_capacity, permit)?;
        let dynamic_properties =
            ExactOrderedMapBuilder::try_new(dynamic_property_capacity, permit)?;
        Ok(Self {
            class_name: class_name.identity,
            properties,
            dynamic_properties,
            selected_class: None,
            native_payload: None,
            payload_claim,
        })
    }

    /// Quote a replacement generation that reuses the source class/text and
    /// child owners. Only the new payload and fixed map blocks are allocated.
    pub fn replacement_quote(
        property_capacity: usize,
        dynamic_property_capacity: usize,
    ) -> Result<ExactObjectQuote, ExactHostNumericAllocationError> {
        let properties = crate::ExactOrderedMap::<Value>::quote(property_capacity)?;
        let dynamic_properties =
            crate::ExactOrderedMap::<DynamicPropertyDef>::quote(dynamic_property_capacity)?;
        let requested_bytes = Self::payload_requested_bytes()
            .checked_add(properties.requested_bytes)
            .and_then(|bytes| bytes.checked_add(dynamic_properties.requested_bytes))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Ok(ExactObjectQuote {
            requested_bytes,
            retained_claims: 1 + properties.retained_claims + dynamic_properties.retained_claims,
        })
    }

    /// Prepare an independently admitted replacement from one uniquely owned,
    /// already-admitted generation. All child, native, selected-class, key,
    /// and class owners move; no Value or payload backing is cloned.
    pub fn prepare_replacement(
        input: &mut ObjectEditInput,
        property_capacity: usize,
        dynamic_property_capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ObjectEditPreparationError> {
        let source = input
            .object
            .as_ref()
            .ok_or(ObjectEditPreparationError::Consumed)?;
        if !source.permits_exact_move_edit() {
            return Err(ObjectEditPreparationError::SharedGeneration);
        }
        let generation = source.generation();
        let metadata_admitted = generation.class_name.has_retained_claim()
            && generation
                .properties
                .entries()
                .iter()
                .all(|entry| entry.key().is_empty() || entry.key().has_retained_claim())
            && generation.dynamic_properties.entries().iter().all(|entry| {
                let definition = entry.value();
                (entry.key().is_empty() || entry.key().has_retained_claim())
                    && (definition.name.is_empty() || definition.name.has_retained_claim())
                    && definition.defining_class.has_retained_claim()
                    && (definition.description.is_empty()
                        || definition.description.has_retained_claim())
            });
        if !metadata_admitted || !source.native_is_admitted_for_copy() {
            return Err(ObjectEditPreparationError::SourceNotAdmitted);
        }
        if property_capacity < generation.properties.len()
            || dynamic_property_capacity < generation.dynamic_properties.len()
        {
            return Err(ObjectEditPreparationError::CapacityInsufficient);
        }

        // Allocate every destination block before taking the source owner.
        let payload_claim = permit
            .reserve_exact_claim(Self::payload_requested_bytes())
            .map_err(ObjectEditPreparationError::Allocation)?;
        let properties = ExactOrderedMapBuilder::try_new(property_capacity, permit)
            .map_err(ObjectEditPreparationError::Allocation)?;
        let dynamic_properties = ExactOrderedMapBuilder::try_new(dynamic_property_capacity, permit)
            .map_err(ObjectEditPreparationError::Allocation)?;

        let source = input.object.take().expect("source was checked above");
        let payload = source
            .into_payload_unique()
            .expect("unique source generation was checked before allocation");
        let ObjectPayload {
            class_name,
            properties: source_properties,
            dynamic_properties: source_dynamic_properties,
            selected_class,
            native_payload,
            payload_claim: source_payload_claim,
        } = payload;
        let source_properties = source_properties
            .try_into_entries()
            .expect("unique property backing was checked before allocation");
        let source_dynamic_properties = source_dynamic_properties
            .try_into_entries()
            .expect("unique dynamic backing was checked before allocation");
        let mut builder = Self {
            class_name,
            properties,
            dynamic_properties,
            selected_class,
            native_payload,
            payload_claim,
        };
        for (key, value) in source_properties {
            builder
                .properties
                .try_insert_new_prepared(key, value)
                .expect("destination property capacity and key admission were checked");
        }
        for (key, value) in source_dynamic_properties {
            builder
                .dynamic_properties
                .try_insert_new_prepared(key, value)
                .expect("destination dynamic capacity and key admission were checked");
        }
        // Retire the prior payload only after every child has moved.
        drop(source_payload_claim);
        Ok(builder)
    }

    /// Prepare an exact copy generation while the original alias remains
    /// live. Canonical child Values, metadata, and native-state owners clone
    /// in O(1); no retained payload backing is duplicated.
    pub fn prepare_copy(
        source: &ObjectInstance,
        property_capacity: usize,
        dynamic_property_capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ObjectEditPreparationError> {
        Self::prepare_copy_filter(
            source,
            property_capacity,
            dynamic_property_capacity,
            None,
            None,
            permit,
        )
    }

    /// Prepare a private generation omitting one existing property. Metadata,
    /// dynamic declarations, and native owners retain their original owners.
    /// Callers validate the named property before admitting this destination.
    pub fn prepare_copy_excluding_property(
        source: &ObjectInstance,
        excluded: &str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ObjectEditPreparationError> {
        if source.property(excluded).is_none() {
            return Err(ObjectEditPreparationError::CapacityInsufficient);
        }
        Self::prepare_copy_filter(
            source,
            source.property_count() - 1,
            source.dynamic_property_count(),
            Some(excluded),
            None,
            permit,
        )
    }

    /// Remove an existing dynamic property's value and declaration in one
    /// unpublished destination; all other original owners remain shared.
    pub fn prepare_copy_excluding_dynamic_property(
        source: &ObjectInstance,
        excluded: &str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ObjectEditPreparationError> {
        if source.property(excluded).is_none() || source.dynamic_property(excluded).is_none() {
            return Err(ObjectEditPreparationError::CapacityInsufficient);
        }
        Self::prepare_copy_filter(
            source,
            source.property_count() - 1,
            source.dynamic_property_count() - 1,
            Some(excluded),
            Some(excluded),
            permit,
        )
    }

    fn prepare_copy_filter(
        source: &ObjectInstance,
        property_capacity: usize,
        dynamic_property_capacity: usize,
        excluded: Option<&str>,
        excluded_dynamic: Option<&str>,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ObjectEditPreparationError> {
        let generation = source.generation();
        let metadata_admitted = generation.class_name.has_retained_claim()
            && generation
                .properties
                .entries()
                .iter()
                .all(|entry| entry.key().is_empty() || entry.key().has_retained_claim())
            && generation.dynamic_properties.entries().iter().all(|entry| {
                let definition = entry.value();
                (entry.key().is_empty() || entry.key().has_retained_claim())
                    && (definition.name.is_empty() || definition.name.has_retained_claim())
                    && definition.defining_class.has_retained_claim()
                    && (definition.description.is_empty()
                        || definition.description.has_retained_claim())
            });
        if !metadata_admitted || !source.native_is_admitted_for_copy() {
            return Err(ObjectEditPreparationError::SourceNotAdmitted);
        }
        if property_capacity < generation.properties.len() - usize::from(excluded.is_some())
            || dynamic_property_capacity
                < generation.dynamic_properties.len() - usize::from(excluded_dynamic.is_some())
        {
            return Err(ObjectEditPreparationError::CapacityInsufficient);
        }
        let payload_claim = permit
            .reserve_exact_claim(Self::payload_requested_bytes())
            .map_err(ObjectEditPreparationError::Allocation)?;
        let mut properties = ExactOrderedMapBuilder::try_new(property_capacity, permit)
            .map_err(ObjectEditPreparationError::Allocation)?;
        let mut dynamic_properties =
            ExactOrderedMapBuilder::try_new(dynamic_property_capacity, permit)
                .map_err(ObjectEditPreparationError::Allocation)?;
        for entry in generation.properties.entries() {
            if excluded == Some(entry.key().as_str()) {
                continue;
            }
            properties
                .try_insert_new_prepared(entry.key().clone(), entry.value().clone())
                .expect("source property keys and destination capacity were checked");
        }
        for entry in generation.dynamic_properties.entries() {
            if excluded_dynamic == Some(entry.key().as_str()) {
                continue;
            }
            dynamic_properties
                .try_insert_new_prepared(entry.key().clone(), entry.value().clone())
                .expect("source dynamic keys and destination capacity were checked");
        }
        Ok(Self {
            class_name: generation.class_name.clone(),
            properties,
            dynamic_properties,
            selected_class: generation.selected_class.clone(),
            native_payload: generation.native_payload.clone(),
            payload_claim,
        })
    }
    pub fn try_insert_prepared(
        &mut self,
        name: Utf8Text,
        value: Value,
    ) -> Result<Option<Value>, MapInsertError<Value>> {
        self.properties.try_insert_prepared(name, value)
    }
    pub fn try_replace_existing_property(
        &mut self,
        name: &str,
        value: Value,
    ) -> Result<Value, Value> {
        self.properties.try_replace_existing(name, value)
    }
    pub fn try_insert_dynamic_prepared(
        &mut self,
        definition: DynamicPropertyDef,
    ) -> Result<Option<DynamicPropertyDef>, MapInsertError<DynamicPropertyDef>> {
        self.try_insert_dynamic_key_prepared(definition.name.clone(), definition)
    }

    /// Replace only an existing declaration in an unpublished exact generation.
    /// The caller admits the definition's original metadata owners first.
    pub fn try_replace_existing_dynamic_property(
        &mut self,
        name: &str,
        definition: DynamicPropertyDef,
    ) -> Result<DynamicPropertyDef, DynamicPropertyDef> {
        if (!definition.name.is_empty() && !definition.name.has_retained_claim())
            || (!definition.description.is_empty() && !definition.description.has_retained_claim())
            || !definition.defining_class.has_retained_claim()
        {
            return Err(definition);
        }
        self.dynamic_properties
            .try_replace_existing(name, definition)
    }

    /// Link a preallocated metadata shell while this destination is unpublished.
    /// Text, class and map owners are unchanged; no backing is allocated.
    pub fn try_set_dynamic_metadata_handle(
        &mut self,
        name: &str,
        handle: runmat_gc_api::GcHandle,
    ) -> bool {
        let Some(definition) = self.dynamic_properties.get_mut(name) else {
            return false;
        };
        definition.metadata_handle = Some(handle);
        true
    }

    /// Preserve a decoded map key independently of its declaration spelling.
    pub fn try_insert_dynamic_key_prepared(
        &mut self,
        key: Utf8Text,
        definition: DynamicPropertyDef,
    ) -> Result<Option<DynamicPropertyDef>, MapInsertError<DynamicPropertyDef>> {
        let name_ready = definition.name.is_empty() || definition.name.has_retained_claim();
        let description_ready =
            definition.description.is_empty() || definition.description.has_retained_claim();
        if !name_ready || !description_ready || !definition.defining_class.has_retained_claim() {
            return Err(MapInsertError::value_not_admitted(key, definition));
        }
        self.dynamic_properties.try_insert_prepared(key, definition)
    }
    pub fn try_set_selected_class(
        &mut self,
        selected: std::rc::Rc<SelectedObjectClassFacts>,
    ) -> bool {
        if self.selected_class.is_some() || selected.class() != self.class_name.as_identity() {
            return false;
        }
        self.selected_class = Some(selected);
        true
    }
    fn try_set_native(&mut self, payload: NativeObjectPayload) -> bool {
        if self.native_payload.is_some() {
            return false;
        }
        self.native_payload = Some(payload);
        true
    }
    pub fn try_set_foreign_host_control(&mut self, value: crate::ForeignRef) -> bool {
        self.try_set_native(NativeObjectPayload::ForeignHostControl(value))
    }
    pub fn try_set_dataset_authority(&mut self, value: crate::ForeignRef) -> bool {
        self.try_set_native(NativeObjectPayload::DatasetAuthority(value))
    }
    pub fn try_set_matfile_authority(&mut self, value: crate::ForeignRef) -> bool {
        self.try_set_native(NativeObjectPayload::MatFileAuthority(value))
    }
    pub fn try_set_tcp_authority(&mut self, value: crate::ForeignRef) -> bool {
        self.try_set_native(NativeObjectPayload::TcpAuthority(value))
    }
    pub fn try_set_graphics_receiver(&mut self, value: super::super::GraphicsReceiverRef) -> bool {
        self.try_set_native(NativeObjectPayload::GraphicsReceiver(value))
    }
    pub fn try_set_numeric_factor_data(
        &mut self,
        value: super::super::NumericFactorData,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<bool, ExactHostNumericAllocationError> {
        let value = NativeObjectPayload::numeric_factor_exact(value, permit)?;
        Ok(self.try_set_native(value))
    }
    pub fn try_set_optimization_native(
        &mut self,
        value: crate::optimization_model::PreparedOptimizationNative,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<bool, ExactHostNumericAllocationError> {
        let native = NativeObjectPayload::optimization_native_exact(value, permit)?;
        Ok(self.try_set_native(native))
    }
    pub fn try_set_containers_map_payload(
        &mut self,
        value: super::super::ContainersMapStorage,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<bool, ExactHostNumericAllocationError> {
        let value = NativeObjectPayload::containers_map_exact(value, permit)?;
        Ok(self.try_set_native(value))
    }
    pub fn try_set_memmap_receiver(
        &mut self,
        value: super::super::MemmapReceiverState,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<bool, ExactHostNumericAllocationError> {
        let value = NativeObjectPayload::memmap_receiver_exact(value, permit)?;
        Ok(self.try_set_native(value))
    }
    pub fn replace_memmap_receiver(
        &mut self,
        value: super::super::MemmapReceiverState,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactHostNumericAllocationError> {
        let value = NativeObjectPayload::memmap_receiver_exact(value, permit)?;
        self.native_payload = Some(value);
        Ok(())
    }
    pub fn finish(self) -> Result<ObjectInstance, ExactHostNumericAllocationError> {
        let payload = ObjectPayload {
            class_name: self.class_name,
            properties: self.properties.finish(),
            dynamic_properties: self.dynamic_properties.finish(),
            selected_class: self.selected_class,
            native_payload: self.native_payload,
            payload_claim: Some(self.payload_claim),
        };
        SharedObjectPayload::admitted(payload)
            .map(ObjectInstance::from_payload)
            .ok_or(ExactHostNumericAllocationError::AllocationFailed)
    }
}
