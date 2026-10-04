use super::{ContainersMapStorage, DynamicPropertyDef, MemmapReceiverState};
use crate::{ForeignRef, Value};
use runmat_gc_api::{Trace, Tracer};
use runmat_types::ClassIdentity;
use std::collections::HashMap;

// Clone and structural equality include native payloads. Public handle
// equality remains identity-based through HandleRef, not this comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectInstance {
    pub class_name: ClassIdentity,
    pub properties: HashMap<String, Value>,
    pub dynamic_properties: Option<Box<HashMap<String, DynamicPropertyDef>>>,
    native_payload: Option<Box<NativeObjectPayload>>,
}

#[derive(Debug, Clone, PartialEq)]
enum NativeObjectPayload {
    ContainersMap(ContainersMapStorage),
    ForeignHostControl(ForeignRef),
    DatasetAuthority(ForeignRef),
    MatFileAuthority(ForeignRef),
    MemmapReceiver(MemmapReceiverState),
}

/// Whether a value object carries host- or session-owned native state.
/// Serialization must match this exhaustive classification, not class text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectPersistencePayloadKind {
    PlainValue,
    ContainersMap,
    ForeignHostControl,
    DatasetAuthority,
    MatFileAuthority,
    MemmapReceiver,
}

impl Trace for NativeObjectPayload {
    fn trace(&self, tracer: &mut dyn Tracer) {
        match self {
            Self::ContainersMap(storage) => storage.trace(tracer),
            Self::ForeignHostControl(_) => {}
            Self::DatasetAuthority(_) => {}
            Self::MatFileAuthority(_) => {}
            Self::MemmapReceiver(_) => {}
        }
    }
}

impl ObjectInstance {
    pub fn persistence_payload_kind(&self) -> ObjectPersistencePayloadKind {
        match self.native_payload.as_deref() {
            None => ObjectPersistencePayloadKind::PlainValue,
            Some(NativeObjectPayload::ContainersMap(_)) => {
                ObjectPersistencePayloadKind::ContainersMap
            }
            Some(NativeObjectPayload::ForeignHostControl(_)) => {
                ObjectPersistencePayloadKind::ForeignHostControl
            }
            Some(NativeObjectPayload::DatasetAuthority(_)) => {
                ObjectPersistencePayloadKind::DatasetAuthority
            }
            Some(NativeObjectPayload::MatFileAuthority(_)) => {
                ObjectPersistencePayloadKind::MatFileAuthority
            }
            Some(NativeObjectPayload::MemmapReceiver(_)) => {
                ObjectPersistencePayloadKind::MemmapReceiver
            }
        }
    }

    pub fn dynamic_property_definitions(
        &self,
    ) -> impl Iterator<Item = (&String, &DynamicPropertyDef)> {
        self.dynamic_properties
            .iter()
            .flat_map(|properties| properties.iter())
    }

    pub fn new(class_name: impl Into<ClassIdentity>) -> Self {
        Self {
            class_name: class_name.into(),
            properties: HashMap::new(),
            dynamic_properties: None,
            native_payload: None,
        }
    }

    pub fn with_containers_map_payload(mut self, storage: ContainersMapStorage) -> Self {
        self.native_payload = Some(Box::new(NativeObjectPayload::ContainersMap(storage)));
        self
    }

    pub fn with_foreign_host_control(mut self, reference: ForeignRef) -> Self {
        self.native_payload = Some(Box::new(NativeObjectPayload::ForeignHostControl(reference)));
        self
    }

    pub fn foreign_host_control(&self) -> Option<&ForeignRef> {
        match self.native_payload.as_deref() {
            Some(NativeObjectPayload::ForeignHostControl(reference)) => Some(reference),
            Some(NativeObjectPayload::ContainersMap(_))
            | Some(NativeObjectPayload::DatasetAuthority(_))
            | Some(NativeObjectPayload::MatFileAuthority(_))
            | Some(NativeObjectPayload::MemmapReceiver(_))
            | None => None,
        }
    }

    pub fn with_dataset_authority(mut self, reference: ForeignRef) -> Self {
        self.native_payload = Some(Box::new(NativeObjectPayload::DatasetAuthority(reference)));
        self
    }

    pub fn dataset_authority(&self) -> Option<&ForeignRef> {
        match self.native_payload.as_deref() {
            Some(NativeObjectPayload::DatasetAuthority(reference)) => Some(reference),
            Some(NativeObjectPayload::ContainersMap(_))
            | Some(NativeObjectPayload::ForeignHostControl(_))
            | Some(NativeObjectPayload::MatFileAuthority(_))
            | Some(NativeObjectPayload::MemmapReceiver(_))
            | None => None,
        }
    }

    pub fn with_matfile_authority(mut self, reference: ForeignRef) -> Self {
        self.native_payload = Some(Box::new(NativeObjectPayload::MatFileAuthority(reference)));
        self
    }

    pub fn matfile_authority(&self) -> Option<&ForeignRef> {
        match self.native_payload.as_deref() {
            Some(NativeObjectPayload::MatFileAuthority(reference)) => Some(reference),
            Some(NativeObjectPayload::ContainersMap(_))
            | Some(NativeObjectPayload::ForeignHostControl(_))
            | Some(NativeObjectPayload::DatasetAuthority(_))
            | Some(NativeObjectPayload::MemmapReceiver(_))
            | None => None,
        }
    }

    pub fn with_memmap_receiver(mut self, receiver: MemmapReceiverState) -> Self {
        self.native_payload = Some(Box::new(NativeObjectPayload::MemmapReceiver(receiver)));
        self
    }

    pub fn memmap_receiver(&self) -> Option<&MemmapReceiverState> {
        match self.native_payload.as_deref() {
            Some(NativeObjectPayload::MemmapReceiver(receiver)) => Some(receiver),
            _ => None,
        }
    }

    pub fn memmap_receiver_mut(&mut self) -> Option<&mut MemmapReceiverState> {
        match self.native_payload.as_deref_mut() {
            Some(NativeObjectPayload::MemmapReceiver(receiver)) => Some(receiver),
            _ => None,
        }
    }

    pub fn containers_map_payload(&self) -> Option<&ContainersMapStorage> {
        match self.native_payload.as_deref() {
            Some(NativeObjectPayload::ContainersMap(storage)) => Some(storage),
            Some(NativeObjectPayload::ForeignHostControl(_))
            | Some(NativeObjectPayload::DatasetAuthority(_))
            | Some(NativeObjectPayload::MatFileAuthority(_))
            | Some(NativeObjectPayload::MemmapReceiver(_))
            | None => None,
        }
    }

    pub fn containers_map_payload_mut(&mut self) -> Option<&mut ContainersMapStorage> {
        match self.native_payload.as_deref_mut() {
            Some(NativeObjectPayload::ContainersMap(storage)) => Some(storage),
            Some(NativeObjectPayload::ForeignHostControl(_))
            | Some(NativeObjectPayload::DatasetAuthority(_))
            | Some(NativeObjectPayload::MatFileAuthority(_))
            | Some(NativeObjectPayload::MemmapReceiver(_))
            | None => None,
        }
    }

    pub fn has_native_payload(&self) -> bool {
        self.native_payload.is_some()
    }

    pub fn for_each_native_value(&self, mut visitor: impl FnMut(&Value)) {
        match self.native_payload.as_deref() {
            Some(NativeObjectPayload::ContainersMap(storage)) => {
                storage.for_each_value(&mut visitor);
            }
            Some(NativeObjectPayload::ForeignHostControl(_))
            | Some(NativeObjectPayload::DatasetAuthority(_))
            | Some(NativeObjectPayload::MatFileAuthority(_))
            | Some(NativeObjectPayload::MemmapReceiver(_))
            | None => {}
        }
    }

    pub(crate) fn trace_native_payload(&self, tracer: &mut dyn Tracer) {
        if let Some(payload) = &self.native_payload {
            payload.trace(tracer);
        }
    }

    pub fn is_class(&self, identity: runmat_types::StaticClassIdentity) -> bool {
        self.class_name.is(identity)
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
