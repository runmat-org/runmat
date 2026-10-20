use runmat_execution::resource::ExactShared;
use std::rc::Rc;

use super::{DynamicPropertyDef, NativeObjectPayload, SelectedObjectClassFacts};
use crate::{ExactHostAllocationClaim, ExactOrderedMap, SharedClassIdentity, Value};

/// One immutable object generation. Every public `ObjectInstance` alias
/// shares this exact payload; child values and metadata are never cloned by
/// cloning the object.
#[derive(Debug)]
pub(super) struct ObjectPayload {
    pub(super) class_name: SharedClassIdentity,
    pub(super) properties: ExactOrderedMap<Value>,
    pub(super) dynamic_properties: ExactOrderedMap<DynamicPropertyDef>,
    pub(super) selected_class: Option<Rc<SelectedObjectClassFacts>>,
    pub(super) native_payload: Option<NativeObjectPayload>,
    // Retire all physical children and the independently shared class owner
    // before refunding the payload claim. ExactShared releases its header
    // before dropping this payload.
    pub(super) payload_claim: Option<ExactHostAllocationClaim>,
}

impl ObjectPayload {
    pub(super) fn can_begin_unbounded_edit(&self) -> bool {
        !self.class_name.has_retained_claim()
            && self.payload_claim.is_none()
            && self.properties.is_unique()
            && self.dynamic_properties.is_unique()
    }
    pub(super) fn can_copy_unbounded(&self) -> bool {
        !self.class_name.has_retained_claim() && self.payload_claim.is_none()
    }
}

impl PartialEq for ObjectPayload {
    fn eq(&self, other: &Self) -> bool {
        self.class_name == other.class_name
            && self.properties == other.properties
            && self.dynamic_properties == other.dynamic_properties
            && self.selected_class == other.selected_class
            && self.native_payload == other.native_payload
    }
}

#[derive(Debug)]
pub(super) struct SharedObjectPayload(pub(super) ExactShared<ObjectPayload>);

impl Clone for SharedObjectPayload {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl PartialEq for SharedObjectPayload {
    fn eq(&self, other: &Self) -> bool {
        self.0.get() == other.0.get()
    }
}

impl SharedObjectPayload {
    pub(super) const fn requested_bytes() -> usize {
        ExactShared::<ObjectPayload>::requested_bytes()
    }

    pub(super) fn unbounded(payload: ObjectPayload) -> Self {
        Self(ExactShared::new(payload))
    }

    pub(super) fn admitted(payload: ObjectPayload) -> Option<Self> {
        match ExactShared::try_new(payload) {
            Ok(payload) => Some(Self(payload)),
            Err(payload) => {
                // Children retire before the final payload claim by field
                // order; the large owner never crosses this small status API.
                drop(payload);
                None
            }
        }
    }

    pub(super) fn get(&self) -> &ObjectPayload {
        self.0.get()
    }

    pub(super) fn get_mut_unique(&mut self) -> Option<&mut ObjectPayload> {
        self.0.is_unique().then(|| self.0.get_mut_unique())
    }

    pub(super) fn is_unique(&self) -> bool {
        self.0.is_unique()
    }

    pub(super) fn shares_allocation_with(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }

    pub(super) fn try_unwrap(self) -> Result<ObjectPayload, Self> {
        self.0.try_unwrap().map_err(Self)
    }
}
