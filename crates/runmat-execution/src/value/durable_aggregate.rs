//! Durable-storage aggregate forms in the canonical ValuePayload schema.
//! V2 execution transport does not admit these tags.

use super::{StructField, ValuePayload};
use runmat_types::{ClassIdentity, MemberAccess};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DurablePlainObject {
    pub class_identity: ClassIdentity,
    pub properties: Vec<StructField>,
    pub dynamic_properties: Vec<DurableDynamicProperty>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DurableDynamicProperty {
    pub name: String,
    pub definition_name: String,
    pub defining_class: ClassIdentity,
    pub get_access: MemberAccess,
    pub set_access: MemberAccess,
    pub dependent: bool,
    pub hidden: bool,
    pub transient: bool,
    pub non_copyable: bool,
    pub abort_set: bool,
    pub set_observable: bool,
    pub get_observable: bool,
    pub description: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DurableStructArrayField {
    pub name: String,
    pub values: Vec<ValuePayload>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DurableStructArray {
    pub shape: Vec<u64>,
    pub fields: Vec<DurableStructArrayField>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DurableObjectArray {
    pub class_identity: ClassIdentity,
    pub shape: Vec<u64>,
    pub elements: Vec<DurablePlainObject>,
}
