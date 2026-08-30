use super::ValueFact;
use crate::{ClassId, ClassIdentity};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectFact {
    pub class: Option<ClassId>,
    pub runtime_class: Option<ClassIdentity>,
    pub properties: BTreeMap<String, ValueFact>,
    pub properties_complete: bool,
    /// `None` means the class/value-semantics relationship is not proven.
    pub handle_semantics: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassReferenceFact {
    pub class: Option<ClassId>,
    pub runtime_class: Option<ClassIdentity>,
}
