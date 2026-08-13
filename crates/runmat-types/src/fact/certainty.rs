use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CertaintyFact {
    Proven,
    Symbolic,
    Dynamic(DynamicReason),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DynamicReason {
    ConflictingControlFlow,
    DynamicDispatch,
    ExternalData,
    ForeignBoundary,
    RuntimeValue,
    UnsupportedRepresentation,
    UnresolvedCallable,
    Unspecified,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct InvalidationVector(pub BTreeSet<InvalidationCause>);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InvalidationCause {
    SourceChanged,
    DependencyChanged,
    CatalogChanged,
    ClassHierarchyChanged,
    ProviderCapabilitiesChanged,
    RuntimePolicyChanged,
}
