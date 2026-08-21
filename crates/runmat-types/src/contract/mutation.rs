use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlaceMutationKind {
    BindOrAssign,
    IndexedAssign,
    CellAssign,
    MemberAssign,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssignmentCreationPolicy {
    ExistingOnly,
    CreateBinding,
    CreateArrayByIndex,
    CreateStructFieldPath,
    Overloaded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssignmentShapePolicy {
    Exact,
    ScalarExpansion,
    MatlabCompatible,
    Overloaded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MutationContract {
    pub kind: PlaceMutationKind,
    pub creation: AssignmentCreationPolicy,
    pub shape: AssignmentShapePolicy,
}
