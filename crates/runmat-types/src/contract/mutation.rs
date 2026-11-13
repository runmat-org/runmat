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

/// Container required by an immutable assignment suffix for a genuinely
/// absent binding, field, or new cell/structure coordinate. This is a recipe,
/// not permission to reinterpret any existing language value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssignmentChildCreation {
    ExistingOnly,
    NumericArray,
    CellArray,
    ScalarStructure,
    StructureArray,
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
