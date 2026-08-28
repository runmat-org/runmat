use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexKind {
    Paren,
    Brace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexResultContext {
    ReadSingle,
    ReadCommaList,
    AssignmentTarget,
    DeletionTarget,
    FunctionArgumentExpansion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexSelectorFact {
    Scalar,
    KnownOneBasedIndex(usize),
    Colon,
    End { offset: isize },
    Numeric(crate::ValueFact),
    Logical(crate::ValueFact),
    Unknown,
}
