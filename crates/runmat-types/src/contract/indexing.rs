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

/// Semantic use of an object subscript path. This travels unchanged across
/// HIR, MIR, executor and runtime protocol boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectIndexingContext {
    Statement,
    Expression,
    Assignment,
}

/// Exact semantic owner of executable class-method code. Executors combine
/// this declaration identity with the active callable identity; registry
/// generations remain runtime-only and are never serialized.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassMethodOwner {
    pub declaring_class: crate::ClassIdentity,
    pub method: crate::MethodName,
    pub is_static: bool,
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
