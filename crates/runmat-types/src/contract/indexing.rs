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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClassMethodRole {
    Ordinary,
    Constructor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassMethodOwner {
    pub declaring_class: crate::ClassIdentity,
    pub method: crate::MethodName,
    pub is_static: bool,
    pub role: ClassMethodRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexSelectorFact {
    Scalar,
    KnownOneBasedIndex(usize),
    BoundedScalar(ScalarIndexBounds),
    Colon,
    End { offset: isize },
    Numeric(crate::ValueFact),
    Logical(crate::ValueFact),
    Unknown,
}

/// An inclusive enclosure of one scalar index, not a selected index vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "UncheckedScalarIndexBounds")]
pub struct ScalarIndexBounds {
    minimum: usize,
    maximum: usize,
}

impl ScalarIndexBounds {
    pub fn new(minimum: usize, maximum: usize) -> Option<Self> {
        (minimum > 0 && minimum <= maximum).then_some(Self { minimum, maximum })
    }

    pub fn minimum(self) -> usize {
        self.minimum
    }
    pub fn maximum(self) -> usize {
        self.maximum
    }

    pub fn enclose(self, other: Self) -> Self {
        Self {
            minimum: self.minimum.min(other.minimum),
            maximum: self.maximum.max(other.maximum),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UncheckedScalarIndexBounds {
    minimum: usize,
    maximum: usize,
}

impl TryFrom<UncheckedScalarIndexBounds> for ScalarIndexBounds {
    type Error = &'static str;
    fn try_from(value: UncheckedScalarIndexBounds) -> Result<Self, Self::Error> {
        Self::new(value.minimum, value.maximum)
            .ok_or("scalar index bounds must be positive and ordered")
    }
}
