use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalInferenceRule {
    Elementwise(LogicalElementwiseRule),
    MetadataPredicate(MetadataPredicate),
    NumericClassification(NumericClassificationPredicate),
    Relational(RelationalOperator),
    ScalarReduction(ScalarLogicalReduction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalElementwiseRule {
    Binary(LogicalBinaryOperator),
    Unary(LogicalUnaryOperator),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalBinaryOperator {
    And,
    Or,
    Xor,
}

impl LogicalBinaryOperator {
    pub const fn name(self) -> &'static str {
        match self {
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalUnaryOperator {
    Not,
}

impl LogicalUnaryOperator {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Not => "not",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RelationalOperator {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl RelationalOperator {
    pub const fn is_equality(self) -> bool {
        matches!(self, Self::Equal | Self::NotEqual)
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Equal => "eq",
            Self::NotEqual => "ne",
            Self::LessThan => "lt",
            Self::LessThanOrEqual => "le",
            Self::GreaterThan => "gt",
            Self::GreaterThanOrEqual => "ge",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MetadataPredicate {
    Cell,
    CellString,
    GpuArray,
    Logical,
    Numeric,
    Real,
    Sparse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NumericClassificationPredicate {
    Finite,
    Infinite,
    Nan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarLogicalReduction {
    AllFinite,
}
