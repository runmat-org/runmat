use serde::{Deserialize, Serialize};

use crate::IntegerLiteral;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FunctionArgDim {
    Any,
    Exact(usize),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct FunctionArgSizeSpec {
    pub rows: FunctionArgDim,
    pub cols: FunctionArgDim,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FunctionArgValidator {
    A(Vec<String>),
    Column,
    Finite,
    Float,
    Folder,
    File,
    NumericOrLogical,
    Numeric,
    Text,
    TextScalar,
    NonzeroLengthText,
    Nonempty,
    ScalarOrEmpty,
    Real,
    Integer,
    Vector { allow_all_empties: bool },
    Positive,
    Negative,
    Nonnegative,
    Nonmissing,
    NonNan,
    Nonzero,
    Nonpositive,
    Nonsparse,
    Sparse,
    ValidVariableName,
    UnderlyingType(Vec<String>),
    Member(Vec<FunctionArgValidationLiteral>),
    InRange(f64, f64, FunctionArgRangeInclusivity),
    GreaterThanOrEqual(f64),
    LessThanOrEqual(f64),
    GreaterThan(f64),
    LessThan(f64),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct FunctionArgRangeInclusivity {
    pub lower: bool,
    pub upper: bool,
}

impl FunctionArgRangeInclusivity {
    pub const CLOSED: Self = Self {
        lower: true,
        upper: true,
    };

    pub const OPEN: Self = Self {
        lower: false,
        upper: false,
    };

    pub const OPEN_LEFT: Self = Self {
        lower: false,
        upper: true,
    };

    pub const OPEN_RIGHT: Self = Self {
        lower: true,
        upper: false,
    };
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FunctionArgValidationLiteral {
    Number(f64),
    Integer(IntegerLiteral),
    Text(String),
    Bool(bool),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FunctionArgDefaultValue {
    Number(f64),
    Integer(IntegerLiteral),
    Bool(bool),
    String(String),
    EmptyArray,
}
