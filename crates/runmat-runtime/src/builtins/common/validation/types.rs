//! Shared validation values and range controls.

use super::IntValue;

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationAtom {
    Number(f64),
    Integer(IntValue),
    ComplexNumber(f64, f64),
    ComplexInteger(IntValue, IntValue),
    Text(String),
    MissingString,
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RangeInclusivity {
    pub lower: bool,
    pub upper: bool,
}

impl RangeInclusivity {
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
