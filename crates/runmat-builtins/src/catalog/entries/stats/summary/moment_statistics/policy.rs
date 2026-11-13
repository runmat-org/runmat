//! Closed identity policy shared by static facts and runtime admission.

use crate::ReductionNaN;
use runmat_types::LiteralValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MomentControl {
    All,
    Missing(ReductionNaN),
}

impl MomentControl {
    pub fn from_public_option(option: &str) -> Option<Self> {
        ReductionNaN::from_public_option(option)
            .map(Self::Missing)
            .or_else(|| option.eq_ignore_ascii_case("all").then_some(Self::All))
    }

    pub fn from_literal(literal: &LiteralValue) -> Option<Self> {
        match literal {
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text) => Self::from_public_option(text),
            _ => None,
        }
    }
}

/// Preserve the public scalar flag tolerance and range at one boundary.
pub fn moment_flag_from_f64(value: f64) -> Option<u8> {
    (value.is_finite() && value.fract().abs() <= 1e-12 && (0.0..=1.0).contains(&value))
        .then_some(value as u8)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MomentStatisticKind {
    MeanAbsoluteDeviation,
    Skewness,
    Kurtosis,
}

impl MomentStatisticKind {
    pub const fn default_flag(self) -> u8 {
        match self {
            Self::MeanAbsoluteDeviation => 0,
            Self::Skewness | Self::Kurtosis => 1,
        }
    }
}
