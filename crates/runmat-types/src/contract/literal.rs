use crate::NumericClass;
use serde::{Deserialize, Serialize};

/// Source-known literal information. Numeric text is retained for exact
/// integer/decimal interpretation; the legacy `Number` variant remains for
/// catalog resolvers until R06 completes their migration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiteralValue {
    Number(f64),
    Real {
        text: String,
        class: NumericClass,
    },
    Integer {
        text: String,
        class: NumericClass,
    },
    Complex {
        real: String,
        imaginary: String,
        class: NumericClass,
    },
    Bool(bool),
    Character(String),
    String(String),
    Keyword(String),
    Symbolic(String),
    Vector(Vec<LiteralValue>),
    Matrix(Vec<Vec<LiteralValue>>),
    Empty,
    Unknown,
}

/// The source-known state of a colon-range step. Keeping an unknown explicit
/// step distinct from an omitted step prevents dynamic `start:step:end`
/// expressions from being inferred as if MATLAB's implicit step of one had
/// been selected.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RangeStepFact {
    Implicit,
    Known(f64),
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LiteralContext {
    pub literal_args: Vec<LiteralValue>,
}

impl LiteralContext {
    pub fn new(literal_args: Vec<LiteralValue>) -> Self {
        Self { literal_args }
    }

    pub fn numeric_dims(&self) -> Vec<Option<usize>> {
        self.numeric_dims_from(0)
    }

    pub fn numeric_dims_from(&self, start: usize) -> Vec<Option<usize>> {
        let slice = self.literal_args.get(start..).unwrap_or(&[]);
        if let Some(LiteralValue::Vector(values)) = slice.first() {
            return values
                .iter()
                .map(Self::numeric_dimension_from_literal)
                .collect();
        }
        slice
            .iter()
            .map(Self::numeric_dimension_from_literal)
            .collect()
    }

    pub fn literal_string_at(&self, index: usize) -> Option<String> {
        match self.literal_args.get(index) {
            Some(
                LiteralValue::Character(value)
                | LiteralValue::String(value)
                | LiteralValue::Keyword(value),
            ) => Some(value.to_ascii_lowercase()),
            _ => None,
        }
    }

    pub fn literal_bool_at(&self, index: usize) -> Option<bool> {
        match self.literal_args.get(index) {
            Some(LiteralValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn literal_vector_at(&self, index: usize) -> Option<Vec<LiteralValue>> {
        match self.literal_args.get(index) {
            Some(LiteralValue::Vector(values)) => Some(values.clone()),
            _ => None,
        }
    }

    pub fn numeric_vector_at(&self, index: usize) -> Option<Vec<Option<usize>>> {
        let values = match self.literal_args.get(index) {
            Some(LiteralValue::Vector(values)) => values,
            _ => return None,
        };
        if values
            .iter()
            .any(|value| matches!(value, LiteralValue::Vector(_) | LiteralValue::Matrix(_)))
        {
            return None;
        }
        Some(
            values
                .iter()
                .map(Self::numeric_dimension_from_literal)
                .collect(),
        )
    }

    pub fn numeric_at(&self, index: usize) -> Option<f64> {
        self.literal_args
            .get(index)
            .and_then(Self::numeric_from_literal)
    }

    /// Project the represented numeric value into binary64. Typed single
    /// literals round to their physical precision before widening; their
    /// authored decimal text remains available on `LiteralValue::Real`.
    pub fn numeric_from_literal(value: &LiteralValue) -> Option<f64> {
        match value {
            LiteralValue::Number(value) => Some(*value),
            LiteralValue::Real {
                text,
                class: NumericClass::Single,
            } => text.parse::<f32>().ok().map(f64::from),
            LiteralValue::Real { text, .. } => text.parse().ok(),
            LiteralValue::Integer { text, .. } => text.parse().ok(),
            _ => None,
        }
    }

    fn numeric_dimension_from_literal(value: &LiteralValue) -> Option<usize> {
        if let LiteralValue::Integer { text, .. } = value {
            return text.parse().ok();
        }
        let value = Self::numeric_from_literal(value)?;
        if !value.is_finite() {
            return None;
        }
        let rounded = value.round();
        ((value - rounded).abs() <= 1e-9 && rounded >= 0.0).then_some(rounded as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real(text: &str, class: NumericClass) -> LiteralValue {
        LiteralValue::Real {
            text: text.into(),
            class,
        }
    }

    #[test]
    fn numeric_projection_respects_represented_single_precision() {
        for (text, represented) in [
            ("1.00000001", 1.0),
            ("16777217", 16777216.0),
            ("1.00000011920928955078125", 1.0 + f64::from(f32::EPSILON)),
        ] {
            let single = real(text, NumericClass::Single);
            assert_eq!(
                LiteralContext::numeric_from_literal(&single),
                Some(represented)
            );
            assert_eq!(
                LiteralContext::new(vec![single]).numeric_at(0),
                Some(represented)
            );
            assert_eq!(
                LiteralContext::numeric_from_literal(&real(text, NumericClass::Double)),
                text.parse::<f64>().ok()
            );
        }
        assert_eq!(
            LiteralContext::numeric_from_literal(&LiteralValue::Number(1.00000001)),
            Some(1.00000001)
        );
        assert_eq!(
            LiteralContext::numeric_from_literal(&LiteralValue::Unknown),
            None
        );
    }

    #[test]
    fn dimension_projection_rounds_single_without_accepting_representable_fractions() {
        let context = LiteralContext::new(vec![LiteralValue::Vector(vec![
            real("1.00000001", NumericClass::Single),
            real("16777217", NumericClass::Single),
            real("1.00000011920928955078125", NumericClass::Single),
        ])]);
        assert_eq!(
            context.numeric_vector_at(0),
            Some(vec![Some(1), Some(16777216), None])
        );
        assert_eq!(context.numeric_dims(), vec![Some(1), Some(16777216), None]);
    }
}
