//! Shared pure port literal admission for TCP constructors.

use runmat_types::{LiteralValue, NumericClass};

pub(super) fn known_invalid_port(literal: &LiteralValue) -> bool {
    let value = match literal {
        LiteralValue::Integer { .. } => {
            return literal
                .integer_value_i128()
                .is_none_or(|n| !(0..=65_535).contains(&n));
        }
        LiteralValue::Number(n) => *n,
        LiteralValue::Real {
            text,
            class: NumericClass::Double,
        } => return text.parse::<f64>().ok().is_none_or(invalid_float_port),
        _ => return false,
    };
    invalid_float_port(value)
}

fn invalid_float_port(value: f64) -> bool {
    !value.is_finite() || value.fract() != 0.0 || !(0.0..=65_535.0).contains(&value)
}

pub(super) fn known_invalid_timeout(literal: &LiteralValue, allow_infinity: bool) -> bool {
    match literal {
        LiteralValue::Integer { .. } => literal.integer_value_i128().is_none_or(|n| n < 0),
        LiteralValue::Number(value) => invalid_timeout(*value, allow_infinity),
        LiteralValue::Real {
            text,
            class: NumericClass::Double,
        } => text
            .parse::<f64>()
            .ok()
            .is_none_or(|value| invalid_timeout(value, allow_infinity)),
        _ => false,
    }
}

fn invalid_timeout(value: f64, allow_infinity: bool) -> bool {
    value.is_nan() || value < 0.0 || (!allow_infinity && !value.is_finite())
}
