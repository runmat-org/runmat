//! Shared exact positive-dimension literal admission for dot and cross.

use runmat_types::{LiteralContext, LiteralValue};

pub(super) fn literal_positive_dimension(value: &LiteralValue) -> Option<usize> {
    match value {
        LiteralValue::Integer { .. } => value
            .integer_value_i128()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0),
        other => LiteralContext::numeric_from_literal(other).and_then(|n| {
            (n.is_finite() && n >= 1.0 && n.fract() == 0.0 && n < usize::MAX as f64)
                .then_some(n as usize)
        }),
    }
}
