use runmat_types::{LiteralValue, NumericClass};

/// Reject only source-known invalid prefix counts. Both runtime routes read
/// fixed-width integers through their typed storage, never through f64.
pub(in crate::catalog::entries::strings::core) fn known_invalid_prefix_count(
    value: &LiteralValue,
) -> bool {
    match value {
        LiteralValue::Number(value) => invalid_float(*value),
        LiteralValue::Real {
            text,
            class: NumericClass::Double,
        } => text.parse::<f64>().is_ok_and(invalid_float),
        LiteralValue::Real {
            text,
            class: NumericClass::Single,
        } => text
            .parse::<f32>()
            .is_ok_and(|value| invalid_float(f64::from(value))),
        LiteralValue::Vector(elements) if elements.len() == 1 => known_invalid_scalar(&elements[0]),
        LiteralValue::Matrix(rows) if rows.len() == 1 && rows[0].len() == 1 => {
            known_invalid_scalar(&rows[0][0])
        }
        LiteralValue::Vector(_) | LiteralValue::Matrix(_) | LiteralValue::Empty => true,
        _ => false,
    }
}

fn known_invalid_scalar(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Number(_)
        | LiteralValue::Real { .. }
        | LiteralValue::Integer { .. }
        | LiteralValue::Bool(_) => known_invalid_prefix_count(value),
        LiteralValue::Unknown => false,
        _ => true,
    }
}

fn invalid_float(value: f64) -> bool {
    if !value.is_finite() || value.fract() != 0.0 {
        return true;
    }
    value >= 0.0 && (value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn represented_float_count_checks_exact_integrality_and_host_range() {
        for value in [
            LiteralValue::Number(1.0 + f64::EPSILON),
            LiteralValue::Number(-0.5),
            LiteralValue::Number(f64::NAN),
            LiteralValue::Number(f64::INFINITY),
            LiteralValue::Number(1e100),
            LiteralValue::Number(usize::MAX as f64 + 1.0),
            LiteralValue::Real {
                text: "1.0000001".into(),
                class: NumericClass::Single,
            },
            LiteralValue::Matrix(vec![vec![LiteralValue::Number(2.5)]]),
            LiteralValue::Vector(vec![LiteralValue::Number(2.5)]),
            LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(2.0)]),
        ] {
            assert!(known_invalid_prefix_count(&value), "{value:?}");
        }
        if usize::BITS == 64 {
            assert!(known_invalid_prefix_count(&LiteralValue::Number(
                usize::MAX as f64
            )));
        }
        for value in [
            LiteralValue::Number(-1.0),
            LiteralValue::Number(0.0),
            LiteralValue::Real {
                text: "1.0".into(),
                class: NumericClass::Double,
            },
            LiteralValue::Integer {
                text: "18446744073709551615".into(),
                class: NumericClass::UInt64,
            },
            LiteralValue::Bool(true),
            LiteralValue::Unknown,
        ] {
            assert!(!known_invalid_prefix_count(&value), "{value:?}");
        }
    }
}
