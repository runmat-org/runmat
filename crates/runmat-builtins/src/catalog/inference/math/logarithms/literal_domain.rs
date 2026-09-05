use runmat_types::{LiteralValue, NumericDomain};

pub(super) fn infer(literal: &LiteralValue, real_boundary: f64) -> Option<NumericDomain> {
    match literal {
        LiteralValue::Number(value) => Some(real_domain(*value, real_boundary)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => text
            .parse()
            .ok()
            .map(|value| real_domain(value, real_boundary)),
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Bool(_) | LiteralValue::Character(_) | LiteralValue::Empty => {
            Some(NumericDomain::Real)
        }
        LiteralValue::Vector(values) => sequence(values.iter(), real_boundary),
        LiteralValue::Matrix(rows) => sequence(rows.iter().flatten(), real_boundary),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn sequence<'a>(
    values: impl Iterator<Item = &'a LiteralValue>,
    real_boundary: f64,
) -> Option<NumericDomain> {
    let mut domain = NumericDomain::Real;
    for value in values {
        if infer(value, real_boundary)? == NumericDomain::Complex {
            domain = NumericDomain::Complex;
        }
    }
    Some(domain)
}

fn real_domain(value: f64, real_boundary: f64) -> NumericDomain {
    if value < real_boundary {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_boundary_controls_real_to_complex_promotion() {
        assert_eq!(
            infer(&LiteralValue::Number(-0.5), 0.0),
            Some(NumericDomain::Complex)
        );
        assert_eq!(
            infer(&LiteralValue::Number(-0.5), -1.0),
            Some(NumericDomain::Real)
        );
        assert_eq!(
            infer(&LiteralValue::Number(-1.5), -1.0),
            Some(NumericDomain::Complex)
        );
    }
}
