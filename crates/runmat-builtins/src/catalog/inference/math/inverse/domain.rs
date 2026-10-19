use crate::{InverseHyperbolicFunction, InverseTrigonometricFunction};
use runmat_types::{LiteralValue, NumericDomain};

pub(super) fn inverse_hyperbolic_literal_domain(
    literal: &LiteralValue,
    function: InverseHyperbolicFunction,
) -> Option<NumericDomain> {
    match literal {
        LiteralValue::Number(value) => Some(inverse_hyperbolic_real_domain(*value, function)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => text
            .parse::<f64>()
            .ok()
            .map(|value| inverse_hyperbolic_real_domain(value, function)),
        LiteralValue::Bool(value) => Some(inverse_hyperbolic_real_domain(
            if *value { 1.0 } else { 0.0 },
            function,
        )),
        LiteralValue::Character(value) => combine_numeric_domains(value.chars().map(|character| {
            inverse_hyperbolic_real_domain(f64::from(u32::from(character)), function)
        })),
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Vector(values) => combine_optional_numeric_domains(
            values
                .iter()
                .map(|value| inverse_hyperbolic_literal_domain(value, function)),
        ),
        LiteralValue::Matrix(rows) => combine_optional_numeric_domains(
            rows.iter()
                .flatten()
                .map(|value| inverse_hyperbolic_literal_domain(value, function)),
        ),
        LiteralValue::Empty => Some(NumericDomain::Real),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn inverse_hyperbolic_real_domain(
    value: f64,
    function: InverseHyperbolicFunction,
) -> NumericDomain {
    let is_real = match function {
        InverseHyperbolicFunction::Cosine => value.is_nan() || value >= 1.0,
        InverseHyperbolicFunction::Sine => true,
        InverseHyperbolicFunction::Tangent => value.is_nan() || (-1.0..=1.0).contains(&value),
    };
    if is_real {
        NumericDomain::Real
    } else {
        NumericDomain::Complex
    }
}

fn combine_optional_numeric_domains(
    domains: impl IntoIterator<Item = Option<NumericDomain>>,
) -> Option<NumericDomain> {
    let mut combined = NumericDomain::Real;
    for domain in domains {
        if domain? == NumericDomain::Complex {
            combined = NumericDomain::Complex;
        }
    }
    Some(combined)
}

fn combine_numeric_domains(
    domains: impl IntoIterator<Item = NumericDomain>,
) -> Option<NumericDomain> {
    combine_optional_numeric_domains(domains.into_iter().map(Some))
}

pub(super) fn inverse_trigonometric_literal_domain(
    literal: &LiteralValue,
    function: InverseTrigonometricFunction,
) -> Option<NumericDomain> {
    if function == InverseTrigonometricFunction::Tangent {
        return literal_is_real(literal).then_some(NumericDomain::Real);
    }
    match literal {
        LiteralValue::Number(value) => Some(unit_interval_domain(*value)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse::<f64>().ok().map(unit_interval_domain)
        }
        LiteralValue::Bool(_) => Some(NumericDomain::Real),
        LiteralValue::Character(value) => Some(if value.chars().all(|ch| u32::from(ch) <= 1) {
            NumericDomain::Real
        } else {
            NumericDomain::Complex
        }),
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Vector(values) => combine_inverse_literal_domains(values, function),
        LiteralValue::Matrix(rows) => {
            let values = rows.iter().flatten().cloned().collect::<Vec<_>>();
            combine_inverse_literal_domains(&values, function)
        }
        LiteralValue::Empty => Some(NumericDomain::Real),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn literal_is_real(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Number(_)
        | LiteralValue::Real { .. }
        | LiteralValue::Integer { .. }
        | LiteralValue::Bool(_)
        | LiteralValue::Character(_)
        | LiteralValue::Empty => true,
        LiteralValue::Vector(values) => values.iter().all(literal_is_real),
        LiteralValue::Matrix(rows) => rows.iter().flatten().all(literal_is_real),
        LiteralValue::Complex { .. }
        | LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => false,
    }
}

fn unit_interval_domain(value: f64) -> NumericDomain {
    if value.is_nan() || (-1.0..=1.0).contains(&value) {
        NumericDomain::Real
    } else {
        NumericDomain::Complex
    }
}

fn combine_inverse_literal_domains(
    values: &[LiteralValue],
    function: InverseTrigonometricFunction,
) -> Option<NumericDomain> {
    let mut domain = NumericDomain::Real;
    for value in values {
        let value_domain = inverse_trigonometric_literal_domain(value, function)?;
        if value_domain == NumericDomain::Complex {
            domain = NumericDomain::Complex;
        }
    }
    Some(domain)
}
