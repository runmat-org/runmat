use runmat_types::{LiteralValue, NumericDomain};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RootLiteralDomain {
    Nonnegative,
    Negative,
    Complex,
}

impl RootLiteralDomain {
    pub(super) const fn principal_output(self) -> NumericDomain {
        match self {
            Self::Nonnegative => NumericDomain::Real,
            Self::Negative | Self::Complex => NumericDomain::Complex,
        }
    }
}

pub(super) fn classify(literal: &LiteralValue) -> Option<RootLiteralDomain> {
    match literal {
        LiteralValue::Number(value) => Some(classify_real(*value)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse().ok().map(classify_real)
        }
        LiteralValue::Complex { .. } => Some(RootLiteralDomain::Complex),
        LiteralValue::Bool(_) | LiteralValue::Character(_) | LiteralValue::Empty => {
            Some(RootLiteralDomain::Nonnegative)
        }
        LiteralValue::Vector(values) => classify_sequence(values.iter()),
        LiteralValue::Matrix(rows) => classify_sequence(rows.iter().flatten()),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn classify_sequence<'a>(
    values: impl Iterator<Item = &'a LiteralValue>,
) -> Option<RootLiteralDomain> {
    let mut domain = RootLiteralDomain::Nonnegative;
    for value in values {
        match classify(value)? {
            RootLiteralDomain::Complex => return Some(RootLiteralDomain::Complex),
            RootLiteralDomain::Negative => domain = RootLiteralDomain::Negative,
            RootLiteralDomain::Nonnegative => {}
        }
    }
    Some(domain)
}

fn classify_real(value: f64) -> RootLiteralDomain {
    if value < 0.0 {
        RootLiteralDomain::Negative
    } else {
        RootLiteralDomain::Nonnegative
    }
}
