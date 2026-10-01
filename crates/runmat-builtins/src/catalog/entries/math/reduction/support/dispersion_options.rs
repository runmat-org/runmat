use super::ReductionNaN;
use runmat_types::LiteralValue;

/// Sample versus population normalization shared by standard deviation and
/// variance. Their value admission and public errors remain owner-specific.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispersionNormalization {
    Sample,
    Population,
}

impl DispersionNormalization {
    pub fn from_finite_flag(value: f64) -> Option<Self> {
        if (value - 0.0).abs() < f64::EPSILON {
            Some(Self::Sample)
        } else if (value - 1.0).abs() < f64::EPSILON {
            Some(Self::Population)
        } else {
            None
        }
    }
}

/// Literal projection of the owners' value-level normalization admission.
/// `Unmatched` may still be an axis; `Invalid` must not prove output facts.
pub(super) enum NormalizationLiteral {
    Value,
    Placeholder,
    Unmatched,
    Invalid,
}

pub(super) fn std_normalization_literal(literal: &LiteralValue) -> NormalizationLiteral {
    normalization_literal(literal, false)
}

pub(super) fn var_normalization_literal(literal: &LiteralValue) -> NormalizationLiteral {
    normalization_literal(literal, true)
}

fn normalization_literal(literal: &LiteralValue, strict: bool) -> NormalizationLiteral {
    match literal {
        LiteralValue::Empty => NormalizationLiteral::Placeholder,
        LiteralValue::Vector(values) if values.is_empty() => NormalizationLiteral::Placeholder,
        LiteralValue::Vector(values) if values.len() == 1 => {
            normalization_literal(&values[0], strict)
        }
        LiteralValue::Vector(_) if strict => NormalizationLiteral::Invalid,
        LiteralValue::Bool(_) => NormalizationLiteral::Value,
        LiteralValue::Number(value) => floating_normalization(*value, strict),
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .map(|value| floating_normalization(value, strict))
            .unwrap_or(NormalizationLiteral::Invalid),
        LiteralValue::Integer { text, .. } => {
            let value = text.parse::<u128>().ok();
            match value {
                Some(0 | 1) => NormalizationLiteral::Value,
                _ if strict => NormalizationLiteral::Invalid,
                _ => NormalizationLiteral::Unmatched,
            }
        }
        _ => NormalizationLiteral::Unmatched,
    }
}

fn floating_normalization(value: f64, strict: bool) -> NormalizationLiteral {
    if !value.is_finite() {
        return NormalizationLiteral::Invalid;
    }
    match DispersionNormalization::from_finite_flag(value) {
        Some(_) => NormalizationLiteral::Value,
        None if strict => NormalizationLiteral::Invalid,
        None => NormalizationLiteral::Unmatched,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispersionControl {
    All,
    Missing(ReductionNaN),
}

impl DispersionControl {
    pub fn from_std_public_option(option: &str) -> Option<Self> {
        if let Some(policy) = ReductionNaN::from_public_option(option) {
            return Some(Self::Missing(policy));
        }
        // The std axis parser has historically accepted surrounding spaces
        // for `all`, unlike the var parser.
        option
            .trim()
            .eq_ignore_ascii_case("all")
            .then_some(Self::All)
    }

    pub fn from_var_public_option(option: &str) -> Option<Self> {
        if let Some(policy) = ReductionNaN::from_public_option(option) {
            return Some(Self::Missing(policy));
        }
        option.eq_ignore_ascii_case("all").then_some(Self::All)
    }

    pub fn from_std_literal(literal: &LiteralValue) -> Option<Self> {
        text_literal(literal).and_then(Self::from_std_public_option)
    }

    pub fn from_var_literal(literal: &LiteralValue) -> Option<Self> {
        text_literal(literal).and_then(Self::from_var_public_option)
    }
}

fn text_literal(literal: &LiteralValue) -> Option<&str> {
    match literal {
        LiteralValue::String(text)
        | LiteralValue::Character(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{DispersionControl, ReductionNaN};
    use runmat_types::LiteralValue;

    #[test]
    fn owner_spellings_map_to_shared_typed_controls() {
        assert_eq!(
            DispersionControl::from_std_public_option(" all "),
            Some(DispersionControl::All)
        );
        assert_eq!(DispersionControl::from_var_public_option(" all "), None);
        assert_eq!(
            DispersionControl::from_var_literal(&LiteralValue::Character("OMITMISSING".into())),
            Some(DispersionControl::Missing(ReductionNaN::Omit))
        );
        assert_eq!(DispersionControl::from_std_public_option("native"), None);
    }
}
