use super::ReductionNaN;
use runmat_types::{LiteralContext, LiteralValue};

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
#[derive(Debug, PartialEq, Eq)]
pub(super) enum NormalizationLiteral {
    Value,
    Placeholder,
    Unmatched,
    Invalid,
}

#[derive(Clone, Copy)]
enum NormalizationGrammar {
    AxisFallback,
    RequiredNormalization,
}

pub(super) fn std_normalization_literal(literal: &LiteralValue) -> NormalizationLiteral {
    normalization_literal(literal, NormalizationGrammar::AxisFallback)
}

pub(super) fn var_normalization_literal(literal: &LiteralValue) -> NormalizationLiteral {
    normalization_literal(literal, NormalizationGrammar::RequiredNormalization)
}

fn normalization_literal(
    literal: &LiteralValue,
    grammar: NormalizationGrammar,
) -> NormalizationLiteral {
    match literal {
        LiteralValue::Empty => NormalizationLiteral::Placeholder,
        LiteralValue::Vector(values) if values.is_empty() => NormalizationLiteral::Placeholder,
        LiteralValue::Vector(values) if values.len() == 1 => {
            normalization_literal(&values[0], grammar)
        }
        LiteralValue::Vector(_)
            if matches!(grammar, NormalizationGrammar::RequiredNormalization) =>
        {
            NormalizationLiteral::Invalid
        }
        LiteralValue::Bool(_) => NormalizationLiteral::Value,
        LiteralValue::Number(value) => floating_normalization(*value, grammar),
        LiteralValue::Real { .. } => LiteralContext::numeric_from_literal(literal)
            .map(|value| floating_normalization(value, grammar))
            .unwrap_or(NormalizationLiteral::Invalid),
        LiteralValue::Integer { text, .. } => {
            let value = text.parse::<u128>().ok();
            match value {
                Some(0 | 1) => NormalizationLiteral::Value,
                _ if matches!(grammar, NormalizationGrammar::RequiredNormalization) => {
                    NormalizationLiteral::Invalid
                }
                _ => NormalizationLiteral::Unmatched,
            }
        }
        _ => NormalizationLiteral::Unmatched,
    }
}

fn floating_normalization(value: f64, grammar: NormalizationGrammar) -> NormalizationLiteral {
    if !value.is_finite() {
        return NormalizationLiteral::Invalid;
    }
    match DispersionNormalization::from_finite_flag(value) {
        Some(_) => NormalizationLiteral::Value,
        None if matches!(grammar, NormalizationGrammar::RequiredNormalization) => {
            NormalizationLiteral::Invalid
        }
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
    use super::{
        std_normalization_literal, var_normalization_literal, DispersionControl,
        NormalizationLiteral, ReductionNaN,
    };
    use runmat_types::{LiteralValue, NumericClass};

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

    #[test]
    fn normalization_respects_physical_single_literal_and_owner_grammar() {
        let single = LiteralValue::Real {
            text: "1.00000001".into(),
            class: NumericClass::Single,
        };
        let double = LiteralValue::Real {
            text: "1.00000001".into(),
            class: NumericClass::Double,
        };
        assert_eq!(
            std_normalization_literal(&single),
            NormalizationLiteral::Value
        );
        assert_eq!(
            var_normalization_literal(&single),
            NormalizationLiteral::Value
        );
        assert_eq!(
            std_normalization_literal(&double),
            NormalizationLiteral::Unmatched
        );
        assert_eq!(
            var_normalization_literal(&double),
            NormalizationLiteral::Invalid
        );
    }
}
