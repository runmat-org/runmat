use super::ReductionNaN;
use runmat_types::LiteralValue;

/// Text controls for the two public `min` call families. A comparison-method
/// control consumes a following value; dimensions remain value-dependent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinControl {
    All,
    Linear,
    Missing(ReductionNaN),
    ComparisonMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinComparisonMethod {
    Auto,
    Real,
    Abs,
}

impl MinControl {
    pub fn from_public_option(option: &str) -> Option<Self> {
        if option.eq_ignore_ascii_case("omitnan") {
            return Some(Self::Missing(ReductionNaN::Omit));
        }
        if option.eq_ignore_ascii_case("includenan") {
            return Some(Self::Missing(ReductionNaN::Include));
        }
        if option.eq_ignore_ascii_case("all") {
            return Some(Self::All);
        }
        if option.eq_ignore_ascii_case("linear") {
            return Some(Self::Linear);
        }
        if option.eq_ignore_ascii_case("comparisonmethod") {
            return Some(Self::ComparisonMethod);
        }
        None
    }

    pub fn from_literal(literal: &LiteralValue) -> Option<Self> {
        match literal {
            LiteralValue::String(text)
            | LiteralValue::Character(text)
            | LiteralValue::Keyword(text) => Self::from_public_option(text),
            _ => None,
        }
    }
}

impl MinComparisonMethod {
    pub fn from_public_option(option: &str) -> Option<Self> {
        if option.eq_ignore_ascii_case("auto") {
            Some(Self::Auto)
        } else if option.eq_ignore_ascii_case("real") {
            Some(Self::Real)
        } else if option.eq_ignore_ascii_case("abs") || option.eq_ignore_ascii_case("magnitude") {
            Some(Self::Abs)
        } else {
            None
        }
    }

    pub fn from_literal(literal: &LiteralValue) -> Option<Self> {
        match literal {
            LiteralValue::String(text)
            | LiteralValue::Character(text)
            | LiteralValue::Keyword(text) => Self::from_public_option(text),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MinComparisonMethod, MinControl, ReductionNaN};
    use runmat_types::LiteralValue;

    #[test]
    fn min_reduction_and_pairwise_controls_have_typed_identities() {
        assert_eq!(
            MinControl::from_public_option("omitnan"),
            Some(MinControl::Missing(ReductionNaN::Omit))
        );
        assert_eq!(MinControl::from_public_option("omitmissing"), None);
        assert_eq!(
            MinControl::from_literal(&LiteralValue::Character("linear".into())),
            Some(MinControl::Linear)
        );
        assert_eq!(
            MinControl::from_public_option("comparisonmethod"),
            Some(MinControl::ComparisonMethod)
        );
        assert_eq!(
            MinComparisonMethod::from_public_option("magnitude"),
            Some(MinComparisonMethod::Abs)
        );
        assert_eq!(MinComparisonMethod::from_public_option("unsupported"), None);
    }
}
