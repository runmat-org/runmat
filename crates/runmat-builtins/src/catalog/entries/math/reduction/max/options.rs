use super::ReductionNaN;
use runmat_types::LiteralValue;

/// Text controls for the two public `max` call families. A comparison-method
/// control consumes a following value; dimensions remain value-dependent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxControl {
    All,
    Linear,
    Missing(ReductionNaN),
    ComparisonMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxComparisonMethod {
    Auto,
    Real,
    Abs,
}

impl MaxControl {
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

impl MaxComparisonMethod {
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
    use super::{MaxComparisonMethod, MaxControl, ReductionNaN};
    use runmat_types::LiteralValue;

    #[test]
    fn max_reduction_and_pairwise_controls_have_typed_identities() {
        assert_eq!(
            MaxControl::from_public_option("omitnan"),
            Some(MaxControl::Missing(ReductionNaN::Omit))
        );
        assert_eq!(MaxControl::from_public_option("omitmissing"), None);
        assert_eq!(
            MaxControl::from_literal(&LiteralValue::Character("linear".into())),
            Some(MaxControl::Linear)
        );
        assert_eq!(
            MaxControl::from_public_option("comparisonmethod"),
            Some(MaxControl::ComparisonMethod)
        );
        assert_eq!(
            MaxComparisonMethod::from_public_option("magnitude"),
            Some(MaxComparisonMethod::Abs)
        );
        assert_eq!(MaxComparisonMethod::from_public_option("unsupported"), None);
    }
}
