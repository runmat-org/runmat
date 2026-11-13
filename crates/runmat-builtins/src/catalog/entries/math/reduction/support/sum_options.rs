use super::ReductionNaN;
use runmat_types::LiteralValue;
pub use runmat_types::SumOutputClass;

/// Public option identities owned by the numerical `sum` reduction.
/// Dimension payloads remain value-dependent and are admitted by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumControl {
    All,
    Missing(ReductionNaN),
    Output(SumOutputClass),
}

impl SumControl {
    /// Decode only the public text boundary; execution carries this identity.
    pub fn from_public_option(option: &str) -> Option<Self> {
        if let Some(policy) = ReductionNaN::from_public_option(option) {
            return Some(Self::Missing(policy));
        }
        if option.eq_ignore_ascii_case("all") {
            return Some(Self::All);
        }
        if option.eq_ignore_ascii_case("default") {
            return Some(Self::Output(SumOutputClass::Default));
        }
        if option.eq_ignore_ascii_case("double") {
            return Some(Self::Output(SumOutputClass::Double));
        }
        if option.eq_ignore_ascii_case("native") {
            return Some(Self::Output(SumOutputClass::Native));
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

#[cfg(test)]
mod tests {
    use super::{ReductionNaN, SumControl, SumOutputClass};
    use runmat_types::LiteralValue;

    #[test]
    fn sum_controls_have_one_public_decoder() {
        assert_eq!(SumControl::from_public_option("all"), Some(SumControl::All));
        assert_eq!(SumControl::from_public_option(" ALL "), None);
        assert_eq!(
            SumControl::from_public_option("OMITMISSING"),
            Some(SumControl::Missing(ReductionNaN::Omit))
        );
        assert_eq!(
            SumControl::from_public_option("default"),
            Some(SumControl::Output(SumOutputClass::Default))
        );
        assert_eq!(
            SumControl::from_literal(&LiteralValue::Character("native".into())),
            Some(SumControl::Output(SumOutputClass::Native))
        );
        assert_eq!(SumControl::from_public_option("unknown"), None);
    }
}
