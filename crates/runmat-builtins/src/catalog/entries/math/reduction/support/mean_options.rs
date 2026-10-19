use super::ReductionNaN;
use runmat_types::LiteralValue;

/// Public text controls admitted by the numerical `mean` owner.
///
/// Axes and weight payloads require runtime values, so these identities name
/// only the source-language controls. The owning runtime parser attaches the
/// value to `Weights` and validates dimensions before any execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanControl {
    All,
    Weights,
    Missing(ReductionNaN),
    Output(MeanOutputClass),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanOutputClass {
    Default,
    Double,
    Native,
}

impl MeanControl {
    /// Decode one public option at the source/value boundary. Unknown text is
    /// not a control and remains subject to the owner's ordinary diagnostics.
    pub fn from_public_option(option: &str) -> Option<Self> {
        if let Some(policy) = ReductionNaN::from_public_option(option) {
            return Some(Self::Missing(policy));
        }
        if option.eq_ignore_ascii_case("weights") {
            return Some(Self::Weights);
        }
        if option.eq_ignore_ascii_case("default") {
            return Some(Self::Output(MeanOutputClass::Default));
        }
        if option.eq_ignore_ascii_case("double") {
            return Some(Self::Output(MeanOutputClass::Double));
        }
        if option.eq_ignore_ascii_case("native") {
            return Some(Self::Output(MeanOutputClass::Native));
        }
        // Existing mean admission accepts surrounding whitespace for `all`
        // via its dimension parser, but not for other option words.
        if option.trim().eq_ignore_ascii_case("all") {
            return Some(Self::All);
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
    use super::{MeanControl, MeanOutputClass, ReductionNaN};
    use runmat_types::LiteralValue;

    #[test]
    fn mean_controls_decode_to_typed_identities() {
        assert_eq!(MeanControl::from_public_option(" Weights "), None);
        assert_eq!(
            MeanControl::from_public_option("Weights"),
            Some(MeanControl::Weights)
        );
        assert_eq!(
            MeanControl::from_public_option(" all "),
            Some(MeanControl::All)
        );
        assert_eq!(
            MeanControl::from_public_option("OMITMISSING"),
            Some(MeanControl::Missing(ReductionNaN::Omit))
        );
        assert_eq!(
            MeanControl::from_public_option("native"),
            Some(MeanControl::Output(MeanOutputClass::Native))
        );
        assert_eq!(MeanControl::from_public_option("unknown"), None);
    }

    #[test]
    fn static_literals_share_the_public_decoder() {
        assert_eq!(
            MeanControl::from_literal(&LiteralValue::Character("double".into())),
            Some(MeanControl::Output(MeanOutputClass::Double))
        );
        assert_eq!(MeanControl::from_literal(&LiteralValue::Number(1.0)), None);
    }
}
