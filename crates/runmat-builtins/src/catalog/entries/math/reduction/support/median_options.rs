use super::ReductionNaN;
use runmat_types::LiteralValue;

/// Public text controls owned by numerical `median`. Weight values and
/// dimensions are decoded at the runtime value boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MedianControl {
    All,
    Weights,
    Missing(ReductionNaN),
}

impl MedianControl {
    pub fn from_public_option(option: &str) -> Option<Self> {
        let axis_option = option.trim();
        if let Some(policy) = ReductionNaN::from_public_option(axis_option) {
            return Some(Self::Missing(policy));
        }
        if axis_option.eq_ignore_ascii_case("all") {
            return Some(Self::All);
        }
        if option.eq_ignore_ascii_case("weights") {
            return Some(Self::Weights);
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
    use super::{MedianControl, ReductionNaN};
    use runmat_types::LiteralValue;

    #[test]
    fn median_public_controls_have_one_identity() {
        assert_eq!(
            MedianControl::from_public_option("ALL"),
            Some(MedianControl::All)
        );
        assert_eq!(
            MedianControl::from_public_option(" all "),
            Some(MedianControl::All)
        );
        assert_eq!(
            MedianControl::from_public_option(" omitnan "),
            Some(MedianControl::Missing(ReductionNaN::Omit))
        );
        assert_eq!(MedianControl::from_public_option(" Weights "), None);
        assert_eq!(
            MedianControl::from_public_option("Weights"),
            Some(MedianControl::Weights)
        );
        assert_eq!(
            MedianControl::from_literal(&LiteralValue::Character("omitmissing".into())),
            Some(MedianControl::Missing(ReductionNaN::Omit))
        );
        assert_eq!(MedianControl::from_public_option("native"), None);
    }
}
