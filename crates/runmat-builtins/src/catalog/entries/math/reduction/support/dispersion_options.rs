use super::ReductionNaN;
use runmat_types::LiteralValue;

/// Sample versus population normalization shared by standard deviation and
/// variance. Their value admission and public errors remain owner-specific.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispersionNormalization {
    Sample,
    Population,
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
