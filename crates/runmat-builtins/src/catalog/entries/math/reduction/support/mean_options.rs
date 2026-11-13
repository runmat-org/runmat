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
        Self::from_public_characters(option.chars())
    }

    /// Decode borrowed Unicode scalars without allocating an option string.
    pub fn from_public_characters(
        option: impl DoubleEndedIterator<Item = char> + Clone,
    ) -> Option<Self> {
        Self::from_public_units(option.map(u32::from))
    }

    /// Decode borrowed UTF-16 units. Surrogates and NUL cannot match controls.
    pub fn from_public_code_units(
        option: impl DoubleEndedIterator<Item = u16> + Clone,
    ) -> Option<Self> {
        Self::from_public_units(option.map(u32::from))
    }

    fn from_public_units(mut option: impl DoubleEndedIterator<Item = u32> + Clone) -> Option<Self> {
        let matches = |word: &str| {
            option
                .clone()
                .map(|unit| {
                    if unit <= u32::from(b'Z') && unit >= u32::from(b'A') {
                        unit + u32::from(b'a' - b'A')
                    } else {
                        unit
                    }
                })
                .eq(word.bytes().map(u32::from))
        };
        for (word, control) in [
            ("weights", Self::Weights),
            ("default", Self::Output(MeanOutputClass::Default)),
            ("double", Self::Output(MeanOutputClass::Double)),
            ("native", Self::Output(MeanOutputClass::Native)),
            ("omitnan", Self::Missing(ReductionNaN::Omit)),
            ("omitmissing", Self::Missing(ReductionNaN::Omit)),
            ("includenan", Self::Missing(ReductionNaN::Include)),
            ("includemissing", Self::Missing(ReductionNaN::Include)),
        ] {
            if matches(word) {
                return Some(control);
            }
        }
        // Existing mean admission accepts surrounding whitespace for `all`
        // via its dimension parser, but not for other option words.
        let whitespace = |unit| char::from_u32(unit).is_some_and(char::is_whitespace);
        while option.clone().next().is_some_and(whitespace) {
            option.next();
        }
        while option.clone().next_back().is_some_and(whitespace) {
            option.next_back();
        }
        option
            .map(|unit| match unit {
                65..=90 => unit + 32,
                _ => unit,
            })
            .eq("all".bytes().map(u32::from))
            .then_some(Self::All)
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
        for word in [
            "weights",
            "default",
            "double",
            "native",
            "omitnan",
            "omitmissing",
            "includenan",
            "includemissing",
            "all",
            "unknown",
            "",
            "\0all",
        ] {
            for text in [
                word.to_owned(),
                word.to_ascii_uppercase(),
                format!("\u{2003}{word}\u{a0}"),
            ] {
                let expected = MeanControl::from_public_option(&text);
                if text.starts_with('\u{2003}') {
                    assert_eq!(expected, (word == "all").then_some(MeanControl::All));
                }
                assert_eq!(MeanControl::from_public_characters(text.chars()), expected);
                let units: Vec<_> = text.encode_utf16().collect();
                assert_eq!(
                    MeanControl::from_public_code_units(units.iter().copied()),
                    expected
                );
                assert_eq!(
                    match expected {
                        Some(MeanControl::Missing(policy)) => Some(policy),
                        _ => None,
                    },
                    ReductionNaN::from_public_option(&text)
                );
            }
        }
        for units in [
            &[0xD800, 97, 108, 108][..],
            &[97, 108, 108, 0xDC00],
            &[0, 97, 108, 108],
        ] {
            assert_eq!(
                MeanControl::from_public_code_units(units.iter().copied()),
                None
            );
        }
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
