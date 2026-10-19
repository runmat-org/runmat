//! The one conversion grammar shared by format validation and execution.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FloatStyle {
    FixedLower,
    FixedUpper,
    ExponentLower,
    ExponentUpper,
    GeneralLower,
    GeneralUpper,
}

impl FloatStyle {
    pub(super) fn permits_grouping_or_decimal(self) -> bool {
        matches!(
            self,
            Self::FixedLower | Self::FixedUpper | Self::GeneralLower | Self::GeneralUpper
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FormatConversion {
    SignedDecimal,
    UnsignedDecimal,
    Octal,
    HexLower,
    HexUpper,
    Binary,
    Float(FloatStyle),
    String,
    Character,
}

impl FormatConversion {
    pub(super) fn from_symbol(symbol: char) -> Option<Self> {
        Some(match symbol {
            'd' | 'i' => Self::SignedDecimal,
            'u' => Self::UnsignedDecimal,
            'o' => Self::Octal,
            'x' => Self::HexLower,
            'X' => Self::HexUpper,
            'b' => Self::Binary,
            'f' => Self::Float(FloatStyle::FixedLower),
            'F' => Self::Float(FloatStyle::FixedUpper),
            'e' => Self::Float(FloatStyle::ExponentLower),
            'E' => Self::Float(FloatStyle::ExponentUpper),
            'g' => Self::Float(FloatStyle::GeneralLower),
            'G' => Self::Float(FloatStyle::GeneralUpper),
            's' => Self::String,
            'c' => Self::Character,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_grammar_is_exact_and_typed() {
        for symbol in "diuoxXbfFeEgGsc".chars() {
            assert!(FormatConversion::from_symbol(symbol).is_some());
        }
        for symbol in ['q', 'p', 'a', 'A', '%'] {
            assert!(FormatConversion::from_symbol(symbol).is_none());
        }
        assert_ne!(
            FormatConversion::from_symbol('x'),
            FormatConversion::from_symbol('X')
        );
        assert_ne!(
            FormatConversion::from_symbol('g'),
            FormatConversion::from_symbol('G')
        );
    }
}
