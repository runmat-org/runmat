//! Public `format` spellings, decoded once into the display domain type.

use runmat_value::FormatMode;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FormatCommand {
    Numeric(FormatMode),
    Compact,
    Loose,
}

impl FormatCommand {
    pub fn decode(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "short" => Some(Self::Numeric(FormatMode::Short)),
            "long" => Some(Self::Numeric(FormatMode::Long)),
            "shorte" => Some(Self::Numeric(FormatMode::ShortE)),
            "longe" => Some(Self::Numeric(FormatMode::LongE)),
            "shortg" => Some(Self::Numeric(FormatMode::ShortG)),
            "longg" => Some(Self::Numeric(FormatMode::LongG)),
            "rat" | "rational" => Some(Self::Numeric(FormatMode::Rational)),
            "hex" => Some(Self::Numeric(FormatMode::Hex)),
            "compact" => Some(Self::Compact),
            "loose" => Some(Self::Loose),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_public_modes_and_alias_decode_without_reparsing() {
        for label in [
            "short", "long", "shortE", "longE", "shortG", "longG", "rat", "rational", "hex",
            "compact", "loose",
        ] {
            assert!(FormatCommand::decode(label).is_some(), "{label}");
            assert!(FormatCommand::decode(&label.to_ascii_uppercase()).is_some());
        }
        assert_eq!(FormatCommand::decode("unsupported"), None);
    }
}
