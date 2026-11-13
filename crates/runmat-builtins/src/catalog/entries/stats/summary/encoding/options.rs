//! One public decoder for categorical encoding option and output identities.
use runmat_types::NumericClass;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncodingOptionName {
    Classes,
    OutputType,
}

impl EncodingOptionName {
    pub fn from_public(text: &str) -> Option<Self> {
        match normalize(text).as_str() {
            "classnames" | "classes" => Some(Self::Classes),
            "outputtype" | "typename" => Some(Self::OutputType),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncodingOutputKind {
    Numeric(NumericClass),
    Logical,
    String,
    Cell,
    Categorical,
}

impl EncodingOutputKind {
    pub fn from_public(text: &str) -> Option<Self> {
        Some(match normalize(text).as_str() {
            "double" => Self::Numeric(NumericClass::Double),
            "single" => Self::Numeric(NumericClass::Single),
            "int8" => Self::Numeric(NumericClass::Int8),
            "int16" => Self::Numeric(NumericClass::Int16),
            "int32" => Self::Numeric(NumericClass::Int32),
            "int64" => Self::Numeric(NumericClass::Int64),
            "uint8" => Self::Numeric(NumericClass::UInt8),
            "uint16" => Self::Numeric(NumericClass::UInt16),
            "uint32" => Self::Numeric(NumericClass::UInt32),
            "uint64" => Self::Numeric(NumericClass::UInt64),
            "logical" => Self::Logical,
            "string" => Self::String,
            "cell" | "cellstr" => Self::Cell,
            "categorical" => Self::Categorical,
            _ => return None,
        })
    }
}

pub(super) fn normalize(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_aliases_decode_to_typed_identities() {
        assert_eq!(
            EncodingOptionName::from_public("Output_Type"),
            Some(EncodingOptionName::OutputType)
        );
        assert_eq!(
            EncodingOutputKind::from_public("cellstr"),
            Some(EncodingOutputKind::Cell)
        );
        assert_eq!(
            EncodingOutputKind::from_public("UINT64"),
            Some(EncodingOutputKind::Numeric(NumericClass::UInt64))
        );
        assert_eq!(EncodingOutputKind::from_public("unknown"), None);
    }
}
