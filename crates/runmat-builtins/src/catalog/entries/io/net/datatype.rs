//! Closed TCP payload datatype identity shared by static and runtime admission.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetDataType {
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Single,
    Double,
    Char,
    String,
}

impl NetDataType {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "uint8" => Some(Self::UInt8),
            "int8" => Some(Self::Int8),
            "uint16" => Some(Self::UInt16),
            "int16" => Some(Self::Int16),
            "uint32" => Some(Self::UInt32),
            "int32" => Some(Self::Int32),
            "uint64" => Some(Self::UInt64),
            "int64" => Some(Self::Int64),
            "single" => Some(Self::Single),
            "double" => Some(Self::Double),
            "char" => Some(Self::Char),
            "string" => Some(Self::String),
            _ => None,
        }
    }

    pub const fn element_size(self) -> usize {
        match self {
            Self::UInt8 | Self::Int8 | Self::Char | Self::String => 1,
            Self::UInt16 | Self::Int16 => 2,
            Self::UInt32 | Self::Int32 | Self::Single => 4,
            Self::UInt64 | Self::Int64 | Self::Double => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NetDataType;

    #[test]
    fn all_supported_text_spellings_have_one_typed_identity() {
        for (text, expected) in [
            (" UINT8 ", NetDataType::UInt8),
            ("int8", NetDataType::Int8),
            ("uint16", NetDataType::UInt16),
            ("int16", NetDataType::Int16),
            ("uint32", NetDataType::UInt32),
            ("int32", NetDataType::Int32),
            ("uint64", NetDataType::UInt64),
            ("int64", NetDataType::Int64),
            ("single", NetDataType::Single),
            ("double", NetDataType::Double),
            ("char", NetDataType::Char),
            ("string", NetDataType::String),
        ] {
            assert_eq!(NetDataType::parse(text), Some(expected));
        }
        assert_eq!(NetDataType::parse(""), None);
        assert_eq!(NetDataType::parse("uint128"), None);
    }
}
