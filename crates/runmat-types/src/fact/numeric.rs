use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericClass {
    Double,
    Single,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

impl NumericClass {
    /// Resolve the canonical source-language name of a built-in numeric class.
    pub fn from_class_name(name: &str) -> Option<Self> {
        let normalized = name.to_ascii_lowercase();
        crate::ClassIdentity::new(normalized)
            .ok()
            .and_then(|identity| Self::from_class_identity(&identity))
    }

    pub fn from_class_identity(identity: &crate::ClassIdentity) -> Option<Self> {
        use crate::standard;

        if identity.is(standard::DOUBLE) {
            Some(Self::Double)
        } else if identity.is(standard::SINGLE) {
            Some(Self::Single)
        } else if identity.is(standard::INT8) {
            Some(Self::Int8)
        } else if identity.is(standard::UINT8) {
            Some(Self::UInt8)
        } else if identity.is(standard::INT16) {
            Some(Self::Int16)
        } else if identity.is(standard::UINT16) {
            Some(Self::UInt16)
        } else if identity.is(standard::INT32) {
            Some(Self::Int32)
        } else if identity.is(standard::UINT32) {
            Some(Self::UInt32)
        } else if identity.is(standard::INT64) {
            Some(Self::Int64)
        } else if identity.is(standard::UINT64) {
            Some(Self::UInt64)
        } else {
            None
        }
    }

    pub const fn class_name(self) -> &'static str {
        match self {
            Self::Double => "double",
            Self::Single => "single",
            Self::Int8 => "int8",
            Self::UInt8 => "uint8",
            Self::Int16 => "int16",
            Self::UInt16 => "uint16",
            Self::Int32 => "int32",
            Self::UInt32 => "uint32",
            Self::Int64 => "int64",
            Self::UInt64 => "uint64",
        }
    }

    /// Width of one real scalar in the class's native representation.
    pub const fn byte_width(self) -> usize {
        match self {
            Self::Double | Self::Int64 | Self::UInt64 => 8,
            Self::Single | Self::Int32 | Self::UInt32 => 4,
            Self::Int16 | Self::UInt16 => 2,
            Self::Int8 | Self::UInt8 => 1,
        }
    }

    pub const fn integer_class(self) -> Option<crate::IntegerClass> {
        use crate::IntegerClass;

        match self {
            Self::Double | Self::Single => None,
            Self::Int8 => Some(IntegerClass::Int8),
            Self::UInt8 => Some(IntegerClass::UInt8),
            Self::Int16 => Some(IntegerClass::Int16),
            Self::UInt16 => Some(IntegerClass::UInt16),
            Self::Int32 => Some(IntegerClass::Int32),
            Self::UInt32 => Some(IntegerClass::UInt32),
            Self::Int64 => Some(IntegerClass::Int64),
            Self::UInt64 => Some(IntegerClass::UInt64),
        }
    }
}

impl From<crate::IntegerClass> for NumericClass {
    fn from(class: crate::IntegerClass) -> Self {
        use crate::IntegerClass;

        match class {
            IntegerClass::Int8 => Self::Int8,
            IntegerClass::UInt8 => Self::UInt8,
            IntegerClass::Int16 => Self::Int16,
            IntegerClass::UInt16 => Self::UInt16,
            IntegerClass::Int32 => Self::Int32,
            IntegerClass::UInt32 => Self::UInt32,
            IntegerClass::Int64 => Self::Int64,
            IntegerClass::UInt64 => Self::UInt64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericDomain {
    Real,
    Complex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericFact {
    pub class: NumericClass,
    pub domain: NumericDomain,
}

#[cfg(test)]
mod tests {
    use super::NumericClass;
    use crate::{standard, ClassIdentity, IntegerClass};

    #[test]
    fn numeric_classes_resolve_from_typed_class_identity() {
        assert_eq!(
            NumericClass::from_class_identity(&standard::UINT64.owned()),
            Some(NumericClass::UInt64)
        );
        assert_eq!(
            NumericClass::from_class_identity(&ClassIdentity::from("table")),
            None
        );
    }

    #[test]
    fn numeric_and_integer_classes_convert_without_names() {
        for integer in [
            IntegerClass::Int8,
            IntegerClass::Int16,
            IntegerClass::Int32,
            IntegerClass::Int64,
            IntegerClass::UInt8,
            IntegerClass::UInt16,
            IntegerClass::UInt32,
            IntegerClass::UInt64,
        ] {
            let numeric = NumericClass::from(integer);
            assert_eq!(numeric.integer_class(), Some(integer));
        }
        assert_eq!(NumericClass::Double.integer_class(), None);
        assert_eq!(NumericClass::Single.integer_class(), None);
    }
}
