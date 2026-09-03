use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntegerClass {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl IntegerClass {
    pub const fn bit_width(self) -> u32 {
        match self {
            Self::Int8 | Self::UInt8 => 8,
            Self::Int16 | Self::UInt16 => 16,
            Self::Int32 | Self::UInt32 => 32,
            Self::Int64 | Self::UInt64 => 64,
        }
    }

    pub fn class_name(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
        }
    }

    pub fn from_class_name(name: &str) -> Option<Self> {
        crate::NumericClass::from_class_name(name)?.integer_class()
    }

    pub const fn is_signed(self) -> bool {
        matches!(self, Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64)
    }

    pub const fn bit_mask(self) -> u64 {
        match self.bit_width() {
            64 => u64::MAX,
            width => (1_u64 << width) - 1,
        }
    }

    pub const fn range(self) -> (i128, i128) {
        match self {
            Self::Int8 => (i8::MIN as i128, i8::MAX as i128),
            Self::Int16 => (i16::MIN as i128, i16::MAX as i128),
            Self::Int32 => (i32::MIN as i128, i32::MAX as i128),
            Self::Int64 => (i64::MIN as i128, i64::MAX as i128),
            Self::UInt8 => (0, u8::MAX as i128),
            Self::UInt16 => (0, u16::MAX as i128),
            Self::UInt32 => (0, u32::MAX as i128),
            Self::UInt64 => (0, u64::MAX as i128),
        }
    }
}

/// Backward-compatible source name for the class carried by a typed integer literal.
///
/// Integer class identity is shared beyond parsing; new code should use
/// [`IntegerClass`] directly.
pub type IntegerLiteralClass = IntegerClass;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntegerLiteral {
    text: String,
    bits: u64,
    class: IntegerClass,
}

impl IntegerLiteral {
    pub fn parse(text: &str) -> Result<Self, String> {
        let (radix, body, kind) =
            if let Some(body) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
                (16, body, "hexadecimal")
            } else if let Some(body) = text.strip_prefix("0b").or_else(|| text.strip_prefix("0B")) {
                (2, body, "binary")
            } else {
                return Err("integer literal must begin with 0x or 0b".to_string());
            };

        let suffixes = [
            ("u64", IntegerClass::UInt64),
            ("s64", IntegerClass::Int64),
            ("u32", IntegerClass::UInt32),
            ("s32", IntegerClass::Int32),
            ("u16", IntegerClass::UInt16),
            ("s16", IntegerClass::Int16),
            ("u8", IntegerClass::UInt8),
            ("s8", IntegerClass::Int8),
        ];
        let (digits, explicit_class) = suffixes
            .iter()
            .find_map(|(suffix, class)| body.strip_suffix(suffix).map(|digits| (digits, *class)))
            .map_or((body, None), |(digits, class)| (digits, Some(class)));

        if digits.is_empty() {
            return Err(format!("{kind} literal requires at least one digit"));
        }
        let valid_digits = match radix {
            16 => digits.bytes().all(|byte| byte.is_ascii_hexdigit()),
            2 => digits.bytes().all(|byte| matches!(byte, b'0' | b'1')),
            _ => unreachable!(),
        };
        if !valid_digits {
            return Err(format!("invalid digit or type suffix in {kind} literal"));
        }

        let max_digits = explicit_class
            .map(|class| class.bit_width() as usize)
            .unwrap_or(64);
        let max_digits = if radix == 16 {
            max_digits.div_ceil(4)
        } else {
            max_digits
        };
        if digits.len() > max_digits {
            let qualifier = if explicit_class.is_some() {
                " for specified type suffix"
            } else {
                ""
            };
            return Err(format!("{kind} literal has too many digits{qualifier}"));
        }

        let bits = u64::from_str_radix(digits, radix)
            .map_err(|_| format!("{kind} literal has too many digits"))?;
        let class = explicit_class.unwrap_or_else(|| {
            if u8::try_from(bits).is_ok() {
                IntegerClass::UInt8
            } else if u16::try_from(bits).is_ok() {
                IntegerClass::UInt16
            } else if u32::try_from(bits).is_ok() {
                IntegerClass::UInt32
            } else {
                IntegerClass::UInt64
            }
        });

        Ok(Self {
            text: text.to_string(),
            bits,
            class,
        })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn bits(&self) -> u64 {
        self.bits
    }

    pub fn class(&self) -> IntegerClass {
        self.class
    }
}

#[cfg(test)]
mod tests {
    use super::{IntegerClass, IntegerLiteral, IntegerLiteralClass};

    #[test]
    fn integer_class_is_shared_with_literal_compatibility_name() {
        let compatibility_name: IntegerLiteralClass = IntegerClass::UInt64;
        assert_eq!(compatibility_name, IntegerClass::UInt64);
        assert_eq!(
            IntegerClass::from_class_name("UINT64"),
            Some(compatibility_name)
        );
        assert_eq!(compatibility_name.bit_width(), 64);
        assert_eq!(compatibility_name.bit_mask(), u64::MAX);
        assert_eq!(compatibility_name.range(), (0, u64::MAX as i128));
        assert!(!compatibility_name.is_signed());
    }

    #[test]
    fn parses_exact_classes_and_bit_patterns() {
        for (text, class, bits) in [
            ("0x2A", IntegerLiteralClass::UInt8, 42),
            ("0X100", IntegerLiteralClass::UInt16, 256),
            ("0b1u64", IntegerLiteralClass::UInt64, 1),
            ("0xFFs8", IntegerLiteralClass::Int8, 255),
            (
                "0xFFFFFFFFFFFFFFFFs64",
                IntegerLiteralClass::Int64,
                u64::MAX,
            ),
        ] {
            let literal = IntegerLiteral::parse(text).expect(text);
            assert_eq!(literal.class(), class);
            assert_eq!(literal.bits(), bits);
            assert_eq!(literal.text(), text);
        }
    }

    #[test]
    fn rejects_invalid_digits_suffixes_and_widths() {
        for text in [
            "0x",
            "0b",
            "0xGG",
            "0b102",
            "0xFFu9",
            "0x100u8",
            "0b100000000s8",
            "0x10000000000000000",
        ] {
            assert!(IntegerLiteral::parse(text).is_err(), "{text}");
        }
    }
}
