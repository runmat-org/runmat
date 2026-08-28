use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntegerLiteralClass {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl IntegerLiteralClass {
    pub fn bit_width(self) -> usize {
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
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntegerLiteral {
    text: String,
    bits: u64,
    class: IntegerLiteralClass,
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
            ("u64", IntegerLiteralClass::UInt64),
            ("s64", IntegerLiteralClass::Int64),
            ("u32", IntegerLiteralClass::UInt32),
            ("s32", IntegerLiteralClass::Int32),
            ("u16", IntegerLiteralClass::UInt16),
            ("s16", IntegerLiteralClass::Int16),
            ("u8", IntegerLiteralClass::UInt8),
            ("s8", IntegerLiteralClass::Int8),
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
            .map(IntegerLiteralClass::bit_width)
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
                IntegerLiteralClass::UInt8
            } else if u16::try_from(bits).is_ok() {
                IntegerLiteralClass::UInt16
            } else if u32::try_from(bits).is_ok() {
                IntegerLiteralClass::UInt32
            } else {
                IntegerLiteralClass::UInt64
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

    pub fn class(&self) -> IntegerLiteralClass {
        self.class
    }
}

#[cfg(test)]
mod tests {
    use super::{IntegerLiteral, IntegerLiteralClass};

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
