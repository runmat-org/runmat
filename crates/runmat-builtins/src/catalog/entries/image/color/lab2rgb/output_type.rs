use runmat_types::NumericClass;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LabOutputType {
    Double,
    Single,
    UInt8,
    UInt16,
}

impl LabOutputType {
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("double") {
            Some(Self::Double)
        } else if name.eq_ignore_ascii_case("single") {
            Some(Self::Single)
        } else if name.eq_ignore_ascii_case("uint8") {
            Some(Self::UInt8)
        } else if name.eq_ignore_ascii_case("uint16") {
            Some(Self::UInt16)
        } else {
            None
        }
    }

    pub const fn class(self) -> NumericClass {
        match self {
            Self::Double => NumericClass::Double,
            Self::Single => NumericClass::Single,
            Self::UInt8 => NumericClass::UInt8,
            Self::UInt16 => NumericClass::UInt16,
        }
    }
}
