/// Public spelling boundary for the two supported JSON encoding controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonEncodeOptionName {
    PrettyPrint,
    ConvertInfAndNaN,
}

impl JsonEncodeOptionName {
    pub fn from_public_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("PrettyPrint") {
            Some(Self::PrettyPrint)
        } else if name.eq_ignore_ascii_case("ConvertInfAndNaN") {
            Some(Self::ConvertInfAndNaN)
        } else {
            None
        }
    }
}
