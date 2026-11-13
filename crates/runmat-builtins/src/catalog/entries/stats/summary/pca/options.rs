#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcaOptionName {
    Algorithm,
    Centered,
    Economy,
    NumComponents,
    Rows,
}
impl PcaOptionName {
    pub fn from_public(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("Algorithm") {
            Some(Self::Algorithm)
        } else if name.eq_ignore_ascii_case("Centered") {
            Some(Self::Centered)
        } else if name.eq_ignore_ascii_case("Economy") {
            Some(Self::Economy)
        } else if name.eq_ignore_ascii_case("NumComponents") {
            Some(Self::NumComponents)
        } else if name.eq_ignore_ascii_case("Rows") {
            Some(Self::Rows)
        } else {
            None
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcaRows {
    Complete,
    All,
}
impl PcaRows {
    pub fn from_public(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("complete") {
            Some(Self::Complete)
        } else if name.eq_ignore_ascii_case("all") {
            Some(Self::All)
        } else {
            None
        }
    }
}
