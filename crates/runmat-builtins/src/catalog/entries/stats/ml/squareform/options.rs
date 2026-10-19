//! Closed squareform conversion directions; textual spellings are decoded only at the call boundary.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SquareformDirection {
    Auto,
    ToMatrix,
    ToVector,
}

impl SquareformDirection {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        if keyword.eq_ignore_ascii_case("tomatrix") {
            Some(Self::ToMatrix)
        } else if keyword.eq_ignore_ascii_case("tovector") {
            Some(Self::ToVector)
        } else {
            None
        }
    }
}
