#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LscovAlgorithm {
    Orth,
    Chol,
}

impl LscovAlgorithm {
    pub fn from_public_keyword(keyword: &str) -> Option<Self> {
        match keyword {
            "orth" => Some(Self::Orth),
            "chol" => Some(Self::Chol),
            _ => None,
        }
    }
}
