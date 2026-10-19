//! Shared typed spelling of factorization options.
//!
//! Runtime value admission remains at each builtin; catalog inference only
//! projects source-known literals through the same spelling boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorOption {
    Matrix,
    Vector,
    Economy,
    Full,
    Balance,
    NoBalance,
}

impl FactorOption {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "matrix" => Some(Self::Matrix),
            "vector" => Some(Self::Vector),
            "econ" | "economy" | "0" => Some(Self::Economy),
            "full" => Some(Self::Full),
            "balance" => Some(Self::Balance),
            "nobalance" => Some(Self::NoBalance),
            _ => None,
        }
    }
}
