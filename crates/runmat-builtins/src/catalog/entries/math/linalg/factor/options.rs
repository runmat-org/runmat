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
        let text = text.trim();
        [
            ("matrix", Self::Matrix),
            ("vector", Self::Vector),
            ("econ", Self::Economy),
            ("economy", Self::Economy),
            ("0", Self::Economy),
            ("full", Self::Full),
            ("balance", Self::Balance),
            ("nobalance", Self::NoBalance),
        ]
        .into_iter()
        .find_map(|(name, option)| text.eq_ignore_ascii_case(name).then_some(option))
    }
}
