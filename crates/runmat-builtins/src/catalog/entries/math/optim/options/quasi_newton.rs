//! Distinct modern/legacy inverse-update value domains.
use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimQuasiNewtonUpdate {
    Bfgs,
    LimitedMemory { history: NonZeroUsize },
    Dfp,
    Steepest,
}
impl OptimQuasiNewtonUpdate {
    pub fn parse(text: &str, legacy: bool) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "bfgs" => Some(Self::Bfgs),
            "lbfgs" => Some(Self::LimitedMemory {
                history: NonZeroUsize::new(10).expect("positive default"),
            }),
            "dfp" if legacy => Some(Self::Dfp),
            "steepdesc" if legacy => Some(Self::Steepest),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bfgs => "bfgs",
            Self::LimitedMemory { .. } => "lbfgs",
            Self::Dfp => "dfp",
            Self::Steepest => "steepdesc",
        }
    }
    pub const fn algorithm(self) -> &'static str {
        match self {
            Self::Bfgs => "quasi-newton bfgs",
            Self::LimitedMemory { .. } => "quasi-newton lbfgs",
            Self::Dfp => "quasi-newton dfp",
            Self::Steepest => "steepest descent",
        }
    }
    pub const fn dense(self) -> bool {
        matches!(self, Self::Bfgs | Self::Dfp)
    }
}
