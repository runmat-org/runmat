//! Closed identity policy shared by static facts and runtime admission.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MomentStatisticKind {
    MeanAbsoluteDeviation,
    Skewness,
    Kurtosis,
}

impl MomentStatisticKind {
    pub const fn default_flag(self) -> u8 {
        match self {
            Self::MeanAbsoluteDeviation => 0,
            Self::Skewness | Self::Kurtosis => 1,
        }
    }
}
