//! Stored optimization-variable domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizationVariableDomain {
    Continuous,
    Integer,
    SemiContinuous,
    SemiInteger,
}
impl OptimizationVariableDomain {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "continuous" => Some(Self::Continuous),
            "integer" => Some(Self::Integer),
            "semi-continuous" => Some(Self::SemiContinuous),
            "semi-integer" => Some(Self::SemiInteger),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Continuous => "continuous",
            Self::Integer => "integer",
            Self::SemiContinuous => "semi-continuous",
            Self::SemiInteger => "semi-integer",
        }
    }
    pub const fn is_semi(self) -> bool {
        matches!(self, Self::SemiContinuous | Self::SemiInteger)
    }
}
