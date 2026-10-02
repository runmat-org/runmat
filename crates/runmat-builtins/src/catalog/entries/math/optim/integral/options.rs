//! Public option spellings are decoded once into quadrature control identities.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegralOptionName {
    AbsTol,
    RelTol,
    MaxFunEvals,
    MaxIntervalCount,
    ArrayValued,
}

impl IntegralOptionName {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "abstol" => Some(Self::AbsTol),
            "reltol" => Some(Self::RelTol),
            "maxfunevals" => Some(Self::MaxFunEvals),
            "maxintervalcount" => Some(Self::MaxIntervalCount),
            "arrayvalued" => Some(Self::ArrayValued),
            _ => None,
        }
    }

    pub const fn is_numeric(self) -> bool {
        !matches!(self, Self::ArrayValued)
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::AbsTol => "AbsTol",
            Self::RelTol => "RelTol",
            Self::MaxFunEvals => "MaxFunEvals",
            Self::MaxIntervalCount => "MaxIntervalCount",
            Self::ArrayValued => "ArrayValued",
        }
    }
}
