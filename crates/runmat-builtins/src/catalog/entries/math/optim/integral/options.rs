//! Public option spellings are decoded once into quadrature control identities.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegralOptionName {
    AbsTol,
    RelTol,
    MaxFunEvals,
    MaxIntervalCount,
    ArrayValued,
    Vectorized,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegralCallbackMode {
    Vectorized,
    Scalar,
}

impl IntegralCallbackMode {
    pub const fn from_vectorized(flag: bool) -> Self {
        if flag {
            Self::Vectorized
        } else {
            Self::Scalar
        }
    }
}

impl IntegralOptionName {
    pub const DEFAULT_ABS_TOL: f64 = 1.0e-10;
    pub const DEFAULT_REL_TOL: f64 = 1.0e-6;

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "abstol" => Some(Self::AbsTol),
            "reltol" => Some(Self::RelTol),
            "maxfunevals" => Some(Self::MaxFunEvals),
            "maxintervalcount" => Some(Self::MaxIntervalCount),
            "arrayvalued" => Some(Self::ArrayValued),
            "vectorized" => Some(Self::Vectorized),
            _ => None,
        }
    }

    pub const fn is_numeric(self) -> bool {
        matches!(
            self,
            Self::AbsTol | Self::RelTol | Self::MaxFunEvals | Self::MaxIntervalCount
        )
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::AbsTol => "AbsTol",
            Self::RelTol => "RelTol",
            Self::MaxFunEvals => "MaxFunEvals",
            Self::MaxIntervalCount => "MaxIntervalCount",
            Self::ArrayValued => "ArrayValued",
            Self::Vectorized => "Vectorized",
        }
    }
}
