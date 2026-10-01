/// Discriminant calculation selected by the public `type` argument.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiscriminantType {
    Linear,
    Quadratic,
    DiagLinear,
    DiagQuadratic,
    Mahalanobis,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PriorKeyword {
    Empirical,
}

impl PriorKeyword {
    pub fn from_public(text: &str) -> Option<Self> {
        text.eq_ignore_ascii_case("empirical")
            .then_some(Self::Empirical)
    }
}

impl DiscriminantType {
    pub fn from_public(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("linear") {
            Some(Self::Linear)
        } else if text.eq_ignore_ascii_case("quadratic") {
            Some(Self::Quadratic)
        } else if text.eq_ignore_ascii_case("diaglinear") {
            Some(Self::DiagLinear)
        } else if text.eq_ignore_ascii_case("diagquadratic") {
            Some(Self::DiagQuadratic)
        } else if text.eq_ignore_ascii_case("mahalanobis") {
            Some(Self::Mahalanobis)
        } else {
            None
        }
    }

    pub fn property_name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
            Self::DiagLinear => "diagLinear",
            Self::DiagQuadratic => "diagQuadratic",
            Self::Mahalanobis => "mahalanobis",
        }
    }

    pub fn uses_shared_covariance(self) -> bool {
        matches!(self, Self::Linear | Self::DiagLinear)
    }

    pub fn uses_diagonal_covariance(self) -> bool {
        matches!(self, Self::DiagLinear | Self::DiagQuadratic)
    }

    pub fn has_quadratic_coefficient(self) -> bool {
        !matches!(self, Self::Linear | Self::DiagLinear)
    }
}
