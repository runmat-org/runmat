//! Closed distribution identities and source-language option decoding.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionKind {
    NormalPdf,
    NormalCdf,
    NormalInv,
    StudentTPdf,
    StudentTCdf,
    StudentTInv,
    BinomialCdf,
    ChiSquareCdf,
    WeibullInv,
    GenericInv,
}

impl DistributionKind {
    pub const fn accepts_numeric_count(self, count: usize) -> bool {
        match self {
            Self::NormalPdf | Self::NormalCdf | Self::NormalInv => count >= 1 && count <= 3,
            Self::StudentTPdf | Self::StudentTCdf | Self::StudentTInv | Self::ChiSquareCdf => {
                count == 2
            }
            Self::BinomialCdf => count == 3,
            Self::WeibullInv => count == 1 || count == 3,
            Self::GenericInv => false,
        }
    }

    pub const fn supports_upper_tail(self) -> bool {
        matches!(
            self,
            Self::NormalCdf | Self::StudentTCdf | Self::BinomialCdf | Self::ChiSquareCdf
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionTail {
    Lower,
    Upper,
}

impl DistributionTail {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        keyword.eq_ignore_ascii_case("upper").then_some(Self::Upper)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InverseDistributionFamily {
    Normal,
    StudentT,
    Weibull,
    ChiSquare,
    Binomial,
    Gamma,
    Exponential,
    Poisson,
    Uniform,
    Lognormal,
    Beta,
    F,
}

impl InverseDistributionFamily {
    /// Decode a public family spelling once; execution and inference use the enum.
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        let normalized: String = keyword
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        match normalized.as_str() {
            "normal" | "norm" | "gaussian" => Some(Self::Normal),
            "t" | "tdistribution" | "studentt" | "student" => Some(Self::StudentT),
            "weibull" | "wbl" => Some(Self::Weibull),
            "chisquare" | "chi2" | "chisquared" => Some(Self::ChiSquare),
            "binomial" | "bino" => Some(Self::Binomial),
            "gamma" | "gam" => Some(Self::Gamma),
            "exponential" | "exp" => Some(Self::Exponential),
            "poisson" | "poiss" => Some(Self::Poisson),
            "uniform" | "unif" => Some(Self::Uniform),
            "lognormal" | "logn" => Some(Self::Lognormal),
            "beta" => Some(Self::Beta),
            "f" | "fdist" | "fdistribution" => Some(Self::F),
            _ => None,
        }
    }

    pub const fn accepts_parameter_count(self, count: usize) -> bool {
        match self {
            Self::Normal | Self::Lognormal => count <= 2,
            Self::Weibull | Self::Uniform => count == 0 || count == 2,
            Self::StudentT | Self::ChiSquare | Self::Exponential | Self::Poisson => count == 1,
            Self::Binomial | Self::Gamma | Self::Beta | Self::F => count == 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_and_parameter_arity_are_closed() {
        assert_eq!(
            InverseDistributionFamily::from_keyword("Student t"),
            Some(InverseDistributionFamily::StudentT)
        );
        assert_eq!(
            InverseDistributionFamily::from_keyword("chi-square"),
            Some(InverseDistributionFamily::ChiSquare)
        );
        assert_eq!(
            InverseDistributionFamily::from_keyword("f_dist"),
            Some(InverseDistributionFamily::F)
        );
        assert_eq!(InverseDistributionFamily::from_keyword("unknown"), None);
        assert!(!InverseDistributionFamily::Weibull.accepts_parameter_count(1));
        assert!(InverseDistributionFamily::Normal.accepts_parameter_count(2));
        assert_eq!(
            DistributionTail::from_keyword("UPPER"),
            Some(DistributionTail::Upper)
        );
        assert_eq!(DistributionTail::from_keyword("lower"), None);
    }
}
