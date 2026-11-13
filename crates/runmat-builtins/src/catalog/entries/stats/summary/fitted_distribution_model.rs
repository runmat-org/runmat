/// Families supported by the fitted-distribution value model. Names are
/// interpreted only at the public text boundary; execution uses this identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FittedDistributionFamily {
    Normal,
    Exponential,
    Lognormal,
    Gamma,
    Weibull,
    Poisson,
}

impl FittedDistributionFamily {
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Exponential => "Exponential",
            Self::Lognormal => "Lognormal",
            Self::Gamma => "Gamma",
            Self::Weibull => "Weibull",
            Self::Poisson => "Poisson",
        }
    }

    pub const fn parameter_names(self) -> &'static [&'static str] {
        match self {
            Self::Normal | Self::Lognormal => &["mu", "sigma"],
            Self::Exponential => &["mu"],
            Self::Gamma | Self::Weibull => &["a", "b"],
            Self::Poisson => &["lambda"],
        }
    }

    pub fn from_keyword(keyword: &str) -> Option<Self> {
        let normalized = keyword
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        match normalized.as_str() {
            "normal" | "norm" | "gaussian" => Some(Self::Normal),
            "exponential" | "exp" => Some(Self::Exponential),
            "lognormal" | "logn" => Some(Self::Lognormal),
            "gamma" | "gam" => Some(Self::Gamma),
            "weibull" | "wbl" => Some(Self::Weibull),
            "poisson" | "poiss" => Some(Self::Poisson),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FittedFitOption {
    Frequency,
    Censoring,
    Options,
    By,
}

impl FittedFitOption {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "frequency" | "freq" => Some(Self::Frequency),
            "censoring" | "censor" => Some(Self::Censoring),
            "options" => Some(Self::Options),
            "by" => Some(Self::By),
            _ => None,
        }
    }
}
