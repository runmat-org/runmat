/// Public `fitlm` name-value identity. Decode spelling once at the API boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitlmOption {
    Intercept,
    ResponseVar,
    PredictorVars,
    VarNames,
    Exclude,
    Weights,
    RobustOpts,
    CategoricalVars,
}

impl FitlmOption {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "intercept" => Self::Intercept,
            "responsevar" => Self::ResponseVar,
            "predictorvars" => Self::PredictorVars,
            "varnames" => Self::VarNames,
            "exclude" => Self::Exclude,
            "weights" => Self::Weights,
            "robustopts" => Self::RobustOpts,
            "categoricalvars" => Self::CategoricalVars,
            _ => return None,
        })
    }

    pub const fn unsupported(self) -> bool {
        matches!(self, Self::RobustOpts | Self::CategoricalVars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_names_are_case_insensitive_and_closed() {
        assert_eq!(
            FitlmOption::from_public_name("INTERCEPT"),
            Some(FitlmOption::Intercept)
        );
        assert_eq!(
            FitlmOption::from_public_name("Weights"),
            Some(FitlmOption::Weights)
        );
        assert_eq!(FitlmOption::from_public_name("Unknown"), None);
    }
}
