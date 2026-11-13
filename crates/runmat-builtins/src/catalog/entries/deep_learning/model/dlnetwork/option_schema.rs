//! Constructor option identities for a sequential deep-learning network.

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepLearningNetworkOption {
    Initialize,
    OutputNames,
}

impl DeepLearningNetworkOption {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Initialize => "Initialize",
            Self::OutputNames => "OutputNames",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case(Self::Initialize.name()) {
            Some(Self::Initialize)
        } else if name.eq_ignore_ascii_case(Self::OutputNames.name()) {
            Some(Self::OutputNames)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DeepLearningNetworkOption as OptionName;

    #[test]
    fn constructor_options_have_one_case_insensitive_identity() {
        assert_eq!(
            OptionName::from_name("INITIALIZE"),
            Some(OptionName::Initialize)
        );
        assert_eq!(
            OptionName::from_name("outputnames"),
            Some(OptionName::OutputNames)
        );
        assert_eq!(OptionName::from_name("InputNames"), None);
        assert_eq!(OptionName::Initialize.name(), "Initialize");
        assert_eq!(OptionName::OutputNames.name(), "OutputNames");
    }
}
