use super::DeepLearningLayerOption as O;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeepLearningConvolutionPaddingMode {
    Same,
    Causal,
}

impl DeepLearningConvolutionPaddingMode {
    pub const ALL: [Self; 2] = [Self::Same, Self::Causal];
    pub const NAMES: &'static [&'static str] =
        &[Self::Same.canonical_name(), Self::Causal.canonical_name()];

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|mode| name.eq_ignore_ascii_case(mode.canonical_name()))
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Same => "same",
            Self::Causal => "causal",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeepLearningConvolutionPaddingValueMode {
    Replicate,
    SymmetricIncludeEdge,
    SymmetricExcludeEdge,
}

impl DeepLearningConvolutionPaddingValueMode {
    pub const ALL: [Self; 3] = [
        Self::SymmetricIncludeEdge,
        Self::SymmetricExcludeEdge,
        Self::Replicate,
    ];
    pub const NAMES: &'static [&'static str] = &[
        Self::SymmetricIncludeEdge.canonical_name(),
        Self::SymmetricExcludeEdge.canonical_name(),
        Self::Replicate.canonical_name(),
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|mode| name.eq_ignore_ascii_case(mode.canonical_name()))
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Replicate => "replicate",
            Self::SymmetricIncludeEdge => "symmetric-include-edge",
            Self::SymmetricExcludeEdge => "symmetric-exclude-edge",
        }
    }
}

pub(super) const fn supports(option: O) -> bool {
    matches!(
        option,
        O::Stride
            | O::DilationFactor
            | O::Padding
            | O::PaddingValue
            | O::NumChannels
            | O::WeightsInitializer
            | O::BiasInitializer
            | O::Weights
            | O::Bias
            | O::WeightLearnRateFactor
            | O::WeightL2Factor
            | O::BiasLearnRateFactor
            | O::BiasL2Factor
    )
}
