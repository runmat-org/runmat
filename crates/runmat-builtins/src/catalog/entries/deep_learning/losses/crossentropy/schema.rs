//! Pure cross-entropy option and mode identities shared by analysis and execution.

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CrossentropyOptionKey {
    ClassificationMode,
    DataFormat,
    Mask,
    NormalizationFactor,
    Reduction,
    TargetCategories,
    Weights,
    WeightsFormat,
}

impl CrossentropyOptionKey {
    pub const ALL: [Self; 8] = [
        Self::ClassificationMode,
        Self::DataFormat,
        Self::Mask,
        Self::NormalizationFactor,
        Self::Reduction,
        Self::TargetCategories,
        Self::Weights,
        Self::WeightsFormat,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "classificationmode" => Some(Self::ClassificationMode),
            "targetcategories" => Some(Self::TargetCategories),
            "reduction" => Some(Self::Reduction),
            "normalizationfactor" => Some(Self::NormalizationFactor),
            "dataformat" => Some(Self::DataFormat),
            "weightsformat" => Some(Self::WeightsFormat),
            "weights" => Some(Self::Weights),
            "mask" => Some(Self::Mask),
            _ => None,
        }
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ClassificationMode => "ClassificationMode",
            Self::TargetCategories => "TargetCategories",
            Self::Reduction => "Reduction",
            Self::NormalizationFactor => "NormalizationFactor",
            Self::DataFormat => "DataFormat",
            Self::WeightsFormat => "WeightsFormat",
            Self::Weights => "Weights",
            Self::Mask => "Mask",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CrossentropyClassificationMode {
    SingleLabel,
    MultiLabel,
}

impl CrossentropyClassificationMode {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
            "single-label" | "singlelabel" | "exclusive" => Some(Self::SingleLabel),
            "multi-label" | "multilabel" | "independent" => Some(Self::MultiLabel),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CrossentropyReductionMode {
    Sum,
    None,
}

impl CrossentropyReductionMode {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "sum" => Some(Self::Sum),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CrossentropyNormalizationMode {
    AllElements,
    BatchSize,
    MaskIncluded,
    None,
}

impl CrossentropyNormalizationMode {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
            "all-elements" | "all" => Some(Self::AllElements),
            "batch-size" | "batch" => Some(Self::BatchSize),
            "mask-included" | "mask" => Some(Self::MaskIncluded),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_and_mode_names_round_trip() {
        let names = CrossentropyOptionKey::ALL.map(|key| key.canonical_name().to_ascii_lowercase());
        assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
        for key in CrossentropyOptionKey::ALL {
            assert_eq!(
                CrossentropyOptionKey::from_name(key.canonical_name()),
                Some(key)
            );
            assert_eq!(
                CrossentropyOptionKey::from_name(&key.canonical_name().to_lowercase()),
                Some(key)
            );
        }
        assert_eq!(CrossentropyOptionKey::from_name("other"), None);
        assert_eq!(
            CrossentropyClassificationMode::from_name("MULTI_LABEL"),
            Some(CrossentropyClassificationMode::MultiLabel)
        );
        assert_eq!(
            CrossentropyReductionMode::from_name("NONE"),
            Some(CrossentropyReductionMode::None)
        );
        assert_eq!(
            CrossentropyNormalizationMode::from_name("Mask Included"),
            Some(CrossentropyNormalizationMode::MaskIncluded)
        );
    }
}
