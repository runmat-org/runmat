use super::DeepLearningLayerKind;

/// Read-only constructor-defined properties, distinct from operational
/// name-value controls. Imported/restored objects use the same identity test.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepLearningLayerStructuralProperty {
    Type,
    InputSize,
    OutputSize,
    FilterSize,
    NumFilters,
    NumHiddenUnits,
    Alpha,
}

impl DeepLearningLayerStructuralProperty {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "Type" => Self::Type,
            "InputSize" => Self::InputSize,
            "OutputSize" => Self::OutputSize,
            "FilterSize" => Self::FilterSize,
            "NumFilters" => Self::NumFilters,
            "NumHiddenUnits" => Self::NumHiddenUnits,
            "Alpha" => Self::Alpha,
            _ => return None,
        })
    }

    pub const fn belongs_to(self, kind: DeepLearningLayerKind) -> bool {
        use DeepLearningLayerKind as K;
        match self {
            Self::Type => true,
            Self::InputSize => matches!(kind, K::FeatureInput | K::SequenceInput),
            Self::OutputSize => matches!(kind, K::FullyConnected),
            Self::FilterSize | Self::NumFilters => matches!(kind, K::Convolution1D),
            Self::NumHiddenUnits => matches!(kind, K::Lstm | K::Bilstm),
            Self::Alpha => matches!(kind, K::Elu),
        }
    }
}
