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
    pub const ALL: &'static [Self] = &[
        Self::Type,
        Self::InputSize,
        Self::OutputSize,
        Self::FilterSize,
        Self::NumFilters,
        Self::NumHiddenUnits,
        Self::Alpha,
    ];

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Type => "Type",
            Self::InputSize => "InputSize",
            Self::OutputSize => "OutputSize",
            Self::FilterSize => "FilterSize",
            Self::NumFilters => "NumFilters",
            Self::NumHiddenUnits => "NumHiddenUnits",
            Self::Alpha => "Alpha",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|property| property.canonical_name() == name)
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
