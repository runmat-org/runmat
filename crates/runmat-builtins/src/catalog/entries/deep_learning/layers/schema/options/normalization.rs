use super::DeepLearningLayerOption as O;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeepLearningNormalizationAxes {
    Auto,
    ChannelOnly,
    SpatialChannel,
    BatchExcluded,
}

impl DeepLearningNormalizationAxes {
    pub const ALL: [Self; 4] = [
        Self::Auto,
        Self::ChannelOnly,
        Self::SpatialChannel,
        Self::BatchExcluded,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|kind| name.eq_ignore_ascii_case(kind.canonical_name()))
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ChannelOnly => "channel-only",
            Self::SpatialChannel => "spatial-channel",
            Self::BatchExcluded => "batch-excluded",
        }
    }
}

pub(super) const fn supports(option: O) -> bool {
    matches!(
        option,
        O::Epsilon
            | O::OperationDimension
            | O::NumChannels
            | O::ScaleInitializer
            | O::OffsetInitializer
            | O::Scale
            | O::Offset
            | O::ScaleLearnRateFactor
            | O::OffsetLearnRateFactor
            | O::ScaleL2Factor
            | O::OffsetL2Factor
    )
}
