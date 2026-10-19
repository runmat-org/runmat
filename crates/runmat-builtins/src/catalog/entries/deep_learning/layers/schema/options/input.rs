use super::{DeepLearningLayerKind as K, DeepLearningLayerOption as O};

pub(super) const fn supports(kind: K, option: O) -> bool {
    matches!(
        option,
        O::Normalization
            | O::NormalizationDimension
            | O::Mean
            | O::StandardDeviation
            | O::Min
            | O::Max
            | O::SplitComplexInputs
    ) || matches!((kind, option), (K::SequenceInput, O::MinLength))
}
