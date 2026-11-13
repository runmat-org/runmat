use super::{DeepLearningLayerKind as K, DeepLearningLayerOption as O};

pub(super) const fn supports(kind: K, option: O) -> bool {
    match kind {
        K::ClassificationOutput => matches!(option, O::Classes | O::ClassWeights),
        K::RegressionOutput => matches!(option, O::ResponseNames),
        _ => false,
    }
}
