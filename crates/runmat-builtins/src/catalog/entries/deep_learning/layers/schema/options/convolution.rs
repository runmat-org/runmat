use super::DeepLearningLayerOption as O;

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
