use super::DeepLearningLayerOption as O;

mod operation_dimension;
pub use operation_dimension::DeepLearningDenseOperationDimension;

pub(super) const fn supports(option: O) -> bool {
    matches!(
        option,
        O::InputSize
            | O::OperationDimension
            | O::WeightsInitializer
            | O::BiasInitializer
            | O::Weights
            | O::Bias
            | O::WeightLearnRateFactor
            | O::WeightL2Factor
            | O::BiasLearnRateFactor
            | O::BiasL2Factor
            | O::InputLearnables
            | O::OutputLearnables
    )
}
