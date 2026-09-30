use super::DeepLearningLayerOption as O;

pub(super) const fn supports(option: O) -> bool {
    matches!(
        option,
        O::OutputMode
            | O::HasStateInputs
            | O::HasStateOutputs
            | O::InputSize
            | O::StateActivationFunction
            | O::GateActivationFunction
            | O::CellState
            | O::HiddenState
            | O::InputWeightsInitializer
            | O::RecurrentWeightsInitializer
            | O::BiasInitializer
            | O::InputWeights
            | O::RecurrentWeights
            | O::Bias
            | O::InputWeightsLearnRateFactor
            | O::RecurrentWeightsLearnRateFactor
            | O::BiasLearnRateFactor
            | O::InputWeightsL2Factor
            | O::RecurrentWeightsL2Factor
            | O::BiasL2Factor
    )
}
