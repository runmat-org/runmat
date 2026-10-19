//! Host forward's explicit operational-option consumption boundary.

use runmat_builtins::{DeepLearningLayerKind as K, DeepLearningLayerOption as O};

use crate::builtins::deep_learning::model::plan::options::LayerOptionRequirement;

pub(super) fn consumed(requirement: &LayerOptionRequirement) -> bool {
    use O::*;
    match requirement.option {
        Name | Description => true,
        // These options affect construction or training, not forward execution
        // of an already initialized canonical network.
        WeightsInitializer
        | BiasInitializer
        | InputWeightsInitializer
        | RecurrentWeightsInitializer
        | ScaleInitializer
        | OffsetInitializer
        | WeightLearnRateFactor
        | WeightL2Factor
        | BiasLearnRateFactor
        | BiasL2Factor
        | InputWeightsLearnRateFactor
        | RecurrentWeightsLearnRateFactor
        | InputWeightsL2Factor
        | RecurrentWeightsL2Factor
        | ScaleLearnRateFactor
        | OffsetLearnRateFactor
        | ScaleL2Factor
        | OffsetL2Factor => true,
        Weights
        | Bias
        | InputWeights
        | RecurrentWeights
        | Scale
        | Offset
        | HiddenState
        | CellState
        | InputSize
        | NumChannels
        | Stride
        | DilationFactor
        | Padding
        | PaddingValue
        | MinLength
        | OutputMode
        | StateActivationFunction
        | GateActivationFunction
        | Epsilon => true,
        OperationDimension => matches!(requirement.kind, K::LayerNormalization),
        // Output metadata is not a numerical forward transform.
        Classes | ClassWeights | ResponseNames => requirement.kind.is_output(),
        // Input transforms, callable controls, state I/O, and any future option
        // need an owning consumer before they may be admitted here.
        _ => false,
    }
}
