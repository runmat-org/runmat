use super::{
    DeepLearningDenseOperationDimension, DeepLearningLayerKind, DeepLearningLayerOption,
    LayerOptionDefault, LayerOptionExecution,
};

impl DeepLearningLayerOption {
    pub const fn execution_for(self, kind: DeepLearningLayerKind) -> LayerOptionExecution {
        use DeepLearningLayerOption as O;
        match self {
            O::Name | O::Description => LayerOptionExecution::Metadata,
            O::Weights | O::Bias if matches!(kind, DeepLearningLayerKind::FullyConnected) => {
                LayerOptionExecution::Executable
            }
            _ => LayerOptionExecution::PendingExecution,
        }
    }

    pub const fn default_for(self, kind: DeepLearningLayerKind) -> Option<LayerOptionDefault> {
        use DeepLearningLayerOption as O;
        use LayerOptionDefault as D;
        match (kind, self) {
            (_, O::Name | O::Description) => Some(D::Text("")),
            (
                DeepLearningLayerKind::FeatureInput | DeepLearningLayerKind::SequenceInput,
                O::Normalization,
            ) => Some(D::Text("none")),
            (
                DeepLearningLayerKind::FeatureInput | DeepLearningLayerKind::SequenceInput,
                O::NormalizationDimension,
            ) => Some(D::Text("auto")),
            (
                DeepLearningLayerKind::FeatureInput | DeepLearningLayerKind::SequenceInput,
                O::SplitComplexInputs,
            ) => Some(D::Bool(false)),
            (DeepLearningLayerKind::SequenceInput, O::MinLength) => Some(D::Number(1.0)),
            (DeepLearningLayerKind::FullyConnected, O::OperationDimension) => Some(D::Text(
                DeepLearningDenseOperationDimension::SpatialChannel.canonical_name(),
            )),
            (
                DeepLearningLayerKind::FullyConnected
                | DeepLearningLayerKind::Lstm
                | DeepLearningLayerKind::Bilstm,
                O::InputSize,
            ) => Some(D::Text("auto")),
            (DeepLearningLayerKind::Convolution1D, O::Stride | O::DilationFactor) => {
                Some(D::Number(1.0))
            }
            // The convolution1dLayer default is zero padding on both sides.
            (DeepLearningLayerKind::Convolution1D, O::Padding) => Some(D::Number(0.0)),
            (DeepLearningLayerKind::Convolution1D, O::PaddingValue) => Some(D::Number(0.0)),
            (DeepLearningLayerKind::Convolution1D, O::NumChannels) => Some(D::Text("auto")),
            (DeepLearningLayerKind::Lstm | DeepLearningLayerKind::Bilstm, O::OutputMode) => {
                Some(D::Text("sequence"))
            }
            (
                DeepLearningLayerKind::Lstm | DeepLearningLayerKind::Bilstm,
                O::HasStateInputs | O::HasStateOutputs,
            ) => Some(D::Bool(false)),
            (
                DeepLearningLayerKind::Lstm | DeepLearningLayerKind::Bilstm,
                O::StateActivationFunction,
            ) => Some(D::Text("tanh")),
            (
                DeepLearningLayerKind::Lstm | DeepLearningLayerKind::Bilstm,
                O::GateActivationFunction,
            ) => Some(D::Text("sigmoid")),
            (DeepLearningLayerKind::LayerNormalization, O::Epsilon) => Some(D::Number(1.0e-5)),
            (DeepLearningLayerKind::LayerNormalization, O::NumChannels) => Some(D::Text("auto")),
            (DeepLearningLayerKind::LayerNormalization, O::OperationDimension) => {
                Some(D::Text("auto"))
            }
            (DeepLearningLayerKind::ClassificationOutput, O::Classes) => Some(D::Text("auto")),
            (DeepLearningLayerKind::ClassificationOutput, O::ClassWeights) => Some(D::Text("none")),
            _ => None,
        }
    }

    /// Closed textual alternatives. Some controls also accept numeric or
    /// callable values; those are validated at the runtime admission boundary.
    pub const fn text_values_for(
        self,
        kind: DeepLearningLayerKind,
    ) -> Option<&'static [&'static str]> {
        use DeepLearningLayerOption as O;
        match self {
            O::Normalization => Some(&[
                "none",
                "zerocenter",
                "zscore",
                "rescale-symmetric",
                "rescale-zero-one",
            ]),
            O::NormalizationDimension if matches!(kind, DeepLearningLayerKind::SequenceInput) => {
                Some(&["auto", "channel", "element", "all"])
            }
            O::NormalizationDimension => Some(&["auto", "channel", "all"]),
            O::OutputMode => Some(super::DeepLearningRecurrentOutputMode::NAMES),
            O::StateActivationFunction => Some(super::DeepLearningRecurrentStateActivation::NAMES),
            O::GateActivationFunction => Some(super::DeepLearningRecurrentGateActivation::NAMES),
            O::OperationDimension if matches!(kind, DeepLearningLayerKind::LayerNormalization) => {
                None
            }
            O::OperationDimension => Some(DeepLearningDenseOperationDimension::NAMES),
            O::Padding => Some(super::DeepLearningConvolutionPaddingMode::NAMES),
            O::PaddingValue => Some(super::DeepLearningConvolutionPaddingValueMode::NAMES),
            O::InputSize | O::NumChannels => Some(&["auto"]),
            _ => None,
        }
    }
}
