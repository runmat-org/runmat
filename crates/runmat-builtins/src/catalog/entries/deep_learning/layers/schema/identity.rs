use crate::{
    BILSTM_LAYER_CLASS, CLASSIFICATION_OUTPUT_LAYER_CLASS, CONVOLUTION_1D_LAYER_CLASS,
    ELU_LAYER_CLASS, FEATURE_INPUT_LAYER_CLASS, FULLY_CONNECTED_LAYER_CLASS,
    GLOBAL_AVERAGE_POOLING_1D_LAYER_CLASS, LAYER_NORMALIZATION_LAYER_CLASS, LSTM_LAYER_CLASS,
    REGRESSION_OUTPUT_LAYER_CLASS, RELU_LAYER_CLASS, SEQUENCE_INPUT_LAYER_CLASS,
    SOFTMAX_LAYER_CLASS,
};
use runmat_types::{ClassIdentity, StaticClassIdentity};

/// The complete finite set of public layer constructors exposed by RunMat.
/// This does not assert that every configured form is executable yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeepLearningLayerKind {
    FeatureInput,
    SequenceInput,
    FullyConnected,
    Convolution1D,
    Lstm,
    Bilstm,
    LayerNormalization,
    GlobalAveragePooling1D,
    Relu,
    Elu,
    Softmax,
    ClassificationOutput,
    RegressionOutput,
}

impl DeepLearningLayerKind {
    pub const fn is_input(self) -> bool {
        matches!(self, Self::FeatureInput | Self::SequenceInput)
    }

    pub const fn is_output(self) -> bool {
        matches!(self, Self::ClassificationOutput | Self::RegressionOutput)
    }

    pub const fn recurrent_gate_count(self) -> Option<usize> {
        match self {
            Self::Lstm => Some(4),
            Self::Bilstm => Some(8),
            _ => None,
        }
    }

    pub const ALL: [Self; 13] = [
        Self::FeatureInput,
        Self::SequenceInput,
        Self::FullyConnected,
        Self::Convolution1D,
        Self::Lstm,
        Self::Bilstm,
        Self::LayerNormalization,
        Self::GlobalAveragePooling1D,
        Self::Relu,
        Self::Elu,
        Self::Softmax,
        Self::ClassificationOutput,
        Self::RegressionOutput,
    ];

    pub const fn class(self) -> StaticClassIdentity {
        match self {
            Self::FeatureInput => FEATURE_INPUT_LAYER_CLASS,
            Self::SequenceInput => SEQUENCE_INPUT_LAYER_CLASS,
            Self::FullyConnected => FULLY_CONNECTED_LAYER_CLASS,
            Self::Convolution1D => CONVOLUTION_1D_LAYER_CLASS,
            Self::Lstm => LSTM_LAYER_CLASS,
            Self::Bilstm => BILSTM_LAYER_CLASS,
            Self::LayerNormalization => LAYER_NORMALIZATION_LAYER_CLASS,
            Self::GlobalAveragePooling1D => GLOBAL_AVERAGE_POOLING_1D_LAYER_CLASS,
            Self::Relu => RELU_LAYER_CLASS,
            Self::Elu => ELU_LAYER_CLASS,
            Self::Softmax => SOFTMAX_LAYER_CLASS,
            Self::ClassificationOutput => CLASSIFICATION_OUTPUT_LAYER_CLASS,
            Self::RegressionOutput => REGRESSION_OUTPUT_LAYER_CLASS,
        }
    }

    pub fn from_identity(identity: &ClassIdentity) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| identity.is(kind.class()))
    }

    pub fn from_constructor(name: &str) -> Option<Self> {
        match name {
            "featureInputLayer" => Some(Self::FeatureInput),
            "sequenceInputLayer" => Some(Self::SequenceInput),
            "fullyConnectedLayer" => Some(Self::FullyConnected),
            "convolution1dLayer" => Some(Self::Convolution1D),
            "lstmLayer" => Some(Self::Lstm),
            "bilstmLayer" => Some(Self::Bilstm),
            "layerNormalizationLayer" => Some(Self::LayerNormalization),
            "globalAveragePooling1dLayer" => Some(Self::GlobalAveragePooling1D),
            "reluLayer" => Some(Self::Relu),
            "eluLayer" => Some(Self::Elu),
            "softmaxLayer" => Some(Self::Softmax),
            "classificationLayer" => Some(Self::ClassificationOutput),
            "regressionLayer" => Some(Self::RegressionOutput),
            _ => None,
        }
    }

    pub const fn required_sizes(self) -> usize {
        match self {
            Self::FeatureInput
            | Self::SequenceInput
            | Self::FullyConnected
            | Self::Lstm
            | Self::Bilstm => 1,
            Self::Convolution1D => 2,
            _ => 0,
        }
    }
}
