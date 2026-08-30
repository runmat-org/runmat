use runmat_types::ClassIdentity;

use super::{
    CLASSIFICATION_OUTPUT_LAYER_CLASS, ELU_LAYER_CLASS, FEATURE_INPUT_LAYER_CLASS,
    FULLY_CONNECTED_LAYER_CLASS, REGRESSION_OUTPUT_LAYER_CLASS, RELU_LAYER_CLASS,
    SOFTMAX_LAYER_CLASS,
};

/// Layer identities understood by the executable forward substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ForwardLayerKind {
    FeatureInput,
    FullyConnected,
    Relu,
    Elu,
    Softmax,
    ClassificationOutput,
    RegressionOutput,
}

impl ForwardLayerKind {
    pub(super) fn from_identity(identity: &ClassIdentity) -> Option<Self> {
        if identity.is(FEATURE_INPUT_LAYER_CLASS) {
            Some(Self::FeatureInput)
        } else if identity.is(FULLY_CONNECTED_LAYER_CLASS) {
            Some(Self::FullyConnected)
        } else if identity.is(RELU_LAYER_CLASS) {
            Some(Self::Relu)
        } else if identity.is(ELU_LAYER_CLASS) {
            Some(Self::Elu)
        } else if identity.is(SOFTMAX_LAYER_CLASS) {
            Some(Self::Softmax)
        } else if identity.is(CLASSIFICATION_OUTPUT_LAYER_CLASS) {
            Some(Self::ClassificationOutput)
        } else if identity.is(REGRESSION_OUTPUT_LAYER_CLASS) {
            Some(Self::RegressionOutput)
        } else {
            None
        }
    }

    pub(super) const fn is_output(self) -> bool {
        matches!(self, Self::ClassificationOutput | Self::RegressionOutput)
    }
}
