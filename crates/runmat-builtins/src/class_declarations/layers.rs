//! Canonical identities for the public deep-learning layer objects.

use runmat_types::StaticClassIdentity;

pub const FULLY_CONNECTED_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.FullyConnectedLayer");
pub const FEATURE_INPUT_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.FeatureInputLayer");
pub const SEQUENCE_INPUT_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.SequenceInputLayer");
pub const RELU_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.ReLULayer");
pub const ELU_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.ELULayer");
pub const SOFTMAX_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.SoftmaxLayer");
pub const CLASSIFICATION_OUTPUT_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.ClassificationOutputLayer");
pub const REGRESSION_OUTPUT_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.RegressionOutputLayer");
pub const LAYER_NORMALIZATION_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.LayerNormalizationLayer");
pub const GLOBAL_AVERAGE_POOLING_1D_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.GlobalAveragePooling1DLayer");
pub const LSTM_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.LSTMLayer");
pub const BILSTM_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.BiLSTMLayer");
pub const CONVOLUTION_1D_LAYER_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("nnet.cnn.layer.Convolution1DLayer");
