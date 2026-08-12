use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::BuiltinResult;

use super::{
    any_type, gather_args, layer_object, numeric_scalar, numeric_vector, positive_usize,
    tensor_value,
};

fn is_numeric_scalar(value: &Value) -> bool {
    match value {
        Value::Num(n) => n.is_finite(),
        Value::Int(_) => true,
        Value::Tensor(t) => {
            crate::builtins::common::tensor::is_scalar_tensor(t)
                && crate::builtins::common::tensor::tensor_value_f64(t, 0).is_finite()
        }
        _ => false,
    }
}

#[runtime_builtin(
    name = "fullyConnectedLayer",
    category = "deep_learning",
    summary = "Create a fully connected layer compatibility object.",
    keywords = "fullyConnectedLayer,deep learning,layer,neural network",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn fully_connected_layer_builtin(
    output_size: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let output_size = positive_usize(&output_size, "fullyConnectedLayer", "outputSize")?;
    layer_object(
        "nnet.cnn.layer.FullyConnectedLayer",
        "Fully Connected",
        vec![("OutputSize", Value::Num(output_size as f64))],
        gather_args(rest).await?,
        "fullyConnectedLayer",
    )
}

#[runtime_builtin(
    name = "featureInputLayer",
    category = "deep_learning",
    summary = "Create a feature input layer compatibility object.",
    keywords = "featureInputLayer,deep learning,input layer,features",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn feature_input_layer_builtin(
    input_size: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let input_size = numeric_vector(&input_size, "featureInputLayer", "inputSize")?;
    layer_object(
        "nnet.cnn.layer.FeatureInputLayer",
        "Feature Input",
        vec![(
            "InputSize",
            tensor_value(
                input_size.iter().map(|v| *v as f64).collect(),
                vec![1, input_size.len()],
                "featureInputLayer",
            )?,
        )],
        gather_args(rest).await?,
        "featureInputLayer",
    )
}

#[runtime_builtin(
    name = "sequenceInputLayer",
    category = "deep_learning",
    summary = "Create a sequence input layer compatibility object.",
    keywords = "sequenceInputLayer,deep learning,sequence,input layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn sequence_input_layer_builtin(
    input_size: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let input_size = numeric_vector(&input_size, "sequenceInputLayer", "inputSize")?;
    layer_object(
        "nnet.cnn.layer.SequenceInputLayer",
        "Sequence Input",
        vec![(
            "InputSize",
            tensor_value(
                input_size.iter().map(|v| *v as f64).collect(),
                vec![1, input_size.len()],
                "sequenceInputLayer",
            )?,
        )],
        gather_args(rest).await?,
        "sequenceInputLayer",
    )
}

#[runtime_builtin(
    name = "reluLayer",
    category = "deep_learning",
    summary = "Create a ReLU layer compatibility object.",
    keywords = "reluLayer,deep learning,activation,layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn relu_layer_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    layer_object(
        "nnet.cnn.layer.ReLULayer",
        "ReLU",
        vec![],
        gather_args(rest).await?,
        "reluLayer",
    )
}

#[runtime_builtin(
    name = "eluLayer",
    category = "deep_learning",
    summary = "Create an ELU layer compatibility object.",
    keywords = "eluLayer,deep learning,activation,layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn elu_layer_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let mut args = gather_args(rest).await?;
    let alpha = if args.first().is_some_and(is_numeric_scalar) {
        numeric_scalar(&args.remove(0), "eluLayer", "alpha")?
    } else {
        1.0
    };
    layer_object(
        "nnet.cnn.layer.ELULayer",
        "ELU",
        vec![("Alpha", Value::Num(alpha))],
        args,
        "eluLayer",
    )
}

#[runtime_builtin(
    name = "softmaxLayer",
    category = "deep_learning",
    summary = "Create a softmax layer compatibility object.",
    keywords = "softmaxLayer,deep learning,softmax,layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn softmax_layer_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    layer_object(
        "nnet.cnn.layer.SoftmaxLayer",
        "Softmax",
        vec![],
        gather_args(rest).await?,
        "softmaxLayer",
    )
}

#[runtime_builtin(
    name = "classificationLayer",
    category = "deep_learning",
    summary = "Create a classification output layer compatibility object.",
    keywords = "classificationLayer,deep learning,classification,output layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn classification_layer_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    layer_object(
        "nnet.cnn.layer.ClassificationOutputLayer",
        "Classification Output",
        vec![("Classes", Value::String("auto".into()))],
        gather_args(rest).await?,
        "classificationLayer",
    )
}

#[runtime_builtin(
    name = "regressionLayer",
    category = "deep_learning",
    summary = "Create a regression output layer compatibility object.",
    keywords = "regressionLayer,deep learning,regression,output layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn regression_layer_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    layer_object(
        "nnet.cnn.layer.RegressionOutputLayer",
        "Regression Output",
        vec![],
        gather_args(rest).await?,
        "regressionLayer",
    )
}

#[runtime_builtin(
    name = "layerNormalizationLayer",
    category = "deep_learning",
    summary = "Create a layer normalization layer compatibility object.",
    keywords = "layerNormalizationLayer,deep learning,normalization,layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn layer_normalization_layer_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    layer_object(
        "nnet.cnn.layer.LayerNormalizationLayer",
        "Layer Normalization",
        vec![("Epsilon", Value::Num(1.0e-5))],
        gather_args(rest).await?,
        "layerNormalizationLayer",
    )
}

#[runtime_builtin(
    name = "globalAveragePooling1dLayer",
    category = "deep_learning",
    summary = "Create a global average pooling 1-D layer compatibility object.",
    keywords = "globalAveragePooling1dLayer,deep learning,pooling,layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn global_average_pooling_1d_layer_builtin(
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    layer_object(
        "nnet.cnn.layer.GlobalAveragePooling1DLayer",
        "Global Average Pooling 1D",
        vec![],
        gather_args(rest).await?,
        "globalAveragePooling1dLayer",
    )
}

#[runtime_builtin(
    name = "lstmLayer",
    category = "deep_learning",
    summary = "Create an LSTM layer compatibility object.",
    keywords = "lstmLayer,deep learning,lstm,recurrent,layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn lstm_layer_builtin(
    num_hidden_units: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    recurrent_layer(
        "lstmLayer",
        "nnet.cnn.layer.LSTMLayer",
        "LSTM",
        num_hidden_units,
        rest,
    )
    .await
}

#[runtime_builtin(
    name = "bilstmLayer",
    category = "deep_learning",
    summary = "Create a bidirectional LSTM layer compatibility object.",
    keywords = "bilstmLayer,deep learning,lstm,bidirectional,recurrent,layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn bilstm_layer_builtin(
    num_hidden_units: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    recurrent_layer(
        "bilstmLayer",
        "nnet.cnn.layer.BiLSTMLayer",
        "BiLSTM",
        num_hidden_units,
        rest,
    )
    .await
}

pub(super) async fn recurrent_layer(
    function: &'static str,
    class_name: &str,
    type_name: &str,
    num_hidden_units: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let units = positive_usize(&num_hidden_units, function, "numHiddenUnits")?;
    layer_object(
        class_name,
        type_name,
        vec![
            ("NumHiddenUnits", Value::Num(units as f64)),
            ("OutputMode", Value::String("last".into())),
        ],
        gather_args(rest).await?,
        function,
    )
}

#[runtime_builtin(
    name = "convolution1dLayer",
    category = "deep_learning",
    summary = "Create a 1-D convolution layer compatibility object.",
    keywords = "convolution1dLayer,deep learning,convolution,layer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn convolution_1d_layer_builtin(
    filter_size: Value,
    num_filters: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let filter_size = positive_usize(&filter_size, "convolution1dLayer", "filterSize")?;
    let num_filters = positive_usize(&num_filters, "convolution1dLayer", "numFilters")?;
    layer_object(
        "nnet.cnn.layer.Convolution1DLayer",
        "Convolution 1D",
        vec![
            ("FilterSize", Value::Num(filter_size as f64)),
            ("NumFilters", Value::Num(num_filters as f64)),
            ("Stride", Value::Num(1.0)),
            ("Padding", Value::String("same".into())),
        ],
        gather_args(rest).await?,
        "convolution1dLayer",
    )
}
