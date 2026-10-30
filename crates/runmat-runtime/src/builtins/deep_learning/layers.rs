use runmat_builtins::{
    DeepLearningLayerKind, DeepLearningLayerOption, LayerOptionDefault,
    CLASSIFICATION_OUTPUT_LAYER_CLASS, CONVOLUTION_1D_LAYER_CLASS, ELU_LAYER_CLASS,
    FEATURE_INPUT_LAYER_CLASS, FULLY_CONNECTED_LAYER_CLASS, GLOBAL_AVERAGE_POOLING_1D_LAYER_CLASS,
    LAYER_NORMALIZATION_LAYER_CLASS, REGRESSION_OUTPUT_LAYER_CLASS, RELU_LAYER_CLASS,
    SEQUENCE_INPUT_LAYER_CLASS, SOFTMAX_LAYER_CLASS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{gather_if_needed_async, BuiltinResult};

use super::{
    deep_learning_error, layer_object, numeric_scalar, numeric_vector, positive_usize, scalar_text,
    tensor_value,
};

// Structural layer dimensions are exposed as double-valued metadata. Admit
// only integers that retain their value through that representation boundary.
fn metadata_size(value: &Value, function: &'static str, label: &str) -> BuiltinResult<usize> {
    let size = positive_usize(value, function, label)?;
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
        return Err(deep_learning_error(
            function,
            format!("{function}: {label} cannot be represented exactly in layer metadata"),
        ));
    }
    Ok(size)
}

fn metadata_sizes(value: &Value, function: &'static str, label: &str) -> BuiltinResult<Vec<usize>> {
    let sizes = numeric_vector(value, function, label)?;
    if sizes.is_empty()
        || !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
    {
        return Err(deep_learning_error(
            function,
            format!("{function}: {label} must contain exactly representable positive dimensions"),
        ));
    }
    Ok(sizes)
}

async fn admitted_layer_options(
    rest: crate::call::BuiltinArguments,
    function: &'static str,
) -> BuiltinResult<std::collections::BTreeMap<DeepLearningLayerOption, Value>> {
    let kind = DeepLearningLayerKind::from_constructor(function)
        .expect("layer constructor must have a catalog identity");
    let mut options = super::layer_helpers::parse_layer_options(rest.view(), kind, function)?;
    for (option, value) in &mut options {
        if super::layer_helpers::options::requires_host_control(*option) {
            *value = gather_if_needed_async(value).await?;
            super::layer_helpers::options::validate(kind, *option, value, function)?;
        }
    }
    Ok(options)
}

fn declared_constructor_default(
    kind: DeepLearningLayerKind,
    option: DeepLearningLayerOption,
) -> Value {
    match option
        .default_for(kind)
        .expect("constructor property must have a declared catalog default")
    {
        LayerOptionDefault::Text(value) => Value::String(value.into()),
        LayerOptionDefault::Number(value) => Value::Num(value),
        LayerOptionDefault::Bool(value) => Value::Bool(value),
    }
}

fn preflight_layer_options(
    rest: crate::call::FunctionArgumentView<'_>,
    function: &'static str,
) -> BuiltinResult<()> {
    let kind = DeepLearningLayerKind::from_constructor(function)
        .expect("layer constructor must have a catalog identity");
    super::layer_helpers::parse_layer_options(rest, kind, function).map(|_| ())
}

fn is_numeric_scalar(value: &Value) -> bool {
    match value {
        Value::Num(n) => n.is_finite(),
        Value::Int(_) => true,
        Value::Tensor(t) => {
            crate::builtins::common::tensor::is_scalar_tensor(t)
                && crate::builtins::common::tensor::tensor_value_f64(t, 0).is_finite()
        }
        Value::GpuTensor(handle) => {
            handle.shape().iter().all(|dimension| *dimension == 1)
                && runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle)
        }
        _ => false,
    }
}

#[runtime_builtin(
    name = "fullyConnectedLayer",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn fully_connected_layer_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    output_size: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output(
        "fullyConnectedLayer",
        &runmat_builtins::LAYER_INVALID_INPUT,
    )?;
    let output_size = metadata_size(&output_size, "fullyConnectedLayer", "outputSize")?;
    let mut options = rest;
    if options.first().is_some_and(|value| {
        is_numeric_scalar(value)
            || scalar_text(value, "fullyConnectedLayer").is_ok_and(|text| {
                runmat_builtins::DeepLearningDenseOperationDimension::from_name(&text).is_some()
            })
    }) {
        let dimension = options.pop_front().expect("validated argument prefix");
        let source = options.into_function_arguments();
        let operation = Value::String(
            crate::call::argument_text::copy_str("OperationDimension", runtime.as_ref())?.into(),
        );
        options = crate::execution::RootedValues::transform_arguments(
            &source,
            source.len() + 2,
            None,
            runtime.as_ref(),
            |index| {
                Ok(match index {
                    0 => operation.clone(),
                    1 => dimension.clone(),
                    index => source
                        .get(index - 2)
                        .expect("fully connected option argument")
                        .clone(),
                })
            },
        )?
        .into();
    }
    layer_object(
        FULLY_CONNECTED_LAYER_CLASS,
        "Fully Connected",
        vec![("OutputSize", Value::Num(output_size as f64))],
        admitted_layer_options(options, "fullyConnectedLayer").await?,
        "fullyConnectedLayer",
    )
}

#[runtime_builtin(
    name = "featureInputLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn feature_input_layer_builtin(
    input_size: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output("featureInputLayer", &runmat_builtins::LAYER_INVALID_INPUT)?;
    let input_size = metadata_sizes(&input_size, "featureInputLayer", "inputSize")?;
    if input_size.len() > 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::FEATURE_INPUT_VECTOR_EXTENSION,
            "featureInputLayer",
        )?;
    }
    layer_object(
        FEATURE_INPUT_LAYER_CLASS,
        "Feature Input",
        vec![(
            "InputSize",
            tensor_value(
                input_size.iter().map(|v| *v as f64).collect(),
                vec![1, input_size.len()],
                "featureInputLayer",
            )?,
        )],
        admitted_layer_options(rest, "featureInputLayer").await?,
        "featureInputLayer",
    )
}

#[runtime_builtin(
    name = "sequenceInputLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn sequence_input_layer_builtin(
    input_size: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output(
        "sequenceInputLayer",
        &runmat_builtins::LAYER_INVALID_INPUT,
    )?;
    let input_size = metadata_sizes(&input_size, "sequenceInputLayer", "inputSize")?;
    layer_object(
        SEQUENCE_INPUT_LAYER_CLASS,
        "Sequence Input",
        vec![(
            "InputSize",
            tensor_value(
                input_size.iter().map(|v| *v as f64).collect(),
                vec![1, input_size.len()],
                "sequenceInputLayer",
            )?,
        )],
        admitted_layer_options(rest, "sequenceInputLayer").await?,
        "sequenceInputLayer",
    )
}

#[runtime_builtin(
    name = "reluLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn relu_layer_builtin(
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output("reluLayer", &runmat_builtins::LAYER_INVALID_INPUT)?;
    layer_object(
        RELU_LAYER_CLASS,
        "ReLU",
        vec![],
        admitted_layer_options(rest, "reluLayer").await?,
        "reluLayer",
    )
}

#[runtime_builtin(
    name = "eluLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn elu_layer_builtin(rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::errors::admit_single_output("eluLayer", &runmat_builtins::LAYER_INVALID_INPUT)?;
    let mut args = rest;
    let leading_alpha = args.first().is_some_and(is_numeric_scalar);
    if leading_alpha {
        preflight_layer_options(
            args.view().range(1..).expect("ELU option suffix"),
            "eluLayer",
        )?;
    }
    let alpha = if args.first().is_some_and(is_numeric_scalar) {
        numeric_scalar(
            &gather_if_needed_async(&args.pop_front().expect("validated argument prefix")).await?,
            "eluLayer",
            "alpha",
        )?
    } else {
        1.0
    };
    layer_object(
        ELU_LAYER_CLASS,
        "ELU",
        vec![("Alpha", Value::Num(alpha))],
        admitted_layer_options(args, "eluLayer").await?,
        "eluLayer",
    )
}

#[runtime_builtin(
    name = "softmaxLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn softmax_layer_builtin(
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output("softmaxLayer", &runmat_builtins::LAYER_INVALID_INPUT)?;
    layer_object(
        SOFTMAX_LAYER_CLASS,
        "Softmax",
        vec![],
        admitted_layer_options(rest, "softmaxLayer").await?,
        "softmaxLayer",
    )
}

#[runtime_builtin(
    name = "classificationLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn classification_layer_builtin(
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output(
        "classificationLayer",
        &runmat_builtins::LAYER_INVALID_INPUT,
    )?;
    layer_object(
        CLASSIFICATION_OUTPUT_LAYER_CLASS,
        "Classification Output",
        vec![("Classes", Value::String("auto".into()))],
        admitted_layer_options(rest, "classificationLayer").await?,
        "classificationLayer",
    )
}

#[runtime_builtin(
    name = "regressionLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn regression_layer_builtin(
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output("regressionLayer", &runmat_builtins::LAYER_INVALID_INPUT)?;
    layer_object(
        REGRESSION_OUTPUT_LAYER_CLASS,
        "Regression Output",
        vec![],
        admitted_layer_options(rest, "regressionLayer").await?,
        "regressionLayer",
    )
}

#[runtime_builtin(
    name = "layerNormalizationLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn layer_normalization_layer_builtin(
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output(
        "layerNormalizationLayer",
        &runmat_builtins::LAYER_INVALID_INPUT,
    )?;
    layer_object(
        LAYER_NORMALIZATION_LAYER_CLASS,
        "Layer Normalization",
        vec![("Epsilon", Value::Num(1.0e-5))],
        admitted_layer_options(rest, "layerNormalizationLayer").await?,
        "layerNormalizationLayer",
    )
}

#[runtime_builtin(
    name = "globalAveragePooling1dLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn global_average_pooling_1d_layer_builtin(
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output(
        "globalAveragePooling1dLayer",
        &runmat_builtins::LAYER_INVALID_INPUT,
    )?;
    layer_object(
        GLOBAL_AVERAGE_POOLING_1D_LAYER_CLASS,
        "Global Average Pooling 1D",
        vec![],
        admitted_layer_options(rest, "globalAveragePooling1dLayer").await?,
        "globalAveragePooling1dLayer",
    )
}

#[runtime_builtin(
    name = "lstmLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn lstm_layer_builtin(
    num_hidden_units: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output("lstmLayer", &runmat_builtins::LAYER_INVALID_INPUT)?;
    recurrent_layer(
        "lstmLayer",
        DeepLearningLayerKind::Lstm,
        "LSTM",
        num_hidden_units,
        rest,
    )
    .await
}

#[runtime_builtin(
    name = "bilstmLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn bilstm_layer_builtin(
    num_hidden_units: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output("bilstmLayer", &runmat_builtins::LAYER_INVALID_INPUT)?;
    recurrent_layer(
        "bilstmLayer",
        DeepLearningLayerKind::Bilstm,
        "BiLSTM",
        num_hidden_units,
        rest,
    )
    .await
}

pub(super) async fn recurrent_layer(
    function: &'static str,
    kind: DeepLearningLayerKind,
    type_name: &str,
    num_hidden_units: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let units = metadata_size(&num_hidden_units, function, "numHiddenUnits")?;
    layer_object(
        kind.class(),
        type_name,
        vec![
            ("NumHiddenUnits", Value::Num(units as f64)),
            (
                DeepLearningLayerOption::OutputMode.canonical_name(),
                declared_constructor_default(kind, DeepLearningLayerOption::OutputMode),
            ),
        ],
        admitted_layer_options(rest, function).await?,
        function,
    )
}

#[runtime_builtin(
    name = "convolution1dLayer",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::layers"
)]
pub(super) async fn convolution_1d_layer_builtin(
    filter_size: Value,
    num_filters: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output(
        "convolution1dLayer",
        &runmat_builtins::LAYER_INVALID_INPUT,
    )?;
    let filter_size = metadata_size(&filter_size, "convolution1dLayer", "filterSize")?;
    let num_filters = metadata_size(&num_filters, "convolution1dLayer", "numFilters")?;
    layer_object(
        CONVOLUTION_1D_LAYER_CLASS,
        "Convolution 1D",
        vec![
            ("FilterSize", Value::Num(filter_size as f64)),
            ("NumFilters", Value::Num(num_filters as f64)),
            (
                DeepLearningLayerOption::Stride.canonical_name(),
                declared_constructor_default(
                    DeepLearningLayerKind::Convolution1D,
                    DeepLearningLayerOption::Stride,
                ),
            ),
            (
                DeepLearningLayerOption::Padding.canonical_name(),
                declared_constructor_default(
                    DeepLearningLayerKind::Convolution1D,
                    DeepLearningLayerOption::Padding,
                ),
            ),
        ],
        admitted_layer_options(rest, "convolution1dLayer").await?,
        "convolution1dLayer",
    )
}

#[cfg(test)]
#[path = "layers/identity_tests.rs"]
mod identity_tests;

#[cfg(test)]
#[path = "layers/residency_tests.rs"]
mod residency_tests;
