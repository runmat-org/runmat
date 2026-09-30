//! Provider-resident forward execution and result ownership validation.
use super::*;

/// Execute the subset of sequential feed-forward networks for which every
/// operation is available through the acceleration provider. The complete
/// layer sequence is validated before the first upload or dispatch so a
/// rejected layer cannot leave behind a partially executed GPU forward pass.
pub(super) async fn evaluate_network_gpu(
    object: &ObjectInstance,
    input: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| {
            unsupported_error(
                function,
                format!("{function}: no acceleration provider owns the GPU dlarray input"),
            )
        })?;
    let layers = object
        .properties
        .get("Layers")
        .map(|value| layer_slice(value, function))
        .transpose()?
        .unwrap_or(&[]);
    validate_gpu_forward_layers(layers, input, function)?;

    let mut current = input.clone();
    for layer in layers {
        let Value::Object(layer) = layer else {
            unreachable!("validate_gpu_forward_layers accepts layer objects only");
        };
        if layer.class_name.is(super::FULLY_CONNECTED_LAYER_CLASS) {
            current = fully_connected_forward_gpu(provider, current, &layer, function).await?;
        } else if layer.class_name.is(super::RELU_LAYER_CLASS) {
            let output = provider
                .scalar_max(&current, 0.0)
                .map_err(|err| provider_failure(function, "provider-resident ReLU", err.into()))?;
            validate_unary_forward_result(&output, &current, provider, function)?;
            current = output;
        } else if layer.class_name.is(super::ELU_LAYER_CLASS) {
            let alpha = elu_alpha(&layer, function)?;
            let output = provider
                .activation_elu(&current, alpha)
                .await
                .map_err(|err| provider_failure(function, "provider-resident ELU", err.into()))?;
            validate_unary_forward_result(&output, &current, provider, function)?;
            current = output;
        } else if layer.class_name.is(super::SOFTMAX_LAYER_CLASS) {
            let output = provider
                .activation_softmax_rows(&current)
                .await
                .map_err(|err| {
                    provider_failure(function, "provider-resident Softmax", err.into())
                })?;
            validate_unary_forward_result(&output, &current, provider, function)?;
            current = output;
        }
    }
    Ok(current)
}

fn validate_unary_forward_result(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    function: &'static str,
) -> BuiltinResult<()> {
    use crate::builtins::common::gpu_helpers::{
        unary_gpu_output_matches, GpuOutputAliasPolicy, UnaryGpuOutputContract,
    };
    let precision = runmat_accelerate_api::handle_precision(input);
    if unary_gpu_output_matches(
        output,
        input,
        provider,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision,
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::AllowInput,
        },
    ) {
        Ok(())
    } else {
        Err(crate::runtime_error::build_runtime_error(format!(
            "{function}: provider returned an invalid forward result descriptor"
        ))
        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build())
    }
}

fn validate_gpu_forward_layers(
    layers: &[Value],
    input: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<()> {
    if runmat_accelerate_api::handle_storage(input) != runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Err(unsupported_error(
            function,
            format!(
                "{function}: provider-resident forward currently requires a real dlarray input"
            ),
        ));
    }
    if input.shape().len() != 2 {
        return Err(unsupported_error(
            function,
            format!("{function}: provider-resident forward currently requires a 2-D dlarray input"),
        ));
    }
    if runmat_accelerate_api::handle_precision(input).is_none()
        || runmat_accelerate_api::handle_is_logical(input)
    {
        return Err(unsupported_error(
            function,
            format!("{function}: provider-resident forward requires single or double dlarray Data"),
        ));
    }
    let mut features = input.shape()[1];
    let mut saw_input = false;
    let mut saw_transform = false;
    for layer in layers {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        match ForwardLayerKind::from_identity(&layer.class_name) {
            Some(ForwardLayerKind::FeatureInput) if !saw_input => {
                let expected = feature_input_width(layer, function)?;
                if features != expected {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: input has {features} features, but featureInputLayer expects {expected}"
                        ),
                    ));
                }
                saw_input = true;
            }
            Some(ForwardLayerKind::FullyConnected) if saw_input => {
                features = fully_connected_output_width(layer, features, function)?;
                let provider =
                    crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
                        .expect("GPU input provider resolved before layer validation");
                for name in ["Weights", "Bias"] {
                    if let Some(handle) = resident_parameter_handle(layer, name) {
                        let owner =
                            crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
                                .ok_or_else(|| {
                                    unsupported_error(
                                        function,
                                        format!("{function}: {name} has no owning provider"),
                                    )
                                })?;
                        if !runmat_accelerate_api::same_provider_instance(provider, owner) {
                            return Err(unsupported_error(
                                function,
                                format!("{function}: {name} belongs to a different provider"),
                            ));
                        }
                    }
                }
                saw_transform = true;
            }
            Some(ForwardLayerKind::Relu) if saw_input => saw_transform = true,
            Some(ForwardLayerKind::Elu) if saw_input => {
                let _ = elu_alpha(layer, function)?;
                saw_transform = true;
            }
            Some(ForwardLayerKind::Softmax) if saw_input => saw_transform = true,
            Some(kind) if kind.is_output() && saw_input && saw_transform => {}
            _ => {
                return Err(unsupported_error(
                    function,
                    format!(
                        "{function}: provider-resident forward does not yet support layer type '{}'",
                        layer.class_name
                    ),
                ));
            }
        }
    }
    if !saw_input {
        return Err(deep_learning_error(
            function,
            format!("{function}: network must start with a supported input layer"),
        ));
    }
    if !saw_transform {
        return Err(unsupported_error(
            function,
            format!(
                "{function}: provider-resident forward requires at least one supported transform layer"
            ),
        ));
    }
    Ok(())
}

fn resident_parameter_handle<'a>(
    layer: &'a ObjectInstance,
    name: &str,
) -> Option<&'a GpuTensorHandle> {
    let value = layer.properties.get(name)?;
    match value {
        Value::GpuTensor(handle) => Some(handle),
        Value::Object(object) if object.class_name.is(super::DLARRAY_CLASS) => {
            match object.properties.get("Data") {
                Some(Value::GpuTensor(handle)) => Some(handle),
                _ => None,
            }
        }
        _ => None,
    }
}

async fn fully_connected_forward_gpu(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    current: GpuTensorHandle,
    layer: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let precision = runmat_accelerate_api::handle_precision(&current).ok_or_else(|| {
        unsupported_error(
            function,
            format!("{function}: input has no floating precision"),
        )
    })?;
    let dtype = match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    };
    // Only these consumed parameters cross the host boundary. Other network
    // metadata and learnables retain their original residency and owner.
    let mut prepared = layer.clone();
    for name in ["Weights", "Bias"] {
        let value = layer
            .properties
            .get(name)
            .expect("preflight admitted parameter");
        prepared
            .properties
            .insert(name.into(), crate::gather_if_needed_async(value).await?);
    }
    let weights = tensor_property(&prepared, "Weights", function)?;
    let bias = tensor_property(&prepared, "Bias", function)?;
    let weights_transposed = transpose_2d(&weights, function)?;
    let weights_transposed = floating_tensor_from_f64(
        tensor::tensor_values_f64(&weights_transposed),
        weights_transposed.shape,
        dtype,
        function,
    )?;
    let bias_row = floating_tensor_from_f64(
        tensor::tensor_values_f64(&bias),
        vec![1, weights.rows],
        dtype,
        function,
    )?;
    let weights_handle = crate::builtins::common::gpu_helpers::upload_tensor_protecting_typed(
        provider,
        &weights_transposed,
        &[&current],
    )
    .map_err(|err| {
        provider_failure(
            function,
            "weight upload",
            ProviderExecutionError::from_source(err),
        )
    })?;
    let bias_handle = crate::builtins::common::gpu_helpers::upload_tensor_protecting_typed(
        provider,
        &bias_row,
        &[&current, &weights_handle],
    )
    .map_err(|err| {
        provider_failure(
            function,
            "bias upload",
            ProviderExecutionError::from_source(err),
        )
    })?;
    let product = provider
        .matmul(&current, &weights_handle)
        .await
        .map_err(|err| provider_failure(function, "fully connected matmul", err.into()))?;
    validate_binary_forward_result(
        &product,
        &current,
        &weights_handle,
        provider,
        &[current.shape()[0], weights.rows],
        precision,
        function,
    )?;
    let output = provider
        .elem_add(&product, &bias_handle)
        .await
        .map_err(|err| provider_failure(function, "fully connected bias add", err.into()))?;
    validate_binary_forward_result(
        &output,
        &product,
        &bias_handle,
        provider,
        &[current.shape()[0], weights.rows],
        precision,
        function,
    )?;
    Ok(output)
}

fn provider_failure(
    function: &'static str,
    operation: &str,
    source: ProviderExecutionError,
) -> crate::RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        deep_learning_error(
            function,
            format!("{function}: {operation} failed: {source}"),
        ),
        source,
    )
}

fn validate_binary_forward_result(
    output: &GpuTensorHandle,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    precision: ProviderPrecision,
    function: &'static str,
) -> BuiltinResult<()> {
    use crate::builtins::common::gpu_helpers::{
        binary_gpu_output_matches, BinaryGpuOutputContract, GpuOutputAliasPolicy,
    };
    if binary_gpu_output_matches(
        output,
        left,
        right,
        provider,
        &BinaryGpuOutputContract {
            shape: shape.to_vec(),
            storage: GpuTensorStorage::Real,
            precision: Some(precision),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) {
        Ok(())
    } else {
        Err(crate::runtime_error::build_runtime_error(format!(
            "{function}: provider returned an invalid forward result descriptor"
        ))
        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build())
    }
}
