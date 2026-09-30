use super::super::*;
use super::validation::validate_output;
use crate::builtins::common::gpu_helpers::propagate_output_provenance;
use crate::builtins::deep_learning::model::plan::normalization::NormalizationStage;
use runmat_accelerate_api::DeepLearningLayerNormCapability;

fn parameter(layer: &ObjectInstance, role: ParameterRole) -> Option<&GpuTensorHandle> {
    resident_parameter_handle(layer, role)
}

async fn prepare_parameter(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    layer: &ObjectInstance,
    role: ParameterRole,
    input: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    if let Some(handle) = parameter(layer, role) {
        return Ok(handle.clone());
    }
    let host = parameters::prepare_host_parameter(layer, role, function).await?;
    parameters::upload_parameter(provider, host, input, protected, function)
}

pub(in crate::builtins::deep_learning) async fn forward(
    network: &ObjectInstance,
    input: &GpuTensorHandle,
    stage: &NormalizationStage,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Result<GpuTensorHandle, GpuCapabilityReason>> {
    // A later composed stage/presentation plan can pre-admit device permutes.
    // This route deliberately declines them before any provider effect.
    if target_axes != &stage.geometry.axes {
        return Ok(Err(GpuCapabilityReason::AxisLayout));
    }
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| unsupported_error(function, "no provider owns the GPU dlarray input"))?;
    let Some(Value::Cell(layers)) = network.properties.get("Layers") else {
        unreachable!("host plan admitted network Layers")
    };
    let Some(Value::Object(layer)) = layers.data.get(stage.layer_index) else {
        return Err(deep_learning_error(
            function,
            "admitted normalization stage no longer matches network Layers",
        ));
    };
    let parameters = match super::admit_parameters(layer, provider, input, function)? {
        Ok(parameters) => parameters,
        Err(reason) => return Ok(Err(reason)),
    };
    let scale = parameters.scale;
    let offset = parameters.offset;
    let request = stage.geometry.resident_request();
    let capability = provider
        .deep_learning_layer_norm_capability(input, scale, offset, &request)
        .map_err(|error| {
            provider_failure(function, "resident normalization admission", error.into())
        })?;
    match capability {
        DeepLearningLayerNormCapability::Ready => {}
        DeepLearningLayerNormCapability::UnsupportedOperation => {
            return Ok(Err(GpuCapabilityReason::ProviderOperation))
        }
        DeepLearningLayerNormCapability::ResourceLimit => {
            return Ok(Err(GpuCapabilityReason::ResourceLimit))
        }
    }
    let scale = prepare_parameter(
        provider,
        layer,
        ParameterRole::Scale,
        input,
        &[input],
        function,
    )
    .await?;
    let offset = prepare_parameter(
        provider,
        layer,
        ParameterRole::Offset,
        input,
        &[input, &scale],
        function,
    )
    .await?;
    let output = provider
        .deep_learning_layer_norm(input, &scale, &offset, &request)
        .await
        .map_err(|error| {
            provider_failure(function, "resident layer normalization", error.into())
        })?;
    let mut output = validate_output(
        output,
        input,
        &scale,
        &offset,
        provider,
        stage.geometry.axes.shape(),
        function,
    )?;
    propagate_output_provenance(&mut output, [input, &scale, &offset])?;
    Ok(Ok(output))
}
