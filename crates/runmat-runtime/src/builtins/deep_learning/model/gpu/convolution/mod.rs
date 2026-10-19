//! Executed learned convolution on resident storage. The canonical host
//! geometry supplies the physical request; this owner only prepares operands
//! and validates the provider's executed results.

mod validation;

use super::stage_failure::StageExecutionFailure;
use super::*;
use crate::builtins::common::gpu_helpers::reject_gpu_outputs;
use runmat_accelerate_api::{
    DeepLearningConv1DCapability, DeepLearningConv1DGradients, DeepLearningConv1DOperands,
    DeepLearningConv1DRequest,
};

pub(in crate::builtins::deep_learning) struct ExecutedConvolution {
    pub output: GpuTensorHandle,
    pub weights: GpuTensorHandle,
    pub bias: GpuTensorHandle,
}

async fn prepared_parameter(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    layer: &ObjectInstance,
    role: ParameterRole,
    input: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    cancel: &dyn Fn() -> BuiltinResult<()>,
    function: &'static str,
) -> Result<GpuTensorHandle, StageExecutionFailure> {
    if let Some(handle) = resident_parameter_handle(layer, role) {
        return Ok(handle.clone());
    }
    let host = parameters::prepare_host_parameter(layer, role, function).await?;
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    Ok(parameters::upload_parameter(
        provider, host, input, protected, function,
    )?)
}

pub(in crate::builtins::deep_learning) async fn forward(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    layer: &ObjectInstance,
    input: &GpuTensorHandle,
    request: &DeepLearningConv1DRequest,
    cancel: &dyn Fn() -> BuiltinResult<()>,
    function: &'static str,
) -> Result<ExecutedConvolution, StageExecutionFailure> {
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    let weights = prepared_parameter(
        provider,
        layer,
        ParameterRole::Weights,
        input,
        &[input],
        cancel,
        function,
    )
    .await?;
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    let bias = prepared_parameter(
        provider,
        layer,
        ParameterRole::Bias,
        input,
        &[input, &weights],
        cancel,
        function,
    )
    .await?;
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    // Recheck the actual leases after upload. A changed capability is an
    // operational failure, never permission to gather/replay the route.
    let capability = provider
        .deep_learning_conv1d_capability(
            DeepLearningConv1DOperands {
                anchor: input,
                input: Some(input),
                weights: Some(&weights),
                bias: Some(&bias),
                upstream: None,
            },
            request,
        )
        .map_err(|error| {
            provider_failure(function, "convolution execution admission", error.into())
        })?;
    match capability {
        DeepLearningConv1DCapability::Ready => {}
        DeepLearningConv1DCapability::UnsupportedOperation => {
            return Err(super::mixed::capability_changed(
                function,
                GpuCapabilityReason::ProviderOperation,
            )
            .into());
        }
        DeepLearningConv1DCapability::ResourceLimit => {
            return Err(super::mixed::capability_changed(
                function,
                GpuCapabilityReason::ResourceLimit,
            )
            .into());
        }
    }
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    let output = provider
        .deep_learning_conv1d(input, &weights, &bias, request)
        .await
        .map_err(|error| {
            provider_failure(function, "resident learned convolution", error.into())
        })?;
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    let output = validation::result(
        output,
        input,
        &weights,
        &bias,
        provider,
        &request.output_shape,
        function,
    )?;
    let output = crate::builtins::common::gpu_helpers::propagate_owned_output_provenance(
        output,
        [input, &weights, &bias],
    )?;
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    Ok(ExecutedConvolution {
        output,
        weights,
        bias,
    })
}

pub(in crate::builtins::deep_learning) async fn adjoint(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    weights: &GpuTensorHandle,
    bias: &GpuTensorHandle,
    upstream: &GpuTensorHandle,
    request: &DeepLearningConv1DRequest,
    cancel: &dyn Fn() -> BuiltinResult<()>,
    function: &'static str,
) -> BuiltinResult<DeepLearningConv1DGradients> {
    cancel()?;
    let capability = provider
        .deep_learning_conv1d_capability(
            DeepLearningConv1DOperands {
                anchor: input,
                input: Some(input),
                weights: Some(weights),
                bias: Some(bias),
                upstream: Some(upstream),
            },
            request,
        )
        .map_err(|error| {
            provider_failure(function, "convolution adjoint admission", error.into())
        })?;
    match capability {
        DeepLearningConv1DCapability::Ready => {}
        DeepLearningConv1DCapability::UnsupportedOperation => {
            return Err(super::mixed::capability_changed(
                function,
                GpuCapabilityReason::ProviderOperation,
            ));
        }
        DeepLearningConv1DCapability::ResourceLimit => {
            return Err(super::mixed::capability_changed(
                function,
                GpuCapabilityReason::ResourceLimit,
            ));
        }
    }
    let result = provider
        .deep_learning_conv1d_adjoint(input, weights, bias, upstream, request)
        .await
        .map_err(|error| {
            provider_failure(function, "resident convolution adjoint", error.into())
        })?;
    let DeepLearningConv1DGradients {
        input: raw_input,
        weights: raw_weights,
        bias: raw_bias,
    } = result;
    if let Err(error) = cancel() {
        return Err(reject_gpu_outputs(
            [raw_input, raw_weights, raw_bias],
            error,
        ));
    }
    let mut input_gradient = match validation::result(
        raw_input,
        input,
        weights,
        bias,
        provider,
        &request.input_shape,
        function,
    ) {
        Ok(gradient) => gradient,
        Err(error) => return Err(reject_gpu_outputs([raw_weights, raw_bias], error)),
    };
    let mut weight_gradient = match validation::result(
        raw_weights,
        input,
        weights,
        bias,
        provider,
        &request.weights_shape,
        function,
    ) {
        Ok(gradient) => gradient,
        Err(error) => return Err(reject_gpu_outputs([input_gradient, raw_bias], error)),
    };
    let mut bias_gradient = match validation::result(
        raw_bias,
        input,
        weights,
        bias,
        provider,
        &request.bias_shape,
        function,
    ) {
        Ok(gradient) => gradient,
        Err(error) => return Err(reject_gpu_outputs([input_gradient, weight_gradient], error)),
    };
    if let Err(error) = validation::distinct_gradients(
        &input_gradient,
        &weight_gradient,
        &bias_gradient,
        upstream,
        function,
    ) {
        return Err(reject_gpu_outputs(
            [input_gradient, weight_gradient, bias_gradient],
            error,
        ));
    }
    if let Err(error) = crate::builtins::common::gpu_helpers::propagate_output_provenance(
        &mut input_gradient,
        [input, weights, bias, upstream],
    ) {
        return Err(reject_gpu_outputs(
            [input_gradient, weight_gradient, bias_gradient],
            error,
        ));
    }
    if let Err(error) = crate::builtins::common::gpu_helpers::propagate_output_provenance(
        &mut weight_gradient,
        [input, weights, bias, upstream],
    ) {
        return Err(reject_gpu_outputs(
            [input_gradient, weight_gradient, bias_gradient],
            error,
        ));
    }
    if let Err(error) = crate::builtins::common::gpu_helpers::propagate_output_provenance(
        &mut bias_gradient,
        [input, weights, bias, upstream],
    ) {
        return Err(reject_gpu_outputs(
            [input_gradient, weight_gradient, bias_gradient],
            error,
        ));
    }
    if let Err(error) = cancel() {
        return Err(reject_gpu_outputs(
            [input_gradient, weight_gradient, bias_gradient],
            error,
        ));
    }
    Ok(DeepLearningConv1DGradients {
        input: input_gradient,
        weights: weight_gradient,
        bias: bias_gradient,
    })
}
