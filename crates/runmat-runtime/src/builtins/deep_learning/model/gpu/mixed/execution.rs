//! Execute a fully pre-admitted sequence. A later unsupported provider result
//! is operational failure, never permission to replay after earlier effects.

use super::super::*;
use super::plan::{Plan, Stage};

fn capability_changed(function: &'static str, reason: GpuCapabilityReason) -> crate::RuntimeError {
    crate::runtime_error::build_runtime_error(format!(
        "{function}: resident provider capability changed during execution: {reason}"
    ))
    .with_identifier("RunMat:gpu:ProviderCapabilityChanged")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

pub(super) async fn execute(
    network: &ObjectInstance,
    input: &GpuTensorHandle,
    plan: Plan,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| admission::invalid_owner(function, admission::GpuOperand::Input))?;
    let Some(Value::Cell(layers)) = network.properties.get("Layers") else {
        unreachable!("host plan admitted Layers")
    };
    let mut current = input.clone();
    for stage in plan.stages {
        current = match stage {
            Stage::Transpose {
                output_axes,
                materialize,
            } => {
                let output = (if materialize {
                    provider.permute(&current, &[1, 0])
                } else {
                    provider.transpose(&current)
                })
                .map_err(|error| {
                    provider_failure(function, "planned layout transpose", error.into())
                })?;
                let output = validate_layout_transpose(output, &current, provider, function)?;
                if materialize
                    && current.shape().iter().all(|extent| *extent > 1)
                    && output.shares_allocation_with(&current)
                {
                    return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                        output,
                        crate::runtime_error::build_runtime_error(format!(
                            "{function}: planned physical permutation returned an unmaterialized view"
                        ))
                        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build(),
                    ));
                }
                if output.shape() != output_axes.shape() {
                    return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                        output,
                        deep_learning_error(
                            function,
                            "planned transpose differs from admitted axes",
                        ),
                    ));
                }
                output
            }
            Stage::Dense { layer_index } => {
                let Value::Object(layer) = &layers.data[layer_index] else {
                    unreachable!("host plan admitted layer objects")
                };
                fully_connected_forward_gpu(provider, current, layer, function).await?
            }
            Stage::Relu => {
                let output = provider.scalar_max(&current, 0.0).map_err(|error| {
                    provider_failure(function, "planned resident ReLU", error.into())
                })?;
                validate_unary_forward_result(output, &current, provider, function)?
            }
            Stage::Elu { alpha } => {
                let output = provider
                    .activation_elu(&current, alpha)
                    .await
                    .map_err(|error| {
                        provider_failure(function, "planned resident ELU", error.into())
                    })?;
                validate_unary_forward_result(output, &current, provider, function)?
            }
            Stage::Softmax => {
                let output = provider
                    .activation_softmax_rows(&current)
                    .await
                    .map_err(|error| {
                        provider_failure(function, "planned resident Softmax", error.into())
                    })?;
                validate_unary_forward_result(output, &current, provider, function)?
            }
            Stage::Normalize { stage } => match normalization::forward(
                network,
                &current,
                &stage,
                &stage.geometry.axes,
                function,
            )
            .await?
            {
                Ok(output) => output,
                Err(reason) => return Err(capability_changed(function, reason)),
            },
            Stage::Pool {
                axis,
                input_axes,
                output_axes,
            } => pooling::forward(
                provider,
                &current,
                axis,
                &input_axes,
                &output_axes,
                &output_axes,
                function,
            )
            .await?
            .ok_or_else(|| capability_changed(function, GpuCapabilityReason::ProviderOperation))?,
        };
    }
    if current.shape() != plan.output_axes.shape() {
        return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
            current,
            deep_learning_error(
                function,
                "resident sequence output differs from admitted axes",
            ),
        ));
    }
    crate::builtins::common::gpu_helpers::propagate_output_provenance(&mut current, [input])?;
    Ok(current)
}
