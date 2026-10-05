//! Execute a fully pre-admitted sequence. A later unsupported provider result
//! is operational failure, never permission to replay after earlier effects.

use super::super::*;
use super::plan::{Plan, Stage};
use super::trace::{ResidentExecution, ResidentStage, ResidentStageKind, ResidentTrace};
use runmat_accelerate_api::GpuTensorLayout;

#[cfg(test)]
#[path = "execution/tests.rs"]
mod tests;

fn materialize_view(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    use crate::builtins::common::gpu_helpers::{
        reject_gpu_output, unary_gpu_output_matches_shape, GpuOutputAliasPolicy,
        UnaryGpuOutputContract,
    };
    let output = provider.reshape(input, input.shape()).map_err(|error| {
        provider_failure(function, "planned view materialization", error.into())
    })?;
    let alias = if input.shape().iter().all(|extent| *extent > 1) {
        GpuOutputAliasPolicy::RequireDistinctAllocation
    } else {
        GpuOutputAliasPolicy::AllowInput
    };
    if output.view().layout() == GpuTensorLayout::Contiguous
        && unary_gpu_output_matches_shape(
            &output,
            input,
            provider,
            input.shape(),
            UnaryGpuOutputContract {
                storage: GpuTensorStorage::Real,
                precision: runmat_accelerate_api::handle_precision(input),
                integer: None,
                logical: false,
                alias,
            },
        )
    {
        Ok(output)
    } else {
        Err(reject_gpu_output(
            output,
            crate::runtime_error::build_runtime_error(format!(
                "{function}: provider returned an invalid materialized view"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ))
    }
}

pub(super) fn capability_changed(
    function: &'static str,
    reason: GpuCapabilityReason,
) -> crate::RuntimeError {
    crate::runtime_error::build_runtime_error(format!(
        "{function}: resident provider capability changed during execution: {reason}"
    ))
    .with_identifier("RunMat:gpu:ProviderCapabilityChanged")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

/// Failure provenance is fixed at the operation that produced the error.
/// A cancellation arriving after a provider error cannot rewrite that error
/// into an abandoned call.
pub(super) enum SelectedExecutionFailure {
    Abandoned(crate::RuntimeError),
    Operational(crate::RuntimeError),
}

impl SelectedExecutionFailure {
    pub(super) fn into_error(self) -> crate::RuntimeError {
        match self {
            Self::Abandoned(error) | Self::Operational(error) => error,
        }
    }
}

impl From<crate::RuntimeError> for SelectedExecutionFailure {
    fn from(error: crate::RuntimeError) -> Self {
        Self::Operational(error)
    }
}

pub(super) async fn execute(
    network: &ObjectInstance,
    input: &GpuTensorHandle,
    plan: Plan,
    capture: bool,
    runtime: Option<&crate::context::RuntimeContext>,
    function: &'static str,
) -> Result<ResidentExecution, SelectedExecutionFailure> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| admission::invalid_owner(function, admission::GpuOperand::Input))?;
    let Some(Value::Cell(layers)) = network.properties.get("Layers") else {
        unreachable!("host plan admitted Layers")
    };
    let mut current = input.clone();
    let mut recorded = capture.then(|| Vec::with_capacity(plan.stages.len()));
    let consumed_resident_parameters = plan.consumed_resident_parameters;
    for stage in plan.stages {
        admit_cancellation(runtime, function).map_err(SelectedExecutionFailure::Abandoned)?;
        let changes_view_layout = matches!(&stage, Stage::Transpose { .. });
        let stage_input = recorded.as_ref().map(|_| current.clone());
        let (next, kind) = match stage {
            Stage::Materialize => (
                materialize_view(provider, &current, function)?,
                Some(ResidentStageKind::Materialize),
            ),
            Stage::Transpose {
                output_axes,
                output_layout,
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
                    && runmat_accelerate_api::handle_transpose_info(&current).is_none()
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
                    )
                    .into());
                }
                if output.shape() != output_axes.shape() {
                    return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                        output,
                        deep_learning_error(
                            function,
                            "planned transpose differs from admitted axes",
                        ),
                    )
                    .into());
                }
                if output.view().layout() != output_layout {
                    return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                        output,
                        crate::runtime_error::build_runtime_error(format!(
                            "{function}: provider transpose layout differs from admitted view"
                        ))
                        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build(),
                    )
                    .into());
                }
                (output, Some(ResidentStageKind::Transpose { materialize }))
            }
            Stage::Dense { layer_index } => {
                let Value::Object(layer) = &layers.data[layer_index] else {
                    unreachable!("host plan admitted layer objects")
                };
                let (output, weights) =
                    fully_connected_forward_with_weights_gpu(provider, current, layer, function)
                        .await?;
                (
                    output,
                    Some(ResidentStageKind::Dense {
                        layer_index,
                        weights,
                    }),
                )
            }
            Stage::Relu => {
                let output = provider.activation_relu(&current).await.map_err(|error| {
                    provider_failure(function, "planned resident ReLU", error.into())
                })?;
                (
                    validate_unary_forward_result(output, &current, provider, function)?,
                    Some(ResidentStageKind::Relu),
                )
            }
            Stage::Elu { alpha } => {
                let output = provider
                    .activation_elu(&current, alpha)
                    .await
                    .map_err(|error| {
                        provider_failure(function, "planned resident ELU", error.into())
                    })?;
                (
                    validate_unary_forward_result(output, &current, provider, function)?,
                    Some(ResidentStageKind::Elu { alpha }),
                )
            }
            Stage::Softmax => {
                let output = provider
                    .activation_softmax_rows(&current)
                    .await
                    .map_err(|error| {
                        provider_failure(function, "planned resident Softmax", error.into())
                    })?;
                (
                    validate_unary_forward_result(output, &current, provider, function)?,
                    Some(ResidentStageKind::Softmax),
                )
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
                Ok(output) => (output, None),
                Err(reason) => return Err(capability_changed(function, reason).into()),
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
            .ok_or_else(|| capability_changed(function, GpuCapabilityReason::ProviderOperation))
            .map(|output| (output, None))?,
        };
        admit_cancellation(runtime, function).map_err(SelectedExecutionFailure::Abandoned)?;
        current = next;
        if !changes_view_layout && current.view().layout() != GpuTensorLayout::Contiguous {
            return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                current,
                crate::runtime_error::build_runtime_error(format!(
                    "{function}: resident stage returned an unexpected lazy view"
                ))
                .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
            )
            .into());
        }
        if let Some(stages) = recorded.as_mut() {
            stages.push(ResidentStage {
                kind: kind.expect("capture admission excludes unsupported adjoints"),
                input: stage_input.expect("capture retains its stage input"),
                output: current.clone(),
            });
        }
    }
    admit_cancellation(runtime, function).map_err(SelectedExecutionFailure::Abandoned)?;
    if current.shape() != plan.output_axes.shape() {
        return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
            current,
            deep_learning_error(
                function,
                "resident sequence output differs from admitted axes",
            ),
        )
        .into());
    }
    crate::builtins::common::gpu_helpers::propagate_output_provenance(
        &mut current,
        std::iter::once(input).chain(consumed_resident_parameters.iter()),
    )?;
    admit_cancellation(runtime, function).map_err(SelectedExecutionFailure::Abandoned)?;
    let trace = recorded.map(|stages| ResidentTrace {
        stages,
        output: current.clone(),
    });
    Ok(ResidentExecution {
        output: current,
        trace,
    })
}

pub(super) fn admit_cancellation(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &'static str,
) -> BuiltinResult<()> {
    if cancellation_requested(runtime) {
        let message = format!("{function}: network forward execution was cancelled");
        return Err(match runtime {
            Some(runtime) => crate::build_runtime_error(message)
                .with_identifier(format!("{}:Cancelled", runtime.error_namespace()))
                .build(),
            None => crate::runtime_error::semantic_error("RunMat:Cancelled", message),
        });
    }
    Ok(())
}

fn cancellation_requested(runtime: Option<&crate::context::RuntimeContext>) -> bool {
    runtime.map_or_else(crate::interrupt::is_cancelled, |context| {
        context.is_cancelled()
    })
}
