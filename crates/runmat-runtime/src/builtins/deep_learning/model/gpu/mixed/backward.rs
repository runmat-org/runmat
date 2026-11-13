//! Native resident adjoints over the exact forward stage leases.
//! No host transfer or numerical replay occurs in this module.

use super::super::*;
use super::trace::{ResidentStageKind, ResidentTrace};
use crate::builtins::common::gpu_helpers;
use crate::builtins::deep_learning::model::parameters::ParameterRole;
use runmat_accelerate_api::{AccelProvider, ProviderActivationAdjoint};

#[cfg(test)]
mod tests;

pub(in crate::builtins::deep_learning) struct ResidentGradients {
    pub input: GpuTensorHandle,
    pub parameters: Vec<(usize, ParameterRole, GpuTensorHandle)>,
}

fn provider_result<E: Into<runmat_accelerate_api::ProviderExecutionError>>(
    result: std::result::Result<GpuTensorHandle, E>,
    operation: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    result.map_err(|error| provider_failure("dlgradient", operation, error.into()))
}

fn checked_binary(
    result: GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
) -> BuiltinResult<GpuTensorHandle> {
    let precision = runmat_accelerate_api::handle_precision(lhs).ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: resident adjoint lost floating class",
        )
    })?;
    validate_binary_forward_result(result, lhs, rhs, provider, shape, precision, "dlgradient")
}

fn checked_reduction(
    result: GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
) -> BuiltinResult<GpuTensorHandle> {
    use crate::builtins::common::gpu_helpers::{
        reject_gpu_output, unary_gpu_output_matches_shape, GpuOutputAliasPolicy,
        UnaryGpuOutputContract,
    };
    if unary_gpu_output_matches_shape(
        &result,
        input,
        provider,
        shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: if shape == input.shape() {
                GpuOutputAliasPolicy::AllowInput
            } else {
                GpuOutputAliasPolicy::RequireDistinctAllocation
            },
        },
    ) {
        Ok(result)
    } else {
        Err(reject_gpu_output(
            result,
            crate::runtime_error::build_runtime_error(
                "dlgradient: provider returned an invalid resident reduction descriptor",
            )
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ))
    }
}

fn checked_native_sum_reduction(
    result: GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
) -> BuiltinResult<GpuTensorHandle> {
    use crate::builtins::common::gpu_helpers::{
        reject_gpu_output, unary_gpu_output_matches_shape, GpuOutputAliasPolicy,
        UnaryGpuOutputContract,
    };
    let valid = unary_gpu_output_matches_shape(
        &result,
        input,
        provider,
        shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: if shape == input.shape() {
                GpuOutputAliasPolicy::AllowInput
            } else {
                GpuOutputAliasPolicy::RequireDistinctAllocation
            },
        },
    ) && result.allocation_lease().validate_live().is_ok()
        && gpu_helpers::expected_handle_numeric_element_type(&result).is_ok();
    if !valid {
        return Err(reject_gpu_output(
            result,
            deep_learning_error(
                "dlgradient",
                "dlgradient: provider returned a malformed native sum stage",
            ),
        ));
    }
    checked_reduction(result, input, provider, shape)
}

fn checked_view(
    result: GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
) -> BuiltinResult<GpuTensorHandle> {
    use crate::builtins::common::gpu_helpers::{
        reject_gpu_output, unary_gpu_output_matches_shape, GpuOutputAliasPolicy,
        UnaryGpuOutputContract,
    };
    if unary_gpu_output_matches_shape(
        &result,
        input,
        provider,
        shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::AllowInput,
        },
    ) {
        Ok(result)
    } else {
        Err(reject_gpu_output(
            result,
            crate::runtime_error::build_runtime_error(
                "dlgradient: provider returned an invalid resident view descriptor",
            )
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ))
    }
}

pub(in crate::builtins::deep_learning) async fn reverse(
    trace: &ResidentTrace,
    seed: GpuTensorHandle,
    cancel: &dyn Fn() -> BuiltinResult<()>,
) -> BuiltinResult<ResidentGradients> {
    cancel()?;
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(&trace.output)
        .ok_or_else(|| {
        deep_learning_error("dlgradient", "dlgradient: invalid resident owner")
    })?;
    crate::builtins::common::gpu_helpers::exact_provider_for_binary_inputs(&seed, &trace.output)
        .map_err(|_| {
            deep_learning_error(
                "dlgradient",
                "dlgradient: adjoint owner differs from forward output",
            )
        })?;
    if seed.shape() != trace.output.shape()
        || runmat_accelerate_api::handle_precision(&seed)
            != runmat_accelerate_api::handle_precision(&trace.output)
    {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: resident seed differs from executed output geometry or class",
        ));
    }
    let mut gradient = seed;
    let mut parameters = Vec::new();
    for stage in trace.stages.iter().rev() {
        cancel()?;
        if gradient.shape() != stage.output.shape() {
            return Err(deep_learning_error(
                "dlgradient",
                "dlgradient: resident stage adjoint differs from recorded output",
            ));
        }
        gradient = match &stage.kind {
            ResidentStageKind::Materialize => gradient,
            ResidentStageKind::Transpose { materialize } => {
                let output = provider_result(
                    if *materialize {
                        provider.permute(&gradient, &[1, 0]).and_then(|output| {
                            output.ok_or_else(|| {
                                runmat_accelerate_api::ProviderExecutionError::from(
                                    anyhow::anyhow!("required resident permutation unavailable"),
                                )
                                .with_unproven_effects()
                            })
                        })
                    } else {
                        provider.transpose(&gradient).map_err(|error| {
                            runmat_accelerate_api::ProviderExecutionError::from(error)
                                .with_unproven_effects()
                        })
                    },
                    "resident transpose adjoint",
                )?;
                let output = validate_layout_transpose(output, &gradient, provider, "dlgradient")?;
                if output.shape() != stage.input.shape() {
                    return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                        output,
                        deep_learning_error(
                            "dlgradient",
                            "dlgradient: inverse transpose shape differs from stage input",
                        ),
                    ));
                }
                output
            }
            ResidentStageKind::Dense {
                layer_index,
                weights,
            } => {
                let weights_transposed = provider_result(
                    provider.transpose(weights),
                    "resident dense weight transpose",
                )?;
                let weights_transposed =
                    validate_layout_transpose(weights_transposed, weights, provider, "dlgradient")?;
                let dx = provider_result(
                    provider.matmul(&gradient, &weights_transposed).await,
                    "resident dense input adjoint",
                )?;
                cancel()?;
                let dx = checked_binary(
                    dx,
                    &gradient,
                    &weights_transposed,
                    provider,
                    stage.input.shape(),
                )?;
                let input_transposed = provider_result(
                    provider.transpose(&stage.input),
                    "resident dense input transpose",
                )?;
                let input_transposed = validate_layout_transpose(
                    input_transposed,
                    &stage.input,
                    provider,
                    "dlgradient",
                )?;
                let dw = provider_result(
                    provider.matmul(&input_transposed, &gradient).await,
                    "resident dense weight adjoint",
                )?;
                cancel()?;
                // The recorded weight is the prepared [Cin,Cout] matmul
                // operand; the canonical Learnables row is [Cout,Cin].
                let dw =
                    checked_binary(dw, &input_transposed, &gradient, provider, weights.shape())?;
                let transposed =
                    provider_result(provider.transpose(&dw), "resident dense weight shape")?;
                let dw = validate_layout_transpose(transposed, &dw, provider, "dlgradient")?;
                let db = provider_result(
                    provider.reduce_real_sum_native_dim(&gradient, 0).await,
                    "resident dense bias adjoint",
                )?;
                cancel()?;
                let db = checked_native_sum_reduction(
                    db,
                    &gradient,
                    provider,
                    &[1, gradient.shape()[1]],
                )?;
                let reshaped = provider_result(
                    provider.reshape(&db, &[gradient.shape()[1], 1]),
                    "resident dense bias shape",
                )?;
                let db = checked_view(reshaped, &db, provider, &[gradient.shape()[1], 1])?;
                parameters.push((*layer_index, ParameterRole::Weights, dw));
                parameters.push((*layer_index, ParameterRole::Bias, db));
                dx
            }
            ResidentStageKind::Conv {
                layer_index,
                request,
                weights,
                bias,
            } => {
                let derivatives = super::super::convolution::adjoint(
                    provider,
                    &stage.input,
                    weights,
                    bias,
                    &gradient,
                    request,
                    cancel,
                    "dlgradient",
                )
                .await?;
                parameters.push((*layer_index, ParameterRole::Weights, derivatives.weights));
                parameters.push((*layer_index, ParameterRole::Bias, derivatives.bias));
                derivatives.input
            }
            ResidentStageKind::Relu | ResidentStageKind::Elu { .. } => {
                let operation = match &stage.kind {
                    ResidentStageKind::Relu => ProviderActivationAdjoint::Relu,
                    ResidentStageKind::Elu { alpha } => {
                        ProviderActivationAdjoint::Elu { alpha: *alpha }
                    }
                    _ => unreachable!("the branch admitted only activation stages"),
                };
                let output = provider_result(
                    provider
                        .activation_adjoint(&stage.input, &gradient, operation)
                        .await,
                    "resident activation adjoint",
                )?;
                cancel()?;
                checked_binary(
                    output,
                    &stage.input,
                    &gradient,
                    provider,
                    stage.input.shape(),
                )?
            }
            ResidentStageKind::Softmax => {
                let weighted = provider_result(
                    provider.elem_mul(&stage.output, &gradient).await,
                    "resident softmax weighted seed",
                )?;
                cancel()?;
                let weighted = checked_binary(
                    weighted,
                    &stage.output,
                    &gradient,
                    provider,
                    stage.output.shape(),
                )?;
                let sum = provider_result(
                    provider.reduce_real_sum_native_dim(&weighted, 1).await,
                    "resident softmax row reduction",
                )?;
                cancel()?;
                let sum = checked_native_sum_reduction(
                    sum,
                    &weighted,
                    provider,
                    &[stage.output.shape()[0], 1],
                )?;
                let centered = provider_result(
                    provider.elem_sub(&gradient, &sum).await,
                    "resident softmax centered seed",
                )?;
                cancel()?;
                let centered =
                    checked_binary(centered, &gradient, &sum, provider, stage.output.shape())?;
                let output = provider_result(
                    provider.elem_mul(&stage.output, &centered).await,
                    "resident softmax adjoint",
                )?;
                cancel()?;
                checked_binary(
                    output,
                    &stage.output,
                    &centered,
                    provider,
                    stage.input.shape(),
                )?
            }
            ResidentStageKind::Normalize { record } => {
                let derivatives = normalization::adjoint(
                    provider,
                    &stage.input,
                    &gradient,
                    record,
                    cancel,
                    "dlgradient",
                )
                .await?;
                parameters.push((record.layer_index, ParameterRole::Scale, derivatives.scale));
                parameters.push((
                    record.layer_index,
                    ParameterRole::Offset,
                    derivatives.offset,
                ));
                derivatives.input
            }
            ResidentStageKind::Pool {
                axis,
                input_axes,
                output_axes,
            } => {
                pooling::adjoint(
                    provider,
                    &stage.input,
                    &gradient,
                    *axis,
                    input_axes,
                    output_axes,
                    output_axes,
                    cancel,
                    "dlgradient",
                )
                .await?
            }
        };
    }
    cancel()?;
    Ok(ResidentGradients {
        input: gradient,
        parameters,
    })
}
