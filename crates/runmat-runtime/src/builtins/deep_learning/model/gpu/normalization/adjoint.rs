//! Reverse one recorded resident normalization using its executed statistics.

use super::super::*;
use super::execution::NormalizationRecord;
use crate::builtins::common::gpu_helpers::{
    propagate_output_provenance, reject_gpu_output, reject_gpu_outputs,
    unary_gpu_output_matches_shape, GpuOutputAliasPolicy, UnaryGpuOutputContract,
};
use runmat_accelerate_api::{DeepLearningLayerNormCapability, DeepLearningLayerNormGradients};

fn checked_gradient(
    result: GpuTensorHandle,
    operand: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    if unary_gpu_output_matches_shape(
        &result,
        operand,
        provider,
        operand.shape(),
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(operand),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) && protected
        .iter()
        .all(|source| !result.shares_allocation_with(source))
    {
        Ok(result)
    } else {
        Err(reject_gpu_output(
            result,
            crate::build_runtime_error(format!(
                "{function}: provider returned an invalid normalization adjoint descriptor"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ))
    }
}

pub(in crate::builtins::deep_learning) async fn adjoint(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    upstream: &GpuTensorHandle,
    record: &NormalizationRecord,
    cancel: &dyn Fn() -> BuiltinResult<()>,
    function: &'static str,
) -> BuiltinResult<DeepLearningLayerNormGradients> {
    cancel()?;
    if input.shape() != record.request.shape.as_slice()
        || upstream.shape() != record.request.shape.as_slice()
        || record.scale.shape() != record.request.scale_shape.as_slice()
        || record.offset.shape() != record.request.offset_shape.as_slice()
    {
        return Err(deep_learning_error(
            function,
            "recorded normalization adjoint geometry changed",
        ));
    }
    let capability = provider
        .deep_learning_layer_norm_adjoint_capability(
            input,
            Some(&record.scale),
            Some(&record.offset),
            Some(&record.statistics),
            Some(upstream),
            &record.request,
        )
        .map_err(|error| {
            provider_failure(function, "normalization adjoint admission", error.into())
        })?;
    if capability != DeepLearningLayerNormCapability::Ready {
        return Err(deep_learning_error(
            function,
            "normalization adjoint capability changed after forward execution",
        ));
    }
    let result = provider
        .deep_learning_layer_norm_adjoint(
            input,
            &record.scale,
            &record.offset,
            &record.statistics,
            upstream,
            &record.request,
        )
        .await
        .map_err(|error| {
            provider_failure(function, "resident normalization adjoint", error.into())
        })?;
    let DeepLearningLayerNormGradients {
        input: raw_input,
        scale: raw_scale,
        offset: raw_offset,
    } = result;
    if let Err(error) = cancel() {
        return Err(reject_gpu_outputs(
            [raw_input, raw_scale, raw_offset],
            error,
        ));
    }
    let protected = [
        input,
        upstream,
        &record.scale,
        &record.offset,
        &record.statistics,
    ];
    let mut dx = match checked_gradient(raw_input, input, &protected, provider, function) {
        Ok(gradient) => gradient,
        Err(error) => return Err(reject_gpu_outputs([raw_scale, raw_offset], error)),
    };
    let mut dscale =
        match checked_gradient(raw_scale, &record.scale, &protected, provider, function) {
            Ok(gradient) => gradient,
            Err(error) => return Err(reject_gpu_outputs([dx, raw_offset], error)),
        };
    let mut doffset =
        match checked_gradient(raw_offset, &record.offset, &protected, provider, function) {
            Ok(gradient) => gradient,
            Err(error) => return Err(reject_gpu_outputs([dx, dscale], error)),
        };
    if dx.shares_allocation_with(&dscale)
        || dx.shares_allocation_with(&doffset)
        || dscale.shares_allocation_with(&doffset)
    {
        return Err(reject_gpu_outputs(
            [dx, dscale, doffset],
            deep_learning_error(function, "normalization adjoint results alias each other"),
        ));
    }
    if let Err(error) = propagate_output_provenance(&mut dx, [input, upstream, &record.scale]) {
        return Err(reject_gpu_outputs([dx, dscale, doffset], error));
    }
    if let Err(error) = propagate_output_provenance(&mut dscale, [input, upstream, &record.scale]) {
        return Err(reject_gpu_outputs([dx, dscale, doffset], error));
    }
    if let Err(error) = propagate_output_provenance(&mut doffset, [input, upstream, &record.offset])
    {
        return Err(reject_gpu_outputs([dx, dscale, doffset], error));
    }
    Ok(DeepLearningLayerNormGradients {
        input: dx,
        scale: dscale,
        offset: doffset,
    })
}
