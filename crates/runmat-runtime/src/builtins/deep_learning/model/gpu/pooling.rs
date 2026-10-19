use super::*;
use crate::builtins::common::gpu_helpers::{
    propagate_owned_output_provenance, reject_gpu_output, unary_gpu_output_matches_shape,
    GpuOutputAliasPolicy, UnaryGpuOutputContract,
};
use runmat_accelerate_api::{
    DeepLearningPool1DRequest, DeepLearningPool1DResultAxis, DeepLearningPool1DRoute,
};

#[path = "pooling/adjoint.rs"]
mod adjoint;
pub(super) use adjoint::adjoint;

pub(super) fn route(
    input_axes: &AxisPlan,
    source_axes: &AxisPlan,
    target_axes: &AxisPlan,
    axis: usize,
    function: &'static str,
) -> BuiltinResult<DeepLearningPool1DRoute> {
    let Some(&pool_size) = input_axes.shape().get(axis) else {
        return Err(deep_learning_error(
            function,
            "pool axis exceeds admitted input rank",
        ));
    };
    if pool_size == 0 {
        return Err(deep_learning_error(
            function,
            "cannot average an empty pool axis",
        ));
    }
    let expected_source = input_axes
        .after_global_average_pooling_1d()
        .map_err(|error| {
            deep_learning_error(
                function,
                format!("{function}: invalid pooling axes: {error:?}"),
            )
        })?;
    if source_axes != &expected_source {
        return Err(deep_learning_error(
            function,
            format!("{function}: pooled output geometry differs from admitted axes"),
        ));
    }
    let mut kept_shape = input_axes.shape().to_vec();
    kept_shape[axis] = 1;
    let result_axis = if source_axes.shape() == kept_shape {
        DeepLearningPool1DResultAxis::KeepSingleton
    } else {
        let mut removed_shape = kept_shape;
        removed_shape.remove(axis);
        if source_axes.shape() != removed_shape {
            return Err(deep_learning_error(
                function,
                format!("{function}: pooled axis shape is not a supported reduction view"),
            ));
        }
        DeepLearningPool1DResultAxis::RemoveReduced
    };
    let lowering = source_axes
        .lower_to(target_axes.layout())
        .map_err(|error| {
            deep_learning_error(
                function,
                format!("{function}: invalid GPU output axes: {error:?}"),
            )
        })?;
    if lowering.target_shape() != target_axes.shape() {
        return Err(deep_learning_error(
            function,
            format!("{function}: presented output shape differs from admitted axes"),
        ));
    }
    Ok(DeepLearningPool1DRoute {
        reduction: DeepLearningPool1DRequest {
            axis_zero_based: axis,
        },
        result_axis,
        source_shape: source_axes.shape().to_vec(),
        target_shape: target_axes.shape().to_vec(),
        target_to_source: lowering.target_to_source().to_vec(),
    })
}

pub(super) async fn forward(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    axis: usize,
    input_axes: &AxisPlan,
    source_axes: &AxisPlan,
    target_axes: &AxisPlan,
    cancel: &dyn Fn() -> BuiltinResult<()>,
    function: &'static str,
) -> Result<Option<GpuTensorHandle>, super::stage_failure::StageExecutionFailure> {
    let planned = route(input_axes, source_axes, target_axes, axis, function)?;
    let precision = runmat_accelerate_api::handle_precision(input);
    let mut keep_shape = input.shape().to_vec();
    keep_shape[axis] = 1;
    cancel().map_err(super::stage_failure::StageExecutionFailure::Abandoned)?;
    let reduced = provider
        .deep_learning_global_average_pool_1d(input, planned.reduction)
        .await
        .map_err(|error| {
            provider_failure(function, "resident global average pooling", error.into())
        })?;
    cancel().map_err(super::stage_failure::StageExecutionFailure::Abandoned)?;
    let mut current = validate(
        reduced,
        input,
        input,
        provider,
        &keep_shape,
        precision,
        GpuOutputAliasPolicy::RequireDistinctAllocation,
        function,
    )?;
    if current.shape() != source_axes.shape() {
        cancel().map_err(super::stage_failure::StageExecutionFailure::Abandoned)?;
        let reshaped = provider
            .reshape(&current, source_axes.shape())
            .map_err(|error| provider_failure(function, "pooled output reshape", error.into()))?;
        current = validate(
            reshaped,
            &current,
            input,
            provider,
            source_axes.shape(),
            precision,
            GpuOutputAliasPolicy::AllowInput,
            function,
        )?;
    }
    if planned
        .target_to_source
        .iter()
        .enumerate()
        .any(|(target, &source)| target != source)
    {
        cancel().map_err(super::stage_failure::StageExecutionFailure::Abandoned)?;
        let alias = presentation_alias_policy(source_axes, &planned);
        let reordered = provider
            .permute(&current, &planned.target_to_source)
            .map_err(|source| source.after_effect())
            .and_then(|output| {
                output.ok_or_else(|| {
                    runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                        "required pooled output presentation unavailable"
                    ))
                    .after_effect()
                })
            })
            .map_err(|error| {
                provider_failure(function, "resident pooled output presentation", error)
            })?;
        current = validate(
            reordered,
            &current,
            input,
            provider,
            target_axes.shape(),
            precision,
            alias,
            function,
        )?;
    }
    cancel().map_err(super::stage_failure::StageExecutionFailure::Abandoned)?;
    let current = propagate_owned_output_provenance(current, [input])?;
    Ok(Some(current))
}

fn presentation_alias_policy(
    source_axes: &AxisPlan,
    route: &DeepLearningPool1DRoute,
) -> GpuOutputAliasPolicy {
    // In column-major storage, moving only singleton axes changes metadata
    // but not the linear element order. Materialization is necessary exactly
    // when the relative order of non-singleton source axes changes.
    let mut last_non_singleton = None;
    let reorders_data = route
        .target_to_source
        .iter()
        .copied()
        .filter(|&source| source_axes.shape()[source] > 1)
        .any(|source| {
            let out_of_order = last_non_singleton.is_some_and(|previous| source < previous);
            last_non_singleton = Some(source);
            out_of_order
        });
    if reorders_data {
        GpuOutputAliasPolicy::RequireDistinctAllocation
    } else {
        GpuOutputAliasPolicy::AllowInput
    }
}

fn validate(
    output: GpuTensorHandle,
    operand: &GpuTensorHandle,
    original_input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    precision: Option<ProviderPrecision>,
    alias: GpuOutputAliasPolicy,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    // Pooling has already required a fresh allocation. Later metadata views
    // may share that reduced allocation, but no later provider result may
    // substitute a view of the caller's original borrowed input.
    if unary_gpu_output_matches_shape(
        &output,
        operand,
        provider,
        shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision,
            integer: None,
            logical: false,
            alias,
        },
    ) && !output.shares_allocation_with(original_input)
    {
        Ok(output)
    } else {
        Err(reject_gpu_output(
            output,
            crate::build_runtime_error(format!(
                "{function}: provider returned an invalid pooling output descriptor"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ))
    }
}

#[cfg(test)]
mod tests;
