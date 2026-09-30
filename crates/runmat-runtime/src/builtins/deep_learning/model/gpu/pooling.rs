use super::*;
use crate::builtins::common::gpu_helpers::{
    propagate_output_provenance, reject_gpu_output, unary_gpu_output_matches_shape,
    GpuOutputAliasPolicy, UnaryGpuOutputContract,
};
use runmat_accelerate_api::{DeepLearningPool1DRequest, DeepLearningPool1DRoute};

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
    function: &'static str,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let planned = route(input_axes, source_axes, target_axes, axis, function)?;
    let precision = runmat_accelerate_api::handle_precision(input);
    let mut keep_shape = input.shape().to_vec();
    keep_shape[axis] = 1;
    let reduced = provider
        .deep_learning_global_average_pool_1d(input, planned.reduction)
        .await
        .map_err(|error| {
            provider_failure(function, "resident global average pooling", error.into())
        })?;
    let mut current = validate(
        reduced,
        input,
        provider,
        &keep_shape,
        precision,
        GpuOutputAliasPolicy::RequireDistinctAllocation,
        function,
    )?;
    if current.shape() != source_axes.shape() {
        let reshaped = provider
            .reshape(&current, source_axes.shape())
            .map_err(|error| provider_failure(function, "pooled output reshape", error.into()))?;
        current = validate(
            reshaped,
            &current,
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
        let reordered = provider
            .permute(&current, &planned.target_to_source)
            .map_err(|error| {
                provider_failure(
                    function,
                    "resident pooled output presentation",
                    error.into(),
                )
            })?;
        current = validate(
            reordered,
            &current,
            provider,
            target_axes.shape(),
            precision,
            GpuOutputAliasPolicy::AllowInput,
            function,
        )?;
    }
    propagate_output_provenance(&mut current, [input])?;
    Ok(Some(current))
}

fn validate(
    output: GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    precision: Option<ProviderPrecision>,
    alias: GpuOutputAliasPolicy,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    if unary_gpu_output_matches_shape(
        &output,
        input,
        provider,
        shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision,
            integer: None,
            logical: false,
            alias,
        },
    ) {
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
