use super::*;
use crate::builtins::common::gpu_helpers::{
    propagate_output_provenance, reject_gpu_output, unary_gpu_output_matches_shape,
    GpuOutputAliasPolicy, UnaryGpuOutputContract,
};
use runmat_accelerate_api::DeepLearningPool1DRequest;

pub(super) async fn forward(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    axis: usize,
    source_axes: &AxisPlan,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let precision = runmat_accelerate_api::handle_precision(input);
    let mut keep_shape = input.shape().to_vec();
    keep_shape[axis] = 1;
    let reduced = match provider
        .deep_learning_global_average_pool_1d(
            input,
            DeepLearningPool1DRequest {
                axis_zero_based: axis,
            },
        )
        .await
    {
        Ok(output) => output,
        Err(error)
            if runmat_accelerate_api::is_unsupported_provider_operation(&error)
                && !runmat_accelerate_api::handle_is_explicit(input) =>
        {
            return Ok(None);
        }
        Err(error) => {
            return Err(provider_failure(
                function,
                "resident global average pooling",
                error.into(),
            ));
        }
    };
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
    if source_axes != target_axes {
        let lowering = source_axes
            .lower_to(target_axes.layout())
            .map_err(|error| {
                deep_learning_error(
                    function,
                    format!("{function}: invalid GPU output axes: {error:?}"),
                )
            })?;
        let reordered = match provider.permute(&current, lowering.target_to_source()) {
            Ok(output) => output,
            Err(error)
                if runmat_accelerate_api::is_unsupported_provider_operation(&error)
                    && !runmat_accelerate_api::handle_is_explicit(input) =>
            {
                return Ok(None);
            }
            Err(error) => {
                return Err(provider_failure(
                    function,
                    "resident pooled output presentation",
                    error.into(),
                ));
            }
        };
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
