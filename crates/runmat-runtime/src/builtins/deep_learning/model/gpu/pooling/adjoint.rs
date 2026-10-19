//! Invert the admitted physical presentation before a resident pool adjoint.
//! The provider receives the same singleton-axis geometry as its forward
//! reduction; semantic axis lowering remains owned by `AxisPlan`.

use super::*;

fn checked_seed_view(
    result: GpuTensorHandle,
    operand: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    alias: GpuOutputAliasPolicy,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    if unary_gpu_output_matches_shape(
        &result,
        operand,
        provider,
        shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(operand),
            integer: None,
            logical: false,
            alias,
        },
    ) {
        Ok(result)
    } else {
        Err(reject_gpu_output(
            result,
            crate::build_runtime_error(format!(
                "{function}: resident pooling adjoint returned an invalid inverse view"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ))
    }
}

pub(in crate::builtins::deep_learning::model::gpu) async fn adjoint(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    upstream: &GpuTensorHandle,
    axis: usize,
    input_axes: &AxisPlan,
    source_axes: &AxisPlan,
    target_axes: &AxisPlan,
    cancel: &dyn Fn() -> BuiltinResult<()>,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    cancel()?;
    let planned = route(input_axes, source_axes, target_axes, axis, function)?;
    if input.shape() != input_axes.shape() || upstream.shape() != target_axes.shape() {
        return Err(deep_learning_error(
            function,
            "resident pooling adjoint differs from executed stage geometry",
        ));
    }
    if !unary_gpu_output_matches_shape(
        upstream,
        input,
        provider,
        target_axes.shape(),
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::AllowInput,
        },
    ) {
        return Err(deep_learning_error(
            function,
            "resident pooling adjoint seed has the wrong owner or numerical class",
        ));
    }
    let mut seed = upstream.clone();
    if planned
        .target_to_source
        .iter()
        .enumerate()
        .any(|(target, &source)| target != source)
    {
        let mut source_to_target = vec![usize::MAX; planned.target_to_source.len()];
        for (target, &source) in planned.target_to_source.iter().enumerate() {
            let Some(slot) = source_to_target.get_mut(source) else {
                return Err(deep_learning_error(
                    function,
                    "invalid pooled axis permutation",
                ));
            };
            if *slot != usize::MAX {
                return Err(deep_learning_error(
                    function,
                    "repeated pooled axis permutation",
                ));
            }
            *slot = target;
        }
        if source_to_target.contains(&usize::MAX) {
            return Err(deep_learning_error(
                function,
                "incomplete pooled axis permutation",
            ));
        }
        let inverse = provider
            .permute(&seed, &source_to_target)
            .and_then(|output| {
                output.ok_or_else(|| {
                    runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                        "required pooling inverse presentation unavailable"
                    ))
                    .with_unproven_effects()
                })
            })
            .map_err(|error| {
                provider_failure(function, "resident pooling inverse presentation", error)
            })?;
        seed = checked_seed_view(
            inverse,
            &seed,
            provider,
            source_axes.shape(),
            presentation_alias_policy(source_axes, &planned),
            function,
        )?;
        cancel()?;
    }
    if seed.shape() != source_axes.shape() {
        return Err(deep_learning_error(
            function,
            "resident pooling inverse presentation has an invalid shape",
        ));
    }
    let mut kept_shape = input_axes.shape().to_vec();
    kept_shape[axis] = 1;
    if seed.shape() != kept_shape {
        let reshaped = provider.reshape(&seed, &kept_shape).map_err(|error| {
            provider_failure(
                function,
                "resident pooling inverse axis removal",
                error.into(),
            )
        })?;
        seed = checked_seed_view(
            reshaped,
            &seed,
            provider,
            &kept_shape,
            GpuOutputAliasPolicy::AllowInput,
            function,
        )?;
        cancel()?;
    }
    let capability = provider
        .deep_learning_global_average_pool_1d_adjoint_capability(
            input,
            input_axes.shape(),
            Some(&seed),
            planned.reduction,
        )
        .map_err(|error| {
            provider_failure(function, "resident pooling adjoint admission", error.into())
        })?;
    if capability != runmat_accelerate_api::DeepLearningPool1DCapability::Ready {
        return Err(deep_learning_error(
            function,
            "resident pooling adjoint capability changed after forward execution",
        ));
    }
    cancel()?;
    let output = provider
        .deep_learning_global_average_pool_1d_adjoint(input, &seed, planned.reduction)
        .await
        .map_err(|error| provider_failure(function, "resident pooling adjoint", error.into()))?;
    cancel()?;
    let output = validate(
        output,
        &seed,
        input,
        provider,
        input_axes.shape(),
        runmat_accelerate_api::handle_precision(input),
        GpuOutputAliasPolicy::RequireDistinctAllocation,
        function,
    )?;
    let output = propagate_owned_output_provenance(output, [input, upstream])?;
    Ok(output)
}
