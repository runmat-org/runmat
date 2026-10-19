//! Physical descriptor and requested-class admission for resident mean.
use super::*;
use gpu_helpers::{GpuOutputAliasPolicy, UnaryGpuOutputContract};
use runmat_accelerate_api::GpuTensorStorage;
#[path = "admission/include.rs"]
mod include;
pub(super) use include::{
    admit_include_permutation, admit_include_reduction, admit_include_reshape,
};

#[derive(Clone, Copy)]
pub(super) enum OmitNanStageAlias<'a> {
    Fresh,
    SingletonIdentity(&'a GpuTensorHandle),
}

#[derive(Clone, Copy)]
pub(super) enum ClassAdaptationStage {
    BeforeReduction,
    AfterReduction,
}

impl<'a> OmitNanStageAlias<'a> {
    pub(super) fn for_reduction(current: &'a GpuTensorHandle, dim: usize) -> Self {
        if current.shape()[dim] == 1
            && runmat_accelerate_api::handle_precision(current) == Some(ProviderPrecision::F64)
        {
            Self::SingletonIdentity(current)
        } else {
            Self::Fresh
        }
    }
}

fn exact_execution_owner(
    provider: &dyn AccelProvider,
    input: &GpuTensorHandle,
) -> Option<&'static dyn AccelProvider> {
    gpu_helpers::exact_provider_for_handle(input)
        .filter(|owner| runmat_accelerate_api::same_provider_instance(*owner, provider))
}

pub(super) fn desired_precision(
    input: &GpuTensorHandle,
    output_class: &OutputTemplate,
) -> Option<ProviderPrecision> {
    match output_class {
        OutputTemplate::Double => Some(ProviderPrecision::F64),
        OutputTemplate::Default | OutputTemplate::Native => {
            runmat_accelerate_api::handle_precision(input)
        }
    }
}

pub(super) async fn adapt_output_class(
    provider: &dyn AccelProvider,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    output_class: &OutputTemplate,
    stage: ClassAdaptationStage,
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    let Some(target) = desired_precision(input, output_class) else {
        return Ok(output);
    };
    if runmat_accelerate_api::handle_precision(&output) == Some(target) {
        return Ok(output);
    }
    let cast = match target {
        ProviderPrecision::F32 => provider.unary_single(&output).await,
        ProviderPrecision::F64 => provider.unary_double(&output).await,
    }
    .map_err(|source| {
        let source = match stage {
            ClassAdaptationStage::BeforeReduction => source,
            ClassAdaptationStage::AfterReduction => source.after_effect(),
        };
        mean_provider_error(source)
    })?;
    let contract = UnaryGpuOutputContract {
        storage: runmat_accelerate_api::handle_storage(input),
        precision: Some(target),
        integer: None,
        logical: false,
        alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    let valid = exact_execution_owner(provider, input).is_some_and(|owner| {
        gpu_helpers::unary_gpu_output_matches_shape(&cast, &output, owner, output.shape(), contract)
    }) && cast.allocation_lease().validate_live().is_ok()
        && cast.device_id() == input.device_id()
        && !cast.shares_allocation_with(input)
        && gpu_helpers::expected_handle_numeric_element_type(&cast).is_ok();
    if valid {
        Ok(cast)
    } else {
        let mut error = mean_internal_error("mean: provider returned a malformed class adaptation");
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(MeanProviderFailure::Execution(Box::new(
            gpu_helpers::reject_gpu_output(cast, error),
        )))
    }
}

pub(super) fn admit_result(
    provider: &dyn AccelProvider,
    input: &GpuTensorHandle,
    mut output: GpuTensorHandle,
    axes: &MeanAxes,
    dims: &[usize],
    precision: Option<ProviderPrecision>,
) -> BuiltinResult<GpuTensorHandle> {
    let mut shape = input.shape().to_vec();
    for &dim in dims {
        shape[dim] = 1;
    }
    if matches!(axes, MeanAxes::All) {
        shape = vec![1, 1];
    }
    // Normalize only equivalent scalar views, not non-scalar shape mismatches.
    if output.shape() != shape
        && output.shape().iter().all(|&dim| dim == 1)
        && shape.iter().all(|&dim| dim == 1)
    {
        output = output
            .with_view(output.view().clone().with_shape(shape.clone()))
            .map_err(|source| {
                let mut error = mean_internal_error(format!(
                    "mean: cannot publish scalar reduction view: {source}"
                ));
                error.source = Some(Box::new(source));
                error
            })?;
    }
    let alias = if dims.iter().all(|&dim| input.shape()[dim] == 1) {
        GpuOutputAliasPolicy::AllowInput
    } else {
        GpuOutputAliasPolicy::RequireDistinctAllocation
    };
    let valid = exact_execution_owner(provider, input).is_some_and(|owner| {
        precision.is_some_and(|precision| {
            gpu_helpers::unary_gpu_output_matches_shape(
                &output,
                input,
                owner,
                &shape,
                UnaryGpuOutputContract {
                    storage: runmat_accelerate_api::handle_storage(input),
                    precision: Some(precision),
                    integer: None,
                    logical: false,
                    alias,
                },
            )
        })
    }) && output.allocation_lease().validate_live().is_ok()
        && gpu_helpers::expected_handle_numeric_element_type(&output).is_ok();
    if valid {
        Ok(output)
    } else {
        let mut error = mean_internal_error(
            "mean: provider returned malformed reduction class, shape, owner, or alias",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(gpu_helpers::reject_gpu_output(output, error))
    }
}

pub(super) fn admit_omitnan_preparation(
    provider: &dyn AccelProvider,
    original: &GpuTensorHandle,
    output: GpuTensorHandle,
    borrowed: &[&GpuTensorHandle],
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    let valid = runmat_accelerate_api::handle_precision(original).is_some()
        && exact_execution_owner(provider, original).is_some_and(|owner| {
            gpu_helpers::unary_gpu_output_matches_shape(
                &output,
                original,
                owner,
                original.shape(),
                UnaryGpuOutputContract {
                    storage: GpuTensorStorage::Real,
                    precision: runmat_accelerate_api::handle_precision(original),
                    integer: None,
                    logical: false,
                    alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
                },
            )
        })
        && output.allocation_lease().validate_live().is_ok()
        && gpu_helpers::expected_handle_numeric_element_type(&output).is_ok()
        && borrowed
            .iter()
            .all(|operand| !output.shares_allocation_with(operand));
    if valid {
        Ok(output)
    } else {
        let mut error = mean_internal_error("mean: provider returned malformed NaN preparation");
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(MeanProviderFailure::Execution(Box::new(
            gpu_helpers::reject_gpu_output(output, error),
        )))
    }
}

pub(super) fn admit_double_omitnan_stage(
    provider: &dyn AccelProvider,
    original: &GpuTensorHandle,
    output: GpuTensorHandle,
    shape: &[usize],
    alias: OmitNanStageAlias<'_>,
    borrowed: &[&GpuTensorHandle],
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    let current = match alias {
        OmitNanStageAlias::Fresh => None,
        OmitNanStageAlias::SingletonIdentity(current) => Some(current),
    };
    let valid = exact_execution_owner(provider, original).is_some_and(|owner| {
        gpu_helpers::unary_gpu_output_matches_shape(
            &output,
            original,
            owner,
            shape,
            UnaryGpuOutputContract {
                storage: GpuTensorStorage::Real,
                precision: Some(ProviderPrecision::F64),
                integer: None,
                logical: false,
                alias: if current.is_some() {
                    GpuOutputAliasPolicy::AllowInput
                } else {
                    GpuOutputAliasPolicy::RequireDistinctAllocation
                },
            },
        )
    }) && output.allocation_lease().validate_live().is_ok()
        && gpu_helpers::expected_handle_numeric_element_type(&output).is_ok()
        && !output.shares_allocation_with(original)
        && current.is_none_or(|current| {
            runmat_accelerate_api::handle_precision(current) == Some(ProviderPrecision::F64)
                && current.shape() == shape
                && current.allocation_lease().validate_live().is_ok()
                && (!output.shares_allocation_with(current)
                    || gpu_helpers::same_gpu_handle(&output, current))
        })
        && borrowed.iter().all(|operand| {
            !output.shares_allocation_with(operand)
                || current.is_some_and(|current| {
                    gpu_helpers::same_gpu_handle(&output, current)
                        && current.shares_allocation_with(operand)
                })
        });
    if valid {
        Ok(output)
    } else {
        let mut error =
            mean_internal_error("mean: provider returned a malformed double accumulation stage");
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(MeanProviderFailure::Execution(Box::new(
            gpu_helpers::reject_gpu_output(output, error),
        )))
    }
}
