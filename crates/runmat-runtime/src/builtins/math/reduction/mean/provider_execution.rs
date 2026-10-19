//! Mean provider strategies retain typed failures and own every intermediate.
use super::*;
use runmat_accelerate_api::GpuTensorStorage;
use runmat_accelerate_api::ProviderRealMeanNaNMode;

#[path = "provider_execution/admission.rs"]
mod admission;
use admission::{
    adapt_output_class, admit_double_omitnan_stage, admit_omitnan_preparation, admit_result,
    desired_precision, ClassAdaptationStage, OmitNanStageAlias,
};

#[cfg(test)]
#[path = "result_tests.rs"]
mod result_tests;

#[derive(Debug)]
pub(super) enum MeanProviderFailure {
    Unsupported,
    Execution(Box<RuntimeError>),
}

fn mean_provider_error(
    source: impl Into<runmat_accelerate_api::ProviderExecutionError>,
) -> MeanProviderFailure {
    let source = source.into();
    if source.is_unsupported() {
        return MeanProviderFailure::Unsupported;
    }
    let mut failure = mean_internal_error(format!("mean: provider execution failed: {source}"));
    failure = crate::runtime_error::with_provider_execution_source(failure, source);
    failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
    MeanProviderFailure::Execution(Box::new(failure))
}

fn mean_provider_error_after_effect(
    source: impl Into<runmat_accelerate_api::ProviderExecutionError>,
) -> MeanProviderFailure {
    mean_provider_error(source.into().after_effect())
}

fn mean_reduction_error(
    source: runmat_accelerate_api::ProviderExecutionError,
) -> MeanProviderFailure {
    // Optional reduction hooks report absence only as None, never by Err.
    mean_provider_error(source.with_unproven_effects())
}

pub(super) async fn execute(
    provider: &dyn AccelProvider,
    input: &GpuTensorHandle,
    axes: &MeanAxes,
    nan_mode: ReductionNaN,
    output_class: &OutputTemplate,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let dims = resolve_native_dims(input.shape(), axes)?;
    // NaN preparation and count masks operate on real scalar lanes. A complex
    // owner needs a complete complex omission route even for a singleton:
    // omitting a NaN pair changes both components of that scalar result.
    if nan_mode == ReductionNaN::Omit
        && runmat_accelerate_api::handle_storage(input) == GpuTensorStorage::ComplexInterleaved
    {
        return Ok(None);
    }
    if dims.is_empty() || is_scalar_shape(input.shape()) {
        return match adapt_output_class(
            provider,
            input,
            input.clone(),
            output_class,
            ClassAdaptationStage::BeforeReduction,
        )
        .await
        {
            Ok(output) => Ok(Some(output)),
            Err(MeanProviderFailure::Unsupported) => Ok(None),
            Err(MeanProviderFailure::Execution(error)) => Err(*error),
        };
    }
    // One ordinary call, whether Dim or Vec, may use the same Double
    // accumulation route. Vec keeps Double scratch across its ordered axes;
    // nested calls re-enter here separately after their own Single publication.
    // `All` has a distinct single linear-scan host order, not sequential axes.
    if !matches!(axes, MeanAxes::All) {
        if let Some(target) = desired_precision(input, output_class) {
            let selected = match provider.prepare_ordered_real_mean(
                input,
                &dims,
                target,
                match nan_mode {
                    ReductionNaN::Include => ProviderRealMeanNaNMode::Include,
                    ReductionNaN::Omit => ProviderRealMeanNaNMode::Omit,
                },
            ) {
                Ok(selected) => selected,
                Err(source) => match mean_provider_error(source) {
                    MeanProviderFailure::Unsupported => None,
                    MeanProviderFailure::Execution(error) => return Err(*error),
                },
            };
            if let Some(selected) = selected {
                match provider.execute_ordered_real_mean(selected, None) {
                    Ok(executed) => {
                        return admit_result(
                            provider,
                            input,
                            executed.output,
                            axes,
                            &dims,
                            Some(target),
                        )
                        .map(Some);
                    }
                    Err(source) => match mean_provider_error(source) {
                        MeanProviderFailure::Unsupported => {}
                        MeanProviderFailure::Execution(error) => return Err(*error),
                    },
                }
            }
        }
    }
    let result = if nan_mode == ReductionNaN::Include {
        mean_gpu_try(provider, input, axes).await
    } else {
        mean_gpu_omitnan(provider, input, axes).await
    };
    match result {
        Ok(output) => {
            let stage_precision = if nan_mode == ReductionNaN::Omit {
                Some(ProviderPrecision::F64)
            } else {
                runmat_accelerate_api::handle_precision(input)
            };
            let output = admit_result(provider, input, output, axes, &dims, stage_precision)?;
            match adapt_output_class(
                provider,
                input,
                output,
                output_class,
                ClassAdaptationStage::AfterReduction,
            )
            .await
            {
                Ok(output) => admit_result(
                    provider,
                    input,
                    output,
                    axes,
                    &dims,
                    desired_precision(input, output_class),
                )
                .map(Some),
                Err(MeanProviderFailure::Unsupported) => Ok(None),
                Err(MeanProviderFailure::Execution(error)) => Err(*error),
            }
        }
        Err(MeanProviderFailure::Unsupported) => Ok(None),
        Err(MeanProviderFailure::Execution(error)) => Err(*error),
    }
}

async fn mean_gpu_try(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    axes: &MeanAxes,
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    match axes {
        MeanAxes::Default => {
            if is_scalar_shape(handle.shape()) {
                return Ok(handle.clone());
            }
            let dim = default_dimension_from_shape(handle.shape());
            reduce_mean_dim_gpu(provider, handle, handle.clone(), dim, false).await
        }
        MeanAxes::Dim(dim) => {
            reduce_mean_dim_gpu(provider, handle, handle.clone(), *dim, false).await
        }
        MeanAxes::Vec(dims) => {
            // Prefer provider N-D reduce if available
            let mut dims0: Vec<usize> = dims
                .iter()
                .filter_map(|&d| {
                    if d > 0 && d <= handle.shape().len() {
                        Some(d - 1)
                    } else {
                        None
                    }
                })
                .collect();
            dims0.sort_unstable();
            dims0.dedup();
            if !dims0.is_empty() {
                match provider
                    .reduce_mean_nd(handle, &dims0)
                    .await
                    .map_err(mean_reduction_error)
                {
                    Ok(Some(out)) => return Ok(out),
                    Ok(None) => {}
                    Err(error) => return Err(error),
                }
            }
            // Try fast permute+2D fallback
            match reduce_mean_vecdim_nd_gpu(provider, handle, dims).await {
                Ok(nd) => return Ok(nd),
                Err(MeanProviderFailure::Unsupported) => {}
                Err(error) => return Err(error),
            }
            // Sequential per-dimension reductions
            let mut result = handle.clone();
            let mut prior_effect = false;
            let mut dims_sorted = dims.clone();
            dims_sorted.sort_unstable();
            dims_sorted.dedup();
            for dim in dims_sorted {
                if is_scalar_shape(result.shape()) {
                    break;
                }
                result = reduce_mean_dim_gpu(provider, handle, result, dim, prior_effect).await?;
                prior_effect = true;
            }
            Ok(result)
        }
        MeanAxes::All => {
            if is_scalar_shape(handle.shape()) {
                return Ok(handle.clone());
            }
            match provider
                .reduce_mean(handle)
                .await
                .map_err(mean_reduction_error)
            {
                Ok(Some(out)) => Ok(out),
                Ok(None) => {
                    let rank = handle.shape().len();
                    if is_scalar_shape(handle.shape()) || rank == 0 {
                        Ok(handle.clone())
                    } else {
                        let dims: Vec<usize> = (1..=rank).collect();
                        match reduce_mean_vecdim_nd_gpu(provider, handle, &dims).await {
                            Ok(result) => Ok(result),
                            Err(MeanProviderFailure::Unsupported) => {
                                let mut result = handle.clone();
                                let mut prior_effect = false;
                                for dim in 1..=rank {
                                    result = reduce_mean_dim_gpu(
                                        provider,
                                        handle,
                                        result,
                                        dim,
                                        prior_effect,
                                    )
                                    .await?;
                                    prior_effect = true;
                                }
                                Ok(result)
                            }
                            Err(error) => Err(error),
                        }
                    }
                }
                Err(error) => Err(error),
            }
        }
    }
}

async fn reduce_mean_dim_gpu(
    provider: &dyn AccelProvider,
    original: &GpuTensorHandle,
    handle: GpuTensorHandle,
    dim: usize,
    prior_effect: bool,
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    if dim == 0 {
        return Err(MeanProviderFailure::Unsupported);
    }
    if handle.shape().len() < dim {
        return Ok(handle);
    }
    let mut expected = handle.shape().to_vec();
    expected[dim - 1] = 1;
    let output = provider
        .reduce_mean_dim(&handle, dim - 1)
        .await
        .map_err(|source| {
            if prior_effect {
                mean_provider_error_after_effect(source)
            } else {
                mean_reduction_error(source)
            }
        })?
        .ok_or_else(|| {
            if prior_effect {
                mean_provider_error_after_effect(anyhow::anyhow!(
                    "mean: required dimensional stage unavailable"
                ))
            } else {
                MeanProviderFailure::Unsupported
            }
        })?;
    admission::admit_include_reduction(provider, original, &handle, output, &expected, dim - 1)
}

#[path = "provider_execution/vector.rs"]
mod vector;
pub(super) use vector::reduce_mean_vecdim_nd_gpu;

async fn mean_gpu_omitnan(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    axes: &MeanAxes,
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    // Early return for empty dim selection
    let dims_in_bounds: Vec<usize> = match axes {
        MeanAxes::Default => {
            if is_scalar_shape(handle.shape()) {
                return Ok(handle.clone());
            }
            vec![default_dimension_from_shape(handle.shape()) - 1]
        }
        MeanAxes::Dim(d) => {
            if *d == 0 || *d > handle.shape().len() {
                return Ok(handle.clone());
            }
            vec![*d - 1]
        }
        MeanAxes::Vec(v) => {
            let mut dims: Vec<usize> = v
                .iter()
                .filter_map(|&d| {
                    if d > 0 && d <= handle.shape().len() {
                        Some(d - 1)
                    } else {
                        None
                    }
                })
                .collect();
            dims.sort_unstable();
            dims.dedup();
            dims
        }
        MeanAxes::All => {
            if is_scalar_shape(handle.shape()) {
                return Ok(handle.clone());
            }
            (0..handle.shape().len()).collect()
        }
    };

    if dims_in_bounds.is_empty() {
        return Ok(handle.clone());
    }

    // Build cleaned values and not-NaN counts on device
    let cleaned = provider
        .map_nan_to_zero(handle)
        .map_err(mean_provider_error)?;
    let cleaned = admit_omitnan_preparation(provider, handle, cleaned, &[])?;
    let mask = provider
        .not_nan_mask(handle)
        .map_err(mean_provider_error_after_effect)?;
    let mask = admit_omitnan_preparation(provider, handle, mask, &[&cleaned])?;

    // Reduce cleaned (sum) and mask (count) along the requested dims
    let mut sum_h = cleaned.clone();
    let mut cnt_h = mask.clone();
    for &dim in &dims_in_bounds {
        let mut shape = sum_h.shape().to_vec();
        shape[dim] = 1;
        let next_sum = provider
            .reduce_sum_dim(&sum_h, dim)
            .await
            .map_err(mean_provider_error_after_effect)?;
        sum_h = admit_double_omitnan_stage(
            provider,
            handle,
            next_sum,
            &shape,
            OmitNanStageAlias::for_reduction(&sum_h, dim),
            &[&sum_h, &cnt_h, &cleaned, &mask],
        )?;
        let next_count = provider
            .reduce_sum_dim(&cnt_h, dim)
            .await
            .map_err(mean_provider_error_after_effect)?;
        cnt_h = admit_double_omitnan_stage(
            provider,
            handle,
            next_count,
            &shape,
            OmitNanStageAlias::for_reduction(&cnt_h, dim),
            &[&cnt_h, &sum_h, &cleaned, &mask],
        )?;
    }

    // mean = sum ./ count (0/0 -> NaN when all NaN)
    let out = provider
        .elem_div(&sum_h, &cnt_h)
        .await
        .map_err(mean_provider_error_after_effect)?;
    admit_double_omitnan_stage(
        provider,
        handle,
        out,
        sum_h.shape(),
        OmitNanStageAlias::Fresh,
        &[&sum_h, &cnt_h, &cleaned, &mask],
    )
}
