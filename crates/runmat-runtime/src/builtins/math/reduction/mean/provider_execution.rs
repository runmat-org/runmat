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
                match provider.execute_ordered_real_mean(selected) {
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
            reduce_mean_dim_gpu(provider, handle.clone(), dim, false).await
        }
        MeanAxes::Dim(dim) => reduce_mean_dim_gpu(provider, handle.clone(), *dim, false).await,
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
                    .map_err(mean_provider_error)
                {
                    Ok(out) => return Ok(out),
                    Err(MeanProviderFailure::Unsupported) => {}
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
                result = reduce_mean_dim_gpu(provider, result, dim, prior_effect).await?;
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
                .map_err(mean_provider_error)
            {
                Ok(out) => Ok(out),
                Err(MeanProviderFailure::Unsupported) => {
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
                                    result =
                                        reduce_mean_dim_gpu(provider, result, dim, prior_effect)
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
    provider
        .reduce_mean_dim(&handle, dim - 1)
        .await
        .map_err(|source| {
            if prior_effect {
                mean_provider_error_after_effect(source)
            } else {
                mean_provider_error(source)
            }
        })
}

/// Reduce mean across multiple (1-based) dimensions in a single device pass by
/// permuting reduce dims to the front, reshaping to 2-D, reducing rows, and
/// reshaping/permuting back to the original order with size-1 dims preserved.
// (N-D mean fast path omitted for now; sequential per-dimension GPU reductions used instead.)
pub(super) async fn reduce_mean_vecdim_nd_gpu(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    dims_1based: &[usize],
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    let rank = handle.shape().len();
    if rank == 0 || dims_1based.is_empty() {
        return Ok(handle.clone());
    }
    // The generic permutation/reshape shortcut is a real-lane strategy.
    // Complex owners may still provide direct N-D, global, or dimensional
    // reduction hooks; do not stage an unqualified pairwise shortcut first.
    if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved {
        return Err(MeanProviderFailure::Unsupported);
    }
    // Convert to 0-based and filter in-bounds
    let mut reduce_dims: Vec<usize> = dims_1based
        .iter()
        .filter_map(|&d| {
            if d > 0 && d <= rank {
                Some(d - 1)
            } else {
                None
            }
        })
        .collect();
    if reduce_dims.is_empty() {
        return Ok(handle.clone());
    }
    reduce_dims.sort_unstable();
    reduce_dims.dedup();
    // Kept dims
    let kept_dims: Vec<usize> = (0..rank).filter(|i| !reduce_dims.contains(i)).collect();
    // Permute reduced dims first
    let mut order: Vec<usize> = Vec::with_capacity(rank);
    order.extend_from_slice(&reduce_dims);
    order.extend_from_slice(&kept_dims);
    // Compute rows/cols
    let mut reduce_len: usize = 1;
    for &d in &reduce_dims {
        reduce_len = reduce_len.saturating_mul(handle.shape()[d]);
    }
    let total_elems: usize = handle.shape().iter().copied().product();
    if reduce_len == 0 || total_elems == 0 {
        let mut output_shape = handle.shape().to_vec();
        for &dim in &reduce_dims {
            output_shape[dim] = 1;
        }
        return provider
            .fill(&output_shape, f64::NAN)
            .map_err(mean_provider_error);
    }
    let permuted = provider
        .permute(handle, &order)
        .map_err(|error| mean_provider_error(error.with_unproven_effects()))?;
    let Some(permuted) = permuted else {
        return Err(MeanProviderFailure::Unsupported);
    };
    if !gpu_helpers::permutation_output_matches(handle, &permuted, provider, &order) {
        return Err(MeanProviderFailure::Execution(Box::new(
            gpu_helpers::reject_gpu_output(
                permuted,
                mean_internal_error("mean: invalid provider permutation output"),
            ),
        )));
    }
    let num_slices = total_elems / reduce_len;
    // Reshape permuted view to [rows, cols]
    let reshaped2d = provider
        .reshape(&permuted, &[reduce_len, num_slices])
        .map_err(mean_provider_error_after_effect)?;
    // Reduce along rows (dim 0) -> [1, num_slices]
    let reduced_rows = provider
        .reduce_mean_dim(&reshaped2d, 0)
        .await
        .map_err(mean_provider_error_after_effect)?;
    // Reshape to kept sizes (permuted order)
    let kept_sizes: Vec<usize> = kept_dims.iter().map(|&d| handle.shape()[d]).collect();
    let kept_shape = if kept_sizes.is_empty() {
        vec![1, 1]
    } else {
        kept_sizes.clone()
    };
    let reshaped_kept = provider
        .reshape(&reduced_rows, &kept_shape)
        .map_err(mean_provider_error_after_effect)?;
    // Expand permuted shape by inserting ones for reduced axes
    let mut expanded_perm_shape: Vec<usize> = Vec::with_capacity(rank);
    expanded_perm_shape.extend(std::iter::repeat_n(1usize, reduce_dims.len()));
    expanded_perm_shape.extend_from_slice(&kept_sizes);
    let expanded = provider
        .reshape(&reshaped_kept, &expanded_perm_shape)
        .map_err(mean_provider_error_after_effect)?;
    // Inverse permute back to original axis order
    let mut inv_order = vec![0usize; rank];
    for (dst, &src) in order.iter().enumerate() {
        inv_order[src] = dst;
    }
    let out = provider
        .permute(&expanded, &inv_order)
        .map_err(mean_provider_error_after_effect)?;
    let out = out.ok_or_else(|| {
        mean_provider_error_after_effect(anyhow::anyhow!(
            "mean: required inverse permutation unavailable"
        ))
    })?;
    if !gpu_helpers::permutation_output_matches(&expanded, &out, provider, &inv_order) {
        return Err(MeanProviderFailure::Execution(Box::new(
            gpu_helpers::reject_gpu_output(
                out,
                mean_internal_error("mean: invalid inverse permutation output"),
            ),
        )));
    }
    Ok(out)
}

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
