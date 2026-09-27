//! Mean provider strategies retain typed failures and own every intermediate.
use super::*;

#[derive(Debug)]
pub(super) enum MeanProviderFailure {
    Unsupported,
    Execution(RuntimeError),
}

fn mean_provider_error(source: anyhow::Error) -> MeanProviderFailure {
    if gpu_helpers::provider_hook_is_unsupported(&source) {
        return MeanProviderFailure::Unsupported;
    }
    let mut failure = mean_internal_error(format!("mean: provider execution failed: {source}"));
    failure.source = Some(source.into_boxed_dyn_error());
    failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
    MeanProviderFailure::Execution(failure)
}

pub(super) async fn execute(
    provider: &dyn AccelProvider,
    input: &GpuTensorHandle,
    axes: &MeanAxes,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let result = if nan_mode == ReductionNaN::Include {
        mean_gpu_try(provider, input, axes).await
    } else {
        mean_gpu_omitnan(provider, input, axes).await
    };
    match result {
        Ok(output) => Ok(Some(output)),
        Err(MeanProviderFailure::Unsupported) => Ok(None),
        Err(MeanProviderFailure::Execution(error)) => Err(error),
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
            reduce_mean_dim_gpu(provider, handle.clone(), dim).await
        }
        MeanAxes::Dim(dim) => reduce_mean_dim_gpu(provider, handle.clone(), *dim).await,
        MeanAxes::Vec(dims) => {
            // Prefer provider N-D reduce if available
            let mut dims0: Vec<usize> = dims
                .iter()
                .filter_map(|&d| if d > 0 { Some(d - 1) } else { None })
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
            let mut dims_sorted = dims.clone();
            dims_sorted.sort_unstable();
            dims_sorted.dedup();
            for dim in dims_sorted {
                if is_scalar_shape(result.shape()) {
                    break;
                }
                result = reduce_mean_dim_gpu(provider, result, dim).await?;
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
                                for dim in 1..=rank {
                                    result = reduce_mean_dim_gpu(provider, result, dim).await?;
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
        .map_err(mean_provider_error)
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
    let permuted = provider
        .permute(handle, &order)
        .map_err(mean_provider_error)?;
    // Compute rows/cols
    let mut reduce_len: usize = 1;
    for &d in &reduce_dims {
        reduce_len = reduce_len.saturating_mul(handle.shape()[d]);
    }
    let total_elems: usize = handle.shape().iter().copied().product();
    if reduce_len == 0 || total_elems == 0 {
        return provider
            .fill(&[1, 1], f64::NAN)
            .map_err(mean_provider_error);
    }
    let num_slices = total_elems / reduce_len;
    // Reshape permuted view to [rows, cols]
    let reshaped2d = provider
        .reshape(&permuted, &[reduce_len, num_slices])
        .map_err(mean_provider_error)?;
    // Reduce along rows (dim 0) -> [1, num_slices]
    let reduced_rows = provider
        .reduce_mean_dim(&reshaped2d, 0)
        .await
        .map_err(mean_provider_error)?;
    // Reshape to kept sizes (permuted order)
    let kept_sizes: Vec<usize> = kept_dims.iter().map(|&d| handle.shape()[d]).collect();
    let kept_shape = if kept_sizes.is_empty() {
        vec![1, 1]
    } else {
        kept_sizes.clone()
    };
    let reshaped_kept = provider
        .reshape(&reduced_rows, &kept_shape)
        .map_err(mean_provider_error)?;
    // Expand permuted shape by inserting ones for reduced axes
    let mut expanded_perm_shape: Vec<usize> = Vec::with_capacity(rank);
    expanded_perm_shape.extend(std::iter::repeat_n(1usize, reduce_dims.len()));
    expanded_perm_shape.extend_from_slice(&kept_sizes);
    let expanded = provider
        .reshape(&reshaped_kept, &expanded_perm_shape)
        .map_err(mean_provider_error)?;
    // Inverse permute back to original axis order
    let mut inv_order = vec![0usize; rank];
    for (dst, &src) in order.iter().enumerate() {
        inv_order[src] = dst;
    }
    let out = provider
        .permute(&expanded, &inv_order)
        .map_err(mean_provider_error)?;
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
    let mask = provider.not_nan_mask(handle).map_err(mean_provider_error)?;

    // Reduce cleaned (sum) and mask (count) along the requested dims
    let mut sum_h = cleaned.clone();
    let mut cnt_h = mask.clone();
    for &dim in &dims_in_bounds {
        sum_h = provider
            .reduce_sum_dim(&sum_h, dim)
            .await
            .map_err(mean_provider_error)?;
        cnt_h = provider
            .reduce_sum_dim(&cnt_h, dim)
            .await
            .map_err(mean_provider_error)?;
    }

    // mean = sum ./ count (0/0 -> NaN when all NaN)
    let out = provider
        .elem_div(&sum_h, &cnt_h)
        .await
        .map_err(mean_provider_error)?;

    Ok(out)
}
