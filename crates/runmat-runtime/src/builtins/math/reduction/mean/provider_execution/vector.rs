//! Checked real-lane permutation/reshape reduction strategy; no original-call replay.
use super::*;

fn checked_elements(shape: &[usize]) -> Result<usize, MeanProviderFailure> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| {
            let mut error =
                mean_internal_error("mean: provider strategy geometry exceeds platform limits");
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            MeanProviderFailure::Execution(Box::new(error))
        })
}

/// Reduce mean across multiple (1-based) dimensions in a single device pass by
/// permuting reduce dims to the front, reshaping to 2-D, reducing rows, and
/// reshaping/permuting back to the original order with size-1 dims preserved.
pub(in super::super) async fn reduce_mean_vecdim_nd_gpu(
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
    let total_elems = checked_elements(handle.shape())?;
    if total_elems == 0 {
        let mut output_shape = handle.shape().to_vec();
        for &dim in &reduce_dims {
            output_shape[dim] = 1;
        }
        checked_elements(&output_shape)?;
        let output = provider
            .fill(&output_shape, f64::NAN)
            .map_err(mean_reduction_error)?;
        let Some(output) = output else {
            return Err(MeanProviderFailure::Unsupported);
        };
        return admission::admit_include_reduction(
            provider,
            handle,
            handle,
            output,
            &output_shape,
            reduce_dims[0],
        );
    }
    let reduce_sizes = reduce_dims
        .iter()
        .map(|&axis| handle.shape()[axis])
        .collect::<Vec<_>>();
    let reduce_len = checked_elements(&reduce_sizes)?;
    let permuted = provider
        .permute(handle, &order)
        .map_err(|error| mean_provider_error(error.with_unproven_effects()))?;
    let Some(permuted) = permuted else {
        return Err(MeanProviderFailure::Unsupported);
    };
    let permuted = admission::admit_include_permutation(
        provider,
        handle,
        handle,
        permuted,
        &order,
        &[handle],
    )?;
    let allocated = std::cell::Cell::new(!permuted.shares_allocation_with(handle));
    let failure = |source: runmat_accelerate_api::ProviderExecutionError| {
        mean_provider_error(if allocated.get() {
            source.after_effect()
        } else {
            source.with_unproven_effects()
        })
    };
    let num_slices = total_elems / reduce_len;
    // Reshape permuted view to [rows, cols]
    let reshaped2d = provider
        .reshape(&permuted, &[reduce_len, num_slices])
        .map_err(|source| failure(source.into()))?;
    if !reshaped2d.shares_allocation_with(&permuted) {
        allocated.set(true);
    }
    let reshaped2d = admission::admit_include_reshape(
        provider,
        handle,
        &permuted,
        reshaped2d,
        &[reduce_len, num_slices],
        &[handle, &permuted],
    )?;
    // Reduce along rows (dim 0) -> [1, num_slices]
    let reduced_rows = provider
        .reduce_mean_dim(&reshaped2d, 0)
        .await
        .map_err(failure)?;
    let reduced_rows = reduced_rows.ok_or_else(|| {
        failure(runmat_accelerate_api::ProviderExecutionError::from(
            anyhow::anyhow!("mean: required dimensional stage unavailable after permutation"),
        ))
    })?;
    if !reduced_rows.shares_allocation_with(&reshaped2d) {
        allocated.set(true);
    }
    let reduced_rows = admission::admit_include_reduction(
        provider,
        handle,
        &reshaped2d,
        reduced_rows,
        &[1, num_slices],
        0,
    )?;
    // Reshape to kept sizes (permuted order)
    let kept_sizes: Vec<usize> = kept_dims.iter().map(|&d| handle.shape()[d]).collect();
    let kept_shape = if kept_sizes.is_empty() {
        vec![1, 1]
    } else {
        kept_sizes.clone()
    };
    let reshaped_kept = provider
        .reshape(&reduced_rows, &kept_shape)
        .map_err(|source| failure(source.into()))?;
    if !reshaped_kept.shares_allocation_with(&reduced_rows) {
        allocated.set(true);
    }
    let reshaped_kept = admission::admit_include_reshape(
        provider,
        handle,
        &reduced_rows,
        reshaped_kept,
        &kept_shape,
        &[handle, &permuted, &reshaped2d, &reduced_rows],
    )?;
    // Expand permuted shape by inserting ones for reduced axes
    let mut expanded_perm_shape: Vec<usize> = Vec::with_capacity(rank);
    expanded_perm_shape.extend(std::iter::repeat_n(1usize, reduce_dims.len()));
    expanded_perm_shape.extend_from_slice(&kept_sizes);
    let expanded = provider
        .reshape(&reshaped_kept, &expanded_perm_shape)
        .map_err(|source| failure(source.into()))?;
    if !expanded.shares_allocation_with(&reshaped_kept) {
        allocated.set(true);
    }
    let expanded = admission::admit_include_reshape(
        provider,
        handle,
        &reshaped_kept,
        expanded,
        &expanded_perm_shape,
        &[
            handle,
            &permuted,
            &reshaped2d,
            &reduced_rows,
            &reshaped_kept,
        ],
    )?;
    // Inverse permute back to original axis order
    let mut inv_order = vec![0usize; rank];
    for (dst, &src) in order.iter().enumerate() {
        inv_order[src] = dst;
    }
    let out = provider.permute(&expanded, &inv_order).map_err(failure)?;
    let out = out.ok_or_else(|| {
        failure(runmat_accelerate_api::ProviderExecutionError::from(
            anyhow::anyhow!("mean: required inverse permutation unavailable"),
        ))
    })?;
    admission::admit_include_permutation(
        provider,
        handle,
        &expanded,
        out,
        &inv_order,
        &[
            handle,
            &permuted,
            &reshaped2d,
            &reduced_rows,
            &reshaped_kept,
            &expanded,
        ],
    )
}
