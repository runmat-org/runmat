//! One selected dimensional mean on a rank-N real tensor.
//!
//! The two-dimensional mean kernel keeps its existing route. Higher-rank
//! means select the same exact-view warm pair or one-submission cold moments
//! recipe exposed to staged consumers. A second source mean is a separate
//! call with its own axis and output conversion boundary.

use super::{geometry::NdReductionGeometry, GpuTensorHandle, WgpuProvider};
use anyhow::{ensure, Result};
use runmat_accelerate_api::{unsupported_provider_operation, GpuTensorStorage};

impl WgpuProvider {
    pub(crate) fn reduce_rank_n_mean_dim_exec(
        &self,
        input: &GpuTensorHandle,
        axis: usize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.real_floating_entry(input)?;
        if entry.shape.len() <= 2
            || axis >= entry.shape.len()
            || entry.shape.contains(&0)
            || entry.storage != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(input)
        {
            return Err(unsupported_provider_operation(
                "rank-N resident dimensional mean requires a nonempty real floating tensor and an in-bounds axis",
            ));
        }
        let axes = [axis];
        let geometry = NdReductionGeometry::new(&entry.shape, &axes, "reduce_mean_dim")?;
        if let Some(pinned) = self.reserve_cached_mean_nd_exec(input, &axes)? {
            ensure!(
                pinned.validate_live_source(input)?
                    && pinned.dims() == axes.as_slice()
                    && !pinned.mean().shares_allocation_with(input),
                "rank-N mean cache pair no longer matches the selected source"
            );
            self.validate_nd_moment_pair(input, &axes, pinned.mean(), pinned.ex2())?;
            return Ok(pinned.into_mean());
        }
        let selected = self
            .prepare_cold_mean_nd_exec(input, &axes)?
            .ok_or_else(|| {
                unsupported_provider_operation("rank-N mean has no qualified N-D route")
            })?;
        ensure!(
            selected.output_shape() == geometry.out_shape.as_slice(),
            "rank-N mean cold route changed its output shape"
        );
        let executed = self.execute_cold_mean_nd_exec(selected)?;
        self.validate_nd_moment_pair(input, &axes, &executed.mean, &executed.ex2)?;
        Ok(executed.mean)
    }
}

#[cfg(test)]
#[path = "rank_n_dim/tests.rs"]
mod tests;
