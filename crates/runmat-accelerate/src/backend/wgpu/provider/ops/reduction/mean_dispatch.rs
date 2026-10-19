//! Existing mean physical routes with pure absence separated from execution faults.
use super::*;

impl WgpuProvider {
    pub(crate) async fn mean_optional_exec(
        &self,
        input: &GpuTensorHandle,
    ) -> Result<Option<GpuTensorHandle>> {
        let entry = self.get_entry_raw(input)?;
        if entry.integer_type().is_some() {
            self.integer_reduce_mean_global_exec("mean", input)
                .map(Some)
        } else if entry.storage == GpuTensorStorage::ComplexInterleaved
            || entry.precision != self.precision
        {
            Ok(None)
        } else {
            self.reduce_mean_global_exec(input).map(Some)
        }
    }

    pub(crate) async fn mean_dim_optional_exec(
        &self,
        input: &GpuTensorHandle,
        axis: usize,
    ) -> Result<Option<GpuTensorHandle>> {
        let entry = self.get_entry_raw(input)?;
        if entry.integer_type().is_some() {
            self.integer_reduce_mean_dim_exec(axis, "mean", input)
                .map(Some)
        } else if entry.storage == GpuTensorStorage::ComplexInterleaved && axis < entry.shape.len()
        {
            Ok(None)
        } else if axis < entry.shape.len()
            && entry.precision != self.precision
            && (entry.shape.len() <= 2 || self.precision != NumericPrecision::F64)
        {
            // The legacy dimensional kernel uses provider precision; only
            // rank-N owns a qualified alternate-F32 route on an F64 provider.
            Ok(None)
        } else if entry.shape.len() > 2 {
            self.reduce_rank_n_mean_dim_exec(input, axis)
        } else {
            self.reduce_dim_sum_mean_exec(
                input,
                axis,
                crate::backend::wgpu::types::DimReduceOp::Mean,
            )
            .map(Some)
        }
    }

    pub(crate) async fn mean_nd_optional_exec(
        &self,
        input: &GpuTensorHandle,
        axes: &[usize],
    ) -> Result<Option<GpuTensorHandle>> {
        let entry = self.get_entry_raw(input)?;
        if entry.integer_type().is_none()
            && entry.precision != self.precision
            && self.precision != NumericPrecision::F64
            && !axes.is_empty()
            && axes.iter().all(|&axis| axis < entry.shape.len())
        {
            return Ok(None);
        }
        // The floating ND kernel geometry is deliberately nonempty. Refuse
        // before caches/materialization, so owning empty-fill semantics do
        // not compute irrelevant strides. Integer routes remain separate.
        if entry.integer_type().is_none()
            && entry.shape.contains(&0)
            && !axes.is_empty()
            && axes.iter().all(|&axis| axis < entry.shape.len())
        {
            return Ok(None);
        }
        if entry.integer_type().is_some() {
            self.integer_reduce_mean_dims_exec(axes, "mean", input)
                .map(Some)
        } else if entry.storage == GpuTensorStorage::ComplexInterleaved
            && !axes.is_empty()
            && axes.iter().all(|&axis| axis < entry.shape.len())
        {
            Ok(None)
        } else {
            self.reduce_nd_mean_exec(input, axes).await
        }
    }
}
