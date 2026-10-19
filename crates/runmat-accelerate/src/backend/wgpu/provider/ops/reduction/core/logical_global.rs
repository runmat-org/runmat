//! Shared owner admission and two-stage execution for global logical reductions.

use super::*;
use crate::backend::wgpu::types::DimReduceOp;
use runmat_builtins::LogicalReductionKind;

impl WgpuProvider {
    pub(super) fn reduce_logical_global_exec(
        &self,
        source: &GpuTensorHandle,
        kind: LogicalReductionKind,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = match (kind, omit_nan) {
            (LogicalReductionKind::Any, false) => DimReduceOp::AnyInclude,
            (LogicalReductionKind::Any, true) => DimReduceOp::AnyOmit,
            (LogicalReductionKind::All, false) => DimReduceOp::AllInclude,
            (LogicalReductionKind::All, true) => DimReduceOp::AllOmit,
        };
        let entry = self.get_entry(source)?;
        if entry.len == 0 {
            return self.logical_fill_exec(&[1, 1], kind == LogicalReductionKind::All);
        }
        // The dimensional shader is two-dimensional. This provider-owned
        // reshape admits a checked flat alias; it never changes source metadata
        // or downloads the resident buffer. Keep it live through both stages.
        let flattened = (entry.shape.len() != 2)
            .then(|| self.reshape_exec(source, &[entry.len, 1]))
            .transpose()?;
        let input = flattened.as_ref().unwrap_or(source);
        let snapshot = FrozenAnySnapshot::capture();
        let first = self
            .reduce_dim_sum_mean_with_snapshot_exec(input, 0, op, snapshot, None)?
            .0;
        let second = self
            .reduce_dim_sum_mean_with_snapshot_exec(&first, 1, op, snapshot, None)?
            .0;
        drop(first);
        Ok(second)
    }
}
