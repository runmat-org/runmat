use anyhow::{anyhow, ensure, Result};

/// Geometry shared by the ordinary ND mean and the two-output moments kernel.
/// Cache reuse and the mean-only large-row fallback remain separate decisions.
pub(super) struct NdReductionGeometry {
    pub(super) rank: usize,
    pub(super) reduce: Vec<usize>,
    pub(super) kept: Vec<usize>,
    pub(super) kept_sizes: Vec<u32>,
    pub(super) reduce_sizes: Vec<u32>,
    pub(super) kept_strides: Vec<u32>,
    pub(super) reduce_strides: Vec<u32>,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) out_shape: Vec<usize>,
}

impl NdReductionGeometry {
    pub(super) fn new(
        shape: &[usize],
        dims_zero_based: &[usize],
        operation: &'static str,
    ) -> Result<Self> {
        let rank = shape.len();
        ensure!(rank > 0, "{operation}: rank must be > 0");
        ensure!(
            rank <= crate::backend::wgpu::params::BCAST_MAX_RANK,
            "{operation}: rank exceeds GPU limits"
        );
        let mut reduce: Vec<usize> = dims_zero_based
            .iter()
            .copied()
            .filter(|&dim| dim < rank)
            .collect();
        reduce.sort_unstable();
        reduce.dedup();
        ensure!(!reduce.is_empty(), "{operation}: no valid dims to reduce");
        let kept: Vec<usize> = (0..rank).filter(|dim| !reduce.contains(dim)).collect();

        let mut strides = Vec::with_capacity(rank);
        let mut stride = 1usize;
        for &extent in shape {
            strides.push(stride);
            stride = stride
                .checked_mul(extent)
                .ok_or_else(|| anyhow!("{operation}: shape too large"))?;
        }
        let rows = reduce.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(shape[dim])
                .ok_or_else(|| anyhow!("{operation}: tensor exceeds GPU limits"))
        })?;
        let cols = kept.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(shape[dim])
                .ok_or_else(|| anyhow!("{operation}: tensor exceeds GPU limits"))
        })?;
        ensure!(rows > 0 && cols > 0, "{operation}: empty tensor");
        ensure!(
            u32::try_from(rows).is_ok() && u32::try_from(cols).is_ok(),
            "{operation}: tensor exceeds GPU limits"
        );
        let sizes = |dims: &[usize]| {
            dims.iter()
                .map(|&dim| u32::try_from(shape[dim]))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| anyhow!("{operation}: tensor exceeds GPU limits"))
        };
        let dimension_strides = |dims: &[usize]| {
            dims.iter()
                .map(|&dim| u32::try_from(strides[dim]))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| anyhow!("{operation}: tensor exceeds GPU limits"))
        };
        let mut out_shape = shape.to_vec();
        for &dim in &reduce {
            out_shape[dim] = 1;
        }
        Ok(Self {
            rank,
            kept_sizes: sizes(&kept)?,
            reduce_sizes: sizes(&reduce)?,
            kept_strides: dimension_strides(&kept)?,
            reduce_strides: dimension_strides(&reduce)?,
            reduce,
            kept,
            rows,
            cols,
            out_shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::NdReductionGeometry;

    #[test]
    fn source_axis_order_is_canonicalized_only_for_nd_kernel_geometry() {
        let forward = NdReductionGeometry::new(&[2, 3, 4], &[1, 2], "reduce_mean_nd").unwrap();
        let reversed = NdReductionGeometry::new(&[2, 3, 4], &[2, 1], "reduce_mean_nd").unwrap();
        assert_eq!(forward.reduce, reversed.reduce);
        assert_eq!(forward.reduce, vec![1, 2]);
        assert_eq!(forward.rows, 12);
        assert_eq!(forward.cols, 2);
        assert_eq!(forward.out_shape, vec![2, 1, 1]);
        assert_eq!(forward.reduce_strides, vec![2, 6]);
    }

    #[test]
    fn invalid_geometry_is_rejected_before_materialization_or_submission() {
        assert!(NdReductionGeometry::new(&[], &[0], "reduce_mean_nd").is_err());
        assert!(NdReductionGeometry::new(&[2, 3], &[2], "reduce_mean_nd").is_err());
        assert!(NdReductionGeometry::new(&[2, 0, 4], &[1], "reduce_mean_nd").is_err());
        #[cfg(target_pointer_width = "64")]
        assert!(
            NdReductionGeometry::new(&[u32::MAX as usize + 1], &[0], "reduce_mean_nd").is_err()
        );
    }
}
