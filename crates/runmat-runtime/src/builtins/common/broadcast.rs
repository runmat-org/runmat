//! Helpers for MATLAB-style implicit expansion and indexing.
//!
//! These utilities are shared by builtins that need to align operands with
//! MATLAB-compatible broadcasting semantics. The routines operate on column-
//! major shapes expressed as `[usize]` vectors.

/// Compute the broadcasted shape for two operands using MATLAB implicit
/// expansion rules.
pub fn broadcast_shapes(
    fn_name: &str,
    left: &[usize],
    right: &[usize],
) -> Result<Vec<usize>, String> {
    let rank = left.len().max(right.len());
    let mut shape = Vec::with_capacity(rank);
    for dim in 0..rank {
        let a = left.get(dim).copied().unwrap_or(1);
        let b = right.get(dim).copied().unwrap_or(1);
        if a == b {
            shape.push(a);
        } else if a == 1 {
            shape.push(b);
        } else if b == 1 {
            shape.push(a);
        } else if a == 0 || b == 0 {
            shape.push(0);
        } else {
            return Err(format!(
                "{fn_name}: size mismatch between inputs (dimension {} has lengths {} and {})",
                dim + 1,
                a,
                b
            ));
        }
    }
    Ok(shape)
}

/// Compute column-major strides for a given shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride.saturating_mul(extent.max(1));
    }
    strides
}

/// Map a linear index in the broadcasted result back to a source operand.
pub fn broadcast_index(
    mut linear: usize,
    out_shape: &[usize],
    in_shape: &[usize],
    strides: &[usize],
) -> usize {
    if in_shape.is_empty() {
        return 0;
    }
    let mut offset = 0usize;
    for dim in 0..out_shape.len() {
        let out_extent = out_shape[dim];
        let coord = if out_extent == 0 {
            0
        } else {
            linear % out_extent
        };
        if out_extent != 0 {
            linear /= out_extent;
        }
        let in_extent = in_shape.get(dim).copied().unwrap_or(1);
        let mapped = if in_extent == 1 || out_extent == 0 {
            0
        } else {
            coord
        };
        if dim < strides.len() {
            offset += mapped * strides[dim];
        }
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_equal_shapes() {
        let out = broadcast_shapes("test", &[2, 3], &[2, 3]).unwrap();
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn broadcast_scalar() {
        let out = broadcast_shapes("test", &[1, 1], &[4, 5]).unwrap();
        assert_eq!(out, vec![4, 5]);
    }

    #[test]
    fn broadcast_mismatched_dimension_errors() {
        let err = broadcast_shapes("test", &[2, 3], &[4, 3]).unwrap_err();
        assert!(err.contains("dimension 1"));
    }

    #[test]
    fn compute_strides_column_major() {
        let strides = compute_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![1, 2, 6]);
    }

    #[test]
    fn broadcast_index_maps_scalar_inputs() {
        let strides = compute_strides(&[1, 1]);
        let idx = broadcast_index(5, &[2, 3], &[1, 1], &strides);
        assert_eq!(idx, 0);
    }
}
