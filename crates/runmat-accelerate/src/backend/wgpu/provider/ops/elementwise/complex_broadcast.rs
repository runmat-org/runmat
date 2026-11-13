//! Pre-effect admission for the fixed-rank complex broadcast shader ABI.

use anyhow::Result;
use runmat_accelerate_api::unsupported_provider_operation;

pub(super) fn admit_shapes(
    left: &[usize],
    right: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rank = left.len().max(right.len());
    // Check semantic shape compatibility with the existing decoder first;
    // shader ABI limits must not turn invalid dimensions into fallback.
    let shapes = super::super::backend_shared::matlab_broadcast_shapes(
        "complex broadcast",
        left,
        right,
        rank,
    )?;
    if rank > crate::backend::wgpu::params::BCAST_MAX_RANK {
        return Err(unsupported_provider_operation(
            "complex broadcast rank exceeds the provider shader ABI",
        ));
    }
    Ok(shapes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_rank_is_typed_unsupported_before_provider_effects() {
        let mut left = vec![1; crate::backend::wgpu::params::BCAST_MAX_RANK + 1];
        let mut right = left.clone();
        left[0] = 2;
        right[1] = 2;
        let error = admit_shapes(&left, &right).expect_err("shader cannot address this rank");
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &error
        ));
    }

    #[test]
    fn invalid_shape_remains_distinct_from_unsupported_rank() {
        let error = admit_shapes(&[2, 3], &[4, 3]).expect_err("incompatible dimension");
        assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
            &error
        ));
        assert!(error.to_string().contains("dimension 1"));
        let mut high_rank_invalid = vec![1; crate::backend::wgpu::params::BCAST_MAX_RANK + 1];
        high_rank_invalid[0] = 2;
        let mut other = high_rank_invalid.clone();
        other[0] = 3;
        let error = admit_shapes(&high_rank_invalid, &other).expect_err("incompatible high rank");
        assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
            &error
        ));
        assert!(error.to_string().contains("dimension 1"));
        let (_, _, output) = admit_shapes(&[2, 1], &[1, 3]).expect("supported broadcast");
        assert_eq!(output, [2, 3]);
    }
}
