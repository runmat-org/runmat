//! Typed ABI for multi-output shaders that index their inputs by domain
//! coordinates. Unlike broadcast fusion, the shader owns each index formula;
//! admission must prove its input roles before allocation or submission.

use anyhow::{ensure, Result};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderElementType};

pub(crate) struct IndexedFusionOperand<'a> {
    pub(crate) shape: &'a [usize],
    pub(crate) lanes: usize,
    pub(crate) storage: GpuTensorStorage,
}

pub(crate) struct IndexedFusionMultiRequest<'a> {
    pub(crate) shader: &'a str,
    pub(crate) inputs: &'a [GpuTensorHandle],
    pub(crate) output_shape: &'a [usize],
    pub(crate) len: usize,
    pub(crate) output_classes: &'a [ProviderElementType],
    pub(crate) operands: &'a [IndexedFusionOperand<'a>],
}

pub(super) enum MultiFusionLayout<'a> {
    Broadcast,
    Indexed(&'a [IndexedFusionOperand<'a>]),
}

pub(super) struct FusionMultiRequest<'a> {
    pub(super) shader: &'a str,
    pub(super) inputs: &'a [GpuTensorHandle],
    pub(super) output_shape: &'a [usize],
    pub(super) len: usize,
    pub(super) output_classes: &'a [ProviderElementType],
    pub(super) layout: MultiFusionLayout<'a>,
}

#[derive(Clone, Copy)]
pub(super) struct IndexedFusionObserved<'a> {
    pub(super) allocation_shape: &'a [usize],
    pub(super) view_shape: &'a [usize],
    pub(super) lanes: usize,
    pub(super) storage: GpuTensorStorage,
    pub(super) view_storage: GpuTensorStorage,
    pub(super) transposed: bool,
}

/// The returned empty broadcast table is deliberate: indexed shaders consume
/// only the common len/offset uniform prefix. The caller performs this check
/// before entering `with_emission`, so rejected roles have no GPU side effect.
pub(super) fn validate_indexed_layout(
    observed: &[IndexedFusionObserved<'_>],
    expected: &[IndexedFusionOperand<'_>],
    output_shape: &[usize],
    len: usize,
    max_rank: usize,
) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
    ensure!(
        observed.len() == expected.len(),
        "indexed fusion operand count mismatch"
    );
    ensure!(
        output_shape.len() <= max_rank
            && output_shape
                .iter()
                .all(|&extent| u32::try_from(extent).is_ok()),
        "indexed fusion output dimensions exceed GPU index limits"
    );
    ensure!(
        output_shape
            .iter()
            .try_fold(1usize, |size, &extent| size.checked_mul(extent))
            == Some(len),
        "indexed fusion output shape mismatch"
    );
    for (actual, role) in observed.iter().zip(expected) {
        ensure!(
            actual.view_shape == actual.allocation_shape && actual.allocation_shape == role.shape,
            "indexed fusion operand shape mismatch"
        );
        ensure!(
            actual.lanes == role.lanes,
            "indexed fusion operand lane mismatch"
        );
        ensure!(
            actual.storage == role.storage && actual.view_storage == role.storage,
            "indexed fusion operand storage mismatch"
        );
        ensure!(
            !actual.transposed,
            "indexed fusion requires contiguous input"
        );
    }
    Ok(vec![(Vec::new(), Vec::new()); observed.len()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_roles_reject_shape_lane_storage_and_lazy_view_before_dispatch() {
        let expected = [IndexedFusionOperand {
            shape: &[4, 2],
            lanes: 8,
            storage: GpuTensorStorage::Real,
        }];
        let valid = IndexedFusionObserved {
            allocation_shape: &[4, 2],
            view_shape: &[4, 2],
            lanes: 8,
            storage: GpuTensorStorage::Real,
            view_storage: GpuTensorStorage::Real,
            transposed: false,
        };
        assert_eq!(
            validate_indexed_layout(&[valid], &expected, &[2, 4], 8, 8)
                .unwrap()
                .len(),
            1
        );
        let reject = |actual: IndexedFusionObserved<'_>, token: &str| {
            assert!(validate_indexed_layout(&[actual], &expected, &[2, 4], 8, 8)
                .unwrap_err()
                .to_string()
                .contains(token));
        };
        reject(
            IndexedFusionObserved {
                allocation_shape: &[2, 4],
                ..valid
            },
            "shape",
        );
        reject(IndexedFusionObserved { lanes: 7, ..valid }, "lane");
        reject(
            IndexedFusionObserved {
                storage: GpuTensorStorage::ComplexInterleaved,
                ..valid
            },
            "storage",
        );
        reject(
            IndexedFusionObserved {
                transposed: true,
                ..valid
            },
            "contiguous",
        );
        assert!(validate_indexed_layout(&[valid], &expected, &[2, 5], 8, 8).is_err());
    }
}
