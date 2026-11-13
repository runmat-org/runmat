//! Shape-only views retain their authentic allocation; lazy layouts materialize
//! through the exact source provider before changing logical geometry.
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorLayout};

use crate::builtins::common::gpu_helpers;

fn elements(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        Some(0)
    } else {
        shape.iter().try_fold(1_usize, |n, &dim| n.checked_mul(dim))
    }
}

pub(super) fn reshape(
    source: GpuTensorHandle,
    shape: &[usize],
    builtin: &'static str,
) -> crate::BuiltinResult<GpuTensorHandle> {
    let error = |message: &str| {
        crate::build_runtime_error(format!("{builtin}: {message}"))
            .with_builtin(builtin)
            .build()
    };
    let count = elements(source.shape()).ok_or_else(|| error("input dimensions overflow"))?;
    if elements(shape) != Some(count) {
        return Err(error(
            "reshape dimensions must preserve the number of elements",
        ));
    }
    if source.shape() == shape {
        return Ok(source);
    }
    if source.view().layout() == GpuTensorLayout::Contiguous {
        return source
            .with_view(source.view().clone().with_shape(shape.to_vec()))
            .map_err(|cause| {
                crate::build_runtime_error(format!(
                    "{builtin}: cannot create reshaped view: {cause}"
                ))
                .with_builtin(builtin)
                .with_source(cause)
                .build()
            });
    }

    let provider = gpu_helpers::require_resident_compute_owner(&source, || {
        error("no acceleration provider owns the lazy-layout input")
    })?;
    let mut output = provider.reshape(&source, shape).map_err(|cause| {
        crate::build_runtime_error(format!("{builtin}: layout materialization failed: {cause}"))
            .with_builtin(builtin)
            .with_source(runmat_accelerate_api::ProviderExecutionError::from(cause))
            .build()
    })?;
    if output.shape() != shape
        || output.device_id() != source.device_id()
        || !gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        || output.physical().element_type() != source.physical().element_type()
        || output.physical().storage() != source.physical().storage()
        || output.view().is_logical() != source.view().is_logical()
        || output.view().layout() != GpuTensorLayout::Contiguous
        || output.shares_allocation_with(&source)
    {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            error("provider returned an invalid materialized reshape"),
        ));
    }
    output = gpu_helpers::propagate_owned_output_provenance(output, [&source])?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };

    #[test]
    fn contiguous_views_retain_exact_storage_and_last_alias_owns_retirement() {
        let owner = TestAllocationOwner::new();
        for (element, storage) in [
            (NumericElementType::F32, GpuTensorStorage::Real),
            (
                NumericElementType::U64,
                GpuTensorStorage::ComplexInterleaved,
            ),
        ] {
            let allocation = owner
                .allocate_zeroed(vec![1, 2, 1], element, storage)
                .unwrap();
            let source = allocation.handle;
            let output = reshape(source.clone(), &[2, 1], "squeeze").unwrap();
            assert!(source.shares_allocation_with(&output));
            assert_eq!(source.physical(), output.physical());
            assert_eq!(source.shape(), &[1, 2, 1]);
            assert_eq!(output.shape(), &[2, 1]);
            drop(source);
            assert_eq!(allocation.probe.retirements(), 0);
            drop(output);
            assert_eq!(allocation.probe.retirements(), 1);
        }
    }

    #[test]
    fn zero_dimension_precedes_product_overflow_and_rejections_preserve_source() {
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(vec![0], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let output = reshape(allocation.handle.clone(), &[usize::MAX, 2, 0], "reshape").unwrap();
        assert_eq!(output.shape(), &[usize::MAX, 2, 0]);
        assert!(reshape(allocation.handle.clone(), &[usize::MAX, 2], "reshape").is_err());
        assert_eq!(allocation.probe.retirements(), 0);
        drop((output, allocation.handle));
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn logical_producer_intent_and_explicit_provenance_survive_shape_views() {
        use runmat_accelerate_api::{GpuHandleProvenance, ProviderPrecision};
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_logical_zeroed(vec![1, 2, 1], ProviderPrecision::F32)
            .unwrap();
        let source = allocation
            .handle
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        drop(allocation.handle);
        let output = reshape(source, &[2, 1], "shiftdim").unwrap();
        assert!(output.view().is_logical());
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        drop(output);
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn lazy_layout_is_not_silently_reinterpreted_without_its_owner() {
        use runmat_accelerate_api::{GpuTensorView, TransposeInfo};
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(vec![2, 3], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let source = allocation
            .handle
            .with_view(GpuTensorView::contiguous(vec![3, 2]).with_layout(
                GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 3,
                }),
            ))
            .unwrap();
        let facts = source.view().clone();
        assert!(reshape(source.clone(), &[6, 1], "reshape").is_err());
        assert_eq!(source.view(), &facts);
        assert_eq!(allocation.probe.retirements(), 0);
        drop((source, allocation.handle));
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
