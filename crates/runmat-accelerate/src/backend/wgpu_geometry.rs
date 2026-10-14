//! Pure launch limits shared by shader execution and feature-independent cost tests.
#[cfg(feature = "wgpu")]
pub(crate) const WORKGROUP_SIZE: u32 = 512;
pub(crate) const MAX_DISPATCH_WORKGROUPS: u32 = 65_535;

pub(crate) fn elementwise_chunk_capacity(workgroup_size: u32) -> Option<u64> {
    (workgroup_size != 0).then(|| {
        (u64::from(MAX_DISPATCH_WORKGROUPS) * u64::from(workgroup_size)).min(u64::from(u32::MAX))
    })
}

/// Physical queue shape of the shared permutation kernel. Nonempty work
/// submits a no-op compute warmup, a noncompute flush, then one compute pass
/// per chunk; empty work submits nothing. QuerySets belong only to compute
/// encoders, whereas the existing fence tracks every queue submission.
#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PermutationSubmissionGeometry {
    pub chunk_capacity: u64,
    pub dispatches: u32,
    pub compute_encoders: u32,
    pub queue_submissions: u32,
}

#[cfg(feature = "wgpu")]
pub(crate) fn permutation_submission_geometry(
    storage_elements: u64,
    workgroup_size: u32,
) -> Option<PermutationSubmissionGeometry> {
    if storage_elements > u64::from(u32::MAX) {
        return None;
    }
    let chunk_capacity = elementwise_chunk_capacity(workgroup_size)?;
    let dispatches = u32::try_from(storage_elements.div_ceil(chunk_capacity)).ok()?;
    let (compute_encoders, queue_submissions) = if dispatches == 0 {
        (0, 0)
    } else {
        (dispatches.checked_add(1)?, dispatches.checked_add(2)?)
    };
    Some(PermutationSubmissionGeometry {
        chunk_capacity,
        dispatches,
        compute_encoders,
        queue_submissions,
    })
}

#[cfg(all(test, feature = "wgpu"))]
#[test]
fn permutation_geometry_separates_empty_warmup_flush_and_chunks() {
    let capacity = elementwise_chunk_capacity(512).unwrap();
    for (elements, expected) in [(0, (0, 0, 0)), (1, (1, 2, 3)), (capacity + 1, (2, 3, 4))] {
        let geometry = permutation_submission_geometry(elements, 512).unwrap();
        assert_eq!(
            (
                geometry.dispatches,
                geometry.compute_encoders,
                geometry.queue_submissions
            ),
            expected
        );
        assert_eq!(geometry.chunk_capacity, capacity);
    }
    assert!(permutation_submission_geometry(1, 0).is_none());
    assert!(permutation_submission_geometry(u64::from(u32::MAX) + 1, 512).is_none());
}

/// The native fill kernel uses the scalar workgroup and the same per-dispatch
/// workgroup limit as other WGPU elementwise constructors.
#[cfg(feature = "wgpu")]
pub(crate) fn fill_chunk_capacity(workgroup_size: u32) -> Option<u64> {
    elementwise_chunk_capacity(workgroup_size)
}

#[cfg(feature = "wgpu")]
pub(crate) struct FillPhysicalWork {
    pub output_bytes: u64,
    pub params_bytes: u64,
    pub dispatches: u32,
}

#[cfg(feature = "wgpu")]
impl FillPhysicalWork {
    pub fn total_bytes(&self) -> Option<u64> {
        self.output_bytes.checked_add(self.params_bytes)
    }
}

#[cfg(feature = "wgpu")]
pub(crate) fn fill_physical_work(
    elements: u64,
    workgroup_size: u32,
    precision: runmat_accelerate_api::ProviderPrecision,
) -> Option<FillPhysicalWork> {
    let scalar_bytes = match precision {
        runmat_accelerate_api::ProviderPrecision::F32 => 4,
        runmat_accelerate_api::ProviderPrecision::F64 => 8,
    };
    let capacity = fill_chunk_capacity(workgroup_size)?;
    let dispatches = if elements == 0 {
        0
    } else {
        u32::try_from(elements.div_ceil(capacity)).ok()?
    };
    let params_width = match precision {
        runmat_accelerate_api::ProviderPrecision::F32 => {
            std::mem::size_of::<crate::backend::wgpu::params::FillParamsF32>() as u64
        }
        runmat_accelerate_api::ProviderPrecision::F64 => {
            std::mem::size_of::<crate::backend::wgpu::params::FillParamsF64>() as u64
        }
    };
    Some(FillPhysicalWork {
        output_bytes: elements.max(1).checked_mul(scalar_bytes)?,
        params_bytes: u64::from(dispatches).checked_mul(params_width)?,
        dispatches,
    })
}

/// Fused broadcast uniform ABI: four scalar words, one output shape, and a
/// shape/stride pair for each input. Packed shape words occupy 16 bytes.
#[cfg(feature = "wgpu")]
pub(crate) fn fusion_broadcast_uniform_bytes(inputs: usize, max_rank: usize) -> Option<u64> {
    let arrays = inputs.checked_mul(2)?.checked_add(1)?;
    let packed = arrays.checked_mul(max_rank)?.checked_mul(16)?;
    u64::try_from(16usize.checked_add(packed)?).ok()
}
