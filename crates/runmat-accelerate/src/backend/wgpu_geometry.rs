//! Pure launch limits shared by shader execution and feature-independent cost tests.
#[cfg(feature = "wgpu")]
pub(crate) const WORKGROUP_SIZE: u32 = 512;
pub(crate) const MAX_DISPATCH_WORKGROUPS: u32 = 65_535;

pub(crate) fn elementwise_chunk_capacity(workgroup_size: u32) -> Option<u64> {
    (workgroup_size != 0).then(|| {
        (u64::from(MAX_DISPATCH_WORKGROUPS) * u64::from(workgroup_size)).min(u64::from(u32::MAX))
    })
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
