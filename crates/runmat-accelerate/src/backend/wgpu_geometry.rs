//! Pure launch limits shared by shader execution and feature-independent cost tests.
#[cfg(feature = "wgpu")]
pub(crate) const WORKGROUP_SIZE: u32 = 512;
pub(crate) const MAX_DISPATCH_WORKGROUPS: u32 = 65_535;

pub(crate) fn elementwise_chunk_capacity(workgroup_size: u32) -> Option<u64> {
    (workgroup_size != 0).then(|| {
        (u64::from(MAX_DISPATCH_WORKGROUPS) * u64::from(workgroup_size)).min(u64::from(u32::MAX))
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
