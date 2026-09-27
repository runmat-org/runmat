//! Pure launch limits shared by shader execution and feature-independent cost tests.
#[cfg(feature = "wgpu")]
pub(crate) const WORKGROUP_SIZE: u32 = 512;
pub(crate) const MAX_DISPATCH_WORKGROUPS: u32 = 65_535;

pub(crate) fn elementwise_chunk_capacity(workgroup_size: u32) -> Option<u64> {
    (workgroup_size != 0).then(|| {
        (u64::from(MAX_DISPATCH_WORKGROUPS) * u64::from(workgroup_size)).min(u64::from(u32::MAX))
    })
}
