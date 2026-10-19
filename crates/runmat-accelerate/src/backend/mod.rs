#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
pub mod cuda;
#[cfg(feature = "wgpu")]
pub mod wgpu;

// Pure storage geometry is also used by capability queries without a device.
pub(crate) mod wgpu_storage;
pub(crate) mod wgpu_geometry;
pub(crate) mod fusion_operand_layout;
