//! The provider and plot admission use the same native-WGPU requirements.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use runmat_accelerate_api::native_wgpu_f64::validate_adapter;
pub(crate) use runmat_accelerate_api::native_wgpu_f64::REQUIRED_FEATURES;
