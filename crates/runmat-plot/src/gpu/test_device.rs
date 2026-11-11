//! One test device policy for both scatter packers. Native F64 tests enter
//! through the same adapter admission used by the production WGPU provider.

use pollster::FutureExt;

pub(super) fn maybe_scatter_device(
    label: &'static str,
) -> Option<runmat_accelerate_api::WgpuDeviceAccess> {
    if std::env::var("RUNMAT_PLOT_SKIP_GPU_TESTS").is_ok()
        || std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").is_err()
    {
        return None;
    }
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .block_on()?;
    #[cfg(not(target_arch = "wasm32"))]
    let features = match runmat_accelerate_api::native_wgpu_f64::validate_adapter(&adapter) {
        Ok(()) => runmat_accelerate_api::native_wgpu_f64::REQUIRED_FEATURES,
        Err(error) => {
            eprintln!("scatter test adapter is not admitted for native F64: {error}");
            wgpu::Features::empty()
        }
    };
    #[cfg(target_arch = "wasm32")]
    let features = wgpu::Features::empty();
    let (device, queue) = adapter
        .request_device(
            &crate::wgpu_compat::device_descriptor(Some(label), features, adapter.limits()),
            None,
        )
        .block_on()
        .ok()?;
    Some(runmat_accelerate_api::WgpuDeviceAccess::new(device, queue))
}
