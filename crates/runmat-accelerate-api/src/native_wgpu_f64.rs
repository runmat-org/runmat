//! Shared native-WGPU admission for provider-owned binary64 kernels.
//!
//! A plot loads F64 input and narrows it to an F32 vertex. Native adapter
//! admission authorizes that route; rendering retains F32 vertex precision.
//! External raw-device callers must validate their adapter before creating an
//! F64 device. Stock provider construction does so.

pub const REQUIRED_FEATURES: wgpu::Features = wgpu::Features::SHADER_F64
    .union(wgpu::Features::SHADER_INT64)
    .union(wgpu::Features::SPIRV_SHADER_PASSTHROUGH);

#[cfg(not(target_arch = "wasm32"))]
pub fn validate_adapter(adapter: &wgpu::Adapter) -> anyhow::Result<()> {
    anyhow::ensure!(
        adapter.features().contains(REQUIRED_FEATURES),
        "portable binary64 shaders require native F64, INT64 and SPIR-V support"
    );
    anyhow::ensure!(
        adapter.get_info().backend == wgpu::Backend::Vulkan,
        "controlled native binary64 shaders require Vulkan"
    );
    validate_vulkan(adapter)
}

#[cfg(all(
    not(target_arch = "wasm32"),
    any(target_os = "linux", target_os = "windows")
))]
fn validate_vulkan(adapter: &wgpu::Adapter) -> anyhow::Result<()> {
    use ash::vk;
    // SAFETY: the borrowed HAL adapter is inspected only within the callback.
    // Neither handle is retained/destroyed, and no reentrant wgpu call is made.
    unsafe {
        adapter.as_hal::<wgpu::hal::api::Vulkan, _, _>(|adapter| {
            let adapter =
                adapter.ok_or_else(|| anyhow::anyhow!("Vulkan adapter is unavailable"))?;
            let instance = adapter.shared_instance();
            anyhow::ensure!(
                instance.instance_api_version() >= vk::API_VERSION_1_2,
                "portable binary64 shaders require Vulkan 1.2 float controls"
            );
            let mut controls = vk::PhysicalDeviceFloatControlsProperties::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut controls);
            instance
                .raw_instance()
                .get_physical_device_properties2(adapter.raw_physical_device(), &mut properties);
            anyhow::ensure!(
                properties.properties.api_version >= vk::API_VERSION_1_2,
                "portable binary64 shaders require a Vulkan 1.2 device"
            );
            anyhow::ensure!(
                controls.shader_rounding_mode_rte_float64 == vk::TRUE
                    && controls.shader_signed_zero_inf_nan_preserve_float64 == vk::TRUE,
                "device cannot preserve required binary64 rounding and special values"
            );
            // DenormPreserve is deliberately not requested: provider kernels
            // normalize and pack subnormal values explicitly, including on T4.
            Ok(())
        })
    }
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(any(target_os = "linux", target_os = "windows"))
))]
fn validate_vulkan(_adapter: &wgpu::Adapter) -> anyhow::Result<()> {
    anyhow::bail!("controlled Vulkan binary64 shaders are unavailable on this target")
}
