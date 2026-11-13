//! Initialize preferred native backends before opening a secondary driver.
use std::sync::Arc;

use anyhow::{anyhow, Result};

use super::super::WgpuProviderOptions;

pub(super) async fn request(
    mut descriptor: wgpu::InstanceDescriptor,
    options: &WgpuProviderOptions,
) -> Result<(Arc<wgpu::Instance>, wgpu::Adapter)> {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "windows")))]
    let (preferred, fallback) = backend_candidates(wgpu::Backends::from_env());
    #[cfg(any(target_arch = "wasm32", target_os = "windows"))]
    let (preferred, fallback) = (descriptor.backends, None);

    descriptor.backends = preferred;
    let adapter_options = wgpu::RequestAdapterOptions {
        power_preference: options.power_preference,
        force_fallback_adapter: options.force_fallback_adapter,
        compatible_surface: None,
    };
    let instance = Arc::new(crate::backend::wgpu::compat::instance_new(
        descriptor.clone(),
    ));
    if let Some(adapter) = instance.request_adapter(&adapter_options).await {
        return Ok((instance, adapter));
    }
    // No device or progress owner exists yet. Retire the unsuccessful instance
    // before creating the secondary backend, rather than retaining two drivers
    // in every successful native provider's shared WGPU context.
    drop(instance);
    if let Some(fallback) = fallback {
        descriptor.backends = fallback;
        let instance = Arc::new(crate::backend::wgpu::compat::instance_new(descriptor));
        if let Some(adapter) = instance.request_adapter(&adapter_options).await {
            return Ok((instance, adapter));
        }
    }
    Err(anyhow!("wgpu: no compatible adapter found"))
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "windows")))]
fn backend_candidates(
    explicit: Option<wgpu::Backends>,
) -> (wgpu::Backends, Option<wgpu::Backends>) {
    match explicit {
        Some(backends) => (backends, None),
        None => (wgpu::Backends::PRIMARY, Some(wgpu::Backends::SECONDARY)),
    }
}

#[cfg(all(test, not(target_arch = "wasm32"), not(target_os = "windows")))]
mod tests {
    use super::*;

    #[test]
    fn secondary_driver_is_a_fallback_and_explicit_backends_are_not_replaced() {
        let (primary, fallback) = backend_candidates(None);
        assert!(!primary.intersects(wgpu::Backends::GL));
        assert!(fallback.unwrap().contains(wgpu::Backends::GL));
        for explicit in [
            wgpu::Backends::GL,
            wgpu::Backends::VULKAN,
            wgpu::Backends::all(),
            wgpu::Backends::empty(),
        ] {
            assert_eq!(backend_candidates(Some(explicit)), (explicit, None));
        }
    }
}
