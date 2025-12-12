//! Shared plotting context registry for zero-copy rendering.
//!
//! This module tracks the GPU device/queue exported by the active acceleration
//! provider so plotting backends (native GUI or wasm/web) can reuse the same
//! `wgpu::Device` without creating duplicate adapters. Call
//! [`ensure_context_from_provider`] or [`install_wgpu_context`] once a provider
//! is initialized; subsequent callers can query [`shared_wgpu_context`] to
//! determine whether zero-copy rendering is possible.

#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::OnceCell;
use runmat_accelerate_api::{AccelContextHandle, AccelContextKind, WgpuContextHandle};
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
use std::fmt;

/// Process-wide WGPU context exported by the acceleration provider (if any).
#[cfg(not(target_arch = "wasm32"))]
static SHARED_WGPU_CONTEXT: OnceCell<WgpuContextHandle> = OnceCell::new();

#[cfg(target_arch = "wasm32")]
thread_local! {
    static SHARED_WGPU_CONTEXT: RefCell<Option<WgpuContextHandle>> = RefCell::new(None);
}

/// Returns the cached shared WGPU context, if one has been installed.
pub fn shared_wgpu_context() -> Option<WgpuContextHandle> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        SHARED_WGPU_CONTEXT.get().cloned()
    }
    #[cfg(target_arch = "wasm32")]
    {
        SHARED_WGPU_CONTEXT.with(|cell| cell.borrow().clone())
    }
}

/// Install a shared context that was exported out-of-band (e.g. from a host
/// application that already called `export_context`).
pub fn install_wgpu_context(context: &WgpuContextHandle) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = SHARED_WGPU_CONTEXT.set(context.clone());
    }
    #[cfg(target_arch = "wasm32")]
    {
        SHARED_WGPU_CONTEXT.with(|cell| {
            *cell.borrow_mut() = Some(context.clone());
        });
    }
    propagate_to_plot_crate(context);
}

/// Ensure the shared context is populated by calling back into the acceleration
/// provider. Returns the cached context when available.
pub fn ensure_context_from_provider() -> Result<WgpuContextHandle, PlotContextError> {
    if let Some(ctx) = shared_wgpu_context() {
        return Ok(ctx);
    }

    let handle = runmat_accelerate_api::export_context(AccelContextKind::Plotting)
        .ok_or(PlotContextError::Unavailable)?;
    match handle {
        AccelContextHandle::Wgpu(ctx) => {
            install_wgpu_context(&ctx);
            Ok(ctx)
        }
    }
}

#[derive(Debug)]
pub enum PlotContextError {
    Unavailable,
}

impl fmt::Display for PlotContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlotContextError::Unavailable => write!(
                f,
                "plotting context unavailable (GPU provider did not export a shared device)"
            ),
        }
    }
}

impl std::error::Error for PlotContextError {}

fn propagate_to_plot_crate(context: &WgpuContextHandle) {
    #[cfg(feature = "plot-core")]
    {
        use runmat_plot::context::{
            install_shared_wgpu_context as install_plot_context, SharedWgpuContext,
        };
        use runmat_plot::gpu::tuning as plot_tuning;
        install_plot_context(SharedWgpuContext {
            instance: context.instance.clone(),
            device: context.device.clone(),
            queue: context.queue.clone(),
            adapter: context.adapter.clone(),
            adapter_info: context.adapter_info.clone(),
            limits: context.limits.clone(),
            features: context.features,
        });
        if let Some(wg) = runmat_accelerate_api::workgroup_size_hint() {
            plot_tuning::set_effective_workgroup_size(wg);
        }
    }
}

#[cfg(all(test, feature = "plot-core"))]
pub(crate) mod tests {
    use super::*;
    use pollster::FutureExt;
    use std::sync::Arc;

    #[test]
    fn install_context_propagates_to_plot_crate() {
        if std::env::var("RUNMAT_PLOT_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let instance = Arc::new(wgpu::Instance::default());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()
        {
            Some(adapter) => adapter,
            None => return,
        };
        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let adapter_features = adapter.features();
        let adapter = Arc::new(adapter);
        let (device, queue) = match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("plotting-context-test-device"),
                    required_features: adapter_features,
                    required_limits: adapter_limits.clone(),
                },
                None,
            )
            .block_on()
        {
            Ok(pair) => pair,
            Err(_) => return,
        };
        let handle = WgpuContextHandle {
            instance: instance.clone(),
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter: adapter.clone(),
            adapter_info,
            limits: adapter_limits.clone(),
            features: adapter_features,
        };
        install_wgpu_context(&handle);
        assert!(shared_wgpu_context().is_some());
        assert!(runmat_plot::shared_wgpu_context().is_some());
    }
}
