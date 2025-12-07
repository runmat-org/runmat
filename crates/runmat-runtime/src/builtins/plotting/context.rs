//! Shared plotting context registry for zero-copy rendering.
//!
//! This module tracks the GPU device/queue exported by the active acceleration
//! provider so plotting backends (native GUI or wasm/web) can reuse the same
//! `wgpu::Device` without creating duplicate adapters. Call
//! [`ensure_context_from_provider`] or [`install_wgpu_context`] once a provider
//! is initialized; subsequent callers can query [`shared_wgpu_context`] to
//! determine whether zero-copy rendering is possible.

use once_cell::sync::OnceCell;
use runmat_accelerate_api::{AccelContextHandle, AccelContextKind, WgpuContextHandle};

/// Process-wide WGPU context exported by the acceleration provider (if any).
static SHARED_WGPU_CONTEXT: OnceCell<WgpuContextHandle> = OnceCell::new();

/// Returns the cached shared WGPU context, if one has been installed.
pub fn shared_wgpu_context() -> Option<WgpuContextHandle> {
    SHARED_WGPU_CONTEXT.get().cloned()
}

/// Install a shared context that was exported out-of-band (e.g. from a host
/// application that already called `export_context`).
pub fn install_wgpu_context(context: &WgpuContextHandle) {
    let _ = SHARED_WGPU_CONTEXT.set(context.clone());
    propagate_to_plot_crate(context);
}

/// Ensure the shared context is populated by calling back into the acceleration
/// provider. Returns the cached context when available.
pub fn ensure_context_from_provider() -> Option<WgpuContextHandle> {
    if let Some(ctx) = shared_wgpu_context() {
        return Some(ctx);
    }

    let handle = runmat_accelerate_api::export_context(AccelContextKind::Plotting)?;
    match handle {
        AccelContextHandle::Wgpu(ctx) => {
            let _ = SHARED_WGPU_CONTEXT.set(ctx.clone());
            propagate_to_plot_crate(&ctx);
            Some(ctx)
        }
    }
}

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
