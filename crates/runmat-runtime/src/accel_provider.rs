use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{trace, warn};

thread_local! {
    static ACCEL_CONTEXT: RefCell<Option<&'static str>> = const { RefCell::new(None) };
}

#[derive(Debug)]
pub struct AccelContextGuard(Option<&'static str>);

impl Drop for AccelContextGuard {
    fn drop(&mut self) {
        ACCEL_CONTEXT.with(|slot| {
            *slot.borrow_mut() = self.0.take();
        });
    }
}

static ACCEL_BOUNDARY_CROSSES: AtomicUsize = AtomicUsize::new(0);

fn record_crossing(context: &str, device_id: Option<u32>) {
    let hits = ACCEL_BOUNDARY_CROSSES.fetch_add(1, Ordering::Relaxed) + 1;
    trace!(
        context,
        device = ?device_id,
        boundary_crossings = hits,
        "entering accelerator boundary"
    );
}

fn trace_missing(context: &str, device_id: Option<u32>) {
    trace!(
        context,
        device = ?device_id,
        "accelerator provider not available in this context"
    );
}

/// Attempt to acquire the registered acceleration provider, logging how often
/// the boundary is crossed.
pub fn maybe_provider(context: &str) -> Option<&'static dyn AccelProvider> {
    let provider = runmat_accelerate_api::provider();
    if provider.is_some() {
        record_crossing(context, None);
    } else {
        trace_missing(context, None);
    }
    provider
}

/// Attempt to acquire the provider that owns the supplied GPU handle, logging
/// the crossing if the provider exists.
pub fn maybe_provider_for_handle(
    context: &str,
    handle: &GpuTensorHandle,
) -> Option<&'static dyn AccelProvider> {
    let provider = runmat_accelerate_api::provider_for_handle(handle);
    if provider.is_some() {
        record_crossing(context, Some(handle.device_id));
    } else {
        trace_missing(context, Some(handle.device_id));
    }
    provider
}

/// Acquire the provider for a GPU handle or return a standardized error if no
/// provider is registered.
pub fn provider_for_handle(
    context: &str,
    handle: &GpuTensorHandle,
) -> Result<&'static dyn AccelProvider, String> {
    maybe_provider_for_handle(context, handle).ok_or_else(|| {
        let message = format!(
            "{}: no acceleration provider registered for handle on device {}",
            context, handle.device_id
        );
        warn!(context, device = ?handle.device_id, "{}", message);
        message
    })
}
