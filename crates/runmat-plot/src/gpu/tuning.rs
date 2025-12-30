use std::sync::atomic::{AtomicU32, Ordering};

/// Default workgroup size used by plotting compute shaders when no hint is
/// provided by the acceleration stack. Matches the WGPU backend defaults so we
/// stay within adapter limits even before calibration.
const DEFAULT_WG_SIZE: u32 = 256;

static WORKGROUP_OVERRIDE: AtomicU32 = AtomicU32::new(0);

/// Returns the currently configured workgroup size for plot compute shaders.
pub fn effective_workgroup_size() -> u32 {
    let value = WORKGROUP_OVERRIDE.load(Ordering::Relaxed);
    if value > 0 {
        value
    } else {
        DEFAULT_WG_SIZE
    }
}

/// Update the workgroup size hint using the acceleration provider's
/// auto-calibrated value. Values of zero are ignored to avoid disabling the
/// fallback default.
pub fn set_effective_workgroup_size(value: u32) {
    if value > 0 {
        WORKGROUP_OVERRIDE.store(value, Ordering::Relaxed);
    }
}
