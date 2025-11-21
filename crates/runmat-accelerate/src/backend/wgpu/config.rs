use std::sync::atomic::{AtomicU32, Ordering};

pub const WORKGROUP_SIZE: u32 = 512;
pub const REDUCE_WORKGROUP_SIZE: u32 = 512;
pub const DEFAULT_TWO_PASS_THRESHOLD: usize = 262_144;
pub const DEFAULT_REDUCTION_WG: u32 = 512;
pub const MATMUL_TILE: u32 = 32;
pub const MAX_DISPATCH_WORKGROUPS: u32 = 65_535;

static EFFECTIVE_WG_OVERRIDE: AtomicU32 = AtomicU32::new(0);
static EFFECTIVE_MATMUL_TILE_OVERRIDE: AtomicU32 = AtomicU32::new(0);

fn parse_env_u32(var: &str) -> Option<u32> {
    std::env::var(var)
        .ok()
        .and_then(|val| val.trim().parse::<u32>().ok())
        .filter(|v| *v > 0)
}

pub(crate) fn env_requested_workgroup_size() -> Option<u32> {
    parse_env_u32("RUNMAT_WG")
}

pub(crate) fn env_requested_reduction_workgroup_size() -> Option<u32> {
    parse_env_u32("RUNMAT_REDUCTION_WG")
}

pub(crate) fn env_requested_matmul_tile() -> Option<u32> {
    parse_env_u32("RUNMAT_MATMUL_TILE")
}

/// Set at runtime once adapter limits are known. Used by shader patching helpers.
pub(crate) fn set_effective_workgroup_size(value: u32) {
    EFFECTIVE_WG_OVERRIDE.store(value, Ordering::Relaxed);
}

pub(crate) fn set_effective_matmul_tile(value: u32) {
    EFFECTIVE_MATMUL_TILE_OVERRIDE.store(value, Ordering::Relaxed);
}

fn unresolved_workgroup_size() -> u32 {
    env_requested_workgroup_size().unwrap_or(WORKGROUP_SIZE)
}

fn unresolved_matmul_tile() -> u32 {
    env_requested_matmul_tile().unwrap_or(MATMUL_TILE)
}

/// Effective global workgroup size for elementwise/fused kernels.
/// Overridable via env `RUNMAT_WG` (u32). Falls back to WORKGROUP_SIZE.
pub fn effective_workgroup_size() -> u32 {
    let override_val = EFFECTIVE_WG_OVERRIDE.load(Ordering::Relaxed);
    if override_val > 0 {
        return override_val;
    }
    unresolved_workgroup_size()
}

/// Effective matmul tile size (square tile), overridable via env `RUNMAT_MATMUL_TILE`.
pub fn effective_matmul_tile() -> u32 {
    let override_val = EFFECTIVE_MATMUL_TILE_OVERRIDE.load(Ordering::Relaxed);
    if override_val > 0 {
        return override_val;
    }
    unresolved_matmul_tile()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_helpers_parse() {
        std::env::set_var("RUNMAT_WG", "256");
        assert_eq!(env_requested_workgroup_size(), Some(256));
        std::env::set_var("RUNMAT_MATMUL_TILE", "16");
        assert_eq!(unresolved_matmul_tile(), 16);
        EFFECTIVE_WG_OVERRIDE.store(64, Ordering::Relaxed);
        EFFECTIVE_MATMUL_TILE_OVERRIDE.store(8, Ordering::Relaxed);
        assert_eq!(effective_workgroup_size(), 64);
        assert_eq!(effective_matmul_tile(), 8);
        EFFECTIVE_WG_OVERRIDE.store(0, Ordering::Relaxed);
        EFFECTIVE_MATMUL_TILE_OVERRIDE.store(0, Ordering::Relaxed);
        std::env::remove_var("RUNMAT_WG");
        std::env::remove_var("RUNMAT_MATMUL_TILE");
    }
}
