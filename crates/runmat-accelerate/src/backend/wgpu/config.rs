pub const WORKGROUP_SIZE: u32 = 512;
pub const REDUCE_WORKGROUP_SIZE: u32 = 512;
pub const DEFAULT_TWO_PASS_THRESHOLD: usize = 1024;
pub const DEFAULT_REDUCTION_WG: u32 = 512;
pub const MATMUL_TILE: u32 = 16;
pub const MAX_DISPATCH_WORKGROUPS: u32 = 65_535;

/// Effective global workgroup size for elementwise/fused kernels.
/// Overridable via env `RUNMAT_WG` (u32). Falls back to WORKGROUP_SIZE.
pub fn effective_workgroup_size() -> u32 {
    if let Ok(val) = std::env::var("RUNMAT_WG") {
        if let Ok(parsed) = val.trim().parse::<u32>() {
            if parsed > 0 { return parsed; }
        }
    }
    WORKGROUP_SIZE
}
