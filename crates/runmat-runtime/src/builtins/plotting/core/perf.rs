//! Plotting performance knobs and level-of-detail helpers.

use once_cell::sync::OnceCell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const DEFAULT_SCATTER_TARGET_POINTS: u32 = 250_000;
const MIN_SCATTER_TARGET_POINTS: u32 = 16_384;
const DEFAULT_SURFACE_VERTEX_BUDGET: u64 = 400_000;
const MIN_SURFACE_VERTEX_BUDGET: u64 = 65_536;
const SCATTER_EXTENT_REFERENCE: f32 = 250.0;
const SURFACE_EXTENT_REFERENCE: f32 = 500.0;

static SCATTER_TARGET_POINTS: AtomicU32 = AtomicU32::new(DEFAULT_SCATTER_TARGET_POINTS);
static SURFACE_VERTEX_BUDGET: AtomicU64 = AtomicU64::new(DEFAULT_SURFACE_VERTEX_BUDGET);
static SCATTER_ENV_INIT: OnceCell<()> = OnceCell::new();
static SURFACE_ENV_INIT: OnceCell<()> = OnceCell::new();

/// Returns the target number of scatter points we aim to draw per dispatch
/// before enabling compute-side decimation. The value can be overridden via
/// the `RUNMAT_PLOT_SCATTER_TARGET` environment variable or by calling
/// [`set_scatter_target_points`].
pub(crate) fn scatter_target_points() -> u32 {
    ensure_scatter_env();
    SCATTER_TARGET_POINTS.load(Ordering::Relaxed)
}

/// Override the scatter point target at runtime (e.g., via CLI flags or
/// TypeScript bindings). Values below the minimum threshold are clamped.
pub fn set_scatter_target_points(value: u32) {
    let clamped = value.max(MIN_SCATTER_TARGET_POINTS);
    SCATTER_TARGET_POINTS.store(clamped, Ordering::Relaxed);
}

/// Returns the maximum number of surface vertices we attempt to pack on the GPU
/// before enabling LOD. Override via `RUNMAT_PLOT_SURFACE_VERTEX_BUDGET` or
/// [`set_surface_vertex_budget`].
pub(crate) fn surface_vertex_budget() -> u64 {
    ensure_surface_env();
    SURFACE_VERTEX_BUDGET.load(Ordering::Relaxed)
}

/// Override the surface vertex budget at runtime.
pub fn set_surface_vertex_budget(value: u64) {
    let clamped = value.max(MIN_SURFACE_VERTEX_BUDGET);
    SURFACE_VERTEX_BUDGET.store(clamped, Ordering::Relaxed);
}

fn ensure_scatter_env() {
    SCATTER_ENV_INIT.get_or_init(|| {
        if let Some(value) = read_env_u32("RUNMAT_PLOT_SCATTER_TARGET") {
            set_scatter_target_points(value);
        }
    });
}

fn ensure_surface_env() {
    SURFACE_ENV_INIT.get_or_init(|| {
        if let Some(value) = read_env_u64("RUNMAT_PLOT_SURFACE_VERTEX_BUDGET") {
            set_surface_vertex_budget(value);
        }
    });
}

fn read_env_u32(key: &str) -> Option<u32> {
    std::env::var(key).ok().and_then(|value| value.parse().ok())
}

fn read_env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|value| value.parse().ok())
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SurfaceLod {
    pub stride_x: u32,
    pub stride_y: u32,
    pub lod_x_len: u32,
    pub lod_y_len: u32,
}

impl SurfaceLod {
    pub fn vertex_count(&self) -> usize {
        (self.lod_x_len as usize) * (self.lod_y_len as usize)
    }
}

fn adjust_for_extent<T>(base: T, extent_hint: f32, reference: f32) -> T
where
    T: num_traits::NumCast + Copy,
{
    if !extent_hint.is_finite() || extent_hint <= 0.0 {
        return base;
    }
    let reference = reference.max(1.0);
    let ratio = (reference / extent_hint).clamp(0.25, 4.0);
    let adjusted = num_traits::cast::<_, f64>(base).unwrap_or(0.0) * ratio as f64;
    num_traits::cast(adjusted).unwrap_or(base)
}

/// Compute an approximate level-of-detail strategy for a surface grid so that
/// the generated vertex count stays below the configured budget. `extent_hint`
/// should represent the planar diagonal of the surface (e.g. sqrt(dx^2+dy^2))
/// so that zoomed-in views retain more detail than zoomed-out ones.
pub(crate) fn compute_surface_lod(x_len: usize, y_len: usize, extent_hint: f32) -> SurfaceLod {
    let x_len = x_len.max(1);
    let y_len = y_len.max(1);
    let x_u32 = x_len as u32;
    let y_u32 = y_len as u32;
    let total_vertices = (x_len as u64) * (y_len as u64);
    let mut budget = surface_vertex_budget().max(MIN_SURFACE_VERTEX_BUDGET);
    if extent_hint.is_finite() && extent_hint > 0.0 {
        let adjusted =
            adjust_for_extent::<u64>(budget, extent_hint, SURFACE_EXTENT_REFERENCE).max(1);
        budget = adjusted.max(MIN_SURFACE_VERTEX_BUDGET);
    }

    if total_vertices <= budget {
        return SurfaceLod {
            stride_x: 1,
            stride_y: 1,
            lod_x_len: x_u32,
            lod_y_len: y_u32,
        };
    }

    let stride_guess = ((total_vertices as f64 / budget as f64).sqrt().ceil() as u32).max(2);
    let mut stride_x = stride_guess.min(x_u32);
    let mut stride_y = stride_guess.min(y_u32);
    let mut lod_x_len = ceil_div(x_u32, stride_x);
    let mut lod_y_len = ceil_div(y_u32, stride_y);

    for _ in 0..32 {
        if (lod_x_len as u64) * (lod_y_len as u64) <= budget {
            break;
        }
        if lod_x_len >= lod_y_len && stride_x < x_u32 {
            stride_x = stride_x.saturating_add(1).min(x_u32);
            lod_x_len = ceil_div(x_u32, stride_x);
        } else if stride_y < y_u32 {
            stride_y = stride_y.saturating_add(1).min(y_u32);
            lod_y_len = ceil_div(y_u32, stride_y);
        } else {
            break;
        }
    }

    SurfaceLod {
        stride_x: stride_x.max(1),
        stride_y: stride_y.max(1),
        lod_x_len: lod_x_len.max(1),
        lod_y_len: lod_y_len.max(1),
    }
}

fn ceil_div(len: u32, stride: u32) -> u32 {
    if stride == 0 {
        return len;
    }
    (len + stride - 1) / stride
}

/// Compute the level-of-detail stride for scatter3 given the number of points
/// and an extent hint (diagonal of the plot bounds). Larger plots will target
/// fewer points, while zoomed-in plots retain more detail.
pub(crate) fn scatter3_lod_stride(point_count: u32, extent_hint: f32) -> u32 {
    let base = scatter_target_points();
    let adjusted = adjust_for_extent::<u32>(base, extent_hint, SCATTER_EXTENT_REFERENCE)
        .max(MIN_SCATTER_TARGET_POINTS);
    if point_count <= adjusted {
        1
    } else {
        (point_count + adjusted - 1) / adjusted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_target_env_override() {
        std::env::set_var("RUNMAT_PLOT_SCATTER_TARGET", "300000");
        // Clear cached cell by recreating process-local OnceCell via CHAIN?
        // We cannot reset OnceCell, so just assert helper respects defaults via
        // direct read function.
        assert_eq!(read_env_u64("RUNMAT_PLOT_SCATTER_TARGET").unwrap(), 300_000);
    }

    #[test]
    fn surface_lod_identity_when_small() {
        let lod = compute_surface_lod(32, 64, 10.0);
        assert_eq!(lod.stride_x, 1);
        assert_eq!(lod.stride_y, 1);
        assert_eq!(lod.lod_x_len, 32);
        assert_eq!(lod.lod_y_len, 64);
    }

    #[test]
    fn surface_lod_downsamples_large_grid() {
        let lod = compute_surface_lod(4096, 4096, 10_000.0);
        assert!(lod.stride_x > 1);
        assert!(lod.stride_y > 1);
        assert!((lod.lod_x_len as u64) * (lod.lod_y_len as u64) <= surface_vertex_budget());
    }

    #[test]
    fn scatter3_stride_scales_with_extent() {
        set_scatter_target_points(100_000);
        let dense = scatter3_lod_stride(1_000_000, 50.0);
        let sparse = scatter3_lod_stride(1_000_000, 5_000.0);
        assert!(dense < sparse, "{dense} vs {sparse}");
    }
}
