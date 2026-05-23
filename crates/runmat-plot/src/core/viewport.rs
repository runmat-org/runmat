use crate::core::BoundingBox;

/// Compute the scale factor to convert pixel-sized geometry into data units.
///
/// Many plot style controls (e.g. `LineWidth`) are expressed in *pixels*, but some geometry
/// generators (like thick polyline extrusion) operate in *data space*. This helper provides a
/// conservative conversion factor based on the current data bounds and viewport size:
///
/// \[
/// \text{data\_units\_per\_px} = \min\left(\frac{\Delta x}{w_{px}}, \frac{\Delta y}{h_{px}}\right)
/// \]
///
/// Using the minimum axis scale avoids pathological over-thick extrusion when one data axis spans
/// orders of magnitude more than the other (for example semilog/loglog plots). In those cases a
/// `max(...)` conversion can turn a modest pixel width into a near-filled ribbon.
pub fn data_units_per_px(bounds: &BoundingBox, viewport_px: (u32, u32)) -> f32 {
    let (w_px, h_px) = viewport_px;
    let w_px = w_px.max(1) as f32;
    let h_px = h_px.max(1) as f32;

    let x_range = (bounds.max.x - bounds.min.x).abs().max(1e-6);
    let y_range = (bounds.max.y - bounds.min.y).abs().max(1e-6);

    (x_range / w_px).min(y_range / h_px)
}

/// 3D variant of [`data_units_per_px`].
///
/// For `plot3` thick-line extrusion we still need a scalar data-unit conversion from a
/// pixel width. We conservatively use the smallest axis scale across X/Y/Z, mapping Z
/// against the shorter viewport side because depth does not correspond to a dedicated
/// pixel axis.
pub fn data_units_per_px_3d(bounds: &BoundingBox, viewport_px: (u32, u32)) -> f32 {
    let (w_px, h_px) = viewport_px;
    let w_px = w_px.max(1) as f32;
    let h_px = h_px.max(1) as f32;
    let d_px = w_px.min(h_px).max(1.0);

    let x_range = (bounds.max.x - bounds.min.x).abs().max(1e-6);
    let y_range = (bounds.max.y - bounds.min.y).abs().max(1e-6);
    let z_range = (bounds.max.z - bounds.min.z).abs().max(1e-6);

    (x_range / w_px).min(y_range / h_px).min(z_range / d_px)
}

#[cfg(test)]
mod tests {
    use super::{data_units_per_px, data_units_per_px_3d};
    use crate::core::BoundingBox;
    use glam::Vec3;

    #[test]
    fn data_units_per_px_uses_min_2d_scale() {
        let bounds = BoundingBox {
            min: Vec3::new(0.0, 0.0, 0.0),
            max: Vec3::new(100.0, 50.0, 0.0),
        };
        let scale = data_units_per_px(&bounds, (1000, 500));
        assert!((scale - 0.1).abs() < 1e-6);
    }

    #[test]
    fn data_units_per_px_3d_includes_z_scale() {
        let bounds = BoundingBox {
            min: Vec3::new(0.0, 0.0, 0.0),
            max: Vec3::new(100.0, 50.0, 10.0),
        };
        let scale = data_units_per_px_3d(&bounds, (1000, 500));
        // x: 0.1, y: 0.1, z: 0.02 -> choose min => 0.02
        assert!((scale - 0.02).abs() < 1e-6);
    }
}
