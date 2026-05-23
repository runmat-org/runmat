use crate::core::{BoundingBox, Camera};
use glam::Vec3;

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
/// For `plot3` thick-line extrusion we convert pixel width to data units in the *screen plane*.
/// Using Z-range in this conversion underestimates line width whenever Z span is much smaller
/// than X/Y span, which makes 3D lines appear thinner than requested. So we use only the X and Y
/// screen axes here.
pub fn data_units_per_px_3d(bounds: &BoundingBox, viewport_px: (u32, u32)) -> f32 {
    let (w_px, h_px) = viewport_px;
    let w_px = w_px.max(1) as f32;
    let h_px = h_px.max(1) as f32;

    let x_range = (bounds.max.x - bounds.min.x).abs().max(1e-6);
    let y_range = (bounds.max.y - bounds.min.y).abs().max(1e-6);

    (x_range / w_px).min(y_range / h_px)
}

/// Camera-aware 3D conversion from pixels to data units.
///
/// This estimates world units per pixel at the scene center using the same
/// camera fitting/view-angle model as 3D plot rendering, which keeps `LineWidth`
/// in `plot3` visually closer to requested pixel thickness.
pub fn data_units_per_px_3d_camera(
    bounds: &BoundingBox,
    viewport_px: (u32, u32),
    view_angles_deg: Option<(f32, f32)>,
) -> f32 {
    let (w_px, h_px) = viewport_px;
    let w = w_px.max(1) as f32;
    let h = h_px.max(1) as f32;

    let center = (bounds.min + bounds.max) * 0.5;
    let mut camera = Camera::new();
    camera.update_aspect_ratio((w / h).max(1e-6));
    camera.fit_bounds(bounds.min, bounds.max);
    if let Some((az, el)) = view_angles_deg {
        camera.set_view_angles_deg(az, el);
    }

    let forward = (camera.target - camera.position).normalize_or_zero();
    if forward.length_squared() <= 1e-8 {
        return data_units_per_px_3d(bounds, viewport_px);
    }
    let right = forward.cross(camera.up).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    if right.length_squared() <= 1e-8 || up.length_squared() <= 1e-8 {
        return data_units_per_px_3d(bounds, viewport_px);
    }

    let unit = ((bounds.max - bounds.min).length() * 1e-3).max(1e-3);
    let p0 = project_to_screen(&mut camera, center, w, h);
    let px = project_to_screen(&mut camera, center + right * unit, w, h);
    let py = project_to_screen(&mut camera, center + up * unit, w, h);
    let (Some(p0), Some(px), Some(py)) = (p0, px, py) else {
        return data_units_per_px_3d(bounds, viewport_px);
    };

    let px_per_unit_x = (px - p0).length() / unit;
    let px_per_unit_y = (py - p0).length() / unit;
    let px_per_unit = px_per_unit_x.min(px_per_unit_y).max(1e-6);
    1.0 / px_per_unit
}

fn project_to_screen(camera: &mut Camera, pos: Vec3, w: f32, h: f32) -> Option<glam::Vec2> {
    let vp = camera.view_proj_matrix();
    let clip = vp * pos.extend(1.0);
    if clip.w.abs() <= 1e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !ndc.is_finite() {
        return None;
    }
    Some(glam::Vec2::new(
        (ndc.x * 0.5 + 0.5) * w,
        (1.0 - (ndc.y * 0.5 + 0.5)) * h,
    ))
}

#[cfg(test)]
mod tests {
    use super::{data_units_per_px, data_units_per_px_3d, data_units_per_px_3d_camera};
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
    fn data_units_per_px_3d_uses_screen_axes_scale() {
        let bounds = BoundingBox {
            min: Vec3::new(0.0, 0.0, 0.0),
            max: Vec3::new(100.0, 50.0, 10.0),
        };
        let scale = data_units_per_px_3d(&bounds, (1000, 500));
        // x: 0.1, y: 0.1 -> choose min => 0.1
        assert!((scale - 0.1).abs() < 1e-6);
    }

    #[test]
    fn data_units_per_px_3d_camera_returns_positive_finite_scale() {
        let bounds = BoundingBox {
            min: Vec3::new(-1.0, -1.0, -1.0),
            max: Vec3::new(1.0, 1.0, 1.0),
        };
        let scale = data_units_per_px_3d_camera(&bounds, (1280, 720), Some((30.0, 20.0)));
        assert!(scale.is_finite());
        assert!(scale > 0.0);
    }
}
