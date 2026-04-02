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
