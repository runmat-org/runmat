//! Size and place of a geometry scene's scalar legend (its color-bar panel). The overlay that
//! draws the legend and the poster renderers that keep the result clear of it share this one
//! layout.

/// Size of the legend's label and range values, in points, before the overlay's font scale.
pub const LABEL_POINTS: f32 = 11.0;

/// The font scale the image exporter draws its overlay at.
pub const EXPORT_FONT_SCALE: f32 = 1.25;

/// The legend panel in the top-right corner of a plot, in points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarLegendPanel {
    /// Gap between the plot's top and right edges and the panel.
    pub pad: f32,
    pub width: f32,
    pub height: f32,
    /// The panel's background reaches this far past the panel on every side.
    pub halo: f32,
    /// Top of the color bar below the top of the panel.
    pub bar_offset: f32,
    pub bar_height: f32,
    /// Size of the label and of the range values.
    pub label_points: f32,
    /// Height of the note line under the range, when the legend has a note.
    pub note_height: f32,
}

impl ScalarLegendPanel {
    /// The panel for a plot `plot_width` points wide drawn at `font_scale`.
    pub fn for_plot(plot_width: f32, font_scale: f32, has_note: bool) -> Self {
        let scale = font_scale.max(0.75);
        let note_height = if has_note { 16.0 * scale } else { 0.0 };
        Self {
            pad: 12.0 * scale,
            width: (plot_width * 0.34).clamp(190.0, 300.0),
            height: 56.0 * scale + note_height,
            halo: 6.0 * scale,
            bar_offset: 23.0 * scale,
            bar_height: 7.0 * scale,
            label_points: LABEL_POINTS * scale,
            note_height,
        }
    }
}

/// What the legend covers of an exported image whose plot fills the image (a geometry scene
/// without a title).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegendFootprint {
    /// Height of the label text, in pixels.
    pub label_px: f32,
    /// Share of the image width covered from its right edge, background included.
    pub width_share: f32,
    /// Share of the image height covered from its top edge, background included.
    pub height_share: f32,
}

/// The legend's footprint in an exported `width_px` × `height_px` image drawn at
/// `pixels_per_point`.
pub fn exported_footprint(
    width_px: u32,
    height_px: u32,
    pixels_per_point: f32,
    has_note: bool,
) -> LegendFootprint {
    // The exporter lays its overlay out in points of this size.
    let ppp = pixels_per_point.max(0.5);
    let (width, height) = (width_px.max(1) as f32, height_px.max(1) as f32);
    let panel = ScalarLegendPanel::for_plot(width / ppp, EXPORT_FONT_SCALE, has_note);
    LegendFootprint {
        label_px: panel.label_points * ppp,
        width_share: ((panel.pad + panel.width + panel.halo) * ppp / width).min(1.0),
        height_share: ((panel.pad + panel.height + panel.halo) * ppp / height).min(1.0),
    }
}

/// The pixels per point at which an exported legend's labels are `label_px` pixels tall.
pub fn export_pixels_per_point_for_label(label_px: f32) -> f32 {
    label_px / (LABEL_POINTS * EXPORT_FONT_SCALE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_footprint_follows_the_pixels_per_point() {
        let ppp = export_pixels_per_point_for_label(18.75);
        let footprint = exported_footprint(1200, 750, ppp, false);
        assert!((footprint.label_px - 18.75).abs() < 1e-4);
        // A 1200 px poster at that density is 880 points wide: the panel is at its widest.
        let panel = ScalarLegendPanel::for_plot(1200.0 / ppp, EXPORT_FONT_SCALE, false);
        assert_eq!(panel.width, 0.34 * 1200.0 / ppp);
        assert!(footprint.width_share > 0.3 && footprint.width_share < 0.45);
        assert!(footprint.height_share > 0.1 && footprint.height_share < 0.25);
        let noted = exported_footprint(1200, 750, ppp, true);
        assert!(noted.height_share > footprint.height_share);
        assert_eq!(noted.width_share, footprint.width_share);
    }
}
