//! Size and place of a geometry scene's scalar legend (its color-bar panel). The overlay that
//! draws the legend and the poster renderers that keep the result clear of it share this one
//! layout.

use crate::geometry_scene::{GeometrySceneScalarLegend, ScalarRangeSource};

impl GeometrySceneScalarLegend {
    /// A legend from `minimum` to `maximum` of the field's extremes, with no limit or note.
    pub fn of_data(label: String, unit: String, minimum: f64, maximum: f64) -> Self {
        Self {
            label,
            unit,
            minimum,
            maximum,
            limit: None,
            note: None,
            range_source: ScalarRangeSource::Data,
            below: None,
            above: None,
            precision: None,
        }
    }

    /// How the legend's numbers are written: the study's significant digits, else the
    /// precision its range needs.
    pub fn number_precision(&self) -> crate::legend_text::LegendPrecision {
        use crate::legend_text::LegendPrecision;
        match self.precision {
            Some(digits) => LegendPrecision::significant(self.minimum, self.maximum, digits),
            None => LegendPrecision::for_range(self.minimum, self.maximum),
        }
    }

    /// The overflow lines of a fixed range the field leaves (the highest first), such as
    /// "▲ 14.2 MPa above the range".
    pub fn overflow_lines(&self) -> Vec<String> {
        let precision = self.number_precision();
        [(self.above, true), (self.below, false)]
            .into_iter()
            .filter_map(|(value, above)| {
                value.map(|value| {
                    crate::legend_text::overflow_line(value, &self.unit, above, precision)
                })
            })
            .collect()
    }

    /// Text lines under the range: one per overflow and one for the note.
    pub fn note_lines(&self) -> usize {
        usize::from(self.above.is_some())
            + usize::from(self.below.is_some())
            + usize::from(self.note.is_some())
    }
}

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
    /// Height of the text lines under the range: the overflow lines of a fixed range and the
    /// note, each one line.
    pub note_height: f32,
}

impl ScalarLegendPanel {
    /// One text line under the range, in points before the font scale.
    pub const LINE_POINTS: f32 = 16.0;

    /// The panel for a plot `plot_width` points wide drawn at `font_scale`, with `note_lines`
    /// text lines under the range ([`GeometrySceneScalarLegend::note_lines`]).
    ///
    /// [`GeometrySceneScalarLegend::note_lines`]: GeometrySceneScalarLegend::note_lines
    pub fn for_plot(plot_width: f32, font_scale: f32, note_lines: usize) -> Self {
        let scale = font_scale.max(0.75);
        let note_height = Self::LINE_POINTS * scale * note_lines as f32;
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
/// `pixels_per_point`, with `note_lines` text lines under the range.
pub fn exported_footprint(
    width_px: u32,
    height_px: u32,
    pixels_per_point: f32,
    note_lines: usize,
) -> LegendFootprint {
    // The exporter lays its overlay out in points of this size.
    let ppp = pixels_per_point.max(0.5);
    let (width, height) = (width_px.max(1) as f32, height_px.max(1) as f32);
    let panel = ScalarLegendPanel::for_plot(width / ppp, EXPORT_FONT_SCALE, note_lines);
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
        let footprint = exported_footprint(1200, 750, ppp, 0);
        assert!((footprint.label_px - 18.75).abs() < 1e-4);
        // A 1200 px poster at that density is 880 points wide: the panel is at its widest.
        let panel = ScalarLegendPanel::for_plot(1200.0 / ppp, EXPORT_FONT_SCALE, 0);
        assert_eq!(panel.width, 0.34 * 1200.0 / ppp);
        assert!(footprint.width_share > 0.3 && footprint.width_share < 0.45);
        assert!(footprint.height_share > 0.1 && footprint.height_share < 0.25);
        let noted = exported_footprint(1200, 750, ppp, 1);
        assert!(noted.height_share > footprint.height_share);
        assert_eq!(noted.width_share, footprint.width_share);
        // A fixed range that the result leaves on both sides adds two lines above the note.
        let overflowing = exported_footprint(1200, 750, ppp, 3);
        assert!(overflowing.height_share > noted.height_share);
    }
}
