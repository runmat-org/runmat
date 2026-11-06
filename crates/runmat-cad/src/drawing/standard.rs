//! Drafting standards (CAD-37): the settings a sheet format's `standard` selects, which the
//! sheet, the exports and the checks all read from here.
//!
//! - `iso`: ISO 128-2 lines (line group 0.5: 0.5 mm visible, 0.25 mm thin), ISO 129-1
//!   dimensions (3.5 mm figures aligned with the dimension line, above it), ISO 5456-2 first-angle
//!   projection by default, and the ISO 1101 geometric characteristics.
//! - `asme`: ASME Y14.2 lines (0.6 mm visible, 0.3 mm thin), ASME Y14.5-2018 dimensions (3 mm
//!   figures read from the bottom of the sheet, in a break in the dimension line, extension
//!   lines with a 1.5 mm gap and 3 mm past the dimension line), ASME Y14.3 third-angle
//!   projection by default, and the Y14.5-2018 characteristics: concentricity and symmetry
//!   were withdrawn in 2018, so a frame showing either is an error on an ASME sheet.
//!
//! Both put no trailing zeros on metric values and a leading zero before the decimal point.

use crate::drawing::Projection;
use crate::pmi::Characteristic;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawingStandard {
    #[default]
    Iso,
    Asme,
}

/// Which way dimension figures read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextOrientation {
    /// Along the dimension line, above it, readable from the bottom or the right (ISO 129-1).
    Aligned,
    /// Upright on the sheet, in a break in the dimension line (ASME Y14.5 unidirectional).
    Unidirectional,
}

/// Line widths in millimetres.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LineWeights {
    pub visible: f64,
    pub hidden: f64,
    pub center: f64,
    pub dimension: f64,
    /// Cutting-plane ends and the sheet frame.
    pub thick: f64,
}

/// What a standard sets for one sheet, in sheet millimetres.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DraftingStyle {
    pub standard: DrawingStandard,
    /// The standards it follows, as the title block's note names them.
    pub name: &'static str,
    /// The projection a new sheet takes.
    pub projection: Projection,
    pub text_height: f64,
    pub arrow_length: f64,
    pub arrow_width: f64,
    /// The gap between an extension line and the outline, and how far it runs past the
    /// dimension line.
    pub extension_gap: f64,
    pub extension_overshoot: f64,
    /// The first dimension line's distance from the outline, and each further one's.
    pub first_offset: f64,
    pub offset_step: f64,
    pub text: TextOrientation,
    pub lines: LineWeights,
    /// The geometric characteristics a feature control frame may show.
    pub characteristics: Vec<Characteristic>,
}

const ALL: [Characteristic; 14] = [
    Characteristic::Straightness,
    Characteristic::Flatness,
    Characteristic::Circularity,
    Characteristic::Cylindricity,
    Characteristic::ProfileOfLine,
    Characteristic::ProfileOfSurface,
    Characteristic::Parallelism,
    Characteristic::Perpendicularity,
    Characteristic::Angularity,
    Characteristic::Position,
    Characteristic::Concentricity,
    Characteristic::Symmetry,
    Characteristic::CircularRunout,
    Characteristic::TotalRunout,
];

impl DrawingStandard {
    pub fn is_iso(&self) -> bool {
        *self == DrawingStandard::Iso
    }

    /// Whether a feature control frame may show `c` under this standard.
    pub fn allows(self, c: Characteristic) -> bool {
        !(self == DrawingStandard::Asme && matches!(c, Characteristic::Concentricity | Characteristic::Symmetry))
    }

    pub fn style(self) -> DraftingStyle {
        let characteristics = ALL.iter().copied().filter(|c| self.allows(*c)).collect();
        match self {
            DrawingStandard::Iso => DraftingStyle {
                standard: self,
                name: "ISO 128, ISO 129-1, ISO 1101",
                projection: Projection::FirstAngle,
                text_height: 3.5,
                arrow_length: 3.0,
                arrow_width: 1.0,
                extension_gap: 1.0,
                extension_overshoot: 2.0,
                first_offset: 10.0,
                offset_step: 8.0,
                text: TextOrientation::Aligned,
                lines: LineWeights { visible: 0.5, hidden: 0.25, center: 0.25, dimension: 0.25, thick: 0.7 },
                characteristics,
            },
            DrawingStandard::Asme => DraftingStyle {
                standard: self,
                name: "ASME Y14.2, ASME Y14.3, ASME Y14.5-2018",
                projection: Projection::ThirdAngle,
                text_height: 3.0,
                arrow_length: 3.0,
                arrow_width: 1.0,
                extension_gap: 1.5,
                extension_overshoot: 3.0,
                first_offset: 10.0,
                offset_step: 8.0,
                text: TextOrientation::Unidirectional,
                lines: LineWeights { visible: 0.6, hidden: 0.3, center: 0.3, dimension: 0.3, thick: 0.6 },
                characteristics,
            },
        }
    }

    /// Why a frame can't show `c` under this standard, with what to use instead.
    pub fn refusal(self, c: Characteristic) -> Option<String> {
        if self.allows(c) {
            return None;
        }
        let instead = match c {
            Characteristic::Symmetry => "position or profile of a surface",
            _ => "position, runout or profile of a surface",
        };
        Some(format!("ASME Y14.5-2018 withdrew the {} tolerance; use {instead}", characteristic_name(c)))
    }
}

/// A characteristic's name in a message.
pub fn characteristic_name(c: Characteristic) -> &'static str {
    match c {
        Characteristic::Straightness => "straightness",
        Characteristic::Flatness => "flatness",
        Characteristic::Circularity => "circularity",
        Characteristic::Cylindricity => "cylindricity",
        Characteristic::ProfileOfLine => "profile of a line",
        Characteristic::ProfileOfSurface => "profile of a surface",
        Characteristic::Parallelism => "parallelism",
        Characteristic::Perpendicularity => "perpendicularity",
        Characteristic::Angularity => "angularity",
        Characteristic::Position => "position",
        Characteristic::Concentricity => "concentricity",
        Characteristic::Symmetry => "symmetry",
        Characteristic::CircularRunout => "circular runout",
        Characteristic::TotalRunout => "total runout",
    }
}

/// A metric value in millimetres as both standards write it: no trailing zeros, a leading
/// zero below one (`0.5`, `12`, `12.25`), at most `places` decimals.
pub fn metric_text(value: f64, places: usize) -> String {
    let mut s = format!("{value:.places$}");
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    if s == "-0" {
        s = "0".into();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_standard_sets_its_lines_figures_projection_and_frames() {
        let iso = DrawingStandard::Iso.style();
        let asme = DrawingStandard::Asme.style();
        assert_eq!((iso.text_height, iso.lines.visible, iso.projection), (3.5, 0.5, Projection::FirstAngle));
        assert_eq!((asme.text_height, asme.lines.visible, asme.projection), (3.0, 0.6, Projection::ThirdAngle));
        assert_eq!((iso.text, asme.text), (TextOrientation::Aligned, TextOrientation::Unidirectional));
        assert_eq!(iso.characteristics.len(), 14);
        assert_eq!(asme.characteristics.len(), 12);
        assert!(DrawingStandard::Asme.refusal(Characteristic::Symmetry).unwrap().contains("withdrew the symmetry"));
        assert_eq!(DrawingStandard::Iso.refusal(Characteristic::Symmetry), None);
        assert_eq!(serde_json::to_value(DrawingStandard::Asme).unwrap(), "asme");
    }

    #[test]
    fn metric_values_drop_trailing_zeros_and_keep_the_leading_one() {
        assert_eq!(metric_text(0.5, 2), "0.5");
        assert_eq!(metric_text(12.0, 2), "12");
        assert_eq!(metric_text(12.25, 2), "12.25");
        assert_eq!(metric_text(-0.001, 2), "0");
    }
}
