//! Reference geometry as the model view draws it: datum planes, axes, points and coordinate
//! systems, mate connectors and reference curves. Each reference is a set of polylines in part
//! coordinates (metres), sized from the part's bounds so a plane or an axis reads at the
//! model's scale:
//!
//! - **Plane**: a square outline about the point of the plane nearest the model's centre.
//! - **Axis**: a line through the point of the axis nearest the model's centre, a little longer
//!   than the model.
//! - **Point**: a small three-way cross.
//! - **Coordinate system and mate connector**: three axis lines from the origin, Z the longest.
//! - **Curve**: its evaluated points, closed when the curve is.

use super::Bounds;
use crate::model::datum::DatumValue;
use crate::model::{FeatureKind, PartDocument};
use crate::regen::engine::RegenResult;
use crate::regen::records::FeatureRecord;
use serde::{Deserialize, Serialize};

/// One reference feature's drawn lines.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceDisplay {
    /// The feature id (`axis1`, `coil`).
    pub id: String,
    /// `plane`, `axis`, `point`, `frame`, `connector` or `curve`.
    pub kind: String,
    /// Polylines as flat `[x, y, z, …]` lists, metres.
    pub lines: Vec<Vec<f32>>,
}

type V3 = [f64; 3];

fn add(a: V3, b: V3, s: f64) -> V3 {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn unit(a: V3) -> V3 {
    let n = dot(a, a).sqrt();
    if n > 0.0 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn line(points: &[V3]) -> Vec<f32> {
    points.iter().flat_map(|p| p.iter().map(|&v| v as f32)).collect()
}

/// The model's centre and the length references are sized from (its box diagonal; 40 mm for a
/// part with no body yet).
fn scale_of(bounds: Option<&Bounds>) -> (V3, f64) {
    match bounds {
        Some(b) => {
            let lo = b.min.map(f64::from);
            let hi = b.max.map(f64::from);
            let d = sub(hi, lo);
            let size = dot(d, d).sqrt();
            ([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0], if size > 0.0 { size } else { 0.04 })
        }
        None => ([0.0; 3], 0.04),
    }
}

fn triad(origin: V3, x: V3, y: V3, z: V3, size: f64) -> Vec<Vec<f32>> {
    vec![
        line(&[origin, add(origin, x, size * 0.12)]),
        line(&[origin, add(origin, y, size * 0.12)]),
        line(&[origin, add(origin, z, size * 0.18)]),
    ]
}

fn datum_lines(value: &DatumValue, centre: V3, size: f64) -> Vec<Vec<f32>> {
    match value {
        DatumValue::Plane { frame } => {
            let (x, y, z) = (frame.x, frame.y, frame.z);
            let c = add(centre, z, -dot(sub(centre, frame.origin), z));
            let h = size * 0.35;
            let corner = |sx: f64, sy: f64| add(add(c, x, sx * h), y, sy * h);
            vec![line(&[corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0), corner(-1.0, -1.0)])]
        }
        DatumValue::Axis { point, direction } => {
            let d = unit(*direction);
            let c = add(*point, d, dot(sub(centre, *point), d));
            let h = size * 0.6;
            vec![line(&[add(c, d, -h), add(c, d, h)])]
        }
        DatumValue::Point { at } => {
            let h = size * 0.03;
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                .iter()
                .map(|&d| line(&[add(*at, d, -h), add(*at, d, h)]))
                .collect()
        }
        DatumValue::Frame { frame } => triad(frame.origin, frame.x, frame.y, frame.z, size),
    }
}

/// The part's reference features as drawn lines, in feature order.
pub fn reference_display(document: &PartDocument, result: &RegenResult, bounds: Option<&Bounds>) -> Vec<ReferenceDisplay> {
    let (centre, size) = scale_of(bounds);
    let mut out = Vec::new();
    for feature in &document.features {
        let kind = match &feature.kind {
            FeatureKind::Plane(_) => "plane",
            FeatureKind::Axis(_) => "axis",
            FeatureKind::Point(_) => "point",
            FeatureKind::Frame(_) => "frame",
            FeatureKind::MateConnector(_) => "connector",
            FeatureKind::Curve(_) => "curve",
            _ => continue,
        };
        let lines = if kind == "curve" {
            match result.state.records.get(&feature.id) {
                Some(FeatureRecord::Curve(c)) if c.points.len() >= 2 => {
                    let mut pts = c.points.clone();
                    if c.closed {
                        pts.push(c.points[0]);
                    }
                    vec![line(&pts)]
                }
                _ => continue,
            }
        } else {
            match result.state.datums.get(&feature.id) {
                Some(value) => datum_lines(value, centre, size),
                None => continue,
            }
        };
        out.push(ReferenceDisplay { id: feature.id.0.to_string(), kind: kind.into(), lines });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::datum::Frame;

    #[test]
    fn planes_and_axes_sit_by_the_model_and_frames_draw_three_axes() {
        let b = Bounds { min: [0.0, 0.0, 0.0], max: [0.04, 0.02, 0.01] };
        let (centre, size) = scale_of(Some(&b));
        // A plane at z = 0.05: its square is centred over the model's centre.
        let frame = Frame { origin: [5.0, 5.0, 0.05], x: [1.0, 0.0, 0.0], y: [0.0, 1.0, 0.0], z: [0.0, 0.0, 1.0] };
        let plane = datum_lines(&DatumValue::Plane { frame }, centre, size);
        let p = &plane[0];
        assert_eq!(p.len(), 15);
        let mid_x = (p[0] + p[6]) / 2.0;
        assert!((mid_x - 0.02).abs() < 1e-6 && (p[2] - 0.05).abs() < 1e-6, "{p:?}");
        // An axis along X through (0, 1, 1): it passes the model's centre at x = 0.02.
        let axis = datum_lines(&DatumValue::Axis { point: [0.0, 1.0, 1.0], direction: [2.0, 0.0, 0.0] }, centre, size);
        assert!(((axis[0][0] + axis[0][3]) / 2.0 - 0.02).abs() < 1e-6);
        assert_eq!(datum_lines(&DatumValue::Frame { frame }, centre, size).len(), 3);
        assert_eq!(datum_lines(&DatumValue::Point { at: [0.0; 3] }, centre, size).len(), 3);
    }
}
