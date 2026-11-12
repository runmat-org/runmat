//! Reference geometry as the model view draws it: datum planes, axes, points and coordinate
//! systems, mate connectors and reference curves. Each reference is a set of polylines in part
//! coordinates (metres), sized from a box so a plane or an axis reads at the model's scale. The
//! box is the part's bounds, or, for a part with no body (a skeleton of grids and levels), the
//! extent of its reference geometry ([`reference_extent`]):
//!
//! - **Plane**: the box projected onto the plane, a little larger, as a rectangle outline.
//! - **Axis**: a line through the box, a little longer than it.
//! - **Point**: a small three-way cross.
//! - **Coordinate system and mate connector**: three axis lines from the origin, Z the longest;
//!   on a part with no body, short beside the extent so a skeleton's many frames read apart.
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
    points
        .iter()
        .flat_map(|p| p.iter().map(|&v| v as f32))
        .collect()
}

/// The box references are sized from: its corners, centre and diagonal (a 40 mm cube about the
/// origin when there is nothing to size from).
struct Sizing {
    lo: V3,
    hi: V3,
    centre: V3,
    size: f64,
    /// A frame's X and Y axis length as a share of `size` (Z is half again as long).
    triad: f64,
}

fn sizing(bounds: Option<&Bounds>) -> Sizing {
    let (lo, hi) = match bounds {
        Some(b) => (b.min.map(f64::from), b.max.map(f64::from)),
        None => ([-0.02; 3], [0.02; 3]),
    };
    let d = sub(hi, lo);
    let size = dot(d, d).sqrt();
    Sizing {
        lo,
        hi,
        centre: [
            (lo[0] + hi[0]) / 2.0,
            (lo[1] + hi[1]) / 2.0,
            (lo[2] + hi[2]) / 2.0,
        ],
        size: if size > 0.0 { size } else { 0.04 },
        triad: 0.12,
    }
}

impl Sizing {
    fn corners(&self) -> impl Iterator<Item = V3> + '_ {
        (0..8).map(move |k| {
            [
                if k & 1 == 0 { self.lo[0] } else { self.hi[0] },
                if k & 2 == 0 { self.lo[1] } else { self.hi[1] },
                if k & 4 == 0 { self.lo[2] } else { self.hi[2] },
            ]
        })
    }
    /// The box's range along `d` from `from`, padded by a twentieth of the box's size (at
    /// least a tenth of it in all when the box is flat along `d`).
    fn range(&self, from: V3, d: V3) -> (f64, f64) {
        let (mut a, mut b) = (f64::MAX, f64::MIN);
        for c in self.corners() {
            let t = dot(sub(c, from), d);
            a = a.min(t);
            b = b.max(t);
        }
        let pad = self.size * 0.05;
        (a - pad, b + pad)
    }
}

/// The extent of a part's reference geometry: plane and frame origins, axis and datum points,
/// and reference curves (the box a part with no body is drawn and framed by).
pub fn reference_extent(document: &PartDocument, result: &RegenResult) -> Option<Bounds> {
    let mut points: Vec<V3> = Vec::new();
    for feature in &document.features {
        match result.state.datums.get(&feature.id) {
            Some(DatumValue::Plane { frame }) | Some(DatumValue::Frame { frame }) => {
                points.push(frame.origin)
            }
            Some(DatumValue::Axis { point, .. }) => points.push(*point),
            Some(DatumValue::Point { at }) => points.push(*at),
            None => {}
        }
        if let Some(FeatureRecord::Curve(c)) = result.state.records.get(&feature.id) {
            points.extend(c.points.iter().copied());
        }
    }
    let first = *points.first()?;
    let (lo, hi) = points.iter().fold((first, first), |(lo, hi), p| {
        (
            [lo[0].min(p[0]), lo[1].min(p[1]), lo[2].min(p[2])],
            [hi[0].max(p[0]), hi[1].max(p[1]), hi[2].max(p[2])],
        )
    });
    Some(Bounds {
        min: lo.map(|v| v as f32),
        max: hi.map(|v| v as f32),
    })
}

fn triad(origin: V3, x: V3, y: V3, z: V3, length: f64) -> Vec<Vec<f32>> {
    vec![
        line(&[origin, add(origin, x, length)]),
        line(&[origin, add(origin, y, length)]),
        line(&[origin, add(origin, z, length * 1.5)]),
    ]
}

fn datum_lines(value: &DatumValue, box_: &Sizing) -> Vec<Vec<f32>> {
    let (centre, size) = (box_.centre, box_.size);
    match value {
        DatumValue::Plane { frame } => {
            let (x, y, z) = (frame.x, frame.y, frame.z);
            let c = add(centre, z, -dot(sub(centre, frame.origin), z));
            let (x0, x1) = box_.range(c, x);
            let (y0, y1) = box_.range(c, y);
            let corner = |u: f64, v: f64| add(add(c, x, u), y, v);
            vec![line(&[
                corner(x0, y0),
                corner(x1, y0),
                corner(x1, y1),
                corner(x0, y1),
                corner(x0, y0),
            ])]
        }
        DatumValue::Axis { point, direction } => {
            let d = unit(*direction);
            let c = add(*point, d, dot(sub(centre, *point), d));
            let (a, b) = box_.range(c, d);
            vec![line(&[add(c, d, a), add(c, d, b)])]
        }
        DatumValue::Point { at } => {
            let h = size * 0.03;
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                .iter()
                .map(|&d| line(&[add(*at, d, -h), add(*at, d, h)]))
                .collect()
        }
        DatumValue::Frame { frame } => {
            triad(frame.origin, frame.x, frame.y, frame.z, size * box_.triad)
        }
    }
}

/// The part's reference features as drawn lines, in feature order, sized from `bounds` (the
/// part's bodies), else from the reference geometry's own extent.
pub fn reference_display(
    document: &PartDocument,
    result: &RegenResult,
    bounds: Option<&Bounds>,
) -> Vec<ReferenceDisplay> {
    let extent = match bounds {
        Some(_) => None,
        None => reference_extent(document, result),
    };
    let mut box_ = sizing(bounds.or(extent.as_ref()));
    if extent.is_some() {
        // A skeleton's many frames (grid intersections, room origins) draw small, so they
        // read apart across the grid.
        box_.triad = 0.04;
    }
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
                Some(value) => datum_lines(value, &box_),
                None => continue,
            }
        };
        out.push(ReferenceDisplay {
            id: feature.id.0.to_string(),
            kind: kind.into(),
            lines,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::datum::Frame;

    #[test]
    fn planes_and_axes_sit_by_the_model_and_frames_draw_three_axes() {
        let b = Bounds {
            min: [0.0, 0.0, 0.0],
            max: [0.04, 0.02, 0.01],
        };
        let box_ = sizing(Some(&b));
        // A plane at z = 0.05: its outline covers the model's box seen from above.
        let frame = Frame {
            origin: [5.0, 5.0, 0.05],
            x: [1.0, 0.0, 0.0],
            y: [0.0, 1.0, 0.0],
            z: [0.0, 0.0, 1.0],
        };
        let plane = datum_lines(&DatumValue::Plane { frame }, &box_);
        let p = &plane[0];
        assert_eq!(p.len(), 15);
        let mid_x = (p[0] + p[6]) / 2.0;
        assert!(
            (mid_x - 0.02).abs() < 1e-6 && (p[2] - 0.05).abs() < 1e-6,
            "{p:?}"
        );
        assert!(
            p[0] < 0.0 && p[3] > 0.04 && p[1] < 0.0 && p[7] > 0.02,
            "{p:?}"
        );
        // An axis along X through (0, 1, 1): it spans the model at its centre's height.
        let axis = datum_lines(
            &DatumValue::Axis {
                point: [0.0, 1.0, 1.0],
                direction: [2.0, 0.0, 0.0],
            },
            &box_,
        );
        assert!(((axis[0][0] + axis[0][3]) / 2.0 - 0.02).abs() < 1e-6);
        assert!(axis[0][0] < 0.0 && axis[0][3] > 0.04);
        assert_eq!(datum_lines(&DatumValue::Frame { frame }, &box_).len(), 3);
        assert_eq!(
            datum_lines(&DatumValue::Point { at: [0.0; 3] }, &box_).len(),
            3
        );
    }

    #[test]
    fn a_skeleton_with_no_body_draws_its_references_at_their_own_extent() {
        // Grid planes 10 m apart and a frame at a corner: the planes span the grid.
        let b = Bounds {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 8.0, 6.0],
        };
        let mut box_ = sizing(Some(&b));
        box_.triad = 0.04;
        let grid = Frame {
            origin: [10.0, 0.0, 0.0],
            x: [0.0, 1.0, 0.0],
            y: [0.0, 0.0, 1.0],
            z: [1.0, 0.0, 0.0],
        };
        let p = &datum_lines(&DatumValue::Plane { frame: grid }, &box_)[0];
        let ys: Vec<f32> = p.chunks(3).map(|c| c[1]).collect();
        let zs: Vec<f32> = p.chunks(3).map(|c| c[2]).collect();
        let span = |v: &[f32]| {
            v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min)
        };
        assert!(span(&ys) > 8.0 && span(&zs) > 6.0, "{p:?}");
        assert!(p.chunks(3).all(|c| (c[0] - 10.0).abs() < 1e-5));
        // A frame's triad stays small beside the grid: well under a metre.
        let t = datum_lines(&DatumValue::Frame { frame: grid }, &box_);
        let len = |l: &Vec<f32>| {
            ((l[3] - l[0]).powi(2) + (l[4] - l[1]).powi(2) + (l[5] - l[2]).powi(2)).sqrt()
        };
        assert!(t.iter().all(|l| len(l) < 1.0 && len(l) > 0.1), "{t:?}");
    }
}
