//! A sheet's section and development: flats and bends of a chain, developed lengths under a
//! bend rule, and the flat layout drawings and exports read.

use super::{BendTable, MaterialSide};
use crate::ids::{FeatureId, SketchEntityId};
use runmat_geometry_kernel::op::{ProfileSegment, SegmentKind};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ------------------------------------------------------------------------- the section

/// A flat of a section: segment `curve` of the chain, from `start` to `end` along it (the
/// stretch between the neighbouring bends' tangent lines), in sketch coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SectionFlat {
    pub curve: SketchEntityId,
    /// The flat's reference-side line (on the chain) and its direction.
    pub start: [f64; 2],
    pub end: [f64; 2],
    /// Unit normal towards the material's far side.
    pub normal: [f64; 2],
    /// Offsets of the two faces from the chain, along `normal` (`near ≤ far`).
    pub near: f64,
    pub far: f64,
}

impl SectionFlat {
    pub fn length(&self) -> f64 {
        (self.end[0] - self.start[0]).hypot(self.end[1] - self.start[1])
    }
}

/// A bend of a section between two consecutive flats.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SectionBend {
    /// `"<curve before>+<curve after>"`: the bend's key from the definition.
    pub key: String,
    pub center: [f64; 2],
    pub inner_radius: f64,
    pub thickness: f64,
    /// The change of direction, signed (positive turns left), radians.
    pub angle: f64,
    /// Direction angles (from the centre) where the bend starts and ends, counter-clockwise
    /// order.
    pub from: f64,
    pub to: f64,
}

impl SectionBend {
    pub fn bend_key(a: &SketchEntityId, b: &SketchEntityId) -> String {
        format!("{}+{}", a, b)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Section {
    pub flats: Vec<SectionFlat>,
    pub bends: Vec<SectionBend>,
    pub thickness: f64,
}

/// The section of a sheet along an open polyline chain: `points[i]..points[i+1]` is curve
/// `curves[i]`. Material on `side` of the chain, thickness `t`, inner bend radius `r`.
pub fn section(
    points: &[[f64; 2]],
    curves: &[SketchEntityId],
    side: MaterialSide,
    t: f64,
    r: f64,
) -> Result<Section, String> {
    if points.len() < 2 || curves.len() + 1 != points.len() {
        return Err("a sheet section needs a chain of at least one line".into());
    }
    if t <= 0.0 || r < 0.0 {
        return Err("sheet thickness must be positive and the bend radius non-negative".into());
    }
    let (o0, o1) = match side {
        MaterialSide::Left => (0.0, t),
        MaterialSide::Right => (-t, 0.0),
        MaterialSide::Mid => (-t / 2.0, t / 2.0),
    };
    let dir = |i: usize| {
        let (a, b) = (points[i], points[i + 1]);
        let l = (b[0] - a[0]).hypot(b[1] - a[1]);
        [(b[0] - a[0]) / l, (b[1] - a[1]) / l]
    };
    let left = |d: [f64; 2]| [-d[1], d[0]];
    let n = curves.len();
    // Parameters along each segment where its flat starts and ends.
    let mut s0 = vec![0.0; n];
    let mut s1: Vec<f64> = (0..n)
        .map(|i| (points[i + 1][0] - points[i][0]).hypot(points[i + 1][1] - points[i][1]))
        .collect();
    let mut bends = Vec::new();
    for k in 1..n {
        let (da, db) = (dir(k - 1), dir(k));
        let cross = da[0] * db[1] - da[1] * db[0];
        let dotp = da[0] * db[0] + da[1] * db[1];
        let phi = cross.atan2(dotp);
        if phi.abs() < 1e-12 {
            continue; // collinear segments: no bend
        }
        if phi.abs() >= PI - 1e-9 {
            return Err("a sheet cannot fold back on itself in one bend; use a hem".into());
        }
        // The centre lies on the inner side, at offset c from both segment lines.
        let c = if phi > 0.0 { o1 + r } else { o0 - r };
        // Corner of the two lines offset by c.
        let v = points[k];
        let (na, nb) = (left(da), left(db));
        // Solve v + c·na + s·da = v + c·nb + u·db for the intersection.
        let pa = [v[0] + c * na[0], v[1] + c * na[1]];
        let pb = [v[0] + c * nb[0], v[1] + c * nb[1]];
        let den = da[0] * db[1] - da[1] * db[0];
        let w = [pb[0] - pa[0], pb[1] - pa[1]];
        let s = (w[0] * db[1] - w[1] * db[0]) / den;
        let center = [pa[0] + s * da[0], pa[1] + s * da[1]];
        // Tangent parameters: the centre's foot on each segment line.
        let foot = |i: usize| {
            let d = dir(i);
            (center[0] - points[i][0]) * d[0] + (center[1] - points[i][1]) * d[1]
        };
        s1[k - 1] = foot(k - 1);
        s0[k] = foot(k);
        let inner = c.abs() - (if phi > 0.0 { o1 } else { -o0 }).abs();
        let inner = if phi > 0.0 {
            c - o1
        } else {
            let _ = inner;
            o0 - c
        };
        // From the centre, the chain point of segment k-1's end lies at angle of -c·na.
        let ang = |n: [f64; 2], c: f64| (-c * n[1]).atan2(-c * n[0]);
        let (a0, a1) = (ang(na, c), ang(nb, c));
        let (from, to) = if phi > 0.0 { (a0, a1) } else { (a1, a0) };
        let to = if to < from { to + 2.0 * PI } else { to };
        bends.push(SectionBend {
            key: SectionBend::bend_key(&curves[k - 1], &curves[k]),
            center,
            inner_radius: inner,
            thickness: t,
            angle: phi,
            from,
            to,
        });
    }
    let mut flats = Vec::new();
    for i in 0..n {
        if s1[i] < s0[i] - 1e-12 {
            return Err(format!(
                "segment `{}` is shorter than its bends' setbacks",
                curves[i]
            ));
        }
        let d = dir(i);
        let p = points[i];
        flats.push(SectionFlat {
            curve: curves[i].clone(),
            start: [p[0] + s0[i] * d[0], p[1] + s0[i] * d[1]],
            end: [p[0] + s1[i] * d[0], p[1] + s1[i] * d[1]],
            normal: left(d),
            near: o0,
            far: o1,
        });
    }
    Ok(Section {
        flats,
        bends,
        thickness: t,
    })
}

// ------------------------------------------------------------------- developed lengths

/// The bend rule evaluated (SI).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "rule")]
pub enum BendRuleValue {
    KFactor {
        k: f64,
    },
    Table {
        table: BendTable,
    },
    /// Deduction for a 90° bend.
    Deduction {
        value: f64,
    },
    /// Allowance for a 90° bend.
    Allowance {
        value: f64,
    },
}

/// Bend allowance for one bend (`angle` unsigned, radians).
pub fn bend_allowance(rule: &BendRuleValue, t: f64, r: f64, angle: f64) -> Result<f64, String> {
    let a = angle.abs();
    Ok(match rule {
        BendRuleValue::KFactor { k } => {
            if !(0.0..=1.0).contains(k) {
                return Err(format!("K-factor {} is outside [0, 1]", k));
            }
            a * (r + k * t)
        }
        BendRuleValue::Table { table } => table.allowance(t, r, a).ok_or_else(|| {
            format!(
                "bend table `{}` has no row for thickness {} m and radius {} m",
                table.name, t, r
            )
        })?,
        BendRuleValue::Allowance { value } => value * a / (PI / 2.0),
        BendRuleValue::Deduction { value } => {
            2.0 * outside_setback(t, r, a) - value * a / (PI / 2.0)
        }
    })
}

/// `tan(θ/2)(R + t)`.
pub fn outside_setback(t: f64, r: f64, angle: f64) -> f64 {
    (angle.abs() / 2.0).tan() * (r + t)
}

/// `2·OSSB − BA`.
pub fn bend_deduction(rule: &BendRuleValue, t: f64, r: f64, angle: f64) -> Result<f64, String> {
    Ok(2.0 * outside_setback(t, r, angle) - bend_allowance(rule, t, r, angle)?)
}

/// Whether a bend turns up or down when the flat is viewed from the reference side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BendDirection {
    Up,
    Down,
}

/// A bend line of the flat pattern: the centre of the bend zone, at `position` along the
/// developed length from the first flat's start.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BendLine {
    pub key: String,
    pub position: f64,
    /// The bend zone: `[position − allowance/2, position + allowance/2]`.
    pub allowance: f64,
    pub deduction: f64,
    pub angle: f64,
    pub inner_radius: f64,
    pub direction: BendDirection,
}

/// One region of the developed strip, in order: a flat or a bend zone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "region")]
pub enum StripRegion {
    Flat {
        curve: SketchEntityId,
        start: f64,
        length: f64,
    },
    Bend {
        key: String,
        start: f64,
        length: f64,
    },
}

/// The developed section: total length, regions in order, bend lines and the bend table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Development {
    pub length: f64,
    pub regions: Vec<StripRegion>,
    pub bend_lines: Vec<BendLine>,
}

/// Develop a section under a bend rule.
pub fn develop(section: &Section, rule: &BendRuleValue) -> Result<Development, String> {
    let mut x = 0.0;
    let mut regions = Vec::new();
    let mut bend_lines = Vec::new();
    for (i, f) in section.flats.iter().enumerate() {
        let len = f.length();
        regions.push(StripRegion::Flat {
            curve: f.curve.clone(),
            start: x,
            length: len,
        });
        x += len;
        if let Some(b) = section.bends.get(i) {
            let ba = bend_allowance(rule, section.thickness, b.inner_radius, b.angle)?;
            let bd = bend_deduction(rule, section.thickness, b.inner_radius, b.angle)?;
            regions.push(StripRegion::Bend {
                key: b.key.clone(),
                start: x,
                length: ba,
            });
            bend_lines.push(BendLine {
                key: b.key.clone(),
                position: x + ba / 2.0,
                allowance: ba,
                deduction: bd,
                angle: b.angle.abs(),
                inner_radius: b.inner_radius,
                direction: if b.angle > 0.0 {
                    BendDirection::Up
                } else {
                    BendDirection::Down
                },
            });
            x += ba;
        }
    }
    if section.bends.len() + 1 != section.flats.len() {
        return Err("a section needs one bend between each pair of flats".into());
    }
    Ok(Development {
        length: x,
        regions,
        bend_lines,
    })
}

// ------------------------------------------------------------------------ flat layouts

/// A bend line of a flat layout: the centre line of the bend zone, from edge to edge.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlatBendLine {
    /// The bend's key: the sketch line (`<line>.2` for a jog's second bend), or `<curve
    /// before>+<curve after>` for a bend of a section.
    pub key: String,
    /// The feature that made the bend.
    pub feature: FeatureId,
    pub start: [f64; 2],
    pub end: [f64; 2],
    /// The bend's angle (unsigned, radians) and inner radius.
    pub angle: f64,
    pub inner_radius: f64,
    /// The bend zone's width (half each side of the line) and the bend deduction.
    pub allowance: f64,
    pub deduction: f64,
    /// Up: the bend turns towards the layout frame's +z side.
    pub direction: BendDirection,
}

/// A flat pattern laid out for drawings and cutting: the outline loops (the first is the
/// outer boundary) and the bend lines in `frame`'s xy plane, in metres. The material spans
/// `0..depth` along the frame's z (`depth` is plus or minus the thickness).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlatLayout {
    pub frame: crate::model::datum::Frame,
    pub depth: f64,
    pub loops: Vec<Vec<ProfileSegment>>,
    pub bend_lines: Vec<FlatBendLine>,
}

impl FlatLayout {
    /// The outline's extents in the layout frame, `([x_min, y_min], [x_max, y_max])`. Lines,
    /// arcs and circles are exact; an ellipse counts its whole ellipse and a spline its
    /// control polygon, so their extents can be larger than the curve's.
    pub fn extents(&self) -> ([f64; 2], [f64; 2]) {
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        let mut take = |p: [f64; 2]| {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        };
        for seg in self.loops.iter().flatten() {
            take(seg.start);
            take(seg.end);
            match &seg.kind {
                SegmentKind::Line => {}
                SegmentKind::Arc { center, clockwise } => {
                    let (s, e) = (seg.start, seg.end);
                    let r = (s[0] - center[0]).hypot(s[1] - center[1]);
                    let a0 = (s[1] - center[1]).atan2(s[0] - center[0]);
                    let a1 = (e[1] - center[1]).atan2(e[0] - center[0]);
                    // The counter-clockwise span from `from` to `to`.
                    let (from, to) = if *clockwise { (a1, a0) } else { (a0, a1) };
                    let span = (to - from).rem_euclid(2.0 * PI);
                    let span = if span < 1e-15 { 2.0 * PI } else { span };
                    for q in 0..4 {
                        let a = q as f64 * PI / 2.0;
                        if (a - from).rem_euclid(2.0 * PI) <= span {
                            take([center[0] + r * a.cos(), center[1] + r * a.sin()]);
                        }
                    }
                }
                SegmentKind::Circle { center, radius } => {
                    take([center[0] - radius, center[1] - radius]);
                    take([center[0] + radius, center[1] + radius]);
                }
                SegmentKind::Ellipse { center, major, .. } => {
                    let a = major[0].hypot(major[1]);
                    take([center[0] - a, center[1] - a]);
                    take([center[0] + a, center[1] + a]);
                }
                SegmentKind::Spline { poles, .. } => poles.iter().for_each(|p| take(*p)),
            }
        }
        (lo, hi)
    }
}
