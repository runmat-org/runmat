//! Assembly features: a cut, holes or a weld bead made in the assembly, through the components
//! they affect. The part files don't change; each affected component draws its part with the
//! assembly's features applied, in the assembly's placement.
//!
//! ```text
//! operation slot = cut(profile = sketch.s1, extent = through_all, components = [lower, upper])
//! operation bolt = hole(profile = sketch.s2, diameter = 6.6 mm, extent = through_all,
//!                       counterbore = { diameter = 11 mm, depth = 6.5 mm })
//! operation seam = weld(path = sketch.s3, size = 5 mm)
//! ```
//!
//! A cut extrudes the sketch's closed profile from its plane and removes it from each affected
//! component. A hole drills one hole at each point of the sketch (a point, or a circle's or
//! arc's center) from its plane along its normal. A weld bead sweeps a fillet weld's triangle of leg
//! `size` along the sketch's lines and arcs, on the side of the sketch's normal (`flip`: the
//! other side of the path); it is drawn with the first affected component. `components` empty:
//! every part component.
//!
//! Regeneration order: placement and mates, then the assembly's sketches, then its features in
//! the order written ([`tool`] builds each one's tool in assembly coordinates).

use crate::assembly::sketches::ResolvedSketch;
use crate::model::feature::{Extent, ProfileRef};
use crate::model::sketch::{PointRole, SketchGeometry};
use crate::units::Units;
use crate::value::{Evaluator, NoDynamicValues, ValueEnv, ValueExpr};
use runmat_geometry_kernel::frame::Frame;
use runmat_geometry_kernel::op::{
    CanonicalProfile, HoleDepth, PrismBound, ProfileSegment, SegmentKind, Transform,
};
use serde::{Deserialize, Serialize};

/// One assembly feature (`operation <name> = cut(…)`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssemblyOperation {
    pub name: String,
    pub op: OperationKind,
    /// The components it affects, by name. Empty: every part component.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub suppressed: bool,
}

/// A counterbore: its diameter and depth.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Counterbore {
    pub diameter: ValueExpr,
    pub depth: ValueExpr,
}

/// A countersink: its diameter and included angle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Countersink {
    pub diameter: ValueExpr,
    pub angle: ValueExpr,
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation")]
pub enum OperationKind {
    /// The sketch's closed profile, extruded from its plane and removed.
    Cut {
        profile: ProfileRef,
        extent: Extent,
        /// Against the sketch's normal.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        reverse: bool,
        /// Through all in both directions (with `extent = through_all`).
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        both: bool,
    },
    /// A hole at each point of the sketch, drilled from its plane along its normal (against it
    /// with `reverse`).
    Hole {
        profile: ProfileRef,
        diameter: ValueExpr,
        extent: Extent,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        reverse: bool,
        /// The fastener size the hole is for (`M6`), as written in the panel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        size: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        counterbore: Option<Counterbore>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        countersink: Option<Countersink>,
    },
    /// A fillet weld bead along the sketch's curves.
    Weld {
        path: ProfileRef,
        /// The leg length.
        size: ValueExpr,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        flip: bool,
    },
}

impl OperationKind {
    /// `cut`, `hole` or `weld`: the command's `type`.
    pub fn word(&self) -> &'static str {
        match self {
            OperationKind::Cut { .. } => "cut",
            OperationKind::Hole { .. } => "hole",
            OperationKind::Weld { .. } => "weld",
        }
    }

    /// The sketch it reads.
    pub fn sketch(&self) -> &str {
        match self {
            OperationKind::Cut { profile, .. } | OperationKind::Hole { profile, .. } => {
                &profile.sketch.0
            }
            OperationKind::Weld { path, .. } => &path.sketch.0,
        }
    }
}

/// An assembly feature's tool, in assembly coordinates.
#[derive(Clone, Debug, PartialEq)]
pub enum OperationTool {
    /// A prism removed from each affected component.
    Cut {
        profile: CanonicalProfile,
        start: f64,
        end: f64,
        start_bound: PrismBound,
        end_bound: PrismBound,
    },
    /// Holes drilled along `-frame.z` at `positions` (frame coordinates).
    Holes {
        frame: Frame,
        positions: Vec<(String, [f64; 2])>,
        diameter: f64,
        end: HoleDepth,
        counterbore: Option<(f64, f64)>,
        countersink: Option<(f64, f64)>,
    },
    /// A bead: `profile` swept along `path` (assembly points).
    Bead {
        profile: CanonicalProfile,
        path: Vec<[f64; 3]>,
    },
}

impl OperationTool {
    /// This tool moved by `t` (assembly coordinates into a component's own, with the inverse of
    /// its placement).
    pub fn transformed(&self, t: &Transform) -> OperationTool {
        let frame = |f: &Frame| t.compose(&Transform::from_frame(f)).to_frame();
        match self {
            OperationTool::Cut {
                profile,
                start,
                end,
                start_bound,
                end_bound,
            } => OperationTool::Cut {
                profile: CanonicalProfile {
                    frame: frame(&profile.frame),
                    ..profile.clone()
                },
                start: *start,
                end: *end,
                start_bound: start_bound.clone(),
                end_bound: end_bound.clone(),
            },
            OperationTool::Holes {
                frame: f,
                positions,
                diameter,
                end,
                counterbore,
                countersink,
            } => OperationTool::Holes {
                frame: frame(f),
                positions: positions.clone(),
                diameter: *diameter,
                end: end.clone(),
                counterbore: *counterbore,
                countersink: *countersink,
            },
            OperationTool::Bead { profile, path } => OperationTool::Bead {
                profile: CanonicalProfile {
                    frame: frame(&profile.frame),
                    ..profile.clone()
                },
                path: path.iter().map(|p| t.apply(*p)).collect(),
            },
        }
    }
}

/// `expr` as an SI number (literals and the assembly's constants; an input isn't read here).
fn si(expr: &ValueExpr, what: &str) -> Result<f64, String> {
    Evaluator {
        units: &Units,
        dynamic: &NoDynamicValues,
    }
    .eval(expr, &ValueEnv::default())
    .map(|q| q.si)
    .map_err(|e| format!("{what}: {e}"))
}

fn positive(expr: &ValueExpr, what: &str) -> Result<f64, String> {
    let v = si(expr, what)?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err(format!("{what} must be more than zero"))
    }
}

/// The frame `frame` turned over (z reversed, x kept): a tool against the sketch's normal.
fn flipped(frame: &Frame) -> Frame {
    Frame {
        origin: frame.origin,
        x: frame.x,
        y: [-frame.y[0], -frame.y[1], -frame.y[2]],
        z: [-frame.z[0], -frame.z[1], -frame.z[2]],
    }
}

/// The prism's caps along the profile frame's z for `extent`.
fn caps(
    extent: &Extent,
    both: bool,
) -> Result<(f64, f64, PrismBound, PrismBound), String> {
    Ok(match extent {
        Extent::Blind { distance } => {
            (0.0, positive(distance, "the depth")?, PrismBound::Blind, PrismBound::Blind)
        }
        Extent::Symmetric { distance } => {
            let d = positive(distance, "the depth")? / 2.0;
            (-d, d, PrismBound::Blind, PrismBound::Blind)
        }
        Extent::ThroughAll if both => (
            -1.0,
            1.0,
            PrismBound::ThroughAll,
            PrismBound::ThroughAll,
        ),
        Extent::ThroughAll => (0.0, 1.0, PrismBound::Blind, PrismBound::ThroughAll),
        _ => {
            return Err(
                "an assembly cut ends blind, mid-plane or through all; it doesn't read a component's faces"
                    .into(),
            )
        }
    })
}

/// The points a hole sketch places holes at: its point entities and its circles' and arcs'
/// centers, construction geometry included (a centerline sketch's points place holes too).
fn hole_points(sketch: &ResolvedSketch) -> Vec<(String, [f64; 2])> {
    let mut out = Vec::new();
    for entity in &sketch.sketch.entities {
        let role = match entity.geometry {
            SketchGeometry::Point { .. } => PointRole::At,
            SketchGeometry::Circle { .. } | SketchGeometry::Arc { .. } => PointRole::Center,
            _ => continue,
        };
        if let Some(p) = sketch.solved.points.get(&(entity.id.clone(), role)) {
            out.push((entity.id.0.clone(), *p));
        }
    }
    out
}

/// The sketch's curves as one chain from end to end, in its plane's coordinates (each arc as
/// short segments), for a weld path.
fn path_points(sketch: &ResolvedSketch) -> Result<Vec<[f64; 2]>, String> {
    let mut segments: Vec<(SketchEntityKey, [f64; 2], [f64; 2])> = Vec::new();
    for entity in sketch.sketch.entities.iter().filter(|e| !e.construction) {
        let get = |role| sketch.solved.points.get(&(entity.id.clone(), role)).copied();
        match entity.geometry {
            SketchGeometry::Line { .. } => {
                if let (Some(a), Some(b)) = (get(PointRole::Start), get(PointRole::End)) {
                    segments.push((SketchEntityKey::Line, a, b));
                }
            }
            SketchGeometry::Arc { .. } => {
                if let (Some(a), Some(b), Some(c)) =
                    (get(PointRole::Start), get(PointRole::End), get(PointRole::Center))
                {
                    segments.push((SketchEntityKey::Arc(c), a, b));
                }
            }
            _ => {}
        }
    }
    if segments.is_empty() {
        return Err(format!(
            "`{}` has no line or arc for the weld bead to follow",
            sketch.id
        ));
    }
    // Chain the segments end to end from the first one.
    let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) < 1e-9;
    let (kind, a, b) = segments.remove(0);
    let mut chain: Vec<[f64; 2]> = vec![a];
    push_segment(&mut chain, &kind, a, b);
    while !segments.is_empty() {
        let tail = *chain.last().unwrap();
        let Some(at) = segments
            .iter()
            .position(|(_, a, b)| close(*a, tail) || close(*b, tail))
        else {
            return Err(format!(
                "the curves of `{}` don't join end to end; draw the weld path as one chain",
                sketch.id
            ));
        };
        let (kind, a, b) = segments.remove(at);
        if close(a, tail) {
            push_segment(&mut chain, &kind, a, b);
        } else {
            push_segment(&mut chain, &kind.reversed(), b, a);
        }
    }
    Ok(chain)
}

#[derive(Clone, Copy)]
enum SketchEntityKey {
    Line,
    /// Its center; counter-clockwise from start to end (`ArcCw`: clockwise).
    Arc([f64; 2]),
    ArcCw([f64; 2]),
}

impl SketchEntityKey {
    fn reversed(&self) -> SketchEntityKey {
        match *self {
            SketchEntityKey::Line => SketchEntityKey::Line,
            SketchEntityKey::Arc(c) => SketchEntityKey::ArcCw(c),
            SketchEntityKey::ArcCw(c) => SketchEntityKey::Arc(c),
        }
    }
}

fn push_segment(chain: &mut Vec<[f64; 2]>, kind: &SketchEntityKey, a: [f64; 2], b: [f64; 2]) {
    match *kind {
        SketchEntityKey::Line => chain.push(b),
        SketchEntityKey::Arc(c) | SketchEntityKey::ArcCw(c) => {
            let ccw = matches!(kind, SketchEntityKey::Arc(_));
            let r = (a[0] - c[0]).hypot(a[1] - c[1]);
            let t0 = (a[1] - c[1]).atan2(a[0] - c[0]);
            let mut sweep = (b[1] - c[1]).atan2(b[0] - c[0]) - t0;
            let tau = std::f64::consts::TAU;
            if ccw {
                while sweep <= 0.0 {
                    sweep += tau;
                }
            } else {
                while sweep >= 0.0 {
                    sweep -= tau;
                }
            }
            let n = ((sweep.abs() / (tau / 48.0)).ceil() as usize).max(2);
            for k in 1..=n {
                let t = t0 + sweep * k as f64 / n as f64;
                chain.push([c[0] + r * t.cos(), c[1] + r * t.sin()]);
            }
        }
    }
}

fn line(curve: &str, a: [f64; 2], b: [f64; 2]) -> ProfileSegment {
    ProfileSegment {
        curve: crate::ids::SketchEntityId::new(curve),
        kind: SegmentKind::Line,
        start: a,
        end: b,
    }
}

/// `op`'s tool in assembly coordinates, from its solved sketch.
pub fn tool(op: &OperationKind, sketch: &ResolvedSketch) -> Result<OperationTool, String> {
    match op {
        OperationKind::Cut {
            extent,
            reverse,
            both,
            ..
        } => {
            let loops = sketch.profile().map_err(|e| {
                format!("`{}` has no closed profile to cut: {e}", sketch.id)
            })?;
            if loops.is_empty() {
                return Err(format!(
                    "`{}` has no closed profile to cut; close its outline",
                    sketch.id
                ));
            }
            let (mut start, mut end, start_bound, end_bound) = caps(extent, *both)?;
            // Against the normal: the caps mirror (an end below the start extrudes that way).
            if *reverse {
                (start, end) = (-start, -end);
            }
            Ok(OperationTool::Cut {
                profile: CanonicalProfile {
                    frame: sketch.frame,
                    loops,
                    merge_collinear: false,
                },
                start,
                end,
                start_bound,
                end_bound,
            })
        }
        OperationKind::Hole {
            diameter,
            extent,
            reverse,
            counterbore,
            countersink,
            ..
        } => {
            let positions = hole_points(sketch);
            if positions.is_empty() {
                return Err(format!(
                    "`{}` has no point or circle center to place a hole at",
                    sketch.id
                ));
            }
            let diameter = positive(diameter, "the hole diameter")?;
            // Holes drill along -frame.z: the flipped frame drills along the sketch's normal.
            let (frame, positions) = if *reverse {
                (sketch.frame, positions)
            } else {
                (
                    flipped(&sketch.frame),
                    positions
                        .into_iter()
                        .map(|(k, p)| (k, [p[0], -p[1]]))
                        .collect(),
                )
            };
            let end = match extent {
                Extent::Blind { distance } => HoleDepth::Blind {
                    depth: positive(distance, "the hole depth")?,
                },
                Extent::ThroughAll => HoleDepth::ThroughAll,
                _ => return Err("an assembly hole ends blind or through all".into()),
            };
            let counterbore = match counterbore {
                Some(c) => Some((
                    positive(&c.diameter, "the counterbore diameter")?,
                    positive(&c.depth, "the counterbore depth")?,
                )),
                None => None,
            };
            let countersink = match countersink {
                Some(c) => Some((
                    positive(&c.diameter, "the countersink diameter")?,
                    positive(&c.angle, "the countersink angle")?,
                )),
                None => None,
            };
            for (what, d) in [
                ("counterbore", counterbore.map(|c| c.0)),
                ("countersink", countersink.map(|c| c.0)),
            ] {
                if d.is_some_and(|d| d <= diameter) {
                    return Err(format!("the {what} must be wider than the hole"));
                }
            }
            Ok(OperationTool::Holes {
                frame,
                positions,
                diameter,
                end,
                counterbore,
                countersink,
            })
        }
        OperationKind::Weld { size, flip, .. } => {
            let leg = positive(size, "the weld size")?;
            let chain = path_points(sketch)?;
            let place = Transform::from_frame(&sketch.frame);
            let path: Vec<[f64; 3]> = chain.iter().map(|p| place.apply([p[0], p[1], 0.0])).collect();
            let along = sub(path[1], path[0]);
            let along = normalize(along);
            // The profile's frame: z along the path, y the sketch's normal, x = y × z to the
            // left of the path. The triangle stands on the sketch plane (v up the normal) and
            // lies to the left (u > 0), or mirrored to the right with `flip`.
            let y = sketch.frame.z;
            let frame = Frame {
                origin: path[0],
                x: normalize(cross(y, along)),
                y,
                z: along,
            };
            let corners: [[f64; 2]; 3] = if *flip {
                [[0.0, 0.0], [0.0, leg], [-leg, 0.0]]
            } else {
                [[0.0, 0.0], [leg, 0.0], [0.0, leg]]
            };
            let profile = CanonicalProfile {
                frame,
                loops: vec![vec![
                    line("leg1", corners[0], corners[1]),
                    line("face", corners[1], corners[2]),
                    line("leg2", corners[2], corners[0]),
                ]],
                merge_collinear: false,
            };
            Ok(OperationTool::Bead { profile, path })
        }
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if n > 0.0 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [0.0, 0.0, 1.0]
    }
}
