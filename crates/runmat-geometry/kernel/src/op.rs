//! The kernel seam: canonical operations in, opaque bodies plus history out.
//!
//! No kernel type crosses this trait. Every value is SI and every frame is in part coordinates.
//! Operands are the input bodies in order; entities in an operation are `(operand, index)`.
//! The kernel returns its output bodies and one [`OpHistory`] obeying the contract in
//! [`crate::history`].

use crate::frame::Frame;
use crate::history::{KernelTag, OpHistory, OperandEntity, Topology};
use crate::ids::SketchEntityId;
use crate::names::{InstanceKey, PlaneSide, Role};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct KernelId {
    pub name: String,
    pub version: String,
}

/// An opaque body owned by a kernel. Cloning shares it.
#[derive(Clone)]
pub struct BodyHandle {
    pub kernel: KernelId,
    payload: Arc<dyn Any + Send + Sync>,
}

impl BodyHandle {
    pub fn new<T: Any + Send + Sync>(kernel: KernelId, payload: T) -> Self {
        Self {
            kernel,
            payload: Arc::new(payload),
        }
    }
    pub fn downcast<T: Any>(&self) -> Option<&T> {
        self.payload.downcast_ref::<T>()
    }
}

impl fmt::Debug for BodyHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BodyHandle({}/{})",
            self.kernel.name, self.kernel.version
        )
    }
}

#[derive(Clone, Debug)]
pub struct KernelBody {
    pub handle: BodyHandle,
    pub topology: Topology,
}

#[derive(Clone, Debug)]
pub struct KernelOutput {
    pub bodies: Vec<KernelBody>,
    pub history: OpHistory,
}

#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "error")]
pub enum KernelError {
    /// The kernel does not support this operation or this case of it yet. Never a quietly
    /// wrong body.
    #[error("not supported yet: {what}")]
    NotSupported { what: String },
    /// The operation failed on valid input (a fillet too large, a self-intersecting sweep).
    #[error("{message}")]
    Failed { message: String },
    /// The operation could not report complete history, so it returned nothing.
    #[error("the kernel could not report history: {message}")]
    HistoryIncomplete { message: String },
    #[error("operand {operand} is not a body this kernel made")]
    ForeignOperand { operand: u32 },
}

/// The kernel interface: one canonical operation over operand bodies this kernel made.
///
/// The design layer calls it `CadKernel` (an alias of this trait, not a second definition).
pub trait Kernel {
    fn id(&self) -> KernelId;
    fn execute(&self, op: &KernelOp, operands: &[&BodyHandle])
        -> Result<KernelOutput, KernelError>;
}

/// The design layer's name for [`Kernel`].
pub use Kernel as CadKernel;

// ------------------------------------------------------------------- canonical operations

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "segment")]
pub enum SegmentKind {
    Line,
    /// About `center`, counter-clockwise from `start` to `end` unless `clockwise`.
    Arc {
        center: [f64; 2],
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        clockwise: bool,
    },
    Circle {
        center: [f64; 2],
        radius: f64,
    },
    /// An elliptical arc about `center`, counter-clockwise from `start` to `end` unless
    /// `clockwise`; `start == end` is the whole ellipse. `major` is the semi-major axis vector,
    /// `ratio` the minor axis over the major (at most 1); the minor axis is a quarter turn
    /// counter-clockwise from the major.
    Ellipse {
        center: [f64; 2],
        major: [f64; 2],
        ratio: f64,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        clockwise: bool,
    },
    /// A clamped B-spline (rational when `weights` are given, one per pole) from its first
    /// pole (`start`) to its last (`end`). `knots` are the distinct knot values with their
    /// `multiplicities`; the end knots have multiplicity `degree + 1`. A closed spline repeats
    /// its first pole last.
    Spline {
        degree: u32,
        poles: Vec<[f64; 2]>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        weights: Vec<f64>,
        knots: Vec<f64>,
        multiplicities: Vec<u32>,
    },
}

impl SegmentKind {
    /// The same curve with every defining point moved by a rigid map of the plane.
    pub fn mapped(&self, map: &dyn Fn([f64; 2]) -> [f64; 2]) -> SegmentKind {
        match self {
            SegmentKind::Line => SegmentKind::Line,
            SegmentKind::Arc { center, clockwise } => SegmentKind::Arc {
                center: map(*center),
                clockwise: *clockwise,
            },
            SegmentKind::Circle { center, radius } => SegmentKind::Circle {
                center: map(*center),
                radius: *radius,
            },
            SegmentKind::Ellipse {
                center,
                major,
                ratio,
                clockwise,
            } => {
                let c = map(*center);
                let m = map([center[0] + major[0], center[1] + major[1]]);
                SegmentKind::Ellipse {
                    center: c,
                    major: [m[0] - c[0], m[1] - c[1]],
                    ratio: *ratio,
                    clockwise: *clockwise,
                }
            }
            SegmentKind::Spline {
                degree,
                poles,
                weights,
                knots,
                multiplicities,
            } => SegmentKind::Spline {
                degree: *degree,
                poles: poles.iter().map(|p| map(*p)).collect(),
                weights: weights.clone(),
                knots: knots.clone(),
                multiplicities: multiplicities.clone(),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProfileSegment {
    /// The sketch curve, echoed back in history tags.
    pub curve: SketchEntityId,
    pub kind: SegmentKind,
    pub start: [f64; 2],
    pub end: [f64; 2],
}

impl ProfileSegment {
    /// The point at fraction `t` (0 at `start`, 1 at `end`) of the curve's parameter range:
    /// the angle for arcs, circles and ellipses, the knot range for splines.
    pub fn point_at(&self, t: f64) -> [f64; 2] {
        let tau = std::f64::consts::TAU;
        match &self.kind {
            SegmentKind::Line => [
                self.start[0] + t * (self.end[0] - self.start[0]),
                self.start[1] + t * (self.end[1] - self.start[1]),
            ],
            SegmentKind::Arc { center, clockwise } => {
                let r = ((self.start[0] - center[0]).powi(2) + (self.start[1] - center[1]).powi(2))
                    .sqrt();
                let (a0, sweep) = sweep_between(*center, self.start, self.end, *clockwise);
                let a = a0 + t * sweep;
                [center[0] + r * a.cos(), center[1] + r * a.sin()]
            }
            SegmentKind::Circle { center, radius } => {
                let a = t * tau;
                [center[0] + radius * a.cos(), center[1] + radius * a.sin()]
            }
            SegmentKind::Ellipse {
                center,
                major,
                ratio,
                clockwise,
            } => {
                // Parametric angle about the major axis.
                let ra = (major[0] * major[0] + major[1] * major[1])
                    .sqrt()
                    .max(1e-300);
                let ux = [major[0] / ra, major[1] / ra];
                let uy = [-ux[1], ux[0]];
                let rb = ra * ratio;
                let param = |p: [f64; 2]| {
                    let d = [p[0] - center[0], p[1] - center[1]];
                    let x = d[0] * ux[0] + d[1] * ux[1];
                    let y = d[0] * uy[0] + d[1] * uy[1];
                    (y / rb.max(1e-300)).atan2(x / ra)
                };
                let u0 = param(self.start);
                let mut u1 = param(self.end);
                let whole = ((self.start[0] - self.end[0]).powi(2)
                    + (self.start[1] - self.end[1]).powi(2))
                .sqrt()
                    <= 1e-12 * ra.max(1.0);
                if *clockwise {
                    while u1 >= u0 - if whole { tau - 1e-12 } else { 0.0 } {
                        u1 -= tau;
                    }
                } else {
                    while u1 <= u0 + if whole { tau - 1e-12 } else { 0.0 } {
                        u1 += tau;
                    }
                }
                let u = u0 + t * (u1 - u0);
                [
                    center[0] + ra * u.cos() * ux[0] + rb * u.sin() * uy[0],
                    center[1] + ra * u.cos() * ux[1] + rb * u.sin() * uy[1],
                ]
            }
            SegmentKind::Spline {
                degree,
                poles,
                weights,
                knots,
                multiplicities,
            } => spline_point(*degree, poles, weights, knots, multiplicities, t),
        }
    }

    /// `n + 1` points from `start` to `end`, evenly spaced in the curve's parameter.
    pub fn polyline(&self, n: usize) -> Vec<[f64; 2]> {
        let n = n.max(1);
        (0..=n)
            .map(|i| self.point_at(i as f64 / n as f64))
            .collect()
    }

    /// Whether the segment is a whole closed curve on its own (a circle, a whole ellipse, a
    /// closed spline).
    pub fn is_closed_curve(&self) -> bool {
        match self.kind {
            SegmentKind::Circle { .. } => true,
            SegmentKind::Ellipse { .. } | SegmentKind::Spline { .. } => {
                (self.start[0] - self.end[0]).abs() <= 1e-12
                    && (self.start[1] - self.end[1]).abs() <= 1e-12
            }
            _ => false,
        }
    }
}

/// An arc's start angle and signed sweep about `c` from `a` to `b`.
fn sweep_between(c: [f64; 2], a: [f64; 2], b: [f64; 2], clockwise: bool) -> (f64, f64) {
    let tau = std::f64::consts::TAU;
    let a0 = (a[1] - c[1]).atan2(a[0] - c[0]);
    let a1 = (b[1] - c[1]).atan2(b[0] - c[0]);
    let mut sweep = a1 - a0;
    if clockwise {
        while sweep >= -1e-12 {
            sweep -= tau;
        }
    } else {
        while sweep <= 1e-12 {
            sweep += tau;
        }
    }
    (a0, sweep)
}

/// A clamped (rational) B-spline at fraction `t` of its knot range (de Boor).
fn spline_point(
    degree: u32,
    poles: &[[f64; 2]],
    weights: &[f64],
    knots: &[f64],
    mults: &[u32],
    t: f64,
) -> [f64; 2] {
    let p = degree as usize;
    let mut flat = Vec::new();
    for (k, m) in knots.iter().zip(mults) {
        for _ in 0..*m {
            flat.push(*k);
        }
    }
    let n = poles.len();
    if n == 0 || flat.len() < n + p + 1 || p == 0 {
        return poles.first().copied().unwrap_or([0.0, 0.0]);
    }
    let (lo, hi) = (flat[p], flat[n]);
    let u = lo + t.clamp(0.0, 1.0) * (hi - lo);
    // The span holding u.
    let mut k = p;
    while k + 1 < n && flat[k + 1] <= u {
        k += 1;
    }
    let w = |i: usize| weights.get(i).copied().unwrap_or(1.0);
    let mut d: Vec<[f64; 3]> = (0..=p)
        .map(|j| {
            let i = j + k - p;
            [poles[i][0] * w(i), poles[i][1] * w(i), w(i)]
        })
        .collect();
    for r in 1..=p {
        for j in (r..=p).rev() {
            let i = j + k - p;
            let den = flat[i + p + 1 - r] - flat[i];
            let a = if den.abs() < 1e-300 {
                0.0
            } else {
                (u - flat[i]) / den
            };
            for c in 0..3 {
                d[j][c] = (1.0 - a) * d[j - 1][c] + a * d[j][c];
            }
        }
    }
    [d[p][0] / d[p][2], d[p][1] / d[p][2]]
}

/// Closed loops in a sketch frame. The first loop is outer; the rest are holes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalProfile {
    pub frame: Frame,
    pub loops: Vec<Vec<ProfileSegment>>,
    /// Side faces of collinear (or co-circular) profile curves merge into one face. Off by
    /// default: each profile curve makes its own side face, as the sketch drew it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub merge_collinear: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Combine {
    /// Output is the tool alone (operands untouched and not passed).
    New,
    Add,
    Cut,
    Intersect,
}

/// A tool shape: what extrude, revolve, sweep, loft and hole make before combining.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "tool")]
pub enum ToolShape {
    /// Prism along the frame's Z from `start` (the start cap) to `end` (the end cap), signed
    /// distances; `end < start` extrudes against the normal. Each cap's [`PrismBound`] can
    /// replace its distance with an end condition read from the operands (through all, up to
    /// next, up to a face or body). `draft` tapers the sides; `thin` makes a wall along the
    /// profile curves instead of filling the profile.
    Prism {
        profile: CanonicalProfile,
        start: f64,
        end: f64,
        #[serde(default, skip_serializing_if = "PrismBound::is_blind")]
        start_bound: PrismBound,
        #[serde(default, skip_serializing_if = "PrismBound::is_blind")]
        end_bound: PrismBound,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        draft: Option<PrismDraft>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thin: Option<ThinWallSpec>,
    },
    Revolve {
        profile: CanonicalProfile,
        axis_point: [f64; 3],
        axis_direction: [f64; 3],
        angle: f64,
    },
    /// The profile swept along a polyline path. `path_curves[i]` is the path curve segment `i`
    /// (from `path[i]` to `path[i + 1]`) belongs to, so a curved path curve sampled into
    /// several segments still makes one face per profile curve; side faces are numbered by
    /// path curve when the path has several. Empty: each segment is its own curve.
    ///
    /// Without `follow_path` the profile is translated along the path and stays parallel to
    /// itself. With it the profile keeps its angle to the path: it turns through each path
    /// curve and is mitred at the corners between straight segments, so a path segment may
    /// lie in the profile's plane.
    Sweep {
        profile: CanonicalProfile,
        path: Vec<[f64; 3]>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        path_curves: Vec<u32>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        follow_path: bool,
    },
    Loft {
        profiles: Vec<CanonicalProfile>,
    },
    /// Holes into a planar face along `-frame.z`, one per keyed position (frame coordinates).
    Holes {
        frame: Frame,
        positions: Vec<(String, [f64; 2])>,
        diameter: f64,
        end: HoleDepth,
        counterbore: Option<(f64, f64)>,
        /// (diameter, included angle).
        countersink: Option<(f64, f64)>,
        /// A blind hole's drill point (included angle); `None`: a flat bottom.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        drill_point: Option<f64>,
        /// The wall tapers (half-angle, narrowing with depth): a pipe tap.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        taper: Option<f64>,
        /// A modelled thread: the threaded wall is its own face ([`Role::ThreadFace`] tagged
        /// by the hole key).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thread: Option<HoleThread>,
        /// A slot: `length` centre to centre along the frame's x turned by `angle`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        slot: Option<(f64, f64)>,
        /// A hole series: per operand in order from the start face, the diameter through that
        /// body (each body's depth range is where the axis crosses it). Empty: one diameter.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        stages: Vec<HoleStage>,
    },
}

/// How far a hole goes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "end")]
pub enum HoleDepth {
    Blind {
        depth: f64,
    },
    ThroughAll,
    /// To the next face below the start: out of the material it starts in.
    UpToNext,
    UpToFace {
        face: OperandEntity,
        offset: f64,
    },
}

/// A modelled hole thread: `length` from the start (or the counterbore floor); `None`: the
/// whole wall.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoleThread {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<f64>,
}

/// One body of a hole series.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoleStage {
    pub operand: u32,
    pub diameter: f64,
    /// A modelled thread through this stage.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub thread: bool,
}

/// Where a prism cap ends, beyond its signed distance.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "bound")]
pub enum PrismBound {
    /// At the distance given.
    #[default]
    Blind,
    /// Past every operand.
    ThroughAll,
    /// At the next operand face along the direction from the profile plane, which must
    /// intercept the whole profile; the cap takes that face's shape.
    UpToNext,
    /// On an operand face (planar or curved), `offset` beyond it along the direction.
    UpToFace { face: OperandEntity, offset: f64 },
    /// Where it first meets an operand body.
    UpToBody { operand: u32, offset: f64 },
    /// The start cap lies on this operand face (extrude from a surface); a blind far cap is
    /// then measured from it.
    FromFace { face: OperandEntity },
}

impl PrismBound {
    pub fn is_blind(&self) -> bool {
        matches!(self, PrismBound::Blind)
    }
    /// The operand faces or bodies this bound reads.
    pub fn reads_operands(&self) -> bool {
        !matches!(self, PrismBound::Blind | PrismBound::ThroughAll)
    }
}

/// Side taper per direction (radians; positive narrows the profile moving away from the
/// profile plane).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrismDraft {
    pub end: f64,
    pub start: f64,
}

/// A thin wall: from `inside` within each profile curve's region to `outside` beyond it.
/// Inner walls are [`Role::ThinWall`] tagged by curve; `cap` closes both ends.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThinWallSpec {
    pub inside: f64,
    pub outside: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cap: Option<f64>,
}

/// A blend's cross-section.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "profile")]
pub enum BlendProfile {
    #[default]
    Circular,
    Conic {
        rho: f64,
    },
    Curvature,
}

impl BlendProfile {
    pub fn is_circular(&self) -> bool {
        matches!(self, BlendProfile::Circular)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "size")]
pub enum ChamferSize {
    Equal {
        distance: f64,
    },
    /// On the edge's first face, then its second (swapped by the design layer for `flip`).
    Distances {
        first: f64,
        second: f64,
    },
    /// On the edge's first face (its second with `on_second`), at `angle` from that face.
    DistanceAngle {
        distance: f64,
        angle: f64,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        on_second: bool,
    },
}

impl ChamferSize {
    /// The larger setback, for reach and area.
    pub fn reach(&self) -> f64 {
        match *self {
            ChamferSize::Equal { distance } => distance,
            ChamferSize::Distances { first, second } => first.max(second),
            ChamferSize::DistanceAngle {
                distance, angle, ..
            } => distance.max(distance * angle.tan().abs()),
        }
    }
}

impl ToolShape {
    /// A blind prism from `start` to `end` along the profile frame's z.
    pub fn prism(profile: CanonicalProfile, start: f64, end: f64) -> Self {
        ToolShape::Prism {
            profile,
            start,
            end,
            start_bound: PrismBound::Blind,
            end_bound: PrismBound::Blind,
            draft: None,
            thin: None,
        }
    }

    /// Whether the definition asked collinear profile sides to merge.
    pub fn merge_collinear(&self) -> bool {
        match self {
            ToolShape::Prism { profile, .. }
            | ToolShape::Revolve { profile, .. }
            | ToolShape::Sweep { profile, .. } => profile.merge_collinear,
            ToolShape::Loft { profiles } => profiles.iter().all(|p| p.merge_collinear),
            ToolShape::Holes { .. } => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolOp {
    pub shape: ToolShape,
    pub combine: Combine,
    /// The last `references` operands are read by end conditions only: they are not combined
    /// and come back unchanged, after the other outputs.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub references: u32,
    /// The profile is this planar operand face: a prism, revolve or sweep of its boundary,
    /// holes included. The kernel reads the face's loops (outer first) into `profile.loops` in
    /// `profile.frame`, which must lie in the face's plane; the shape's own loops are then
    /// empty. Each face edge's side face is tagged with the curve `e<k>` (`k` counting the
    /// loops' edges from 1 in order).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_face: Option<OperandEntity>,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

impl ToolOp {
    /// This tool with its face profile read: `loops` (outer first, in the profile frame) become
    /// the profile and `profile_face` is cleared. A tool without a face profile comes back
    /// as it is; a face profile on a loft or holes is an error.
    pub fn with_face_loops(
        &self,
        read: impl FnOnce(OperandEntity, &Frame) -> Result<Vec<Vec<ProfileSegment>>, KernelError>,
    ) -> Result<ToolOp, KernelError> {
        let Some(face) = self.profile_face else {
            return Ok(self.clone());
        };
        let mut out = self.clone();
        out.profile_face = None;
        let profile = match &mut out.shape {
            ToolShape::Prism { profile, .. }
            | ToolShape::Revolve { profile, .. }
            | ToolShape::Sweep { profile, .. } => profile,
            _ => {
                return Err(KernelError::NotSupported {
                    what: "a face profile for a loft or holes".into(),
                })
            }
        };
        profile.loops = read(face, &profile.frame)?;
        if profile.loops.is_empty() {
            return Err(KernelError::Failed {
                message: "the profile face has no boundary".into(),
            });
        }
        Ok(out)
    }

    /// The curve id a face profile gives its `k`-th edge (from 0, loops in order).
    pub fn face_edge_curve(k: usize) -> SketchEntityId {
        SketchEntityId::new(format!("e{}", k + 1))
    }

    pub fn new(shape: ToolShape, combine: Combine) -> Self {
        ToolOp {
            shape,
            combine,
            references: 0,
            profile_face: None,
        }
    }
}

/// A rigid transform `p ↦ R p + t` (rows of `R`, then `t`). Mirrors have `det R = −1`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

impl Transform {
    pub fn translation(t: [f64; 3]) -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: t,
        }
    }
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + self.translation[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + self.translation[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + self.translation[2],
        ]
    }
    pub fn apply_vector(&self, v: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        [
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        ]
    }
    pub const IDENTITY: Transform = Transform {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };

    /// `self ∘ other`: apply `other` first, then `self`.
    pub fn compose(&self, other: &Transform) -> Transform {
        let a = &self.rotation;
        let b = &other.rotation;
        let mut r = [[0.0; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        Transform {
            rotation: r,
            translation: self.apply(other.translation),
        }
    }

    /// The inverse of a rigid transform (or of a reflection: `R` is orthogonal either way).
    pub fn inverse(&self) -> Transform {
        let r = &self.rotation;
        let rt = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ];
        let t = self.translation;
        let inv = Transform {
            rotation: rt,
            translation: [0.0; 3],
        };
        let mt = inv.apply_vector(t);
        Transform {
            rotation: rt,
            translation: [-mt[0], -mt[1], -mt[2]],
        }
    }

    pub fn determinant(&self) -> f64 {
        let r = &self.rotation;
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    }

    /// The transform taking frame-local coordinates to the frame's parent coordinates.
    pub fn from_frame(f: &Frame) -> Transform {
        Transform {
            rotation: [
                [f.x[0], f.y[0], f.z[0]],
                [f.x[1], f.y[1], f.z[1]],
                [f.x[2], f.y[2], f.z[2]],
            ],
            translation: f.origin,
        }
    }

    /// The frame this transform places (its image of the world frame).
    pub fn to_frame(&self) -> Frame {
        let r = &self.rotation;
        Frame {
            origin: self.translation,
            x: [r[0][0], r[1][0], r[2][0]],
            y: [r[0][1], r[1][1], r[2][1]],
            z: [r[0][2], r[1][2], r[2][2]],
        }
    }

    /// Rotation by `angle` radians about the unit `axis` through `point`.
    pub fn rotation_about(point: [f64; 3], axis: [f64; 3], angle: f64) -> Transform {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let (x, y, z) = if n > 0.0 {
            (axis[0] / n, axis[1] / n, axis[2] / n)
        } else {
            (0.0, 0.0, 1.0)
        };
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        let rot = Transform {
            rotation: [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ],
            translation: [0.0; 3],
        };
        Transform::translation(point)
            .compose(&rot)
            .compose(&Transform::translation([-point[0], -point[1], -point[2]]))
    }

    /// The unit quaternion `[w, x, y, z]` of a proper rotation (Shepperd's method).
    pub fn quaternion(&self) -> [f64; 4] {
        let m = &self.rotation;
        let tr = m[0][0] + m[1][1] + m[2][2];
        let q = if tr > 0.0 {
            let s = (tr + 1.0).sqrt() * 2.0;
            [
                0.25 * s,
                (m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s,
            ]
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
            [
                (m[2][1] - m[1][2]) / s,
                0.25 * s,
                (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s,
            ]
        } else if m[1][1] > m[2][2] {
            let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
            [
                (m[0][2] - m[2][0]) / s,
                (m[0][1] + m[1][0]) / s,
                0.25 * s,
                (m[1][2] + m[2][1]) / s,
            ]
        } else {
            let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
            [
                (m[1][0] - m[0][1]) / s,
                (m[0][2] + m[2][0]) / s,
                (m[1][2] + m[2][1]) / s,
                0.25 * s,
            ]
        };
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        // A canonical sign keeps keys and the index deterministic.
        let sign = if q[0] < 0.0 { -1.0 } else { 1.0 };
        [
            sign * q[0] / n,
            sign * q[1] / n,
            sign * q[2] / n,
            sign * q[3] / n,
        ]
    }

    /// A proper rigid transform from a unit quaternion `[w, x, y, z]` and a translation.
    pub fn from_quaternion(q: [f64; 4], translation: [f64; 3]) -> Transform {
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        let (w, x, y, z) = if n > 0.0 {
            (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
        } else {
            (1.0, 0.0, 0.0, 0.0)
        };
        Transform {
            rotation: [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            translation,
        }
    }

    /// Largest absolute entry difference: rotation and translation together.
    pub fn max_difference(&self, other: &Transform) -> f64 {
        let mut m: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                m = m.max((self.rotation[i][j] - other.rotation[i][j]).abs());
            }
            m = m.max((self.translation[i] - other.translation[i]).abs());
        }
        m
    }

    /// Reflection across the plane through `point` with unit `normal`.
    pub fn reflection(point: [f64; 3], normal: [f64; 3]) -> Self {
        let n = normal;
        let mut r = [[0.0; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = if i == j { 1.0 } else { 0.0 } - 2.0 * n[i] * n[j];
            }
        }
        let d = n[0] * point[0] + n[1] * point[1] + n[2] * point[2];
        Self {
            rotation: r,
            translation: [2.0 * d * n[0], 2.0 * d * n[1], 2.0 * d * n[2]],
        }
    }
}

/// The label an operation asks the kernel to give the faces one tool surface makes.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FaceLabel {
    pub role: Role,
    pub tag: KernelTag,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<u32>,
}

impl FaceLabel {
    pub fn key(role: Role, key: impl Into<String>) -> Self {
        FaceLabel {
            role,
            tag: KernelTag::Key { key: key.into() },
            ordinal: None,
        }
    }
    pub fn with_ordinal(mut self, ordinal: u32) -> Self {
        self.ordinal = Some(ordinal);
        self
    }
}

/// A tool whose faces carry labels the design layer chose: what sheet metal flanges and
/// bends, structural members, end caps, gussets, weld beads, ribs, embosses and form tools
/// lower to. The shape is built as for [`ToolOp`]; each face it makes takes the label of the
/// surface that made it (`sides` by profile curve, `start`/`end` caps). A curve with no entry,
/// or a cap with none, keeps the default label ([`Role::Side`] with the curve,
/// [`Role::StartCap`], [`Role::EndCap`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabeledTool {
    pub shape: ToolShape,
    #[serde(default)]
    pub sides: Vec<(SketchEntityId, FaceLabel)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<FaceLabel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<FaceLabel>,
    /// The body's label when this tool alone makes a new body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<FaceLabel>,
    /// Half-spaces the tool is clipped to (mitred member ends): material is kept where
    /// `(p − origin)·normal ≤ 0`; the cut face takes the label.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clip: Vec<ClipPlane>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipPlane {
    pub origin: [f64; 3],
    pub normal: [f64; 3],
    pub label: FaceLabel,
}

impl LabeledTool {
    pub fn new(shape: ToolShape) -> Self {
        LabeledTool {
            shape,
            sides: Vec::new(),
            start: None,
            end: None,
            body: None,
            clip: Vec::new(),
        }
    }
}

/// How the moved faces of a direct edit move.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "motion")]
pub enum FaceMotion {
    /// Along each face's own normal by a signed distance (offset face, push-pull).
    Offset { distance: f64 },
    /// A rigid translation of the faces; neighbours stretch to follow.
    Translate { vector: [f64; 3] },
    /// A rotation about an axis; neighbours stretch to follow.
    Rotate {
        axis_point: [f64; 3],
        axis_direction: [f64; 3],
        angle: f64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    pub key: InstanceKey,
    pub transform: Transform,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum KernelOp {
    /// Make a tool and combine it with every operand (none for `Combine::New`).
    Tool(ToolOp),
    /// Blend edges. `variable`: radii at parameters `(t, r)` along every edge, `t` the fraction
    /// of the edge's length from its [`EdgeGeom`](crate::history::EdgeGeom) start (`radius` at
    /// both ends where none is given). `setbacks`: vertices where the blends stop short and a
    /// corner blend ([`Role::Blend`] tagged [`KernelTag::InputVertex`]) joins them.
    Fillet {
        edges: Vec<OperandEntity>,
        radius: f64,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        variable: Vec<(f64, f64)>,
        #[serde(default, skip_serializing_if = "BlendProfile::is_circular")]
        profile: BlendProfile,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        setbacks: Vec<(OperandEntity, f64)>,
    },
    /// A blend tangent to two face sets; blend faces are tagged by the edges they replace.
    FaceBlend {
        first: Vec<OperandEntity>,
        second: Vec<OperandEntity>,
        radius: f64,
        #[serde(default, skip_serializing_if = "BlendProfile::is_circular")]
        profile: BlendProfile,
    },
    /// Replace the centre faces by one blend tangent to both side sets ([`Role::Blend`] tagged
    /// [`KernelTag::InputFace`] by each centre face).
    FullRound {
        first: Vec<OperandEntity>,
        center: Vec<OperandEntity>,
        second: Vec<OperandEntity>,
    },
    Chamfer {
        edges: Vec<OperandEntity>,
        size: ChamferSize,
    },
    /// Cut corners through points at these distances along each vertex's edges (in the
    /// vertex's edge order); faces [`Role::Chamfer`] tagged [`KernelTag::InputVertex`].
    VertexChamfer {
        vertices: Vec<OperandEntity>,
        distances: [f64; 3],
    },
    /// Hollow a body: walls `thickness` thick, or the thickness given per face. `outward`
    /// grows the walls outside the body, which becomes the cavity (outer walls are
    /// [`Role::ShellOuter`]).
    Shell {
        open_faces: Vec<OperandEntity>,
        thickness: f64,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        faces: Vec<(OperandEntity, f64)>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        outward: bool,
    },
    /// Taper faces about the neutral plane (pull along `neutral.z`). With `parting` edges,
    /// each face hinges about the parting edges it touches instead; `step` adds step faces
    /// ([`Role::DraftStep`] tagged by the face) where neighbours hinge at different heights.
    Draft {
        faces: Vec<OperandEntity>,
        neutral: Frame,
        angle: f64,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        parting: Vec<OperandEntity>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        step: bool,
    },
    /// Re-apply seed tools at each instance (tags carry `seed` and `instance`). `varied`:
    /// per instance (in step with `instances`), the seed tools rebuilt with that instance's
    /// values (a variable pattern); empty: `seeds` at every instance.
    PatternTools {
        seeds: Vec<ToolOp>,
        instances: Vec<Instance>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        varied: Vec<Vec<ToolOp>>,
    },
    /// Copy every operand at each instance; the originals are kept.
    PatternBodies {
        instances: Vec<Instance>,
    },
    /// Repeat the material `faces` (all of one operand) bound, a boss or a pocket, at each
    /// instance: the faces are lifted off the body and healed, the added and removed material
    /// moved per instance, fused and cut. Instance faces are copies tagged by instance.
    PatternFaces {
        faces: Vec<OperandEntity>,
        instances: Vec<Instance>,
    },
    /// Operand 0 is the target; the rest are tools.
    Boolean {
        combine: Combine,
        keep_tools: bool,
    },
    /// Split every operand by a plane (`frame.z` is the tool normal; positive is along it).
    Split {
        frame: Frame,
    },
    /// Split operand 0 by the surfaces of `faces` (each extended across the body) or, with no
    /// faces, by operand 1 (a sheet or a solid). New faces are [`Role::SplitFace`]; the tool
    /// operands are kept.
    SplitWith {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        faces: Vec<OperandEntity>,
    },
    /// Cut operand 1, grown by `clearance`, out of operand 0 (an indent); operand 1 is kept.
    Indent {
        clearance: f64,
    },
    Transform {
        transform: Transform,
    },
    Import {
        source: String,
        digest: String,
    },
    /// Labelled tools combined with every operand. `carve` tools are removed from the
    /// operands first (a flange trimming the sheet it grows from), then `tools` are combined.
    /// With `Combine::New` and `separate`, each tool makes its own body (structural members).
    Labeled {
        tools: Vec<LabeledTool>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        carve: Vec<LabeledTool>,
        combine: Combine,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        separate: bool,
    },
    /// Move faces of one body (direct edit); neighbours extend or trim to keep it closed.
    /// History: moved faces continue their inputs (modified).
    MoveFaces {
        faces: Vec<OperandEntity>,
        motion: FaceMotion,
    },
    /// Delete faces of one body. `heal`: neighbours extend to close the gap (a removed hole,
    /// pocket, boss or blend); otherwise the result is a surface body with the opening.
    DeleteFaces {
        faces: Vec<OperandEntity>,
        heal: bool,
    },
    /// Replace faces of operand 0 by a face of operand 1 (a surface body); the material
    /// between them is added or removed.
    ReplaceFaces {
        faces: Vec<OperandEntity>,
        with: OperandEntity,
    },
    /// Scale every operand about `center` by per-axis factors (a mold shrink allowance).
    Scale {
        center: [f64; 3],
        factors: [f64; 3],
    },
    /// Surface bodies (open sheets) from canonical patches.
    Surface(SurfaceOp),
}

/// Surface-body operations. Faces of surface bodies follow the same history contract;
/// a surface body's boundary edges name their one face twice ([`crate::history::EdgeTopo`]
/// with `faces[0] == faces[1]`, `laminar`).
// One per operation; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "surface")]
pub enum SurfaceOp {
    /// Planar patches, one per labelled polygon (frame plus loops; first loop outer).
    Planar { patches: Vec<SurfacePatch> },
    /// Open or closed profile chains swept by a tool shape into sheets: every profile curve
    /// gives a face (no caps).
    Swept { tool: LabeledTool },
    /// Offset the operands' faces (all, or `faces`) by a signed distance along their normals.
    Offset {
        #[serde(default)]
        faces: Vec<OperandEntity>,
        distance: f64,
    },
    /// Knit sheets sharing boundary edges into one; `solid` closes a watertight set into a solid.
    Knit { solid: bool },
    /// Thicken a sheet into a solid along its normal (`distance` signed; `both` centres it).
    Thicken {
        distance: f64,
        #[serde(default)]
        both: bool,
    },
    /// Trim sheets by a plane, keeping the side `keep` (by the frame's z).
    Trim { frame: Frame, keep: PlaneSide },
    /// Extend boundary edges of a sheet by a distance in their face's plane.
    Extend {
        edges: Vec<OperandEntity>,
        distance: f64,
    },
    /// The mid-surface of a thin solid: one patch per pair of opposite parallel faces.
    MidSurface {
        pairs: Vec<(OperandEntity, OperandEntity)>,
    },
    /// Close holes of sheets: each closed loop of the boundary edges is filled with its
    /// face's surface.
    DeleteHole { edges: Vec<OperandEntity> },
    /// A new sheet patching the closed loop of edges (of sheets or solids), meeting each edge
    /// with its continuity. The operands are kept.
    Fill {
        edges: Vec<(OperandEntity, crate::names::Continuity)>,
    },
    /// A new sheet through boundary curves: with `second` empty, a smooth surface through the
    /// `first` edges in order (at least two); with both, a patch bounded by the outer edges of
    /// each direction (the first and last of each, closing one loop) that passes through the
    /// inner ones. A bounding edge meets its face with its continuity. The operands are kept.
    Boundary {
        first: Vec<(OperandEntity, crate::names::Continuity)>,
        #[serde(default)]
        second: Vec<(OperandEntity, crate::names::Continuity)>,
    },
    /// A new sheet of ruled strips, one face per edge, `distance` wide: along the bounded
    /// face's normal turned `angle` outward across the edge. The operands are kept.
    Ruled {
        edges: Vec<OperandEntity>,
        distance: f64,
        angle: f64,
    },
    /// Cut the solid operand (0) with the sheet operand (1), keeping the side `keep` of the
    /// sheet's normal; the sheet is kept.
    CutWithSurface { keep: PlaneSide },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfacePatch {
    pub profile: CanonicalProfile,
    pub label: FaceLabel,
}

impl KernelOp {
    pub fn name(&self) -> &'static str {
        match self {
            KernelOp::Tool(_) => "tool",
            KernelOp::Fillet { .. } => "fillet",
            KernelOp::FaceBlend { .. } => "face_blend",
            KernelOp::FullRound { .. } => "full_round",
            KernelOp::Chamfer { .. } => "chamfer",
            KernelOp::VertexChamfer { .. } => "vertex_chamfer",
            KernelOp::Shell { .. } => "shell",
            KernelOp::Draft { .. } => "draft",
            KernelOp::PatternTools { .. } => "pattern_tools",
            KernelOp::PatternBodies { .. } => "pattern_bodies",
            KernelOp::PatternFaces { .. } => "pattern_faces",
            KernelOp::Boolean { .. } => "boolean",
            KernelOp::Split { .. } => "split",
            KernelOp::SplitWith { .. } => "split_with",
            KernelOp::Indent { .. } => "indent",
            KernelOp::Transform { .. } => "transform",
            KernelOp::Import { .. } => "import",
            KernelOp::Labeled { .. } => "labeled",
            KernelOp::MoveFaces { .. } => "move_faces",
            KernelOp::DeleteFaces { .. } => "delete_faces",
            KernelOp::ReplaceFaces { .. } => "replace_faces",
            KernelOp::Scale { .. } => "scale",
            KernelOp::Surface(_) => "surface",
        }
    }
}
