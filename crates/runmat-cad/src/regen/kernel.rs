//! The kernel seam: canonical operations in, opaque bodies plus history out.
//!
//! No kernel type crosses this trait. Every value is SI and every frame is in part coordinates.
//! Operands are the input bodies in order; entities in an operation are `(operand, index)`.
//! The kernel returns its output bodies and one [`OpHistory`] obeying the contract in
//! [`crate::refs::history`].

use crate::ids::SketchEntityId;
use crate::model::datum::Frame;
use crate::refs::history::{OpHistory, OperandEntity, Topology};
use crate::refs::name::InstanceKey;
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

/// The kernel interface.
pub trait CadKernel {
    fn id(&self) -> KernelId;
    fn execute(&self, op: &KernelOp, operands: &[&BodyHandle])
        -> Result<KernelOutput, KernelError>;
}

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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProfileSegment {
    /// The sketch curve, echoed back in history tags.
    pub curve: SketchEntityId,
    pub kind: SegmentKind,
    pub start: [f64; 2],
    pub end: [f64; 2],
}

/// Closed loops in a sketch frame. The first loop is outer; the rest are holes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalProfile {
    pub frame: Frame,
    pub loops: Vec<Vec<ProfileSegment>>,
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
    /// distances; `end < start` extrudes against the normal. `through_all` extends the end cap
    /// past every operand, in the direction of `end - start`.
    Prism {
        profile: CanonicalProfile,
        start: f64,
        end: f64,
        through_all: bool,
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
    Sweep {
        profile: CanonicalProfile,
        path: Vec<[f64; 3]>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        path_curves: Vec<u32>,
    },
    Loft {
        profiles: Vec<CanonicalProfile>,
    },
    /// Holes into a planar face along `-frame.z`, one per keyed position (frame coordinates).
    Holes {
        frame: Frame,
        positions: Vec<(String, [f64; 2])>,
        diameter: f64,
        /// `None`: through all.
        depth: Option<f64>,
        counterbore: Option<(f64, f64)>,
        countersink: Option<(f64, f64)>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolOp {
    pub shape: ToolShape,
    pub combine: Combine,
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
    Fillet {
        edges: Vec<OperandEntity>,
        radius: f64,
    },
    Chamfer {
        edges: Vec<OperandEntity>,
        distance: f64,
        distance2: Option<f64>,
    },
    Shell {
        open_faces: Vec<OperandEntity>,
        thickness: f64,
    },
    Draft {
        faces: Vec<OperandEntity>,
        neutral: Frame,
        angle: f64,
    },
    /// Re-apply seed tools at each instance (tags carry `seed` and `instance`).
    PatternTools {
        seeds: Vec<ToolOp>,
        instances: Vec<Instance>,
    },
    /// Copy every operand at each instance; the originals are kept.
    PatternBodies {
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
    Transform {
        transform: Transform,
    },
    Import {
        source: String,
        digest: String,
    },
}

impl KernelOp {
    pub fn name(&self) -> &'static str {
        match self {
            KernelOp::Tool(_) => "tool",
            KernelOp::Fillet { .. } => "fillet",
            KernelOp::Chamfer { .. } => "chamfer",
            KernelOp::Shell { .. } => "shell",
            KernelOp::Draft { .. } => "draft",
            KernelOp::PatternTools { .. } => "pattern_tools",
            KernelOp::PatternBodies { .. } => "pattern_bodies",
            KernelOp::Boolean { .. } => "boolean",
            KernelOp::Split { .. } => "split",
            KernelOp::Transform { .. } => "transform",
            KernelOp::Import { .. } => "import",
        }
    }
}
