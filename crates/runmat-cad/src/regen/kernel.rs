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
    /// Counter-clockwise about `center`.
    Arc {
        center: [f64; 2],
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
    Sweep {
        profile: CanonicalProfile,
        path: Vec<[f64; 3]>,
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
