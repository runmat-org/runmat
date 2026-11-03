//! The history record: the contract every kernel operation must emit (K1 implements it).
//!
//! # Contract
//!
//! For one operation, the kernel returns its output bodies, each with a [`Topology`] summary,
//! plus one [`OpHistory`]:
//!
//! 1. **Every output face has exactly one [`FaceRecord`]**, listing every source it came from:
//!    faces of input operands it kept, modified, split from or merged ([`HistorySource::Input`]);
//!    what the operation itself generated, with a [`Role`] and a [`KernelTag`] taken from the
//!    operation's canonical inputs ([`HistorySource::Generated`]); or a transformed image of an
//!    input face ([`HistorySource::Copy`]).
//! 2. **Every output body has exactly one [`BodyRecord`]** with its sources the same way.
//! 3. **Every input face with no image is listed in `deleted_faces`**, and every input body with
//!    no image in `deleted_bodies`.
//! 4. **Tags and ordinals come from the operation's inputs, never from output order.** A
//!    profile side is tagged by the curve id passed in; a split side by the side of the tool
//!    normal; a tie between faces of one role gets an ordinal from the definition (seam pieces
//!    by curve parameter, caps by distance along the direction).
//! 5. **Topology adjacency is complete**: each edge names its two faces; each vertex names its
//!    incident edges. Edge and vertex names are relational and derived from it, so the kernel
//!    does not record edge history.
//! 6. **Entity ids are positions** in the output body's topology lists. Output order must be
//!    deterministic for identical inputs (the regeneration cache and the determinism gate rely
//!    on it), but nothing may depend on it for identity.
//! 7. **An operation that cannot report history fails** with
//!    [`KernelError::HistoryIncomplete`](crate::regen::kernel::KernelError). It never returns a
//!    body with guessed history.
//!
//! Geometry summaries ([`FaceGeom`], [`EdgeGeom`]) are informative: they feed sketch-on-face
//! frames, the explicit geometric fallback and the index. They never decide identity on their own.

use crate::ids::SketchEntityId;
use crate::refs::name::{InstanceKey, PlaneSide, Role};
use serde::{Deserialize, Serialize};

/// An entity's position in its body's topology list.
pub type EntityIndex = u32;

/// An entity of an operation's input: the operand's position in the input list, plus the
/// entity's index in that operand's topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OperandEntity {
    pub operand: u32,
    pub entity: EntityIndex,
}

/// Tags the kernel reports, all echoed from the canonical operation it was given.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "tag")]
pub enum KernelTag {
    None,
    /// Profile `profile` (index into the operation's profiles), curve `curve`.
    Curve {
        profile: u32,
        curve: SketchEntityId,
    },
    /// A key passed in (hole position key).
    Key {
        key: String,
    },
    Side {
        side: PlaneSide,
    },
    /// The input edge a blend or chamfer replaced.
    InputEdge {
        edge: OperandEntity,
    },
    /// The input face a shell wall offsets.
    InputFace {
        face: OperandEntity,
    },
    /// A name read from an imported CAD file.
    CadName {
        name: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum HistorySource {
    /// A face (or body, in a [`BodyRecord`]) of an input operand that this entity continues.
    Input { from: OperandEntity },
    /// Made by this operation.
    Generated {
        role: Role,
        tag: KernelTag,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ordinal: Option<u32>,
        /// For pattern and mirror operations over tools: which seed operation made it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<u32>,
        /// For pattern and mirror operations: which instance.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instance: Option<InstanceKey>,
    },
    /// A transformed image of an input entity (pattern or mirror of bodies).
    Copy {
        from: OperandEntity,
        instance: InstanceKey,
    },
}

/// How an output entity relates to its sources. Informative; naming derives everything from
/// `sources`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Evolution {
    /// Unchanged from one input entity.
    Kept,
    /// Changed, from one input entity.
    Modified,
    /// One of several pieces of one input entity.
    Split,
    /// From several input entities.
    Merged,
    /// Made by the operation.
    Generated,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceRecord {
    /// Output body index.
    pub body: u32,
    pub face: EntityIndex,
    pub evolution: Evolution,
    pub sources: Vec<HistorySource>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyRecord {
    pub body: u32,
    pub sources: Vec<HistorySource>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpHistory {
    pub faces: Vec<FaceRecord>,
    pub bodies: Vec<BodyRecord>,
    #[serde(default)]
    pub deleted_faces: Vec<OperandEntity>,
    #[serde(default)]
    pub deleted_bodies: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "surface")]
pub enum SurfaceKind {
    Plane {
        normal: [f64; 3],
    },
    Cylinder {
        axis_point: [f64; 3],
        axis_direction: [f64; 3],
        radius: f64,
    },
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceGeom {
    pub surface: SurfaceKind,
    pub centroid: [f64; 3],
    pub area: f64,
    pub bbox_min: [f64; 3],
    pub bbox_max: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeGeom {
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub length: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceTopo {
    pub geom: FaceGeom,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeTopo {
    pub faces: [EntityIndex; 2],
    pub geom: EdgeGeom,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VertexTopo {
    pub edges: Vec<EntityIndex>,
    pub point: [f64; 3],
}

/// A body's topology summary: faces, edges with their two faces, vertices with their edges.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Topology {
    pub faces: Vec<FaceTopo>,
    pub edges: Vec<EdgeTopo>,
    pub vertices: Vec<VertexTopo>,
}

impl Topology {
    /// Faces sharing an edge with `face`.
    pub fn face_neighbours(&self, face: EntityIndex) -> Vec<EntityIndex> {
        let mut out: Vec<EntityIndex> = self
            .edges
            .iter()
            .filter_map(|e| {
                if e.faces[0] == face {
                    Some(e.faces[1])
                } else if e.faces[1] == face {
                    Some(e.faces[0])
                } else {
                    None
                }
            })
            .filter(|f| *f != face)
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// Vertices at the ends of `edge`.
    pub fn edge_vertices(&self, edge: EntityIndex) -> Vec<EntityIndex> {
        self.vertices
            .iter()
            .enumerate()
            .filter(|(_, v)| v.edges.contains(&edge))
            .map(|(i, _)| i as EntityIndex)
            .collect()
    }

    /// Faces around a vertex.
    pub fn vertex_faces(&self, vertex: EntityIndex) -> Vec<EntityIndex> {
        let mut out: Vec<EntityIndex> = self.vertices[vertex as usize]
            .edges
            .iter()
            .flat_map(|e| self.edges[*e as usize].faces)
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// Faces touching the ends of `edge`, other than its own two faces.
    pub fn edge_neighbour_faces(&self, edge: EntityIndex) -> Vec<EntityIndex> {
        let own = self.edges[edge as usize].faces;
        let mut out: Vec<EntityIndex> = self
            .edge_vertices(edge)
            .into_iter()
            .flat_map(|v| self.vertex_faces(v))
            .filter(|f| !own.contains(f))
            .collect();
        out.sort();
        out.dedup();
        out
    }
}
