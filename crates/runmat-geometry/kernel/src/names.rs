//! The names a kernel echoes in history: roles, split sides, pattern instance keys and the
//! continuity a fill keeps.

use serde::{Deserialize, Serialize};
use std::fmt;

/// What a feature made an entity as. The role table is per feature family; two faces of one
/// feature never share a role and tag unless the feature numbers them with an ordinal from its
/// own definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// A face swept from a profile curve (extrude, revolve, sweep, loft). A sweep along a path
    /// with corners numbers the pieces by path segment, an ordinal from the definition. A solid
    /// loft makes one smooth face per curve through all its sections; a surface loft through
    /// more than two sections numbers its ruled pieces by span.
    Side,
    /// The cap at the profile plane (extrude, sweep start, revolve start).
    StartCap,
    /// The far cap.
    EndCap,
    /// A fillet face, tagged by the edge it replaced.
    Blend,
    /// A chamfer face, tagged by the edge it replaced.
    Chamfer,
    /// A shell's inner wall, tagged by the face it offsets.
    ShellInner,
    /// A hole's wall, tagged by the hole key.
    HoleWall,
    /// A blind hole's bottom.
    HoleBottom,
    /// A counterbore's floor.
    HoleCounterbore,
    /// A counterbore's wall.
    HoleCounterboreWall,
    /// A countersink's cone.
    HoleCountersink,
    /// The rim a shell leaves where it opened a face, tagged by the opened face.
    ShellRim,
    /// A shell's outer wall when shelled outward, tagged by the face it offsets.
    ShellOuter,
    /// A thin feature's inner wall, tagged by profile curve; ordinal 0 and 1 for the inner
    /// faces of its start and end caps.
    ThinWall,
    /// A step face a step draft leaves between faces hinged at different heights, tagged by
    /// the drafted face.
    DraftStep,
    /// A face made where a split tool cut a body, tagged by side.
    SplitFace,
    /// A body a feature made.
    Body,
    // ----------------------------------------------------------------- direct edits
    /// The face that replaced another (replace face), tagged by the face it replaced.
    Replacement,
    // ---------------------------------------------------------------------- surfaces
    /// A surface body's boundary edge, tagged by the profile curve it runs along or ends;
    /// ordinal 0 at the start cap, 1 at the end cap, 2 at the curve's start point, 3 at its
    /// end point (none: the curve itself, for a planar patch).
    Boundary,
    /// A patch a surface feature made (planar, boundary, fill), tagged by key or curve.
    SurfacePatch,
    /// An offset surface's face, tagged by the face it offsets.
    OffsetFace,
    /// The strip an extend added past a boundary edge, tagged by the edge's key.
    Extension,
    /// A thicken's far face, tagged by the face it offsets.
    ThickenFace,
    /// A thicken's rim along a boundary edge, tagged by the edge's key.
    ThickenRim,
    /// A mid-surface face, tagged by the face pair key.
    MidSurface,
    /// A ruled surface along an edge, tagged by the edge's key.
    Ruled,
    // ------------------------------------------------------------------ sheet metal
    /// A sheet's flat face, tagged by flange key; ordinal 0 is the reference side, 1 the
    /// opposite side.
    SheetFace,
    /// A sheet's thickness face (an edge of the sheet), tagged by flange key and edge.
    SheetEdge,
    /// A bend's cylindrical face, tagged by bend key; ordinal 0 inside, 1 outside.
    BendFace,
    /// A relief cut's wall, tagged by the relief key.
    Relief,
    /// A face a form tool made, tagged by tool face key.
    FormFace,
    // -------------------------------------------------------------------- weldments
    /// A structural member's side, tagged `member/profile-curve`.
    MemberSide,
    /// A structural member's end face; ordinal 0 start, 1 end.
    MemberEnd,
    /// A weldment end cap's face, tagged by cap key.
    CapPlate,
    /// A gusset's face, tagged by gusset key.
    Gusset,
    /// A weld bead's face, tagged by bead key.
    WeldBead,
    // ------------------------------------------------------------------------ mold
    /// A parting or shut-off face of a core/cavity split, tagged by key.
    PartingFace,
    // ---------------------------------------------------------------------- details
    /// A modelled thread's face, tagged by the face it threads.
    ThreadFace,
    /// The top face of an emboss or deboss, tagged by profile key.
    EmbossTop,
    /// A rib's face, tagged by profile curve.
    Rib,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Side => "side",
            Role::StartCap => "start",
            Role::EndCap => "end",
            Role::Blend => "blend",
            Role::Chamfer => "chamfer",
            Role::ShellInner => "inner",
            Role::HoleWall => "wall",
            Role::HoleBottom => "bottom",
            Role::HoleCounterbore => "counterbore",
            Role::HoleCounterboreWall => "counterbore_wall",
            Role::HoleCountersink => "countersink",
            Role::ShellRim => "rim",
            Role::ShellOuter => "outer",
            Role::ThinWall => "thin_wall",
            Role::DraftStep => "draft_step",
            Role::SplitFace => "split_face",
            Role::Body => "body",
            Role::Replacement => "replacement",
            Role::Boundary => "boundary",
            Role::SurfacePatch => "patch",
            Role::OffsetFace => "offset",
            Role::Extension => "extension",
            Role::ThickenFace => "thicken",
            Role::ThickenRim => "thicken_rim",
            Role::MidSurface => "mid_surface",
            Role::Ruled => "ruled",
            Role::SheetFace => "sheet",
            Role::SheetEdge => "sheet_edge",
            Role::BendFace => "bend",
            Role::Relief => "relief",
            Role::FormFace => "form",
            Role::MemberSide => "member_side",
            Role::MemberEnd => "member_end",
            Role::CapPlate => "cap_plate",
            Role::Gusset => "gusset",
            Role::WeldBead => "weld_bead",
            Role::PartingFace => "parting",
            Role::ThreadFace => "thread",
            Role::EmbossTop => "emboss_top",
            Role::Rib => "rib",
        }
    }
}

/// Which side of a split tool (by the tool's normal), from the feature's definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaneSide {
    Negative,
    Positive,
}

/// A pattern instance's semantic key. Never an array index into kernel output: linear and grid
/// keys are the step counts along the layout's directions, table keys are row keys.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "instance")]
pub enum InstanceKey {
    Linear {
        step: u32,
    },
    Grid {
        first: u32,
        second: u32,
    },
    Circular {
        step: u32,
    },
    Row {
        key: String,
    },
    /// The single image of a mirror.
    Mirror,
    /// A sketch-driven instance, by the sketch point it sits on.
    Point {
        point: String,
    },
    /// A curve-driven instance: the step along the curve.
    Curve {
        step: u32,
    },
    /// A fill instance: lattice steps from the seed along the fill's two directions.
    Lattice {
        i: i32,
        j: i32,
    },
    /// A circular fill instance: ring, then step around it.
    Ring {
        ring: u32,
        step: u32,
    },
}

impl fmt::Display for InstanceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstanceKey::Linear { step } => write!(f, "{}", step),
            InstanceKey::Grid { first, second } => write!(f, "{},{}", first, second),
            InstanceKey::Circular { step } => write!(f, "@{}", step),
            InstanceKey::Row { key } => write!(f, "{:?}", key),
            InstanceKey::Mirror => f.write_str("mirror"),
            InstanceKey::Point { point } => write!(f, "pt:{}", point),
            InstanceKey::Curve { step } => write!(f, "~{}", step),
            InstanceKey::Lattice { i, j } => write!(f, "[{},{}]", i, j),
            InstanceKey::Ring { ring, step } => write!(f, "r{}@{}", ring, step),
        }
    }
}

/// Continuity a fill or boundary surface keeps with the faces along an edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Continuity {
    /// Position only.
    #[default]
    G0,
    /// Tangent.
    G1,
    /// Curvature.
    G2,
}
