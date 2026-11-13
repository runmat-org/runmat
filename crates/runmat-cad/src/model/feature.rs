//! The feature list: typed parameters plus the references each feature makes.

use crate::ids::{FeatureId, SketchEntityId};
use crate::model::curves::{CurveDef, MateConnectorDef};
use crate::model::datum::{AxisDef, AxisRef, FrameDef, PlaneDef, PlaneRef, PointDef, PointRef};
use crate::model::detail::{Emboss, LibraryInstance, Rib, Thread};
use crate::model::direct::DirectEdit;
use crate::model::mold::MoldFeature;
use crate::model::sheet_metal::SheetMetalFeature;
use crate::model::sketch::Sketch;
use crate::model::surface::SurfaceFeature;
use crate::model::visit::{Params, Visit};
use crate::model::weldment::WeldmentFeature;
use crate::refs::{InstanceKey, Reference};
use crate::regen::hash::CacheKey;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Feature {
    pub id: FeatureId,
    pub kind: FeatureKind,
    #[serde(default)]
    pub suppressed: bool,
    /// A frozen feature keeps the cached result with this key, whatever changes upstream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frozen: Option<CacheKey>,
}

impl Feature {
    pub fn new(id: &str, kind: FeatureKind) -> Self {
        Self {
            id: FeatureId::new(id),
            kind,
            suppressed: false,
            frozen: None,
        }
    }
}

/// How a body-producing feature combines with existing bodies.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum BodyMode {
    /// A new body.
    #[default]
    New,
    /// Union with `targets` (empty: every body at this point in the history).
    Add {
        #[serde(default)]
        targets: Vec<Reference>,
    },
    Cut {
        #[serde(default)]
        targets: Vec<Reference>,
    },
    Intersect {
        #[serde(default)]
        targets: Vec<Reference>,
    },
}

impl BodyMode {
    pub fn targets(&self) -> &[Reference] {
        match self {
            BodyMode::New => &[],
            BodyMode::Add { targets }
            | BodyMode::Cut { targets }
            | BodyMode::Intersect { targets } => targets,
        }
    }
    pub fn add() -> Self {
        BodyMode::Add {
            targets: Vec::new(),
        }
    }
    pub fn cut() -> Self {
        BodyMode::Cut {
            targets: Vec::new(),
        }
    }
}

/// Which closed loops of a sketch a feature uses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProfileRef {
    pub sketch: FeatureId,
    /// Each loop listed by its curves. Empty: every closed loop of the sketch.
    #[serde(default)]
    pub loops: Vec<Vec<SketchEntityId>>,
    /// Merge the side faces of collinear profile curves into one face. Off by default: each
    /// sketch curve makes its own face.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub merge_collinear: bool,
}

impl ProfileRef {
    pub fn sketch(id: &str) -> Self {
        Self {
            sketch: FeatureId::new(id),
            loops: Vec::new(),
            merge_collinear: false,
        }
    }
}

/// What extrude, revolve and sweep shape: closed loops of a sketch (`sketch.s1`), or a planar
/// face of the part (`feature.base.face.end`), its outer boundary with its holes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProfileSource {
    Sketch(ProfileRef),
    Face(Reference),
}

impl ProfileSource {
    /// The sketch profile, when it is one.
    pub fn sketch(&self) -> Option<&ProfileRef> {
        match self {
            ProfileSource::Sketch(p) => Some(p),
            ProfileSource::Face(_) => None,
        }
    }
}

impl From<ProfileRef> for ProfileSource {
    fn from(p: ProfileRef) -> Self {
        ProfileSource::Sketch(p)
    }
}

/// An extrude's end condition in one direction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "extent")]
pub enum Extent {
    Blind {
        distance: ValueExpr,
    },
    /// Mid-plane: both directions, half each way. Not valid as a second direction.
    #[serde(alias = "mid_plane")]
    Symmetric {
        distance: ValueExpr,
    },
    ThroughAll,
    /// Up to a face: the end cap takes the face's shape (planar or curved).
    #[serde(alias = "up_to_face")]
    UpTo {
        face: Reference,
    },
    /// Up to the next face the extrude meets past its start, which must intercept the whole
    /// profile.
    UpToNext,
    /// Up to a plane (a datum plane or a planar face) parallel to the profile plane.
    UpToPlane {
        plane: PlaneRef,
    },
    /// Up to the plane through a vertex (or point), parallel to the profile plane.
    UpToVertex {
        vertex: PointRef,
    },
    /// Up to a face, `offset` short of it (towards the start) unless `beyond`.
    OffsetFromFace {
        face: Reference,
        offset: ValueExpr,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        beyond: bool,
    },
    /// Up to where the extrude first meets a body.
    UpToBody {
        body: Reference,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<ValueExpr>,
    },
}

/// Where an extrude starts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "from")]
pub enum ExtrudeFrom {
    #[default]
    SketchPlane,
    /// A plane offset from the sketch plane along its normal.
    Offset { distance: ValueExpr },
    /// A face (planar or curved): the start cap takes its shape and blind depths are measured
    /// from it.
    Surface { face: Reference },
    /// The plane through a vertex (or point), parallel to the sketch plane.
    Vertex { vertex: PointRef },
}

impl ExtrudeFrom {
    pub fn is_sketch_plane(&self) -> bool {
        matches!(self, ExtrudeFrom::SketchPlane)
    }
}

/// Draft on an extrude's side faces, per direction. Positive angles taper the profile inward
/// moving away from the start; `outward` flips it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtrudeDraft {
    pub angle: ValueExpr,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub outward: bool,
    /// The second direction's draft (`None`: the same angle and sense).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second_angle: Option<ValueExpr>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub second_outward: bool,
}

/// Which side of the profile curves a thin feature's wall lies on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinSide {
    /// Inside the profile (into the region it bounds).
    #[default]
    One,
    /// Outside the profile.
    Other,
    /// Centred on the profile curves.
    Mid,
    /// `thickness` inside and `second_thickness` outside.
    Two,
}

/// A thin feature: a wall along the profile curves instead of a filled profile.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThinWall {
    pub thickness: ValueExpr,
    #[serde(default)]
    pub side: ThinSide,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second_thickness: Option<ValueExpr>,
    /// Close both ends with caps this thick (a hollow closed profile).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cap_ends: Option<ValueExpr>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Extrude {
    pub profile: ProfileSource,
    pub extent: Extent,
    /// Extrude against the sketch normal.
    #[serde(default)]
    pub reverse: bool,
    #[serde(default)]
    pub mode: BodyMode,
    #[serde(default, skip_serializing_if = "ExtrudeFrom::is_sketch_plane")]
    pub from: ExtrudeFrom,
    /// The second direction (opposite the first), with its own end condition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second: Option<Extent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft: Option<ExtrudeDraft>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thin: Option<ThinWall>,
}

impl Extrude {
    /// A plain extrude: from the sketch plane, one direction, no draft, not thin.
    pub fn new(profile: impl Into<ProfileSource>, extent: Extent, mode: BodyMode) -> Self {
        Extrude {
            profile: profile.into(),
            extent,
            reverse: false,
            mode,
            from: ExtrudeFrom::SketchPlane,
            second: None,
            draft: None,
            thin: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Revolve {
    pub profile: ProfileSource,
    pub axis: AxisRef,
    pub angle: ValueExpr,
    #[serde(default)]
    pub mode: BodyMode,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sweep {
    pub profile: ProfileSource,
    /// The path: a sketch's open chain.
    pub path: ProfileRef,
    #[serde(default)]
    pub mode: BodyMode,
    /// The profile keeps its angle to the path, turning through its bends and corners (a
    /// pipe or cable run). Without it the profile stays parallel to itself.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub follow_path: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Loft {
    pub profiles: Vec<ProfileRef>,
    #[serde(default)]
    pub guides: Vec<Reference>,
    #[serde(default)]
    pub mode: BodyMode,
}

/// A hole's form. Sizes left out come from the hole's standard [`HoleSpec`] (clearance,
/// counterbore and countersink for the fastener, tap drill and thread for tapped holes).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "hole")]
pub enum HoleForm {
    Simple,
    Counterbore {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        diameter: Option<ValueExpr>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        depth: Option<ValueExpr>,
    },
    Countersink {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        diameter: Option<ValueExpr>,
        /// Included angle.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        angle: Option<ValueExpr>,
    },
    /// A tapped hole: drilled at the tap drill, threaded for `thread_depth` (the hole's depth
    /// less one and a half pitches when blind, the full hole when through).
    Tapped {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thread_depth: Option<ValueExpr>,
        /// Model the thread (the threaded wall becomes its own face); otherwise cosmetic.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        modelled: bool,
        /// A near-side countersink on the tapped hole (included angle; diameter from the
        /// major diameter when left out).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        near_countersink: Option<ValueExpr>,
    },
    /// A tapered pipe tap (NPT/NPTF 1:16 on diameter, BSPT): the drill tapers at the thread
    /// taper over the thread length.
    PipeTap {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thread_depth: Option<ValueExpr>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        modelled: bool,
    },
    /// A slot of the hole's width, `length` centre to centre, along the face frame's x turned
    /// by `angle`; optionally counterbored.
    Slot {
        length: ValueExpr,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        angle: Option<ValueExpr>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        counterbore: Option<(ValueExpr, ValueExpr)>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HolePosition {
    /// The semantic key for this hole (instance identity; never an array index).
    pub key: String,
    pub at: PointRef,
}

/// How deep a hole goes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "end")]
pub enum HoleEnd {
    /// To a drilled depth (the full-diameter depth; a drill point adds its cone below).
    Blind {
        depth: ValueExpr,
    },
    ThroughAll,
    /// Through the material the hole starts in, to the next face it meets.
    UpToNext,
    /// To a face (planar or curved), with an offset past it.
    UpToFace {
        face: Reference,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<ValueExpr>,
    },
}

/// A hole through several bodies of the part (a hole series): the start form on the first
/// body, clearance through the middle ones and the end form on the last.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoleSeries {
    /// The bodies in order from the start face. Empty: every body the hole axis crosses, in
    /// order along it.
    #[serde(default)]
    pub bodies: Vec<Reference>,
    /// The form in the last body (tapped for a bolted stack, or the same clearance).
    pub end: HoleForm,
    /// The middle bodies' diameter (`None`: the start clearance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub middle_diameter: Option<ValueExpr>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hole {
    /// The planar face the holes start on.
    pub face: Reference,
    pub positions: Vec<HolePosition>,
    /// The drilled diameter (`None`: from `spec`: clearance, tap drill or pipe tap drill).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diameter: Option<ValueExpr>,
    pub end: HoleEnd,
    #[serde(default = "simple_hole")]
    pub form: HoleForm,
    /// The standard size the hole is for (`M6`, `1/4-20`, `1/8 NPT`) and its fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<crate::model::hole_tables::HoleSpec>,
    /// The drill point's included angle for blind holes (`None`: a flat bottom).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drill_point: Option<ValueExpr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub series: Option<HoleSeries>,
}

fn simple_hole() -> HoleForm {
    HoleForm::Simple
}

/// The cross-section of a blend.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "profile")]
pub enum FilletProfile {
    #[default]
    Circular,
    /// A conic section of shape `rho` (0.5 a parabola, below an ellipse, above a hyperbola).
    Conic { rho: ValueExpr },
    /// Curvature-continuous (G2) to both faces.
    Curvature,
}

impl FilletProfile {
    fn is_circular(&self) -> bool {
        matches!(self, FilletProfile::Circular)
    }
}

/// A radius at a point along each filleted edge (`at` from 0 at its start to 1 at its end).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RadiusAt {
    pub at: ValueExpr,
    pub radius: ValueExpr,
}

/// A setback at a vertex where filleted edges meet: the blends stop `distance` short of it and
/// a corner blend joins them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Setback {
    pub vertex: Reference,
    pub distance: ValueExpr,
}

/// Edge fillet: constant or variable radius, any profile, with setbacks at corners.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fillet {
    pub edges: Vec<Reference>,
    /// The radius; with `variable`, the radius at both ends of each edge where no point gives
    /// one.
    pub radius: ValueExpr,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variable: Vec<RadiusAt>,
    #[serde(default, skip_serializing_if = "FilletProfile::is_circular")]
    pub profile: FilletProfile,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub setbacks: Vec<Setback>,
    /// Extend the selection along tangent-continuous edges.
    #[serde(default = "yes", skip_serializing_if = "Clone::clone")]
    pub propagate: bool,
}

fn yes() -> bool {
    true
}

impl Fillet {
    pub fn constant(edges: Vec<Reference>, radius: ValueExpr) -> Self {
        Fillet {
            edges,
            radius,
            variable: Vec::new(),
            profile: FilletProfile::Circular,
            setbacks: Vec::new(),
            propagate: true,
        }
    }
}

/// Face fillet: a blend tangent to two face sets, consuming what lies between them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceFillet {
    pub first: Vec<Reference>,
    pub second: Vec<Reference>,
    pub radius: ValueExpr,
    #[serde(default, skip_serializing_if = "FilletProfile::is_circular")]
    pub profile: FilletProfile,
}

/// Full round: the centre faces are replaced by a blend tangent to both side face sets; its
/// radius follows from the gap.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FullRound {
    pub first: Vec<Reference>,
    pub center: Vec<Reference>,
    pub second: Vec<Reference>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "chamfer")]
pub enum ChamferForm {
    /// The same setback on both faces.
    Equal { distance: ValueExpr },
    /// `distance` on the first face of each edge, `distance2` on the second.
    Distances {
        distance: ValueExpr,
        distance2: ValueExpr,
    },
    /// `distance` on the first face, at `angle` from it.
    DistanceAngle {
        distance: ValueExpr,
        angle: ValueExpr,
    },
    /// Vertex chamfer: a corner cut through the points at these distances along each of the
    /// vertex's three edges (in the vertex's edge order).
    Vertex { distances: [ValueExpr; 3] },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chamfer {
    /// The edges chamfered (for a vertex chamfer, the vertices).
    pub edges: Vec<Reference>,
    pub form: ChamferForm,
    /// Swap the faces the distances (and angle) are measured on.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub flip: bool,
}

impl Chamfer {
    pub fn equal(edges: Vec<Reference>, distance: ValueExpr) -> Self {
        Chamfer {
            edges,
            form: ChamferForm::Equal { distance },
            flip: false,
        }
    }
}

/// Faces of a shell with their own wall thickness.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceThickness {
    pub faces: Vec<Reference>,
    pub thickness: ValueExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shell {
    pub body: Reference,
    /// Faces removed to open the shell.
    #[serde(default)]
    pub open_faces: Vec<Reference>,
    pub thickness: ValueExpr,
    /// Walls with their own thickness (multi-thickness shell).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub faces: Vec<FaceThickness>,
    /// Grow the walls outward: the body becomes the cavity.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub outward: bool,
}

impl Shell {
    pub fn new(body: Reference, open_faces: Vec<Reference>, thickness: ValueExpr) -> Self {
        Shell {
            body,
            open_faces,
            thickness,
            faces: Vec::new(),
            outward: false,
        }
    }
}

/// How drafted faces hinge.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "draft")]
pub enum DraftKind {
    /// About their line on the neutral plane.
    #[default]
    NeutralPlane,
    /// About the parting-line edges each face touches; the neutral plane gives only the pull.
    PartingLine { edges: Vec<Reference> },
    /// As a parting-line draft, with step faces where neighbouring faces hinge at different
    /// heights, so the parting line need not be planar.
    Step { edges: Vec<Reference> },
}

impl DraftKind {
    fn is_neutral(&self) -> bool {
        matches!(self, DraftKind::NeutralPlane)
    }
    pub fn edges(&self) -> &[Reference] {
        match self {
            DraftKind::NeutralPlane => &[],
            DraftKind::PartingLine { edges } | DraftKind::Step { edges } => edges,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Draft {
    pub faces: Vec<Reference>,
    /// The neutral plane; its normal is the pull direction.
    pub neutral: PlaneRef,
    pub angle: ValueExpr,
    #[serde(default, skip_serializing_if = "DraftKind::is_neutral")]
    pub kind: DraftKind,
    /// Pull the other way.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub flip: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "seeds")]
pub enum PatternSeeds {
    /// Re-apply these features' tools at each instance.
    Features { features: Vec<FeatureId> },
    /// Copy these bodies at each instance.
    Bodies { bodies: Vec<Reference> },
    /// Repeat the material these faces of one body bound (a boss or a pocket) at each instance.
    Faces { faces: Vec<Reference> },
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "layout")]
pub enum PatternLayout {
    /// Instances `1..count` along `direction` (the seed is instance 0).
    Linear {
        direction: AxisRef,
        spacing: ValueExpr,
        count: ValueExpr,
    },
    /// Instances `(i, j)` over two directions.
    Grid {
        first: AxisRef,
        first_spacing: ValueExpr,
        first_count: ValueExpr,
        second: AxisRef,
        second_spacing: ValueExpr,
        second_count: ValueExpr,
    },
    /// Instances `1..count` about `axis`: equally spaced over `angle`, or `angle` apart when
    /// not `equal_spacing`.
    Circular {
        axis: AxisRef,
        count: ValueExpr,
        angle: ValueExpr,
        #[serde(default = "yes", skip_serializing_if = "Clone::clone")]
        equal_spacing: bool,
    },
    /// Instances keyed by table row: each row is a key plus an offset.
    Table { rows: Vec<PatternRow> },
    /// Instances `1..count` along a curve from its start (the seed sits at the start): equally
    /// spaced over its length, or `spacing` apart.
    Curve {
        curve: crate::model::datum::CurveRef,
        count: ValueExpr,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        spacing: Option<ValueExpr>,
        #[serde(default)]
        orientation: CurveOrientation,
    },
    /// One instance per point of a sketch (all its point entities, or `points`), placed by
    /// the offset from `reference` (default: the first point, where the seed is).
    Sketch {
        sketch: FeatureId,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        points: Vec<SketchEntityId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reference: Option<SketchEntityId>,
    },
    /// Fill a region (a sketch's closed loops) with instances on a lattice through the seed:
    /// square, staggered (perforation) or concentric rings, kept `margin` inside the boundary.
    Fill {
        boundary: ProfileRef,
        spacing: ValueExpr,
        #[serde(default)]
        lattice: FillLattice,
        /// The lattice's first direction (default: the boundary sketch's x).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        direction: Option<AxisRef>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        margin: Option<ValueExpr>,
        /// Where the seed sits (default: the boundary's centroid).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<PointRef>,
    },
    /// Instances each with their own placement and their own values: the seed features are
    /// rebuilt per instance with these inputs replaced (a varying depth, a varying size).
    Variable { instances: Vec<VariableInstance> },
}

/// How curve-driven instances turn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurveOrientation {
    /// Translated only.
    #[default]
    Translate,
    /// Turned with the curve (a rotation-minimizing frame along it).
    AlignToCurve,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FillLattice {
    #[default]
    Square,
    /// Rows offset by half a pitch, `spacing` between neighbours (60° perforation).
    Staggered,
    /// Rings `spacing` apart about the seed, instances `spacing` apart on each ring.
    Circular,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariableInstance {
    pub key: String,
    pub dx: ValueExpr,
    pub dy: ValueExpr,
    pub dz: ValueExpr,
    /// Inputs read by the seed (and its sketches) replaced for this instance.
    #[serde(default)]
    pub values: std::collections::BTreeMap<crate::ids::InputName, ValueExpr>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternRow {
    pub key: String,
    pub dx: ValueExpr,
    pub dy: ValueExpr,
    pub dz: ValueExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pattern {
    pub seeds: PatternSeeds,
    pub layout: PatternLayout,
    /// Instances left out, by key.
    #[serde(default)]
    pub skip: Vec<InstanceKey>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mirror {
    pub seeds: PatternSeeds,
    pub plane: PlaneRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BooleanKind {
    Union,
    Subtract,
    Intersect,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Boolean {
    pub kind: BooleanKind,
    pub target: Reference,
    pub tools: Vec<Reference>,
    #[serde(default)]
    pub keep_tools: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Split {
    pub body: Reference,
    pub tool: PlaneRef,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum BodyOp {
    Delete {
        bodies: Vec<Reference>,
    },
    Translate {
        bodies: Vec<Reference>,
        dx: ValueExpr,
        dy: ValueExpr,
        dz: ValueExpr,
    },
    /// Turn bodies about an axis.
    Rotate {
        bodies: Vec<Reference>,
        axis: AxisRef,
        angle: ValueExpr,
    },
    /// Copies of bodies, the originals kept: copy `k` (1 to `count`) is moved `k` times by the
    /// offset, then turned `k` times by `angle` about `axis`.
    Copy {
        bodies: Vec<Reference>,
        dx: ValueExpr,
        dy: ValueExpr,
        dz: ValueExpr,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        axis: Option<AxisRef>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        angle: Option<ValueExpr>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        count: Option<ValueExpr>,
    },
    /// Press `tool` into `target`: the tool, grown by `clearance`, is cut from the target and
    /// the tool body is kept.
    Indent {
        target: Reference,
        #[serde(rename = "tool_body")]
        tool: Reference,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        clearance: Option<ValueExpr>,
    },
}

/// Imported CAD geometry. Its names come from the CAD file; there is no sidecar.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Import {
    /// Project-relative path to the STEP (or other CAD) file, the source of truth.
    pub source: String,
    /// The file's content digest, which joins the cache key.
    pub digest: String,
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "feature")]
pub enum FeatureKind {
    Sketch(Sketch),
    Plane(PlaneDef),
    Axis(AxisDef),
    Point(PointDef),
    Frame(FrameDef),
    Extrude(Extrude),
    Revolve(Revolve),
    Sweep(Sweep),
    Loft(Loft),
    Hole(Hole),
    Fillet(Fillet),
    FaceFillet(FaceFillet),
    FullRound(FullRound),
    Chamfer(Chamfer),
    Shell(Shell),
    Draft(Draft),
    Pattern(Pattern),
    Mirror(Mirror),
    Boolean(Boolean),
    Split(Split),
    BodyOp(BodyOp),
    Import(Import),
    Direct(DirectEdit),
    Surface(SurfaceFeature),
    SheetMetal(SheetMetalFeature),
    Weldment(WeldmentFeature),
    Mold(MoldFeature),
    Thread(Thread),
    Emboss(Emboss),
    Rib(Rib),
    Library(LibraryInstance),
    Curve(CurveDef),
    MateConnector(MateConnectorDef),
}

impl FeatureKind {
    /// The parameters of every kind that reports them through [`Params`] (all but sketches,
    /// the original datum definitions and the plain body operations).
    pub fn family(&self) -> Option<&dyn Params> {
        Some(match self {
            FeatureKind::Extrude(x) => x,
            FeatureKind::Hole(x) => x,
            FeatureKind::Fillet(x) => x,
            FeatureKind::FaceFillet(x) => x,
            FeatureKind::FullRound(x) => x,
            FeatureKind::Chamfer(x) => x,
            FeatureKind::Shell(x) => x,
            FeatureKind::Draft(x) => x,
            FeatureKind::Pattern(x) => x,
            FeatureKind::Direct(x) => x,
            FeatureKind::Surface(x) => x,
            FeatureKind::SheetMetal(x) => x,
            FeatureKind::Weldment(x) => x,
            FeatureKind::Mold(x) => x,
            FeatureKind::Thread(x) => x,
            FeatureKind::Emboss(x) => x,
            FeatureKind::Rib(x) => x,
            FeatureKind::Library(x) => x,
            FeatureKind::Curve(x) => x,
            FeatureKind::MateConnector(x) => x,
            _ => return None,
        })
    }

    fn family_visit(&self) -> Option<Visit<'_>> {
        let f = self.family()?;
        let mut v = Visit::default();
        f.visit(&mut v);
        Some(v)
    }

    /// The datum-definition variants added with reference geometry.
    fn datum_visit(&self) -> Option<Visit<'_>> {
        let mut v = Visit::default();
        match self {
            FeatureKind::Plane(PlaneDef::ThreePoint { a, b, c }) => {
                v.point("a", a);
                v.point("b", b);
                v.point("c", c);
            }
            FeatureKind::Plane(PlaneDef::Mid { a, b }) => {
                v.plane("a", a);
                v.plane("b", b);
            }
            FeatureKind::Plane(PlaneDef::NormalToCurve { curve, parameter }) => {
                curve.visit(&mut v, "curve");
                v.value(parameter);
            }
            FeatureKind::Plane(PlaneDef::TangentToFace { face, at }) => {
                v.reference("face", face);
                v.point("at", at);
            }
            FeatureKind::Axis(AxisDef::TwoPlanes { a, b }) => {
                v.plane("a", a);
                v.plane("b", b);
            }
            FeatureKind::Axis(AxisDef::Normal { plane, through }) => {
                v.plane("plane", plane);
                v.point("through", through);
            }
            FeatureKind::Point(PointDef::CenterOf { of }) => v.reference("of", of),
            FeatureKind::Point(PointDef::OnCurve { curve, parameter }) => {
                curve.visit(&mut v, "curve");
                v.value(parameter);
            }
            FeatureKind::Point(PointDef::Intersection { axis, plane }) => {
                v.axis("axis", axis);
                v.plane("plane", plane);
            }
            _ => return None,
        }
        Some(v)
    }
}

impl FeatureKind {
    pub fn name(&self) -> &'static str {
        match self {
            FeatureKind::Sketch(_) => "sketch",
            FeatureKind::Plane(_) => "plane",
            FeatureKind::Axis(_) => "axis",
            FeatureKind::Point(_) => "point",
            FeatureKind::Frame(_) => "frame",
            FeatureKind::Extrude(_) => "extrude",
            FeatureKind::Revolve(_) => "revolve",
            FeatureKind::Sweep(_) => "sweep",
            FeatureKind::Loft(_) => "loft",
            FeatureKind::Hole(_) => "hole",
            FeatureKind::Fillet(_) => "fillet",
            FeatureKind::FaceFillet(_) => "face_fillet",
            FeatureKind::FullRound(_) => "full_round",
            FeatureKind::Chamfer(_) => "chamfer",
            FeatureKind::Shell(_) => "shell",
            FeatureKind::Draft(_) => "draft",
            FeatureKind::Pattern(_) => "pattern",
            FeatureKind::Mirror(_) => "mirror",
            FeatureKind::Boolean(_) => "boolean",
            FeatureKind::Split(_) => "split",
            FeatureKind::BodyOp(_) => "body_op",
            FeatureKind::Import(_) => "import",
            other => other.family().map(|f| f.kind_name()).unwrap_or("feature"),
        }
    }

    /// Whether the feature produces or changes bodies (as opposed to sketches and datums).
    pub fn is_body_feature(&self) -> bool {
        if let Some(f) = self.family() {
            return f.body_feature();
        }
        !matches!(
            self,
            FeatureKind::Sketch(_)
                | FeatureKind::Plane(_)
                | FeatureKind::Axis(_)
                | FeatureKind::Point(_)
                | FeatureKind::Frame(_)
        )
    }

    /// Every value expression in the feature's parameters, for dependency analysis.
    pub fn values(&self) -> Vec<&ValueExpr> {
        if let Some(v) = self.family_visit().or_else(|| self.datum_visit()) {
            return v.values;
        }
        fn point<'a>(out: &mut Vec<&'a ValueExpr>, p: &'a PointRef) {
            if let PointRef::Coords { x, y, z } = p {
                out.push(x);
                out.push(y);
                out.push(z);
            }
        }
        fn plane_def<'a>(out: &mut Vec<&'a ValueExpr>, p: &'a PlaneDef) {
            match p {
                PlaneDef::Offset { distance, .. } => out.push(distance),
                PlaneDef::Angle { angle, .. } => out.push(angle),
                _ => {}
            }
        }
        let mut ptrs: Vec<&ValueExpr> = Vec::new();
        match self {
            FeatureKind::Sketch(s) => {
                use crate::model::sketch::{
                    PointExpr, SketchConstraintKind as K, SketchGeometry as G,
                };
                fn pe<'a>(out: &mut Vec<&'a ValueExpr>, p: &'a PointExpr) {
                    if let PointExpr::Coords { x, y } = p {
                        out.push(x);
                        out.push(y);
                    }
                }
                for e in &s.entities {
                    match &e.geometry {
                        G::Point { at } => pe(&mut ptrs, at),
                        G::Line { start, end } => {
                            pe(&mut ptrs, start);
                            pe(&mut ptrs, end);
                        }
                        G::Arc { center, start, end } => {
                            pe(&mut ptrs, center);
                            pe(&mut ptrs, start);
                            pe(&mut ptrs, end);
                        }
                        G::Circle { center, radius } => {
                            pe(&mut ptrs, center);
                            ptrs.push(radius);
                        }
                        G::Spline { points, .. } => points.iter().for_each(|p| pe(&mut ptrs, p)),
                    }
                }
                for c in &s.constraints {
                    match &c.kind {
                        K::Distance { value, .. }
                        | K::Length { value, .. }
                        | K::Radius { value, .. }
                        | K::Diameter { value, .. }
                        | K::Angle { value, .. } => ptrs.push(value),
                        K::Limit { min, max, .. } => {
                            ptrs.push(min);
                            ptrs.push(max);
                        }
                        _ => {}
                    }
                }
            }
            FeatureKind::Plane(p) => plane_def(&mut ptrs, p),
            FeatureKind::Axis(AxisDef::Through { a, b }) => {
                point(&mut ptrs, a);
                point(&mut ptrs, b);
            }
            FeatureKind::Point(p) => match p {
                PointDef::At { at } => point(&mut ptrs, at),
                PointDef::Offset { base, dx, dy, dz } => {
                    point(&mut ptrs, base);
                    ptrs.push(dx);
                    ptrs.push(dy);
                    ptrs.push(dz);
                }
                _ => {}
            },
            FeatureKind::Frame(FrameDef::At { origin, .. }) => point(&mut ptrs, origin),
            FeatureKind::Revolve(r) => ptrs.push(&r.angle),
            FeatureKind::Sweep(_) | FeatureKind::Loft(_) | FeatureKind::Import(_) => {}
            FeatureKind::Mirror(_) | FeatureKind::Boolean(_) | FeatureKind::Split(_) => {}
            FeatureKind::BodyOp(BodyOp::Translate { dx, dy, dz, .. }) => {
                ptrs.push(dx);
                ptrs.push(dy);
                ptrs.push(dz);
            }
            FeatureKind::BodyOp(BodyOp::Rotate { angle, .. }) => ptrs.push(angle),
            FeatureKind::BodyOp(BodyOp::Copy {
                dx,
                dy,
                dz,
                angle,
                count,
                ..
            }) => {
                ptrs.push(dx);
                ptrs.push(dy);
                ptrs.push(dz);
                ptrs.extend(angle.iter());
                ptrs.extend(count.iter());
            }
            FeatureKind::BodyOp(BodyOp::Indent { clearance, .. }) => ptrs.extend(clearance.iter()),
            _ => {}
        }
        ptrs
    }

    /// Every topological reference the feature makes, with the parameter it sits in.
    pub fn references(&self) -> Vec<(String, &Reference)> {
        if let Some(v) = self.family_visit().or_else(|| self.datum_visit()) {
            return v.references;
        }
        let mut out: Vec<(String, &Reference)> = Vec::new();
        fn list<'a>(out: &mut Vec<(String, &'a Reference)>, name: &str, refs: &'a [Reference]) {
            for (i, r) in refs.iter().enumerate() {
                out.push((format!("{}[{}]", name, i), r));
            }
        }
        fn plane<'a>(out: &mut Vec<(String, &'a Reference)>, name: &str, p: &'a PlaneRef) {
            if let PlaneRef::Face { face } = p {
                out.push((name.to_owned(), face));
            }
        }
        fn axis<'a>(out: &mut Vec<(String, &'a Reference)>, name: &str, a: &'a AxisRef) {
            if let AxisRef::Edge { edge: r } | AxisRef::Face { face: r } = a {
                out.push((name.to_owned(), r));
            }
        }
        fn point<'a>(out: &mut Vec<(String, &'a Reference)>, name: &str, p: &'a PointRef) {
            if let PointRef::Vertex { vertex } = p {
                out.push((name.to_owned(), vertex));
            }
        }
        fn mode<'a>(out: &mut Vec<(String, &'a Reference)>, m: &'a BodyMode) {
            list(out, "targets", m.targets());
        }
        match self {
            FeatureKind::Sketch(s) => plane(&mut out, "on", &s.plane),
            FeatureKind::Plane(p) => match p {
                PlaneDef::Offset { base, .. } => plane(&mut out, "base", base),
                PlaneDef::Angle { base, axis: a, .. } => {
                    plane(&mut out, "base", base);
                    axis(&mut out, "axis", a);
                }
                _ => {}
            },
            FeatureKind::Axis(a) => match a {
                AxisDef::Through { a, b } => {
                    point(&mut out, "a", a);
                    point(&mut out, "b", b);
                }
                AxisDef::OfFace { face } => out.push(("face".into(), face)),
                AxisDef::OfEdge { edge } => out.push(("edge".into(), edge)),
                _ => {}
            },
            FeatureKind::Point(p) => match p {
                PointDef::At { at } => point(&mut out, "at", at),
                PointDef::Offset { base, .. } => point(&mut out, "base", base),
                _ => {}
            },
            FeatureKind::Frame(f) => match f {
                FrameDef::At { origin, z, x } => {
                    point(&mut out, "origin", origin);
                    axis(&mut out, "z", z);
                    if let Some(x) = x {
                        axis(&mut out, "x", x);
                    }
                }
                FrameDef::OnFace { face } => out.push(("face".into(), face)),
            },
            FeatureKind::Revolve(r) => {
                if let ProfileSource::Face(face) = &r.profile {
                    out.push(("profile".into(), face));
                }
                axis(&mut out, "axis", &r.axis);
                mode(&mut out, &r.mode);
            }
            FeatureKind::Sweep(s) => {
                if let ProfileSource::Face(face) = &s.profile {
                    out.push(("profile".into(), face));
                }
                mode(&mut out, &s.mode);
            }
            FeatureKind::Loft(l) => {
                list(&mut out, "guides", &l.guides);
                mode(&mut out, &l.mode);
            }
            FeatureKind::Mirror(m) => {
                match &m.seeds {
                    PatternSeeds::Bodies { bodies } => list(&mut out, "bodies", bodies),
                    PatternSeeds::Faces { faces } => list(&mut out, "faces", faces),
                    PatternSeeds::Features { .. } => {}
                }
                plane(&mut out, "plane", &m.plane);
            }
            FeatureKind::Boolean(b) => {
                out.push(("target".into(), &b.target));
                list(&mut out, "tools", &b.tools);
            }
            FeatureKind::Split(s) => {
                out.push(("body".into(), &s.body));
                plane(&mut out, "tool", &s.tool);
            }
            FeatureKind::BodyOp(op) => match op {
                BodyOp::Delete { bodies } | BodyOp::Translate { bodies, .. } => {
                    list(&mut out, "bodies", bodies)
                }
                BodyOp::Rotate {
                    bodies, axis: a, ..
                } => {
                    list(&mut out, "bodies", bodies);
                    axis(&mut out, "axis", a);
                }
                BodyOp::Copy {
                    bodies, axis: a, ..
                } => {
                    list(&mut out, "bodies", bodies);
                    if let Some(a) = a {
                        axis(&mut out, "axis", a);
                    }
                }
                BodyOp::Indent { target, tool, .. } => {
                    out.push(("target".into(), target));
                    out.push(("tool_body".into(), tool));
                }
            },
            FeatureKind::Import(_) => {}
            _ => {}
        }
        out
    }

    /// Features this feature names directly, other than through topological references:
    /// profiles, pattern seeds, datum features and sketch points.
    pub fn feature_dependencies(&self) -> Vec<FeatureId> {
        if let Some(v) = self.family_visit().or_else(|| self.datum_visit()) {
            let mut out = v.features;
            out.sort();
            out.dedup();
            return out;
        }
        let mut out = Vec::new();
        fn datum(out: &mut Vec<FeatureId>, d: &crate::model::datum::DatumRef) {
            if let crate::model::datum::DatumRef::Feature { feature } = d {
                out.push(feature.clone());
            }
        }
        fn plane(out: &mut Vec<FeatureId>, p: &PlaneRef) {
            if let PlaneRef::Datum { datum: d } = p {
                datum(out, d);
            }
        }
        fn axis(out: &mut Vec<FeatureId>, a: &AxisRef) {
            if let AxisRef::Datum { datum: d } = a {
                datum(out, d);
            }
        }
        fn point(out: &mut Vec<FeatureId>, p: &PointRef) {
            match p {
                PointRef::Datum { datum: d } => datum(out, d),
                PointRef::Sketch { sketch, .. } => out.push(sketch.clone()),
                _ => {}
            }
        }
        fn profile(out: &mut Vec<FeatureId>, p: &ProfileRef) {
            out.push(p.sketch.clone());
        }
        fn source(out: &mut Vec<FeatureId>, p: &ProfileSource) {
            if let ProfileSource::Sketch(p) = p {
                out.push(p.sketch.clone());
            }
        }
        fn seeds(out: &mut Vec<FeatureId>, s: &PatternSeeds) {
            if let PatternSeeds::Features { features } = s {
                out.extend(features.iter().cloned());
            }
        }
        match self {
            FeatureKind::Sketch(s) => plane(&mut out, &s.plane),
            FeatureKind::Plane(p) => match p {
                PlaneDef::Offset { base, .. } => plane(&mut out, base),
                PlaneDef::Angle { base, axis: a, .. } => {
                    plane(&mut out, base);
                    axis(&mut out, a);
                }
                _ => {}
            },
            FeatureKind::Axis(AxisDef::Through { a, b }) => {
                point(&mut out, a);
                point(&mut out, b);
            }
            FeatureKind::Axis(_) => {}
            FeatureKind::Point(PointDef::At { at }) => point(&mut out, at),
            FeatureKind::Point(PointDef::Offset { base, .. }) => point(&mut out, base),
            FeatureKind::Frame(FrameDef::At { origin, z, x }) => {
                point(&mut out, origin);
                axis(&mut out, z);
                if let Some(x) = x {
                    axis(&mut out, x);
                }
            }
            FeatureKind::Frame(_) => {}
            FeatureKind::Revolve(r) => {
                source(&mut out, &r.profile);
                axis(&mut out, &r.axis);
            }
            FeatureKind::Sweep(s) => {
                source(&mut out, &s.profile);
                profile(&mut out, &s.path);
            }
            FeatureKind::Loft(l) => l.profiles.iter().for_each(|p| profile(&mut out, p)),
            FeatureKind::Mirror(m) => {
                seeds(&mut out, &m.seeds);
                plane(&mut out, &m.plane);
            }
            FeatureKind::Split(s) => plane(&mut out, &s.tool),
            FeatureKind::BodyOp(BodyOp::Rotate { axis: a, .. }) => axis(&mut out, a),
            FeatureKind::BodyOp(BodyOp::Copy { axis: Some(a), .. }) => axis(&mut out, a),
            FeatureKind::Boolean(_) | FeatureKind::BodyOp(_) | FeatureKind::Import(_) => {}
            _ => {}
        }
        out.sort();
        out.dedup();
        out
    }

    /// Whether the feature acts on every body when it names no targets (implicit operands).
    pub fn has_implicit_targets(&self) -> bool {
        let mode = match self {
            FeatureKind::Extrude(e) => &e.mode,
            FeatureKind::Revolve(r) => &r.mode,
            FeatureKind::Sweep(s) => &s.mode,
            FeatureKind::Loft(l) => &l.mode,
            FeatureKind::Mirror(m) => return matches!(m.seeds, PatternSeeds::Features { .. }),
            other => {
                return other
                    .family()
                    .map(|f| f.implicit_targets())
                    .unwrap_or(false)
            }
        };
        !matches!(mode, BodyMode::New) && mode.targets().is_empty()
    }
}
