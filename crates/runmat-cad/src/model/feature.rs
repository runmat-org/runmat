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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "extent")]
pub enum Extent {
    Blind {
        distance: ValueExpr,
    },
    /// Both directions, half each way.
    Symmetric {
        distance: ValueExpr,
    },
    ThroughAll,
    /// Up to a face.
    UpTo {
        face: Reference,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Extrude {
    pub profile: ProfileRef,
    pub extent: Extent,
    /// Extrude against the sketch normal.
    #[serde(default)]
    pub reverse: bool,
    #[serde(default)]
    pub mode: BodyMode,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Revolve {
    pub profile: ProfileRef,
    pub axis: AxisRef,
    pub angle: ValueExpr,
    #[serde(default)]
    pub mode: BodyMode,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sweep {
    pub profile: ProfileRef,
    /// The path: a sketch's open chain.
    pub path: ProfileRef,
    #[serde(default)]
    pub mode: BodyMode,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Loft {
    pub profiles: Vec<ProfileRef>,
    #[serde(default)]
    pub guides: Vec<Reference>,
    #[serde(default)]
    pub mode: BodyMode,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "hole")]
pub enum HoleForm {
    Simple,
    Counterbore {
        diameter: ValueExpr,
        depth: ValueExpr,
    },
    Countersink {
        diameter: ValueExpr,
        angle: ValueExpr,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HolePosition {
    /// The semantic key for this hole (instance identity; never an array index).
    pub key: String,
    pub at: PointRef,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hole {
    /// The planar face the holes start on.
    pub face: Reference,
    pub positions: Vec<HolePosition>,
    pub diameter: ValueExpr,
    /// `None`: through all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<ValueExpr>,
    #[serde(default = "simple_hole")]
    pub form: HoleForm,
}

fn simple_hole() -> HoleForm {
    HoleForm::Simple
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fillet {
    pub edges: Vec<Reference>,
    pub radius: ValueExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chamfer {
    pub edges: Vec<Reference>,
    pub distance: ValueExpr,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance2: Option<ValueExpr>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shell {
    pub body: Reference,
    /// Faces removed to open the shell.
    #[serde(default)]
    pub open_faces: Vec<Reference>,
    pub thickness: ValueExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Draft {
    pub faces: Vec<Reference>,
    pub neutral: PlaneRef,
    pub angle: ValueExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "seeds")]
pub enum PatternSeeds {
    /// Re-apply these features' tools at each instance.
    Features { features: Vec<FeatureId> },
    /// Copy these bodies at each instance.
    Bodies { bodies: Vec<Reference> },
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
    /// Instances `1..count` about `axis`, equally spaced over `angle`.
    Circular {
        axis: AxisRef,
        count: ValueExpr,
        angle: ValueExpr,
    },
    /// Instances keyed by table row: each row is a key plus an offset.
    Table { rows: Vec<PatternRow> },
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
    /// The family parameters of the newer feature kinds.
    pub fn family(&self) -> Option<&dyn Params> {
        Some(match self {
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
        fn extent<'a>(out: &mut Vec<&'a ValueExpr>, e: &'a Extent) {
            match e {
                Extent::Blind { distance } | Extent::Symmetric { distance } => out.push(distance),
                Extent::ThroughAll | Extent::UpTo { .. } => {}
            }
        }
        fn plane_def<'a>(out: &mut Vec<&'a ValueExpr>, p: &'a PlaneDef) {
            match p {
                PlaneDef::Offset { distance, .. } => out.push(distance),
                PlaneDef::Angle { angle, .. } => out.push(angle),
                _ => {}
            }
        }
        fn layout<'a>(out: &mut Vec<&'a ValueExpr>, l: &'a PatternLayout) {
            match l {
                PatternLayout::Linear { spacing, count, .. } => {
                    out.push(spacing);
                    out.push(count);
                }
                PatternLayout::Grid {
                    first_spacing,
                    first_count,
                    second_spacing,
                    second_count,
                    ..
                } => {
                    out.push(first_spacing);
                    out.push(first_count);
                    out.push(second_spacing);
                    out.push(second_count);
                }
                PatternLayout::Circular { count, angle, .. } => {
                    out.push(count);
                    out.push(angle);
                }
                PatternLayout::Table { rows } => {
                    for r in rows {
                        out.push(&r.dx);
                        out.push(&r.dy);
                        out.push(&r.dz);
                    }
                }
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
                    }
                }
                for c in &s.constraints {
                    match &c.kind {
                        K::Distance { value, .. }
                        | K::Length { value, .. }
                        | K::Radius { value, .. }
                        | K::Angle { value, .. } => ptrs.push(value),
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
            FeatureKind::Extrude(e) => extent(&mut ptrs, &e.extent),
            FeatureKind::Revolve(r) => ptrs.push(&r.angle),
            FeatureKind::Sweep(_) | FeatureKind::Loft(_) | FeatureKind::Import(_) => {}
            FeatureKind::Hole(h) => {
                ptrs.push(&h.diameter);
                if let Some(d) = &h.depth {
                    ptrs.push(d);
                }
                match &h.form {
                    HoleForm::Simple => {}
                    HoleForm::Counterbore { diameter, depth } => {
                        ptrs.push(diameter);
                        ptrs.push(depth);
                    }
                    HoleForm::Countersink { diameter, angle } => {
                        ptrs.push(diameter);
                        ptrs.push(angle);
                    }
                }
                for p in &h.positions {
                    point(&mut ptrs, &p.at);
                }
            }
            FeatureKind::Fillet(f) => ptrs.push(&f.radius),
            FeatureKind::Chamfer(c) => {
                ptrs.push(&c.distance);
                if let Some(d) = &c.distance2 {
                    ptrs.push(d);
                }
            }
            FeatureKind::Shell(s) => ptrs.push(&s.thickness),
            FeatureKind::Draft(d) => ptrs.push(&d.angle),
            FeatureKind::Pattern(p) => layout(&mut ptrs, &p.layout),
            FeatureKind::Mirror(_) | FeatureKind::Boolean(_) | FeatureKind::Split(_) => {}
            FeatureKind::BodyOp(BodyOp::Translate { dx, dy, dz, .. }) => {
                ptrs.push(dx);
                ptrs.push(dy);
                ptrs.push(dz);
            }
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
            if let AxisRef::Edge { edge } = a {
                out.push((name.to_owned(), edge));
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
        fn seeds<'a>(out: &mut Vec<(String, &'a Reference)>, s: &'a PatternSeeds) {
            if let PatternSeeds::Bodies { bodies } = s {
                list(out, "bodies", bodies);
            }
        }
        fn layout<'a>(out: &mut Vec<(String, &'a Reference)>, l: &'a PatternLayout) {
            match l {
                PatternLayout::Linear { direction, .. } => axis(out, "direction", direction),
                PatternLayout::Grid { first, second, .. } => {
                    axis(out, "first", first);
                    axis(out, "second", second);
                }
                PatternLayout::Circular { axis: a, .. } => axis(out, "axis", a),
                PatternLayout::Table { .. } => {}
            }
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
            FeatureKind::Extrude(e) => {
                if let Extent::UpTo { face } = &e.extent {
                    out.push(("up_to".into(), face));
                }
                mode(&mut out, &e.mode);
            }
            FeatureKind::Revolve(r) => {
                axis(&mut out, "axis", &r.axis);
                mode(&mut out, &r.mode);
            }
            FeatureKind::Sweep(s) => mode(&mut out, &s.mode),
            FeatureKind::Loft(l) => {
                list(&mut out, "guides", &l.guides);
                mode(&mut out, &l.mode);
            }
            FeatureKind::Hole(h) => {
                out.push(("face".into(), &h.face));
                for (i, p) in h.positions.iter().enumerate() {
                    point(&mut out, &format!("positions[{}]", i), &p.at);
                }
            }
            FeatureKind::Fillet(f) => list(&mut out, "edges", &f.edges),
            FeatureKind::Chamfer(c) => list(&mut out, "edges", &c.edges),
            FeatureKind::Shell(s) => {
                out.push(("body".into(), &s.body));
                list(&mut out, "open_faces", &s.open_faces);
            }
            FeatureKind::Draft(d) => {
                list(&mut out, "faces", &d.faces);
                plane(&mut out, "neutral", &d.neutral);
            }
            FeatureKind::Pattern(p) => {
                seeds(&mut out, &p.seeds);
                layout(&mut out, &p.layout);
            }
            FeatureKind::Mirror(m) => {
                seeds(&mut out, &m.seeds);
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
            FeatureKind::Extrude(e) => profile(&mut out, &e.profile),
            FeatureKind::Revolve(r) => {
                profile(&mut out, &r.profile);
                axis(&mut out, &r.axis);
            }
            FeatureKind::Sweep(s) => {
                profile(&mut out, &s.profile);
                profile(&mut out, &s.path);
            }
            FeatureKind::Loft(l) => l.profiles.iter().for_each(|p| profile(&mut out, p)),
            FeatureKind::Hole(h) => h.positions.iter().for_each(|p| point(&mut out, &p.at)),
            FeatureKind::Draft(d) => plane(&mut out, &d.neutral),
            FeatureKind::Pattern(p) => {
                seeds(&mut out, &p.seeds);
                match &p.layout {
                    PatternLayout::Linear { direction, .. } => axis(&mut out, direction),
                    PatternLayout::Grid { first, second, .. } => {
                        axis(&mut out, first);
                        axis(&mut out, second);
                    }
                    PatternLayout::Circular { axis: a, .. } => axis(&mut out, a),
                    PatternLayout::Table { .. } => {}
                }
            }
            FeatureKind::Mirror(m) => {
                seeds(&mut out, &m.seeds);
                plane(&mut out, &m.plane);
            }
            FeatureKind::Split(s) => plane(&mut out, &s.tool),
            FeatureKind::Fillet(_)
            | FeatureKind::Chamfer(_)
            | FeatureKind::Shell(_)
            | FeatureKind::Boolean(_)
            | FeatureKind::BodyOp(_)
            | FeatureKind::Import(_) => {}
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
            FeatureKind::Hole(_) => return true,
            FeatureKind::Pattern(p) => return matches!(p.seeds, PatternSeeds::Features { .. }),
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
