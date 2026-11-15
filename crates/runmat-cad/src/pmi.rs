//! Semantic PMI on parts: datum features and targets, geometric and dimensional tolerances,
//! surface finish and treatments, attached by the same references as features and regions.
//!
//! PMI is canonical in the `.part` (AP242 semantic PMI maps onto it). Drawings inherit it
//! (a GD&T frame on a view names a tolerance here rather than restating it) and `.fea` reads it
//! (fits, contact, finish). [`resolve_pmi`] resolves every attachment against a regenerated
//! state and checks the semantics: datum labels unique and declared before use, form tolerances
//! without datums, orientation and location tolerances with them, tolerance values of the
//! right dimension, and modifiers only where they mean something.

use crate::assembly::views::ModelView;
use crate::diagnostics::{
    describe_ref_problem, Cause, DiagCode, DiagnosticReport, Severity, Site, Subject,
};
use crate::drawing::AnnotationKind;
use crate::model::items::Cardinality;
use crate::model::PartDocument;
use crate::refs::lock::LockSection;
use crate::refs::name::{EntityName, TopoKind};
use crate::refs::publish::PublishedCatalog;
use crate::refs::reference::Reference;
use crate::refs::resolve::{ResolutionPolicy, ResolveOutcome, Resolver};
use crate::regen::engine::RegenResult;
use crate::units::{Dimension, Units};
use crate::value::{DynamicValueEvaluator, Evaluator, ValueExpr};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatumFeature {
    /// `A`, `B`, `C` …
    pub label: String,
    pub on: Reference,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target")]
pub enum TargetShape {
    Point,
    Line { length: ValueExpr },
    Circle { diameter: ValueExpr },
    Rectangle { width: ValueExpr, height: ValueExpr },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatumTarget {
    pub datum: String,
    pub index: u32,
    pub shape: TargetShape,
    pub on: Reference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Characteristic {
    Straightness,
    Flatness,
    Circularity,
    Cylindricity,
    ProfileOfLine,
    ProfileOfSurface,
    Parallelism,
    Perpendicularity,
    Angularity,
    Position,
    Concentricity,
    Symmetry,
    CircularRunout,
    TotalRunout,
}

impl Characteristic {
    /// Form tolerances take no datums.
    pub fn is_form(self) -> bool {
        matches!(
            self,
            Self::Straightness | Self::Flatness | Self::Circularity | Self::Cylindricity
        )
    }
    /// Orientation, location and runout tolerances need at least one datum.
    pub fn needs_datum(self) -> bool {
        matches!(
            self,
            Self::Parallelism
                | Self::Perpendicularity
                | Self::Angularity
                | Self::Position
                | Self::Concentricity
                | Self::Symmetry
                | Self::CircularRunout
                | Self::TotalRunout
        )
    }
    /// Material-condition modifiers apply to features of size under these.
    pub fn takes_material_condition(self) -> bool {
        matches!(
            self,
            Self::Straightness
                | Self::Parallelism
                | Self::Perpendicularity
                | Self::Angularity
                | Self::Position
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialCondition {
    /// Regardless of feature size.
    #[default]
    Rfs,
    Mmc,
    Lmc,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatumReference {
    pub label: String,
    #[serde(default)]
    pub modifier: MaterialCondition,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeometricTolerance {
    pub name: String,
    pub characteristic: Characteristic,
    pub on: Vec<Reference>,
    pub tolerance: ValueExpr,
    /// A diametral (cylindrical) zone.
    #[serde(default)]
    pub diametral: bool,
    #[serde(default)]
    pub modifier: MaterialCondition,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected: Option<ValueExpr>,
    /// Primary, secondary, tertiary.
    #[serde(default)]
    pub datums: Vec<DatumReference>,
}

/// A toleranced size or a fit (`10 mm ± [0.02 mm, 0 mm]` is +0.02/-0, `H7`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SizeTolerance {
    pub name: String,
    pub on: Vec<Reference>,
    /// Usually [`ValueExpr::Toleranced`].
    pub size: ValueExpr,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<String>,
}

/// A toleranced distance between two features (a dimensional location):
/// `location slot = { from = feature.body.face.start, to = feature.cut.face.side(s.c1), distance = 20 mm ± 0.1 mm }`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocationTolerance {
    pub name: String,
    pub from: Reference,
    pub to: Reference,
    /// Usually [`ValueExpr::Toleranced`].
    pub distance: ValueExpr,
}

/// A surface finish requirement:
/// `finish seat = { on = [feature.pad.face.end], ra = 0.8 um, process = "lapped" }`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceFinish {
    #[serde(default)]
    pub name: String,
    pub on: Vec<Reference>,
    /// Roughness average (a length).
    pub ra: ValueExpr,
    /// The production method the finish names (`lapped`, `ground`), written beside the symbol.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process: Option<String>,
    /// The direction of the lay as ISO 1302 marks it (`=`, `⊥`, `X`, `M`, `C`, `R`, `P`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lay: Option<String>,
    /// Whether the surface must, or must not, be machined (ISO 1302's basic symbol, its bar
    /// or its circle).
    #[serde(default, skip_serializing_if = "Removal::is_any")]
    pub removal: Removal,
}

/// The material-removal requirement of a surface finish (ISO 1302, ASME Y14.36).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Removal {
    /// Any process.
    #[default]
    Any,
    /// Material removal by machining is required.
    Required,
    /// Material removal is not allowed.
    Prohibited,
}

impl Removal {
    pub fn is_any(&self) -> bool {
        *self == Removal::Any
    }
}

/// A finish, coating or treatment by specification (`anodize per MIL-A-8625 type II`):
/// `treatment coat1 = { on = [ … ], specification = "anodize type II, black" }`. No faces: the
/// whole part.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Treatment {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub on: Vec<Reference>,
    pub specification: String,
}

/// A 3D annotation on the part (model-based definition): the drawing's annotation kinds
/// ([`AnnotationKind`]), placed in the model instead of on a sheet. A datum symbol, GD&T frame,
/// toleranced dimension, finish or coating shows the semantic PMI item it names; a note and a
/// weld symbol carry their own content. References are the part's canonical face and edge
/// names, so an annotation follows its geometry through every rebuild:
/// `annotation note1 = note(text = "Deburr all edges", leader = feature.body.face.end, view = front_pmi, at = [0, 0, 12])`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelAnnotation {
    /// Stable across edits and merges.
    pub id: String,
    pub kind: AnnotationKind,
    /// The annotation view (a [`ModelView`] in [`PmiSet::views`]) it belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view: Option<String>,
    /// Where its label sits, as an offset from the point it is attached to (mm, part axes).
    /// Absent: off the attached face along its normal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<[f64; 3]>,
}

impl ModelAnnotation {
    /// The model references the annotation points at directly (a weld's edge, a note's leader,
    /// a finish's face, a dimension's references).
    pub fn references(&self) -> Vec<&Reference> {
        match &self.kind {
            AnnotationKind::Dimension { references, .. } => references.iter().collect(),
            AnnotationKind::SurfaceFinish { on }
            | AnnotationKind::Weld { on, .. }
            | AnnotationKind::CenterMark { on } => vec![on],
            AnnotationKind::Note { leader, .. } => leader.iter().collect(),
            AnnotationKind::HoleCallout { hole } => vec![hole],
            AnnotationKind::Centerline { between } => between.iter().collect(),
            _ => Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PmiSet {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datums: Vec<DatumFeature>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<DatumTarget>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tolerances: Vec<GeometricTolerance>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sizes: Vec<SizeTolerance>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<LocationTolerance>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub finishes: Vec<SurfaceFinish>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub treatments: Vec<Treatment>,
    /// How the PMI shows in the 3D model, and notes and weld symbols on it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<ModelAnnotation>,
    /// Annotation views: a saved view direction each, grouping the annotations that name it
    /// (`view front_pmi = { from = [0, -1, 0], up = [0, 0, 1] }`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub views: Vec<ModelView>,
}

/// PMI resolved against one regeneration: entity names per item, for drawings and `.fea`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResolvedPmi {
    pub datums: BTreeMap<String, Vec<Vec<EntityName>>>,
    pub tolerances: BTreeMap<String, Vec<Vec<EntityName>>>,
    pub sizes: BTreeMap<String, Vec<Vec<EntityName>>>,
    /// Per location: the `from` entities, then the `to` entities.
    #[serde(default)]
    pub locations: BTreeMap<String, [Vec<Vec<EntityName>>; 2]>,
    /// Per 3D annotation: the entities its own references name (a weld's edge, a note's leader).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub annotations: BTreeMap<String, Vec<Vec<EntityName>>>,
    pub diagnostics: DiagnosticReport,
}

fn site(item: &str, param: &str) -> Site {
    Site::with_param(
        Subject::Value {
            name: format!("pmi.{}", item),
        },
        param,
    )
}

impl PmiSet {
    pub fn is_empty(&self) -> bool {
        self.datums.is_empty()
            && self.targets.is_empty()
            && self.tolerances.is_empty()
            && self.sizes.is_empty()
            && self.locations.is_empty()
            && self.finishes.is_empty()
            && self.treatments.is_empty()
            && self.annotations.is_empty()
            && self.views.is_empty()
    }
}

/// Resolve and check a part's PMI against its regeneration. Regeneration calls this for
/// [`PartDocument::pmi`] itself ([`RegenResult::pmi`]); call it directly for a PMI set held
/// elsewhere (an imported AP242 set before it is written into the part).
pub fn resolve_pmi(
    doc: &PartDocument,
    pmi: &PmiSet,
    result: &RegenResult,
    units: &Units,
    dynamic: &dyn DynamicValueEvaluator,
) -> ResolvedPmi {
    resolve_pmi_with(
        doc,
        pmi,
        result,
        units,
        dynamic,
        None,
        None,
        ResolutionPolicy::default(),
    )
}

/// [`resolve_pmi`] with the regeneration's lock, catalog and policy, so PMI on published or
/// renamed entities resolves exactly as the part's own references do.
#[allow(clippy::too_many_arguments)]
pub fn resolve_pmi_with(
    doc: &PartDocument,
    pmi: &PmiSet,
    result: &RegenResult,
    units: &Units,
    dynamic: &dyn DynamicValueEvaluator,
    lock: Option<&LockSection>,
    catalog: Option<&dyn PublishedCatalog>,
    policy: ResolutionPolicy,
) -> ResolvedPmi {
    let st = &result.state;
    let resolver = Resolver {
        file: &doc.path,
        bodies: st.views(),
        features: &st.table,
        consumer: None,
        burials: &st.burials,
        publishes: &doc.publishes,
        lock,
        catalog,
        policy,
        names: Some(&st.names),
    };
    let ev = Evaluator { units, dynamic };
    let mut out = ResolvedPmi::default();
    let resolve = |out: &mut ResolvedPmi,
                   s: Site,
                   r: &Reference,
                   kind: Option<TopoKind>|
     -> Vec<Vec<EntityName>> {
        match resolver.resolve(r, kind, Cardinality::AtLeastOne) {
            ResolveOutcome::Resolved(res) => {
                res.entities().iter().map(|h| st.entity_names(h)).collect()
            }
            ResolveOutcome::Waiting(_) => {
                out.diagnostics.push(
                    DiagCode::PmiReference,
                    Severity::Info,
                    s,
                    format!("`{}` waits on a suppressed feature", r),
                    Cause::Root,
                    Vec::new(),
                );
                Vec::new()
            }
            ResolveOutcome::Failed(p) => {
                let (_, msg, repairs) = describe_ref_problem(&s, r, &p);
                out.diagnostics.push(
                    DiagCode::PmiReference,
                    Severity::Error,
                    s,
                    msg,
                    Cause::Root,
                    repairs,
                );
                Vec::new()
            }
        }
    };
    let mut labels = BTreeSet::new();
    for d in &pmi.datums {
        if !labels.insert(d.label.clone()) {
            out.diagnostics.push(
                DiagCode::PmiInvalid,
                Severity::Error,
                site(&format!("datum.{}", d.label), "label"),
                format!("datum `{}` is declared twice", d.label),
                Cause::Root,
                Vec::new(),
            );
        }
        let e = resolve(
            &mut out,
            site(&format!("datum.{}", d.label), "on"),
            &d.on,
            None,
        );
        out.datums.insert(d.label.clone(), e);
    }
    for t in &pmi.targets {
        if !labels.contains(&t.datum) {
            out.diagnostics.push(
                DiagCode::PmiInvalid,
                Severity::Error,
                site(&format!("target.{}{}", t.datum, t.index), "datum"),
                format!("target {}{} names no declared datum", t.datum, t.index),
                Cause::Root,
                Vec::new(),
            );
        }
        resolve(
            &mut out,
            site(&format!("target.{}{}", t.datum, t.index), "on"),
            &t.on,
            None,
        );
    }
    for g in &pmi.tolerances {
        let s = |p: &str| site(&g.name, p);
        if g.characteristic.is_form() && !g.datums.is_empty() {
            out.diagnostics.push(
                DiagCode::PmiInvalid,
                Severity::Error,
                s("datums"),
                format!(
                    "{:?} is a form tolerance and takes no datums",
                    g.characteristic
                ),
                Cause::Root,
                Vec::new(),
            );
        }
        if g.characteristic.needs_datum() && g.datums.is_empty() {
            out.diagnostics.push(
                DiagCode::PmiInvalid,
                Severity::Error,
                s("datums"),
                format!("{:?} needs a datum reference frame", g.characteristic),
                Cause::Root,
                Vec::new(),
            );
        }
        if g.modifier != MaterialCondition::Rfs && !g.characteristic.takes_material_condition() {
            out.diagnostics.push(
                DiagCode::PmiInvalid,
                Severity::Error,
                s("modifier"),
                format!("{:?} does not take a material condition", g.characteristic),
                Cause::Root,
                Vec::new(),
            );
        }
        let mut seen = BTreeSet::new();
        for d in &g.datums {
            if !labels.contains(&d.label) {
                out.diagnostics.push(
                    DiagCode::PmiInvalid,
                    Severity::Error,
                    s("datums"),
                    format!("`{}` references undeclared datum `{}`", g.name, d.label),
                    Cause::Root,
                    Vec::new(),
                );
            }
            if !seen.insert(&d.label) {
                out.diagnostics.push(
                    DiagCode::PmiInvalid,
                    Severity::Error,
                    s("datums"),
                    format!("datum `{}` appears twice in `{}`", d.label, g.name),
                    Cause::Root,
                    Vec::new(),
                );
            }
        }
        if let Err(e) = ev.eval_as(&g.tolerance, &result.values, Dimension::LENGTH) {
            out.diagnostics.push(
                DiagCode::PmiInvalid,
                Severity::Error,
                s("tolerance"),
                format!("`{}`: {}", g.name, e),
                Cause::Root,
                Vec::new(),
            );
        }
        let mut all = Vec::new();
        for (k, r) in g.on.iter().enumerate() {
            all.extend(resolve(&mut out, s(&format!("on[{}]", k)), r, None));
        }
        out.tolerances.insert(g.name.clone(), all);
    }
    for z in &pmi.sizes {
        let s = |p: &str| site(&z.name, p);
        let _ = match ev.eval(&z.size, &result.values) {
            Ok(q) if q.dimension == Dimension::LENGTH || q.dimension == Dimension::ANGLE => 0,
            Ok(q) => out.diagnostics.push(
                DiagCode::PmiInvalid,
                Severity::Error,
                s("size"),
                format!("`{}` is a {}, not a size", z.name, crate::value::kind_name(q.dimension)),
                Cause::Root,
                Vec::new(),
            ),
            Err(e) => out.diagnostics.push(
                DiagCode::PmiInvalid,
                Severity::Error,
                s("size"),
                format!("`{}`: {}", z.name, e),
                Cause::Root,
                Vec::new(),
            ),
        };
        let mut all = Vec::new();
        for (k, r) in z.on.iter().enumerate() {
            all.extend(resolve(&mut out, s(&format!("on[{}]", k)), r, None));
        }
        out.sizes.insert(z.name.clone(), all);
    }
    for l in &pmi.locations {
        let s = |p: &str| site(&l.name, p);
        if let Err(e) = ev.eval_as(&l.distance, &result.values, Dimension::LENGTH) {
            out.diagnostics.push(
                DiagCode::PmiInvalid,
                Severity::Error,
                s("distance"),
                format!("`{}`: {}", l.name, e),
                Cause::Root,
                Vec::new(),
            );
        }
        let from = resolve(&mut out, s("from"), &l.from, None);
        let to = resolve(&mut out, s("to"), &l.to, None);
        out.locations.insert(l.name.clone(), [from, to]);
    }
    for (i, f) in pmi.finishes.iter().enumerate() {
        if let Err(e) = ev.eval_as(&f.ra, &result.values, Dimension::LENGTH) {
            out.diagnostics.push(
                DiagCode::PmiInvalid,
                Severity::Error,
                site(&format!("finish[{}]", i), "ra"),
                e.to_string(),
                Cause::Root,
                Vec::new(),
            );
        }
        for (k, r) in f.on.iter().enumerate() {
            resolve(
                &mut out,
                site(&format!("finish[{}]", i), &format!("on[{}]", k)),
                r,
                Some(TopoKind::Face),
            );
        }
    }
    for (i, t) in pmi.treatments.iter().enumerate() {
        for (k, r) in t.on.iter().enumerate() {
            resolve(
                &mut out,
                site(&format!("treatment[{}]", i), &format!("on[{}]", k)),
                r,
                Some(TopoKind::Face),
            );
        }
    }
    check_annotations(pmi, &labels, &mut out, |out, s, r| resolve(out, s, r, None));
    out
}

/// The 3D annotations: each names a declared PMI item and annotation view, and its own
/// references resolve.
fn check_annotations(
    pmi: &PmiSet,
    labels: &BTreeSet<String>,
    out: &mut ResolvedPmi,
    mut resolve: impl FnMut(&mut ResolvedPmi, Site, &Reference) -> Vec<Vec<EntityName>>,
) {
    let mut ids = BTreeSet::new();
    let invalid = |out: &mut ResolvedPmi, id: &str, param: &str, message: String| {
        out.diagnostics.push(
            DiagCode::PmiInvalid,
            Severity::Error,
            site(&format!("annotation.{id}"), param),
            message,
            Cause::Root,
            Vec::new(),
        );
    };
    for a in &pmi.annotations {
        if !ids.insert(a.id.as_str()) {
            invalid(out, &a.id, "id", format!("annotation `{}` is declared twice", a.id));
        }
        if let Some(view) = &a.view {
            if !pmi.views.iter().any(|v| &v.name == view) {
                invalid(
                    out,
                    &a.id,
                    "view",
                    format!("annotation `{}` names view `{view}`, which the part does not declare", a.id),
                );
            }
        }
        let missing = match &a.kind {
            AnnotationKind::Gdt { tolerance } => (!pmi.tolerances.iter().any(|t| &t.name == tolerance))
                .then(|| format!("tolerance `{tolerance}`")),
            AnnotationKind::DatumSymbol { label } => {
                (!labels.contains(label)).then(|| format!("datum `{label}`"))
            }
            AnnotationKind::Dimension { pmi: Some(name), .. } => (!pmi
                .sizes
                .iter()
                .any(|z| &z.name == name)
                && !pmi.locations.iter().any(|l| &l.name == name))
            .then(|| format!("size or location `{name}`")),
            AnnotationKind::Coating { treatment } => (!pmi
                .treatments
                .iter()
                .any(|t| &t.name == treatment))
            .then(|| format!("treatment `{treatment}`")),
            _ => None,
        };
        if let Some(what) = missing {
            invalid(
                out,
                &a.id,
                "kind",
                format!("annotation `{}` shows {what}, which the part's PMI does not declare", a.id),
            );
        }
        let mut all = Vec::new();
        for (k, r) in a.references().into_iter().enumerate() {
            all.extend(resolve(out, site(&format!("annotation.{}", a.id), &format!("on[{k}]")), r));
        }
        out.annotations.insert(a.id.clone(), all);
    }
}
