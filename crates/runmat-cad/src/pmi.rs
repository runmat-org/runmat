//! Semantic PMI on parts: datum features and targets, geometric and dimensional tolerances,
//! surface finish and treatments, attached by the same references as features and regions.
//!
//! PMI is canonical in the `.part` (AP242 semantic PMI maps onto it). Drawings inherit it
//! (a GD&T frame on a view names a tolerance here rather than restating it) and `.fea` reads it
//! (fits, contact, finish). [`resolve_pmi`] resolves every attachment against a regenerated
//! state and checks the semantics: datum labels unique and declared before use, form tolerances
//! without datums, orientation and location tolerances with them, tolerance values of the
//! right dimension, and modifiers only where they mean something.

use crate::diagnostics::{
    describe_ref_problem, Cause, DiagCode, DiagnosticReport, Severity, Site, Subject,
};
use crate::model::items::Cardinality;
use crate::model::PartDocument;
use crate::refs::lock::LockSection;
use crate::refs::name::{EntityName, TopoKind};
use crate::refs::publish::PublishedCatalog;
use crate::refs::reference::Reference;
use crate::refs::resolve::{ResolutionPolicy, ResolveOutcome, Resolver};
use crate::regen::engine::RegenResult;
use crate::units::{Dimension, UnitAuthority};
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

/// A toleranced size or a fit (`10 mm +0.02 -0`, `H7`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SizeTolerance {
    pub name: String,
    pub on: Vec<Reference>,
    /// Usually [`ValueExpr::Toleranced`].
    pub size: ValueExpr,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceFinish {
    pub on: Vec<Reference>,
    /// Roughness average (a length).
    pub ra: ValueExpr,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lay: Option<String>,
}

/// A finish, coating or treatment by specification (`anodize per MIL-A-8625 type II`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Treatment {
    pub on: Vec<Reference>,
    pub specification: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PmiSet {
    #[serde(default)]
    pub datums: Vec<DatumFeature>,
    #[serde(default)]
    pub targets: Vec<DatumTarget>,
    #[serde(default)]
    pub tolerances: Vec<GeometricTolerance>,
    #[serde(default)]
    pub sizes: Vec<SizeTolerance>,
    #[serde(default)]
    pub finishes: Vec<SurfaceFinish>,
    #[serde(default)]
    pub treatments: Vec<Treatment>,
}

/// PMI resolved against one regeneration: entity names per item, for drawings and `.fea`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResolvedPmi {
    pub datums: BTreeMap<String, Vec<Vec<EntityName>>>,
    pub tolerances: BTreeMap<String, Vec<Vec<EntityName>>>,
    pub sizes: BTreeMap<String, Vec<Vec<EntityName>>>,
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
            && self.finishes.is_empty()
            && self.treatments.is_empty()
    }
}

/// Resolve and check a part's PMI against its regeneration. Regeneration calls this for
/// [`PartDocument::pmi`] itself ([`RegenResult::pmi`]); call it directly for a PMI set held
/// elsewhere (an imported AP242 set before it is written into the part).
pub fn resolve_pmi(
    doc: &PartDocument,
    pmi: &PmiSet,
    result: &RegenResult,
    units: &dyn UnitAuthority,
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
    units: &dyn UnitAuthority,
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
                format!("`{}` is a {}, not a size", z.name, q.dimension),
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
    out
}
