//! Design-for-manufacture check packs: machining, sheet metal, molding and additive.
//!
//! A regenerated part becomes a [`DfmGraph`] of its manufacturing features: the part bodies
//! (extents, wall thickness, undercuts, draft, overhangs, enclosed voids), holes (from the hole
//! features' records: diameter, depth ratios, edge distance, threads), pockets (cut extrudes:
//! depth over width, sharp internal corners), fillets, sheets (thickness, bend radius over
//! thickness, K-factor), flanges and ribs, each with measured attributes. The packs are
//! ordinary [`CheckPack`]s over that graph with `runmat-rules` selectors, run by
//! [`run_checks_on`], so a company pack can tighten or extend them. Geometry measurements go
//! through the [`GeometryAnalyzer`] seam; without one, those attributes are missing and their
//! checks report unknown rather than fail.

use crate::diagnostics::{DiagCode, DiagnosticReport, Site, Subject};
use crate::evaluate::{AnalysisRequest, AnalysisResult, GeometryAnalyzer};
use crate::ids::FeatureId;
use crate::model::feature::*;
use crate::model::sheet_metal::{BendRuleValue, SheetMetalFeature};
use crate::model::{FeatureKind, PartDocument};
use crate::props::{
    run_checks_on, to_rules_dim, CheckDef, CheckGraph, CheckPack, CheckResult, CompareOp, Expected,
    Gate,
};
use crate::refs::history::SurfaceKind;
use crate::regen::records::FeatureRecord;
use crate::regen::RegenResult;
use crate::units::{Dimension, UnitAuthority};
use crate::value::Evaluator;
use runmat_rules as rules;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Process {
    Machining,
    SheetMetal,
    Molding,
    Additive,
}

/// How the part is made, for the measurements that depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DfmOptions {
    /// Mold pull direction.
    pub pull: [f64; 3],
    /// Draft below this (radians) is insufficient.
    pub min_draft: f64,
    /// Additive build direction.
    pub build: [f64; 3],
    /// Overhangs steeper than this from the build direction (radians) need support.
    pub overhang: f64,
}

impl Default for DfmOptions {
    fn default() -> Self {
        DfmOptions {
            pull: [0.0, 0.0, 1.0],
            min_draft: 0.5f64.to_radians(),
            build: [0.0, 0.0, 1.0],
            overhang: 45f64.to_radians(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DfmItem {
    pub kind: &'static str,
    pub label: String,
    pub feature: Option<FeatureId>,
    pub classes: Vec<&'static str>,
    pub attrs: BTreeMap<&'static str, rules::AttrValue>,
}

/// A part's manufacturing features as an item graph for check packs.
#[derive(Clone, Debug, Default)]
pub struct DfmGraph {
    pub items: Vec<DfmItem>,
}

fn len(v: f64) -> rules::AttrValue {
    rules::AttrValue::Num(rules::Quantity::new(v, to_rules_dim(Dimension::LENGTH)))
}

fn num(v: f64) -> rules::AttrValue {
    rules::AttrValue::Num(rules::Quantity::new(v, rules::Dim::NONE))
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl DfmGraph {
    /// Measure a regenerated part. `analyzer` runs the geometry analyses (wall thickness,
    /// undercuts, draft); `ev` evaluates feature parameters against the regeneration's values.
    pub fn measure(
        doc: &PartDocument,
        r: &RegenResult,
        ev: &Evaluator<'_>,
        analyzer: Option<&dyn GeometryAnalyzer>,
        opts: &DfmOptions,
    ) -> DfmGraph {
        let mut g = DfmGraph::default();
        let value = |v: &crate::value::ValueExpr, d: Dimension| ev.eval_as(v, &r.values, d).ok();
        // Sheet thickness, for ratios.
        let sheet_t = r.state.records.values().find_map(|x| match x {
            FeatureRecord::Sheet { rules, .. } => Some(rules.thickness),
            _ => None,
        });
        // Nominal wall: a shell's thickness, else the thinnest wall measured.
        let shell_t = doc.features.iter().find_map(|f| match &f.kind {
            FeatureKind::Shell(s) if !f.suppressed => value(&s.thickness, Dimension::LENGTH),
            _ => None,
        });
        let mut nominal = shell_t;
        // ---------------------------------------------------------------- part bodies
        for (bi, body) in r.state.bodies.iter().enumerate() {
            let mut attrs = BTreeMap::new();
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for f in &body.topology.faces {
                for a in 0..3 {
                    lo[a] = lo[a].min(f.geom.bbox_min[a]);
                    hi[a] = hi[a].max(f.geom.bbox_max[a]);
                }
            }
            if lo[0].is_finite() {
                attrs.insert("size_x", len(hi[0] - lo[0]));
                attrs.insert("size_y", len(hi[1] - lo[1]));
                attrs.insert("size_z", len(hi[2] - lo[2]));
            }
            // Overhangs: downward faces steeper than the limit, above the build plate.
            let bottom = dot(lo, opts.build).min(dot(hi, opts.build));
            let overhangs = body
                .topology
                .faces
                .iter()
                .filter(|f| {
                    let n = match f.geom.surface {
                        SurfaceKind::Plane { normal } => normal,
                        _ => return false,
                    };
                    let down = -dot(n, opts.build);
                    let on_plate = (dot(f.geom.centroid, opts.build) - bottom).abs() < 1e-9;
                    down > opts.overhang.cos() + 1e-12 && !on_plate
                })
                .count();
            attrs.insert("overhangs", num(overhangs as f64));
            let voids = doc
                .features
                .iter()
                .filter(|f| {
                    !f.suppressed
                        && matches!(&f.kind, FeatureKind::Shell(s) if s.open_faces.is_empty())
                })
                .count();
            attrs.insert("enclosed_voids", num(voids as f64));
            if let Some(a) = analyzer {
                let h = [&body.handle];
                if let Ok(AnalysisResult::Thickness { faces }) = a.analyze(
                    &AnalysisRequest::Thickness {
                        min: f64::INFINITY,
                        max: None,
                    },
                    &h,
                ) {
                    let wmin = faces.iter().map(|v| v.min).fold(f64::INFINITY, f64::min);
                    let wmax = faces.iter().map(|v| v.max).fold(0.0, f64::max);
                    if wmin.is_finite() {
                        attrs.insert("min_wall", len(wmin));
                        attrs.insert("max_wall", len(wmax));
                        attrs.insert("wall_ratio", num(wmax / wmin.max(1e-300)));
                        nominal = nominal.or(Some(wmin));
                    }
                }
                if let Ok(AnalysisResult::Undercut { faces }) =
                    a.analyze(&AnalysisRequest::Undercut { pull: opts.pull }, &h)
                {
                    attrs.insert("undercuts", num(faces.len() as f64));
                }
                if let Ok(AnalysisResult::Draft(d)) = a.analyze(
                    &AnalysisRequest::Draft {
                        pull: opts.pull,
                        min: opts.min_draft,
                    },
                    &h,
                ) {
                    attrs.insert("draft_insufficient", num(d.insufficient.len() as f64));
                }
            }
            g.items.push(DfmItem {
                kind: "part",
                label: format!("body[{}]", bi),
                feature: None,
                classes: Vec::new(),
                attrs,
            });
        }
        // ---------------------------------------------------------------------- holes
        let mut centres: Vec<([f64; 3], f64)> = Vec::new();
        for (fid, rec) in &r.state.records {
            let FeatureRecord::Holes { holes } = rec else {
                continue;
            };
            for h in holes {
                centres.push((h.position, h.diameter));
                let mut attrs = BTreeMap::new();
                attrs.insert("diameter", len(h.diameter));
                let mut classes = Vec::new();
                if let Some(d) = h.depth {
                    attrs.insert("depth", len(d));
                    attrs.insert("depth_ratio", num(d / h.diameter));
                    classes.push("blind");
                }
                if let Some(t) = &h.thread {
                    classes.push("tapped");
                    if let Some(td) = t.depth {
                        attrs.insert("thread_depth_ratio", num(td / h.diameter));
                    }
                }
                // Wall from the hole's edge to the nearest outer face parallel to its axis.
                let edge = r
                    .state
                    .bodies
                    .iter()
                    .flat_map(|b| b.topology.faces.iter())
                    .filter_map(|f| match f.geom.surface {
                        SurfaceKind::Plane { normal } if dot(normal, h.axis).abs() < 1e-6 => {
                            let d = [
                                h.position[0] - f.geom.centroid[0],
                                h.position[1] - f.geom.centroid[1],
                                h.position[2] - f.geom.centroid[2],
                            ];
                            // Outer faces face away from the hole.
                            let s = -dot(d, normal);
                            (s > h.diameter / 2.0).then_some(s - h.diameter / 2.0)
                        }
                        _ => None,
                    })
                    .fold(f64::INFINITY, f64::min);
                if edge.is_finite() {
                    attrs.insert("edge_distance", len(edge));
                    attrs.insert("edge_ratio", num(edge / h.diameter));
                    if let Some(t) = sheet_t {
                        attrs.insert("edge_ratio_t", num(edge / t));
                    }
                }
                if let Some(t) = sheet_t {
                    attrs.insert("diameter_ratio_t", num(h.diameter / t));
                }
                g.items.push(DfmItem {
                    kind: "hole",
                    label: format!("{}:{}", fid, h.key),
                    feature: Some(fid.clone()),
                    classes,
                    attrs,
                });
            }
        }
        // Ligament between neighbouring holes.
        for (k, item) in g.items.iter_mut().filter(|i| i.kind == "hole").enumerate() {
            let (p, d) = centres[k];
            let lig = centres
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != k)
                .map(|(_, (q, e))| {
                    let v = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                    dot(v, v).sqrt() - (d + e) / 2.0
                })
                .fold(f64::INFINITY, f64::min);
            if lig.is_finite() {
                item.attrs.insert("ligament", len(lig));
            }
        }
        // ------------------------------------------- pockets, fillets, flanges and ribs
        for f in doc.features.iter().filter(|f| !f.suppressed) {
            let ran = r.feature(&f.id.0).map(|o| o.status.ran()).unwrap_or(false);
            if !ran {
                continue;
            }
            let mut attrs = BTreeMap::new();
            let kind = match &f.kind {
                FeatureKind::Extrude(e) if matches!(e.mode, BodyMode::Cut { .. }) => {
                    let Some(sketch) = e.profile.sketch().map(|p| &p.sketch) else {
                        continue;
                    };
                    let Some((solved, _)) = r.state.sketches.get(sketch) else {
                        continue;
                    };
                    let pts: Vec<[f64; 2]> = solved.points.values().copied().collect();
                    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
                    for p in &pts {
                        for a in 0..2 {
                            lo[a] = lo[a].min(p[a]);
                            hi[a] = hi[a].max(p[a]);
                        }
                    }
                    let width = (hi[0] - lo[0]).min(hi[1] - lo[1]);
                    if let Extent::Blind { distance } = &e.extent {
                        if let Some(d) = value(distance, Dimension::LENGTH) {
                            attrs.insert("depth", len(d));
                            if width > 0.0 {
                                attrs.insert("depth_ratio", num(d / width));
                            }
                        }
                    }
                    attrs.insert("width", len(width));
                    // Corners where two straight profile curves meet leave a sharp internal
                    // vertical corner no rotating tool can cut.
                    let sharp = match doc.feature(sketch).map(|s| &s.kind) {
                        Some(FeatureKind::Sketch(s)) => {
                            use crate::model::sketch::{PointRole, SketchGeometry};
                            let ends: Vec<[f64; 2]> = s
                                .entities
                                .iter()
                                .filter(|x| {
                                    !x.construction
                                        && matches!(x.geometry, SketchGeometry::Line { .. })
                                })
                                .flat_map(|x| {
                                    [PointRole::Start, PointRole::End]
                                        .into_iter()
                                        .filter_map(|role| solved.point(&x.id, role))
                                })
                                .collect();
                            let mut n = 0;
                            for (i, a) in ends.iter().enumerate() {
                                for b in &ends[i + 1..] {
                                    if (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9 {
                                        n += 1;
                                    }
                                }
                            }
                            n
                        }
                        _ => 0,
                    };
                    attrs.insert("sharp_corners", num(sharp as f64));
                    "pocket"
                }
                FeatureKind::Fillet(fl) => {
                    if let Some(v) = value(&fl.radius, Dimension::LENGTH) {
                        attrs.insert("radius", len(v));
                    }
                    "fillet"
                }
                FeatureKind::SheetMetal(SheetMetalFeature::EdgeFlange { length, .. }) => {
                    if let Some(v) = value(length, Dimension::LENGTH) {
                        attrs.insert("length", len(v));
                        if let Some(t) = sheet_t {
                            attrs.insert("length_ratio", num(v / t));
                        }
                    }
                    "flange"
                }
                FeatureKind::Rib(rb) => {
                    let t = value(&rb.thickness, Dimension::LENGTH);
                    let h = value(&rb.depth, Dimension::LENGTH);
                    if let Some(t) = t {
                        attrs.insert("thickness", len(t));
                    }
                    if let (Some(t), Some(w)) = (t, nominal) {
                        attrs.insert("thickness_ratio", num(t / w));
                    }
                    if let (Some(h), Some(w)) = (h, nominal) {
                        attrs.insert("height_ratio", num(h / w));
                    }
                    "rib"
                }
                _ => continue,
            };
            g.items.push(DfmItem {
                kind,
                label: f.id.0.clone(),
                feature: Some(f.id.clone()),
                classes: Vec::new(),
                attrs,
            });
        }
        for (fid, rec) in &r.state.records {
            if let FeatureRecord::Sheet { rules, .. } = rec {
                let mut attrs = BTreeMap::new();
                attrs.insert("thickness", len(rules.thickness));
                attrs.insert("bend_radius", len(rules.bend_radius));
                attrs.insert("bend_ratio", num(rules.bend_radius / rules.thickness));
                if let BendRuleValue::KFactor { k } = rules.rule {
                    attrs.insert("k_factor", num(k));
                }
                g.items.push(DfmItem {
                    kind: "sheet",
                    label: fid.0.clone(),
                    feature: Some(fid.clone()),
                    classes: Vec::new(),
                    attrs,
                });
            }
        }
        g
    }
}

impl rules::ItemGraph for DfmGraph {
    type Id = u32;
    fn items(&self) -> Vec<u32> {
        (0..self.items.len() as u32).collect()
    }
    fn kind(&self, id: u32) -> &str {
        self.items[id as usize].kind
    }
    fn name(&self, id: u32) -> Option<&str> {
        Some(self.items[id as usize].label.as_str())
    }
    fn has_class(&self, id: u32, class: &str) -> bool {
        self.items[id as usize].classes.contains(&class)
    }
    fn attr(&self, id: u32, name: &str) -> Option<rules::AttrValue> {
        self.items[id as usize].attrs.get(name).cloned()
    }
    fn parent(&self, _id: u32) -> Option<u32> {
        None
    }
    fn children(&self, _id: u32) -> Vec<u32> {
        Vec::new()
    }
    fn pseudo(&self, _id: u32, _name: &str, _args: &[rules::PseudoArg]) -> Option<bool> {
        None
    }
}

impl CheckGraph for DfmGraph {
    fn label(&self, id: u32) -> String {
        self.items[id as usize].label.clone()
    }
    fn site(&self, id: u32) -> Site {
        match &self.items[id as usize].feature {
            Some(f) => Site::feature(f, None),
            None => Site::of(Subject::Value {
                name: self.items[id as usize].label.clone(),
            }),
        }
    }
    fn fail_code(&self) -> DiagCode {
        DiagCode::DfmCheckFailed
    }
}

fn check(
    name: &str,
    select: &str,
    attribute: &str,
    op: CompareOp,
    value: &str,
    gate: Gate,
) -> CheckDef {
    CheckDef {
        name: name.into(),
        select: select.into(),
        attribute: attribute.into(),
        op,
        expected: Expected::Quantity { text: value.into() },
        gate,
        // A measurement the part does not have (no analyzer, no such feature) is unknown.
        missing_fails: false,
    }
}

/// The built-in pack for a process.
pub fn pack(process: Process) -> CheckPack {
    use CompareOp::*;
    use Gate::*;
    let (name, checks) = match process {
        Process::Machining => (
            "dfm.machining",
            vec![
                // Twist drills reach about ten diameters; deeper needs gun drilling.
                check("hole_depth", "hole.blind", "depth_ratio", Le, "10", Publish),
                check("hole_diameter", "hole", "diameter", Ge, "1 mm", Publish),
                // Thread engagement past three diameters adds no strength.
                check(
                    "tap_depth",
                    "hole.tapped",
                    "thread_depth_ratio",
                    Le,
                    "3",
                    None,
                ),
                check("hole_edge", "hole", "edge_ratio", Ge, "0.5", None),
                // Internal vertical corners need a radius a rotating tool can leave.
                check(
                    "pocket_corners",
                    "pocket",
                    "sharp_corners",
                    Eq,
                    "0",
                    Release,
                ),
                // End mill reach: pocket depth within four times its width.
                check("pocket_depth", "pocket", "depth_ratio", Le, "4", Publish),
                check("min_wall", "part", "min_wall", Ge, "0.8 mm", Release),
            ],
        ),
        Process::SheetMetal => (
            "dfm.sheet_metal",
            vec![
                // Inside bend radius at least the thickness to avoid cracking.
                check("bend_radius", "sheet", "bend_ratio", Ge, "1", Release),
                check("k_factor_low", "sheet", "k_factor", Ge, "0.25", None),
                check("k_factor_high", "sheet", "k_factor", Le, "0.5", None),
                // A flange shorter than about four thicknesses cannot sit across the V-die.
                check("flange_length", "flange", "length_ratio", Ge, "4", Release),
                // Punched holes at least the sheet thickness, two thicknesses from an edge.
                check(
                    "hole_diameter",
                    "hole",
                    "diameter_ratio_t",
                    Ge,
                    "1",
                    Release,
                ),
                check("hole_edge", "hole", "edge_ratio_t", Ge, "2", Publish),
            ],
        ),
        Process::Molding => (
            "dfm.molding",
            vec![
                check("draft", "part", "draft_insufficient", Eq, "0", Release),
                check("undercuts", "part", "undercuts", Eq, "0", Release),
                check("wall_min", "part", "min_wall", Ge, "0.8 mm", Release),
                check("wall_max", "part", "max_wall", Le, "4 mm", Publish),
                // Walls within a factor of two of each other fill and cool evenly.
                check("wall_uniformity", "part", "wall_ratio", Le, "2", None),
                // Ribs at most 60 % of the wall (sink marks), at most three walls tall.
                check(
                    "rib_thickness",
                    "rib",
                    "thickness_ratio",
                    Le,
                    "0.6",
                    Publish,
                ),
                check("rib_height", "rib", "height_ratio", Le, "3", None),
                check("fillet_radius", "fillet", "radius", Ge, "0.5 mm", None),
            ],
        ),
        Process::Additive => (
            "dfm.additive",
            vec![
                check("min_wall", "part", "min_wall", Ge, "0.8 mm", Release),
                check("min_hole", "hole", "diameter", Ge, "2 mm", Publish),
                check("overhangs", "part", "overhangs", Eq, "0", None),
                // Closed cavities trap powder or resin.
                check("enclosed_voids", "part", "enclosed_voids", Eq, "0", Release),
                check("build_x", "part", "size_x", Le, "250 mm", Release),
                check("build_y", "part", "size_y", Le, "250 mm", Release),
                check("build_z", "part", "size_z", Le, "300 mm", Release),
            ],
        ),
    };
    CheckPack {
        name: name.into(),
        version: "1".into(),
        checks,
    }
}

/// Run DFM packs (built-in or company) over a measured part.
pub fn run(
    packs: &[CheckPack],
    graph: &DfmGraph,
    units: &dyn UnitAuthority,
) -> (Vec<CheckResult>, DiagnosticReport) {
    run_checks_on(packs, graph, units)
}
