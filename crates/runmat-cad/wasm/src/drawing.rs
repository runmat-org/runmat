//! `.drawing` files compiled from their own text: the drawing document checked against the
//! parts and assemblies its views show, and those models' displays, which the sheet canvas
//! projects into each view.
//!
//! - **Models.** Every view names a model file; each is compiled from the text in `files`
//!   (parts on the reference kernel, assemblies over their parts), once per file.
//! - **Checking.** [`build_drawing_index`] runs against [`CompiledModels`]: a reference by name
//!   resolves when the drawn model has a body or face of that name; occurrence paths resolve
//!   against the assembly's resolved tree; properties read the part's output (`part_number`,
//!   `material`, its declared properties). References built from other references
//!   (`edges_between`, pieces) resolve when every name inside them does.
//! - **Lines.** A part's views are projected by the kernel's exact hidden-line removal
//!   ([`crate::hlr::exact`], OCCT); assemblies, and kernels without it, project their display
//!   meshes ([`crate::hlr`]).
//! - **Output.** `{ projection: { kind: "drawing", data }, models: { <file as written>:
//!   <display> }, diagnostics }`; a view whose model is missing or doesn't compile is reported
//!   as a warning and the sheet draws its frame without geometry.

pub mod auto;
mod values;

use crate::runtime::{compile_assembly, regenerate_part, resolve_path, Diagnostic, Files, RegeneratedPart};
use runmat_cad::assembly::bom::Bom;
use runmat_cad::assembly::model::RelPath;
use runmat_cad::drawing::index::build_drawing_index;
use runmat_cad::drawing::{DrawingDocument, ModelRef, ModelReferences};
use runmat_cad::ids::{FilePath, PropertyName};
use runmat_cad::model::PartDocument;
use runmat_cad::refs::name::EntityName;
use runmat_cad::refs::reference::Reference;
use runmat_cad::syntax::parse_drawing;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

/// A part's PMI names: tolerances, sizes and datums.
type PmiNames = (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>);

/// What the drawn models are, as checking reads them.
#[derive(Default)]
pub struct CompiledModels {
    /// File as the drawing wrote it → every body and face name, serialized.
    names: BTreeMap<String, BTreeSet<String>>,
    /// File → property name → text.
    properties: BTreeMap<String, BTreeMap<String, String>>,
    /// File → occurrence paths (assemblies).
    occurrences: BTreeMap<String, BTreeSet<String>>,
    /// File → (tolerances, sizes, datums) the part's PMI declares.
    pmi: BTreeMap<String, PmiNames>,
    /// File → the assembly's flat BOM (its index's), for BOM tables and balloons.
    boms: BTreeMap<String, Bom>,
}

fn names_in(reference: &Value, out: &mut Vec<Value>) {
    match reference {
        Value::Object(o) => {
            if let Some(name) = o.get("name") {
                if o.contains_key("kind") {
                    out.push(name.clone());
                    return;
                }
            }
            for v in o.values() {
                names_in(v, out);
            }
        }
        Value::Array(items) => items.iter().for_each(|v| names_in(v, out)),
        _ => {}
    }
}

impl ModelReferences for CompiledModels {
    fn resolve(
        &self,
        model: &ModelRef,
        reference: &Reference,
    ) -> Result<Vec<Vec<EntityName>>, String> {
        let file = model.file.as_str();
        let Some(known) = self.names.get(file) else {
            return Err(format!("{file} isn't compiled"));
        };
        let mut names = Vec::new();
        names_in(
            &serde_json::to_value(reference).unwrap_or(Value::Null),
            &mut names,
        );
        for name in &names {
            if !known.contains(&name.to_string()) {
                return Err(format!("names {name}, which {file} doesn't have"));
            }
        }
        if let Reference::Name { name, .. } = reference {
            return Ok(vec![vec![name.clone()]]);
        }
        Ok(vec![Vec::new()])
    }
    fn property(&self, model: &ModelRef, name: &PropertyName) -> Option<String> {
        self.properties
            .get(model.file.as_str())?
            .get(name.as_str())
            .cloned()
    }
    fn occurrence(&self, model: &ModelRef, path: &RelPath) -> bool {
        self.occurrences
            .get(model.file.as_str())
            .is_some_and(|o| o.contains(&path.to_string()))
    }
    fn pmi(&self, model: &ModelRef) -> PmiNames {
        self.pmi
            .get(model.file.as_str())
            .cloned()
            .unwrap_or_default()
    }
    fn bom(&self, model: &ModelRef) -> Option<Bom> {
        self.boms.get(model.file.as_str()).cloned()
    }
}

fn model_files(path: &str, text: &str) -> Vec<String> {
    let Some(doc) = parse_drawing(path, text).document else {
        return Vec::new();
    };
    let mut seen = BTreeSet::new();
    doc.views
        .iter()
        .map(|v| v.model.file.as_str().to_owned())
        .filter(|f| seen.insert(f.clone()))
        .collect()
}

/// The model files a drawing's views show, resolved against the drawing's folder.
pub fn dependencies(path: &str, text: &str) -> Vec<String> {
    model_files(path, text)
        .iter()
        .map(|f| resolve_path(path, f, &Files::new()))
        .collect()
}

/// `compile` for a `.drawing`, as JSON.
pub fn compile_json(path: &str, text: &str, files: &Files) -> String {
    let parsed = parse_drawing(path, text);
    // A new, empty file is an empty drawing named after the file: the editor shows its empty
    // prompt (Auto-drawing, add a view), not a syntax error.
    let empty = text.trim().is_empty().then(|| DrawingDocument {
        path: FilePath::new(path),
        header: Default::default(),
        formats: Vec::new(),
        sheets: Vec::new(),
        views: Vec::new(),
        annotations: Vec::new(),
        tables: Vec::new(),
    });
    let Some(document) = parsed.document.or(empty) else {
        let diagnostics: Vec<Value> = parsed
            .diagnostics
            .iter()
            .map(|d| json!({ "path": path, "start": d.span.start, "end": d.span.end, "severity": "error", "message": d.message }))
            .collect();
        return json!({ "diagnostics": diagnostics }).to_string();
    };
    let mut models = CompiledModels::default();
    let mut displays = serde_json::Map::new();
    let mut documents: BTreeMap<String, PartDocument> = BTreeMap::new();
    // Each drawn part as regenerated, its bodies on the kernel that made them.
    let mut exact: BTreeMap<String, RegeneratedPart> = BTreeMap::new();
    let mut problems: Vec<Diagnostic> = Vec::new();
    let warn = |message: String| Diagnostic {
        path: path.to_owned(),
        start: 0,
        end: 0,
        severity: "warning".into(),
        message,
    };
    for file in model_files(path, text) {
        let at = resolve_path(path, &file, files);
        let Some(source) = files.get(&at) else {
            problems.push(warn(format!(
                "{file} isn't in the workspace, so its views are empty"
            )));
            continue;
        };
        if at.ends_with(".assembly") {
            match compile_assembly(&at, source, files) {
                Ok((index, display, _)) => {
                    let occ = index["occurrences"]
                        .as_array()
                        .map(|rows| {
                            rows.iter()
                                .filter_map(|r| r["path"].as_str().map(str::to_owned))
                                .collect()
                        })
                        .unwrap_or_default();
                    models.occurrences.insert(file.clone(), occ);
                    if let Ok(bom) = serde_json::from_value::<Bom>(index["bom"].clone()) {
                        models.boms.insert(file.clone(), bom);
                    }
                    models.names.insert(file.clone(), BTreeSet::new());
                    displays.insert(file.clone(), json!({ "kind": "assembly", "data": display }));
                }
                Err(_) => problems.push(warn(format!(
                    "{file} doesn't compile, so its views are empty"
                ))),
            }
            continue;
        }
        match regenerate_part(&at, source) {
            Ok(part) => {
                let mut names = BTreeSet::new();
                for body in &part.index.bodies {
                    for name in body.names.iter().chain(body.faces.iter().flatten()) {
                        names.insert(
                            serde_json::to_value(name)
                                .unwrap_or(Value::Null)
                                .to_string(),
                        );
                    }
                }
                models.names.insert(file.clone(), names);
                let mut props = BTreeMap::new();
                if let Some(output) = part.document.parts.first() {
                    if let Some(pn) = &output.part_number {
                        props.insert("part_number".to_owned(), pn.clone());
                    }
                    if let Some(material) = &output.material {
                        // `material.aluminum` names the material the part uses: its name shows.
                        let text = serde_json::to_value(material).ok().and_then(|v| match v {
                            Value::String(s) => {
                                Some(s.strip_prefix("material.").map(capitalized).unwrap_or(s))
                            }
                            other => other.get("name").and_then(Value::as_str).map(str::to_owned),
                        });
                        if let Some(text) = text {
                            props.insert("material".to_owned(), text);
                        }
                    }
                    for p in &output.properties {
                        let value = serde_json::to_value(&p.value).unwrap_or(Value::Null);
                        let text = value
                            .as_str()
                            .map(str::to_owned)
                            .or_else(|| {
                                value
                                    .get("value")
                                    .map(|v| v.to_string().trim_matches('"').to_owned())
                            })
                            .unwrap_or_else(|| value.to_string());
                        props.insert(p.name.as_str().to_owned(), text);
                    }
                }
                models.properties.insert(file.clone(), props);
                let pmi = &part.document.pmi;
                models.pmi.insert(
                    file.clone(),
                    (
                        pmi.tolerances.iter().map(|t| t.name.clone()).collect(),
                        pmi.sizes.iter().map(|t| t.name.clone()).collect(),
                        pmi.datums.iter().map(|d| d.label.clone()).collect(),
                    ),
                );
                documents.insert(file.clone(), part.document.clone());
                displays.insert(
                    file.clone(),
                    json!({ "kind": "part", "data": part.display }),
                );
                exact.insert(file.clone(), part);
            }
            Err(_) => problems.push(warn(format!(
                "{file} doesn't compile, so its views are empty"
            ))),
        }
    }
    let index = build_drawing_index(
        &document,
        &models,
        &values::Values {
            displays: &displays,
            documents: &documents,
        },
    );
    let mut data = serde_json::to_value(&index).unwrap_or(Value::Null);
    // Each view's edges, visible and hidden, from the drawn model's display.
    let views = serde_json::to_value(&document.views).unwrap_or(Value::Null);
    let mut hlr = serde_json::Map::new();
    // A view's lines: exact on the part's kernel when it projects, else from the display.
    let lines_of = |file: &str, display: &Value, axes: &crate::hlr::Axes| -> crate::hlr::ViewLines {
        exact
            .get(file)
            .and_then(|part| {
                let bodies: Vec<_> = part
                    .result
                    .state
                    .bodies
                    .iter()
                    .map(|b| (&b.handle, runmat_cad::regen::kernel::Transform::IDENTITY))
                    .collect();
                crate::hlr::exact::view_lines(part.kernel.as_ref(), &bodies, display, axes)
            })
            .unwrap_or_else(|| crate::hlr::view_lines(display, axes))
    };
    for view in views.as_array().into_iter().flatten() {
        let (Some(name), Some(file)) = (view["name"].as_str(), view["model"]["file"].as_str())
        else {
            continue;
        };
        if let Some(display) = displays.get(file) {
            let kind = &view["kind"];
            let parent = kind["from"].as_str().unwrap_or("");
            let parent_axes = crate::hlr::axes(crate::hlr::orientation(&views, parent, 0));
            let entry = match kind["view"].as_str() {
                // A section through the plane the view names, seen across the parent's cutting
                // line: the parent's right stays right and its depth becomes up.
                Some("section") => {
                    section_entry(display, &document, name, &parent_axes, &views, parent)
                }
                // A detail: the parent's lines, clipped by the sheet to the circle around the
                // centre the view names.
                Some("detail") => {
                    let axes = parent_axes;
                    let mut lines =
                        serde_json::to_value(lines_of(file, display, &axes)).unwrap_or(Value::Null);
                    let center = document
                        .views
                        .iter()
                        .find(|v| v.name == name)
                        .and_then(|v| match &v.kind {
                            runmat_cad::drawing::ViewKind::Detail { center, radius, .. } => {
                                Some((center, radius))
                            }
                            _ => None,
                        })
                        .and_then(|(center, radius)| {
                            Some((point_of(display, center)?, radius_mm(radius)?))
                        });
                    if let (Some((c, r)), Value::Object(o)) = (center, &mut lines) {
                        o.insert(
                            "detail".into(),
                            json!({ "center": [dot(c, axes.u), dot(c, axes.v)], "radius": r }),
                        );
                    }
                    lines
                }
                _ => {
                    let axes = crate::hlr::axes(crate::hlr::orientation(&views, name, 0));
                    serde_json::to_value(lines_of(file, display, &axes)).unwrap_or(Value::Null)
                }
            };
            hlr.insert(name.to_owned(), entry);
        }
    }
    // Where each dimension's references sit in the model (mm): the sheet measures between them.
    let mut anchors = serde_json::Map::new();
    // And how far each planar reference reaches (its box, mm), so extension lines start at the
    // feature's end nearest the dimension line. Holes and edges have none: their extension
    // lines leave the anchor (a hole's centre line).
    let mut extents = serde_json::Map::new();
    for a in &document.annotations {
        let runmat_cad::drawing::AnnotationKind::Dimension { references, .. } = &a.kind else {
            continue;
        };
        let Some(view) = document.views.iter().find(|v| v.name == a.view) else {
            continue;
        };
        let Some(display) = displays.get(view.model.file.as_str()) else {
            continue;
        };
        let points: Option<Vec<[f64; 3]>> =
            references.iter().map(|r| point_of(display, r)).collect();
        if let Some(points) = points {
            anchors.insert(a.id.clone(), json!(points));
            let boxes: Vec<Value> = references.iter().map(|r| extent_of(display, r)).collect();
            extents.insert(a.id.clone(), Value::Array(boxes));
        }
    }
    if let Value::Object(o) = &mut data {
        o.insert("hlr".into(), Value::Object(hlr));
        o.insert("anchors".into(), Value::Object(anchors));
        o.insert("extents".into(), Value::Object(extents));
    }
    json!({
        "projection": { "kind": "drawing", "data": data },
        "models": displays,
        "diagnostics": problems,
    })
    .to_string()
}

fn capitalized(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().chain(c).collect())
        .unwrap_or_default()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn radius_mm(v: &runmat_cad::value::ValueExpr) -> Option<f64> {
    match v {
        runmat_cad::value::ValueExpr::Quantity { value, unit } if unit == "mm" => Some(*value),
        runmat_cad::value::ValueExpr::Number { value } => Some(*value),
        _ => None,
    }
}

/// A model point (mm) a reference names: a cylindrical face's axis at its middle, a face's
/// centre, or where two faces meet (`edges_between`, also after a blend rounded it away).
/// A planar reference's box `[min, max]` (mm), or null for cylinders and edges.
fn extent_of(display: &Value, reference: &runmat_cad::refs::reference::Reference) -> Value {
    let Some(text) = runmat_cad::syntax::refs::print_reference(reference) else {
        return Value::Null;
    };
    let Some((pts, surface)) = values::face(display, &text) else {
        return Value::Null;
    };
    if values::cylinder(surface).is_some() || pts.is_empty() {
        return Value::Null;
    }
    let lo = [0, 1, 2].map(|i| pts.iter().map(|p| p[i]).fold(f64::MAX, f64::min));
    let hi = [0, 1, 2].map(|i| pts.iter().map(|p| p[i]).fold(f64::MIN, f64::max));
    json!([lo, hi])
}

fn point_of(
    display: &Value,
    reference: &runmat_cad::refs::reference::Reference,
) -> Option<[f64; 3]> {
    use runmat_cad::refs::reference::Reference as R;
    let text = |r: &R| runmat_cad::syntax::refs::print_reference(r);
    let mean = |pts: &[[f64; 3]]| {
        let n = pts.len().max(1) as f64;
        [0, 1, 2].map(|i| pts.iter().map(|p| p[i]).sum::<f64>() / n)
    };
    if let R::EdgesBetween { a, b, .. } = reference {
        let (pa, _) = values::face(display, &text(a.as_ref())?)?;
        let (pb, _) = values::face(display, &text(b.as_ref())?)?;
        let mut best = (f64::MAX, [0.0; 3]);
        for p in &pa {
            for q in &pb {
                let d = (0..3).map(|i| (p[i] - q[i]).powi(2)).sum::<f64>();
                if d < best.0 {
                    best = (d, [0, 1, 2].map(|i| (p[i] + q[i]) / 2.0));
                }
            }
        }
        return (best.0 < f64::MAX).then_some(best.1);
    }
    let (pts, surface) = values::face(display, &text(reference)?)?;
    if let Some((p, d, _)) = values::cylinder(surface) {
        let c = mean(&pts);
        let t = dot([c[0] - p[0], c[1] - p[1], c[2] - p[2]], d);
        return Some([0, 1, 2].map(|i| p[i] + d[i] * t));
    }
    Some(mean(&pts))
}

/// A section view's lines and hatch, and its cutting line on the parent view (at `v` in the
/// parent's view millimetres, with the label and the direction the arrows point).
fn section_entry(
    display: &Value,
    document: &runmat_cad::drawing::DrawingDocument,
    name: &str,
    parent: &crate::hlr::Axes,
    views: &Value,
    from: &str,
) -> Value {
    let Some((plane, label)) =
        document
            .views
            .iter()
            .find(|v| v.name == name)
            .and_then(|v| match &v.kind {
                runmat_cad::drawing::ViewKind::Section { plane, label, .. } => {
                    Some((plane, label.clone()))
                }
                _ => None,
            })
    else {
        return Value::Null;
    };
    let _ = views;
    let Some(p0) = point_of(display, plane) else {
        return Value::Null;
    };
    let axes = crate::hlr::Axes {
        u: parent.u,
        v: parent.depth.map(|x| -x),
        depth: parent.v,
    };
    let mut lines =
        serde_json::to_value(crate::hlr::section_lines(display, &axes, p0)).unwrap_or(Value::Null);
    if let Value::Object(o) = &mut lines {
        o.insert(
            "cut".into(),
            json!({ "view": from, "at": dot(p0, parent.v), "label": label, "toward": [dot(axes.depth, parent.u), dot(axes.depth, parent.v)] }),
        );
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    const BRACKET: &str = include_str!("../../tests/corpus/documents/bracket.part");
    pub(crate) const DRAWING: &str = "runmat 0.7\nstd 1\n\ndrawing bracket {\n    format a3 = {\n        size = { size = iso, name = A3 },\n        title_block = [\n            { label = \"Part number\", source = { source = property, name = part_number } }\n        ]\n    }\n\n    sheet \"1\" = { format = a3, scale = [1, 1], projection = third_angle }\n\n    view front = standard(\n        orientation = front,\n        sheet = \"1\",\n        model = { file = \"bracket.part\" },\n        at = [100, 100],\n        display = hidden_lines\n    )\n    view top = projected(\n        from = front,\n        direction = up,\n        sheet = \"1\",\n        model = { file = \"bracket.part\" },\n        at = [100, 200],\n        display = hidden_lines\n    )\n}\n";

    /// The Desktop's sheet tests read what the worker gets for its fixtures. Set
    /// `RUNMAT_DESKTOP_FIXTURES=1` to write it; otherwise the checked-in copy must be current.
    fn desktop_fixture(name: &str, json: &str) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../runmat-private-cad/desktop/src/app/components/cad/fixtures");
        if !dir.exists() {
            return;
        }
        let path = dir.join(name);
        if std::env::var("RUNMAT_DESKTOP_FIXTURES").is_ok() {
            std::fs::write(&path, json).unwrap();
            return;
        }
        let checked_in = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            checked_in == json,
            "{} is stale; rerun with RUNMAT_DESKTOP_FIXTURES=1",
            path.display()
        );
    }

    #[test]
    fn a_drawing_compiles_from_its_text_with_the_displays_of_its_models() {
        let deps = dependencies("/p/bracket.drawing", DRAWING);
        assert_eq!(deps, vec!["/p/bracket.part".to_owned()]);
        let files: Files = deps
            .iter()
            .map(|d| (d.clone(), BRACKET.to_owned()))
            .collect();
        let out: Value =
            serde_json::from_str(&compile_json("/p/bracket.drawing", DRAWING, &files)).unwrap();
        assert_eq!(out["projection"]["kind"], "drawing", "{out}");
        let data = &out["projection"]["data"];
        assert_eq!(data["document"]["views"].as_array().map(Vec::len), Some(2));
        let model = &out["models"]["bracket.part"];
        assert_eq!(model["kind"], "part", "{out}");
        assert!(model["data"]["bodies"]
            .as_array()
            .is_some_and(|b| !b.is_empty()));
        assert_eq!(
            out["diagnostics"].as_array().map(Vec::len),
            Some(0),
            "{out}"
        );
    }

    /// The Desktop's standard drawing of the showcase bracket: three views at 2:1, section
    /// A–A through the hole, detail B of the round, model dimensions, a hole callout and a
    /// GD&T frame from the part's PMI.
    #[test]
    fn the_standard_drawing_measures_its_model_and_cuts_its_section() {
        const PART: &str = include_str!(
            "../../../../../runmat-private-cad/desktop/src/app/components/cad/fixtures/bracket.part"
        );
        const STANDARD: &str = include_str!(
            "../../../../../runmat-private-cad/desktop/src/app/components/cad/fixtures/standard.drawing"
        );
        let files: Files = [("/p/bracket.part".to_owned(), PART.to_owned())]
            .into_iter()
            .collect();
        let compiled = compile_json("/p/standard.drawing", STANDARD, &files);
        desktop_fixture("standard.drawing.compiled.json", &compiled);
        let out: Value = serde_json::from_str(&compiled).unwrap();
        let data = &out["projection"]["data"];
        assert_eq!(
            out["diagnostics"].as_array().map(Vec::len),
            Some(0),
            "{}",
            out["diagnostics"]
        );
        assert_eq!(
            data["diagnostics"].as_array().map(Vec::len),
            Some(0),
            "{}",
            data["diagnostics"]
        );
        let text = &data["text"];
        assert_eq!(text["width"], "80");
        assert_eq!(text["thickness"], "10");
        assert_eq!(text["depth"], "50");
        assert_eq!(text["hole"], "⌀4 THRU");
        assert_eq!(text["parallel"], "∥|0.05|A");
        // The width's references: the two side faces, 80 mm apart along x.
        let w = &data["anchors"]["width"];
        assert!(
            (w[1][0].as_f64().unwrap() - w[0][0].as_f64().unwrap() - 80.0).abs() < 1e-3,
            "{w}"
        );
        // Every feature located and sized (ASME Y14.5): the cut's 20 x 10 and its position
        // from the datum edges, and the hole's position to its axis.
        for (id, value) in [
            ("cut_width", "20"),
            ("cut_height", "10"),
            ("cut_x", "20"),
            ("cut_y", "20"),
            ("hole_x", "60"),
            ("hole_y", "10"),
        ] {
            assert_eq!(text[id], value, "{id}: {text}");
        }
        // A planar reference carries its box; the hole (a cylinder) locates by its axis.
        let e = &data["extents"]["hole_x"];
        assert!(e[0].is_array() && e[1].is_null(), "{e}");
        let rows = data["title_blocks"]["1"].to_string();
        assert!(
            rows.contains("Aluminum") && rows.contains("BR-100") && rows.contains("2:1"),
            "{rows}"
        );
        // Section A–A: the cut through the hole's axis, hatched in two pieces either side of
        // the hole, and its cutting line on the top view at y = 10 mm.
        let section = &data["hlr"]["a_a"];
        let hatch = section["hatch"].as_array().map(Vec::len).unwrap_or(0);
        assert!(hatch >= 2, "{} hatch loops", hatch);
        assert_eq!(section["cut"]["view"], "top");
        assert!(
            (section["cut"]["at"].as_f64().unwrap() - 10.0).abs() < 1e-6,
            "{}",
            section["cut"]
        );
        assert!(section["hidden"].as_array().is_some_and(|h| h.is_empty()));
        // Detail B: centred on the rounded edge as the right view sees it (y = 0, z = 10).
        let c = &data["hlr"]["b"]["detail"]["center"];
        let (u, v) = (c[0].as_f64().unwrap(), c[1].as_f64().unwrap());
        assert!(u.abs() < 1.6 && (v - 10.0).abs() < 1.6, "{c}");
    }

    #[test]
    fn a_missing_model_is_a_warning_and_the_drawing_still_compiles() {
        let out: Value =
            serde_json::from_str(&compile_json("/p/bracket.drawing", DRAWING, &Files::new()))
                .unwrap();
        assert_eq!(out["projection"]["kind"], "drawing");
        assert!(out["models"].as_object().is_some_and(|m| m.is_empty()));
        assert!(out["diagnostics"]
            .as_array()
            .is_some_and(|d| d.iter().any(|d| d["severity"] == "warning")));
    }

    #[test]
    fn an_empty_file_is_an_empty_drawing() {
        let out: Value =
            serde_json::from_str(&compile_json("/p/new.drawing", "\n  \n", &Files::new())).unwrap();
        assert_eq!(out["projection"]["kind"], "drawing", "{out}");
        let document = &out["projection"]["data"]["document"];
        assert_eq!(
            document["sheets"].as_array().map(Vec::len),
            Some(0),
            "{out}"
        );
        assert!(out["diagnostics"]
            .as_array()
            .is_some_and(|d| d.iter().all(|d| d["severity"] != "error")));
    }

    #[test]
    fn syntax_errors_are_diagnostics() {
        let out: Value =
            serde_json::from_str(&compile_json("/p/x.drawing", "drawing x {", &Files::new()))
                .unwrap();
        assert!(out.get("projection").is_none());
        assert!(!out["diagnostics"].as_array().unwrap().is_empty());
    }

    #[test]
    fn auto_drawing_lays_out_views_a_section_and_hole_callouts() {
        const PART: &str = include_str!(
            "../../../../../runmat-private-cad/desktop/src/app/components/cad/fixtures/bracket.part"
        );
        let files: Files = [("/p/bracket.part".to_owned(), PART.to_owned())]
            .into_iter()
            .collect();
        let values = serde_json::json!({ "model": "bracket.part" });
        let (change, summary) =
            auto::auto("/p/auto.drawing", "", &files, values.as_object().unwrap()).unwrap();
        assert!(
            summary.contains("2:1") && summary.contains("section A-A"),
            "{summary}"
        );
        let out: Value =
            serde_json::from_str(&compile_json("/p/auto.drawing", &change.source, &files)).unwrap();
        let data = &out["projection"]["data"];
        assert_eq!(
            data["diagnostics"].as_array().map(Vec::len),
            Some(0),
            "{}\n{}",
            data["diagnostics"],
            change.source
        );
        assert_eq!(data["document"]["views"].as_array().map(Vec::len), Some(4));
        assert_eq!(data["text"]["width"], "80");
        assert_eq!(data["text"]["depth"], "50");
        assert_eq!(data["text"]["thickness"], "10");
        assert_eq!(data["text"]["hole1"], "⌀4 THRU");
        assert!(data["hlr"]["a_a"]["hatch"]
            .as_array()
            .is_some_and(|h| !h.is_empty()));
        // Run again: the drawing keeps its sheet and gains fresh names.
        let (again, _) = auto::auto(
            "/p/auto.drawing",
            &change.source,
            &files,
            values.as_object().unwrap(),
        )
        .unwrap();
        assert!(
            again.source.contains("view front2 = standard("),
            "{}",
            again.source
        );
        assert_eq!(again.source.matches("sheet \"1\" =").count(), 1);
    }
}
