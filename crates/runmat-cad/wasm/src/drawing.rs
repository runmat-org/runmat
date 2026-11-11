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
mod holes;
pub mod layout;
pub mod progress;
pub(crate) mod props;
pub mod title;
pub(crate) mod values;
mod view_picks;

use crate::runtime::{compile_assembly, resolve_path, Diagnostic, Files, RegeneratedPart};
use runmat_cad::assembly::bom::Bom;
use runmat_cad::assembly::model::RelPath;
use runmat_cad::drawing::index::build_drawing_index;
use runmat_cad::drawing::{DrawingDocument, ModelRef, ModelReferences};
use runmat_cad::ids::{FeatureId, FilePath, PropertyName};
use runmat_cad::model::sheet_metal::FlatLayout;
use runmat_cad::model::PartDocument;
use runmat_cad::refs::name::EntityName;
use runmat_cad::refs::reference::Reference;
use runmat_cad::syntax::parse_drawing_in;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

/// A part's PMI names: tolerances, sizes and locations (what a dimension's `pmi` may name), and datums.
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
    /// File → (tolerances, sizes and locations, datums) the part's PMI declares.
    pmi: BTreeMap<String, PmiNames>,
    /// File → each PMI tolerance's geometric characteristic, by name.
    characteristics: BTreeMap<String, BTreeMap<String, runmat_cad::pmi::Characteristic>>,
    /// File → the assembly's flat BOM (its index's), for BOM tables and balloons.
    boms: BTreeMap<String, Bom>,
    /// Assembly file → BOM item definition file → its properties (parts lists' description).
    item_properties: BTreeMap<String, BTreeMap<String, BTreeMap<String, String>>>,
    /// Part file → flat pattern or unfold feature → its layout, the outline as cut.
    flats: BTreeMap<String, BTreeMap<FeatureId, FlatLayout>>,
    /// Part file → each structural-member feature's cut list, in feature order.
    cut_lists: BTreeMap<String, Vec<(String, Vec<runmat_cad::model::weldment::CutListRow>)>>,
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
    fn tolerance_characteristic(&self, model: &ModelRef, name: &str) -> Option<runmat_cad::pmi::Characteristic> {
        self.characteristics.get(model.file.as_str())?.get(name).copied()
    }
    fn bom(&self, model: &ModelRef) -> Option<Bom> {
        self.boms.get(model.file.as_str()).cloned()
    }
    fn flat_layout(&self, model: &ModelRef, feature: &FeatureId) -> Option<FlatLayout> {
        self.flats.get(model.file.as_str())?.get(feature).cloned()
    }
    fn cut_lists(&self, model: &ModelRef) -> Option<Vec<(String, Vec<runmat_cad::model::weldment::CutListRow>)>> {
        self.cut_lists.get(model.file.as_str()).cloned()
    }
    fn item_property(
        &self,
        model: &ModelRef,
        item: &runmat_cad::assembly::bom::BomItem,
        name: &PropertyName,
    ) -> Option<String> {
        let items = self.item_properties.get(model.file.as_str())?;
        let file = item.definition.file.as_str();
        items
            .get(&props::item_key(file, &item.configuration))
            .or_else(|| items.get(file))?
            .get(name.as_str())
            .cloned()
    }
}

fn model_files(path: &str, text: &str, files: &Files) -> Vec<String> {
    let scope = crate::runtime::module_scope(path, text, files);
    let Some(doc) = parse_drawing_in(path, text, &scope).document else {
        return Vec::new();
    };
    let mut seen = BTreeSet::new();
    doc.views
        .iter()
        .map(|v| v.model.file.as_str().to_owned())
        .filter(|f| seen.insert(f.clone()))
        .collect()
}

/// The folders and model files a drawing's views can name (`runmat_cad::modules::dependencies`),
/// and the SVG files its title-block logos read.
pub fn dependencies(path: &str, text: &str) -> Vec<String> {
    let mut out = runmat_cad::modules::dependencies(path, text);
    // The SVG files its title-block logos draw.
    for file in title::image_files(path, text) {
        if !out.contains(&file) {
            out.push(file);
        }
    }
    out
}

/// [`dependencies`] resolved among `files`: the model files the views show in place of the
/// folders they're named in, and the title-block images.
pub fn dependency_files(path: &str, text: &str, files: &Files) -> Vec<String> {
    let mut out: Vec<String> = dependencies(path, text).into_iter().filter(|d| !d.ends_with('/')).collect();
    for file in model_files(path, text, files) {
        let at = resolve_path(path, &file, files);
        if !out.contains(&at) {
            out.push(at);
        }
    }
    out
}

/// `compile` for a `.drawing`, as JSON.
pub fn compile_json(path: &str, text: &str, files: &Files) -> String {
    compile_json_in(path, text, files, None)
}

/// [`compile_json`] with every part the drawing shows regenerated in the named configuration
/// `configuration` (a product family's sheet per configuration).
pub fn compile_json_in(path: &str, text: &str, files: &Files, configuration: Option<&str>) -> String {
    let parsed = parse_drawing_in(path, text, &crate::runtime::module_scope(path, text, files));
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
    let mut exact: BTreeMap<String, Rc<RegeneratedPart>> = BTreeMap::new();
    // Each drawn model's input, which its views' projections are kept by (progress.rs).
    let mut model_keys: BTreeMap<String, String> = BTreeMap::new();
    let stage = progress::stage();
    let mut problems: Vec<Diagnostic> = Vec::new();
    let warn = |message: String| Diagnostic {
        path: path.to_owned(),
        start: 0,
        end: 0,
        severity: "warning".into(),
        message,
    };
    // The sheet stage paints the sheets before any model is regenerated.
    let drawn = if stage == progress::Stage::Sheet { Vec::new() } else { model_files(path, text, files) };
    for file in drawn {
        let at = resolve_path(path, &file, files);
        let Some(source) = files.get(&at) else {
            problems.push(warn(format!(
                "{file} isn't in the workspace, so its views are empty"
            )));
            continue;
        };
        if at.ends_with(".assembly") {
            model_keys.insert(file.clone(), progress::model_key(&at, source, None, Some(files)));
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
                        models
                            .item_properties
                            .insert(file.clone(), props::item_properties(&at, &bom, files));
                        models.boms.insert(file.clone(), bom);
                    }
                    models.names.insert(file.clone(), BTreeSet::new());
                    displays.insert(file.clone(), tagged("assembly", display));
                }
                Err(_) => problems.push(warn(format!(
                    "{file} doesn't compile, so its views are empty"
                ))),
            }
            continue;
        }
        model_keys.insert(file.clone(), progress::model_key(&at, source, configuration, None));
        match progress::regenerated_part(&at, source, configuration) {
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
                let props = props::output_properties(&part.document);
                models.properties.insert(file.clone(), props);
                let pmi = &part.document.pmi;
                models.pmi.insert(
                    file.clone(),
                    (
                        pmi.tolerances.iter().map(|t| t.name.clone()).collect(),
                        pmi.sizes.iter().map(|t| t.name.clone()).chain(pmi.locations.iter().map(|t| t.name.clone())).collect(),
                        pmi.datums.iter().map(|d| d.label.clone()).collect(),
                    ),
                );
                models.characteristics.insert(file.clone(), pmi.tolerances.iter().map(|t| (t.name.clone(), t.characteristic)).collect());
                let flats = part
                    .document
                    .features
                    .iter()
                    .filter_map(|f| Some((f.id.clone(), crate::exchange::flat::flat_layout_of(&part, &f.id)?)))
                    .collect();
                models.flats.insert(file.clone(), flats);
                let cut_lists: Vec<_> = part
                    .document
                    .features
                    .iter()
                    .filter_map(|f| match part.result.state.records.get(&f.id)? {
                        runmat_cad::regen::records::FeatureRecord::Weldment { cut_list, .. } if !cut_list.is_empty() => {
                            Some((f.id.to_string(), cut_list.clone()))
                        }
                        _ => None,
                    })
                    .collect();
                models.cut_lists.insert(file.clone(), cut_lists);
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
    // The models as the output carries them (in the browser host, mesh arrays as references to
    // the bytes beside the JSON). The drawing measures and projects their numbers in place: its
    // readers take each mesh array through `wire::numbers`, which reads a reference from the
    // bytes, so no inline copy of an assembly's display (tens of megabytes) is held.
    let models_out = displays;
    let displays = &models_out;
    let index = build_drawing_index(
        &document,
        &models,
        &values::Values {
            displays,
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
        if let Some(full) = displays.get(file) {
            // A view's projection is kept by its model's input and its definition (progress.rs).
            // A view this stage leaves pending comes anyway when the cache holds it; otherwise
            // it is only asked for, so a probe fetches it with the others.
            let sheet_scale = document.sheets.iter().find(|s| view["sheet"] == s.name.as_str()).map(|s| s.scale).filter(|s| s.1 != 0).map(|s| s.0 as f64 / s.1 as f64);
            let key = model_keys.get(file).map(|model| progress::view_key(model, &views, name, sheet_scale.unwrap_or(1.0)));
            if !stage.projects(name) && !key.as_deref().is_some_and(crate::runtime::part_cache::holds) {
                if let Some(key) = &key {
                    crate::runtime::part_cache::want(key);
                }
                continue;
            }
            let project = || -> Value {
                // The occurrences the view leaves out (`exclude`), and everything under them.
                let excluded: Vec<String> = document
                    .views
                    .iter()
                    .find(|v| v.name == name)
                    .map(|v| v.exclude.iter().map(|x| x.to_string()).collect())
                    .unwrap_or_default();
                let shown = without_occurrences(full, &excluded);
                let display = shown.as_ref().unwrap_or(full);
                let kind = &view["kind"];
                let parent = kind["from"].as_str().unwrap_or("");
                let parent_axes = crate::hlr::axes(crate::hlr::orientation(&views, parent, 0));
                // The projection each view's lines are drawn in, so the sheet places dimensions and
                // callouts in the same frame (a section's and an auxiliary view's differ from their
                // parent's orientation).
                let mut drawn_in: Option<crate::hlr::Axes> = Some(parent_axes);
                let entry = match kind["view"].as_str() {
                    // A section through the plane the view names, seen across the parent's cutting
                    // line: the parent's right stays right and its depth becomes up.
                    Some("section") => {
                        drawn_in = Some(section_axes(&parent_axes));
                        section_entry(display, &document, name, &parent_axes, &views, parent)
                    }
                    // A detail: the parent's lines, clipped by the sheet to the circle around the
                    // centre the view names.
                    // A detail of a section shows the cut (its hatch and the faces beyond it), seen
                    // as the section sees it.
                    Some("detail") => {
                        let parent_kind = views.as_array().and_then(|vs| vs.iter().find(|v| v["name"] == parent)).map(|v| &v["kind"]);
                        let (axes, mut lines) = match parent_kind {
                            Some(pk) if pk["view"] == "section" => {
                                let from = pk["from"].as_str().unwrap_or("");
                                let from_axes = crate::hlr::axes(crate::hlr::orientation(&views, from, 0));
                                let mut cut = section_entry(display, &document, parent, &from_axes, &views, from);
                                if let Value::Object(o) = &mut cut {
                                    o.remove("cut");
                                }
                                (section_axes(&from_axes), cut)
                            }
                            _ => (parent_axes, serde_json::to_value(lines_of(file, display, &parent_axes)).unwrap_or(Value::Null)),
                        };
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
                        drawn_in = Some(axes);
                        if let (Some((c, r)), Value::Object(o)) = (center, &mut lines) {
                            o.insert(
                                "detail".into(),
                                json!({ "center": [dot(c, axes.u), dot(c, axes.v)], "radius": r }),
                            );
                            o.insert("picks".into(), view_picks::picks(display, &axes));
                        }
                        lines
                    }
                    // A flat pattern: the developed sheet (outline as cut, bend lines and notes,
                    // the bend table), in the layout's millimetres.
                    Some("flat_pattern") => {
                        let sheet = document.views.iter().find(|v| v.name == name).and_then(|v| {
                            let runmat_cad::drawing::ViewKind::FlatPattern { feature, .. } = &v.kind else {
                                return None;
                            };
                            let layout = models.flat_layout(&v.model, feature)?;
                            runmat_cad::drawing::flat_pattern::view_sheet(&v.kind, &layout)
                        });
                        drawn_in = None;
                        json!({ "visible": [], "hidden": [], "flat": sheet })
                    }
                    // An auxiliary view: square to the face its edge shows edge-on in the parent.
                    Some("auxiliary") => {
                        let axes = document
                            .views
                            .iter()
                            .find(|v| v.name == name)
                            .and_then(|v| match &v.kind {
                                runmat_cad::drawing::ViewKind::Auxiliary { edge, .. } => runmat_cad::syntax::refs::print_reference(edge),
                                _ => None,
                            })
                            .and_then(|r| view_picks::auxiliary_axes(display, &parent_axes, &r))
                            .unwrap_or(parent_axes);
                        drawn_in = Some(axes);
                        let mut lines = serde_json::to_value(lines_of(file, display, &axes)).unwrap_or(Value::Null);
                        if let Value::Object(o) = &mut lines {
                            o.insert("picks".into(), view_picks::picks(display, &axes));
                        }
                        lines
                    }
                    // A broken-out section: the parent's lines with the cut inside its boundary.
                    Some("broken_out") => {
                        let parent_lines = lines_of(file, display, &parent_axes);
                        let cut = document.views.iter().find(|v| v.name == name).and_then(|v| match &v.kind {
                            runmat_cad::drawing::ViewKind::BrokenOut { boundary, depth, .. } => Some((boundary.clone(), radius_mm(depth)?)),
                            _ => None,
                        });
                        let lines = match cut {
                            Some((boundary, depth)) => view_picks::broken_out(display, &parent_axes, &parent_lines, &boundary, depth),
                            None => parent_lines,
                        };
                        let mut lines = serde_json::to_value(lines).unwrap_or(Value::Null);
                        if let Value::Object(o) = &mut lines {
                            o.insert("picks".into(), view_picks::picks(display, &parent_axes));
                        }
                        lines
                    }
                    _ => {
                        let axes = crate::hlr::axes(crate::hlr::orientation(&views, name, 0));
                        drawn_in = Some(axes);
                        let mut lines = serde_json::to_value(lines_of(file, display, &axes)).unwrap_or(Value::Null);
                        // What a click on each line means (the face it shows edge-on, or the edge).
                        if let Value::Object(o) = &mut lines {
                            o.insert("picks".into(), view_picks::picks(display, &axes));
                        }
                        lines
                    }
                };
                let mut entry = entry;
                if let (Some(a), Value::Object(o)) = (drawn_in, &mut entry) {
                    o.insert("axes".into(), json!({ "u": a.u, "v": a.v, "depth": a.depth }));
                }
                // As few points as the sheet needs at the view's scale; hidden lines only where
                // the view draws them.
                let ratio = |v: &Value| v.as_array().filter(|a| a.len() == 2).and_then(|a| Some(a[0].as_f64()? / a[1].as_f64().filter(|d| *d != 0.0)?));
                let scale = ratio(&view["scale"]).or(sheet_scale).unwrap_or(1.0);
                let shows_hidden = !matches!(view["display"].as_str(), Some("visible_lines" | "shaded"));
                crate::hlr::thin::thin(&mut entry, scale, shows_hidden);
                entry
            };
            let entry = match key {
                Some(key) => match crate::runtime::part_cache::view_lines(key, project) {
                    Some(entry) => entry,
                    // A probe left it out: the page fetches it, then compiles again.
                    None => continue,
                },
                None => project(),
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
    let mut radii = serde_json::Map::new();
    // An angular dimension's two planar faces' normals, so the sheet draws its arc between the
    // lines the faces show edge-on.
    let mut normals = serde_json::Map::new();
    // A diameter or radius dimension's cylinder: its axis, radius and the span of its face along
    // the axis (mm, from the anchor), so a view that shows the cylinder side-on dimensions it
    // across its two outline lines.
    let mut cylinders = serde_json::Map::new();
    for a in &document.annotations {
        let Some(view) = document.views.iter().find(|v| v.name == a.view) else {
            continue;
        };
        let Some(display) = displays.get(view.model.file.as_str()) else {
            continue;
        };
        // Frames, datum symbols, finish marks and note leaders point at what they annotate.
        if let Some(target) = values::leader_target(&a.kind, documents.get(view.model.file.as_str())) {
            if let Some(p) = point_of(display, &target) {
                anchors.insert(a.id.clone(), json!([p]));
                extents.insert(a.id.clone(), json!([extent_of(display, &target)]));
                // A cylinder's radius (mm), so the leader ends on the bore, not its boss.
                if let Some(r) = values::radius_of(display, &target) {
                    radii.insert(a.id.clone(), json!(r));
                }
            }
            continue;
        }
        let (references, condition) = match &a.kind {
            runmat_cad::drawing::AnnotationKind::Dimension { references, condition, .. } => (references, *condition),
            runmat_cad::drawing::AnnotationKind::Centerline { between } => (between, runmat_cad::drawing::ArcCondition::Center),
            _ => continue,
        };
        let points: Option<Vec<[f64; 3]>> =
            references.iter().map(|r| point_of(display, r)).collect();
        if let Some(mut points) = points {
            values::arc_anchors(display, references, &mut points, condition);
            anchors.insert(a.id.clone(), json!(points));
            let boxes: Vec<Value> = references.iter().map(|r| extent_of(display, r)).collect();
            extents.insert(a.id.clone(), Value::Array(boxes));
            if let (runmat_cad::drawing::AnnotationKind::Dimension { style: runmat_cad::drawing::DimensionStyle::Diameter | runmat_cad::drawing::DimensionStyle::Radius, .. }, [r], [p]) =
                (&a.kind, references.as_slice(), points.as_slice())
            {
                if let Some(c) = cylinder_of(display, r, *p) {
                    cylinders.insert(a.id.clone(), c);
                }
            }
            if matches!(&a.kind, runmat_cad::drawing::AnnotationKind::Dimension { style: runmat_cad::drawing::DimensionStyle::Angular, .. }) {
                let ns: Vec<Value> = references.iter().map(|r| normal_of(display, r)).collect();
                normals.insert(a.id.clone(), Value::Array(ns));
            }
        }
    }
    let titles = title::title_layouts(path, &document, &index.title_blocks, files, &mut problems);
    // Each sheet's drafting style from its format's standard: text height, arrows, extension
    // lines, dimension offsets, figure orientation and line weights. Exports and the Desktop
    // sheet draw annotations by it.
    let styles: serde_json::Map<String, Value> = document
        .sheets
        .iter()
        .filter_map(|sheet| {
            let format = document.formats.iter().find(|f| f.name == sheet.format)?;
            Some((sheet.name.clone(), serde_json::to_value(format.standard.style()).ok()?))
        })
        .collect();
    if let Value::Object(o) = &mut data {
        o.insert("title_layouts".into(), titles);
        o.insert("styles".into(), Value::Object(styles));
        o.insert("hlr".into(), Value::Object(hlr));
        // Views still to come (a staged compile, progress.rs): the sheet draws them as pending.
        if stage.partial() {
            o.insert("pending".into(), Value::Bool(true));
        }
        o.insert("anchors".into(), Value::Object(anchors));
        o.insert("extents".into(), Value::Object(extents));
        o.insert("radii".into(), Value::Object(radii));
        o.insert("normals".into(), Value::Object(normals));
        o.insert("cylinders".into(), Value::Object(cylinders));
        o.insert("hole_tables".into(), holes::hole_tables(&document, &exact, displays));
    }
    // The sheets laid out once, here: the Desktop's sheet and every export draw this layout.
    let layouts = layout::layouts(&data, displays);
    if let Value::Object(o) = &mut data {
        o.insert("layouts".into(), layouts);
    }
    // The models' numbers as JSON values and the regenerated parts are done with: free them
    // before the answer is written, the step that holds the most at once.
    drop(exact);
    drop(documents);
    let mut answer = serde_json::Map::new();
    answer.insert("projection".into(), tagged("drawing", data));
    // A staged compile's sheet draws from its layout alone; the models come with the last stage.
    if !stage.partial() {
        answer.insert("models".into(), Value::Object(models_out));
    }
    answer.insert("diagnostics".into(), serde_json::to_value(&problems).unwrap_or(Value::Null));
    json_text(&Value::Object(answer))
}

/// `{ "kind": kind, "data": data }`, holding `data` as it is: `json!` would serialize a copy of
/// it, and a drawing's data and an assembly's display run to hundreds of thousands of values.
pub(crate) fn tagged(kind: &str, data: Value) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("kind".into(), Value::from(kind));
    out.insert("data".into(), data);
    Value::Object(out)
}

/// `value` as compact JSON text, written once into a buffer of its exact size. Growing the
/// buffer by doubling holds the old and the new buffer while it copies, up to three times the
/// text (an assembly drawing's text runs to tens of megabytes). One byte more is left for the
/// NUL the host adds when it hands the text out as a C string.
pub(crate) fn json_text(value: &Value) -> String {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    if serde_json::to_writer(&mut count, value).is_err() {
        return value.to_string();
    }
    let mut out = Vec::with_capacity(count.0 + 1);
    match serde_json::to_writer(&mut out, value).map(|()| String::from_utf8(out)) {
        Ok(Ok(text)) => text,
        _ => value.to_string(),
    }
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
/// A planar face's normal, or null for any other reference.
fn normal_of(display: &Value, reference: &runmat_cad::refs::reference::Reference) -> Value {
    runmat_cad::syntax::refs::print_reference(reference)
        .and_then(|text| values::face(display, &text).map(|(_, surface)| surface["normal"].clone()))
        .filter(Value::is_array)
        .unwrap_or(Value::Null)
}

/// A cylindrical reference's axis, radius and the span of its face along the axis from `at`
/// (mm): `{ axis, radius, span: [from, to] }`.
fn cylinder_of(display: &Value, reference: &runmat_cad::refs::reference::Reference, at: [f64; 3]) -> Option<Value> {
    let text = runmat_cad::syntax::refs::print_reference(reference)?;
    let (pts, surface) = values::face(display, &text)?;
    let (_, d, r) = values::cylinder(surface)?;
    let along = |p: &[f64; 3]| (0..3).map(|i| (p[i] - at[i]) * d[i]).sum::<f64>();
    let lo = pts.iter().map(along).fold(f64::MAX, f64::min);
    let hi = pts.iter().map(along).fold(f64::MIN, f64::max);
    (lo <= hi).then(|| json!({ "axis": d, "radius": r, "span": [lo, hi] }))
}

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
/// A section's axes, seen across its parent's cutting line: the parent's right stays right and
/// its depth becomes up.
fn section_axes(parent: &crate::hlr::Axes) -> crate::hlr::Axes {
    crate::hlr::Axes { u: parent.u, v: parent.depth.map(|x| -x), depth: parent.v }
}

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
    let axes = section_axes(parent);
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
mod tests;

/// An assembly display without the occurrences at these paths or under them; `None` when the
/// view leaves nothing out.
fn without_occurrences(display: &Value, paths: &[String]) -> Option<Value> {
    if paths.is_empty() || display["kind"] != "assembly" {
        return None;
    }
    let mut out = display.clone();
    let occurrences = out["data"]["occurrences"].as_array_mut()?;
    occurrences.retain(|occ| {
        let path = occ["path"].as_str().unwrap_or("");
        !paths.iter().any(|x| path == x || path.starts_with(&format!("{x}/")))
    });
    Some(out)
}

#[cfg(test)]
mod cut_list_tests;

#[cfg(test)]
mod exclude_tests {
    use super::without_occurrences;
    use serde_json::json;

    #[test]
    fn a_view_leaves_out_excluded_occurrences_and_everything_under_them() {
        let display = json!({ "kind": "assembly", "data": { "prototypes": [], "occurrences": [
            { "path": "housing" }, { "path": "lid" }, { "path": "lid_screw1" }, { "path": "pair/s1" }, { "path": "pair/s2" }, { "path": "pairs" }
        ] } });
        let shown = without_occurrences(&display, &["lid".into(), "pair".into()]).unwrap();
        let paths: Vec<_> = shown["data"]["occurrences"].as_array().unwrap().iter().map(|o| o["path"].as_str().unwrap().to_owned()).collect();
        assert_eq!(paths, ["housing", "lid_screw1", "pairs"]);
        assert!(without_occurrences(&display, &[]).is_none());
    }
}
