//! `design.inspect`: a design's structure by name, so a client can address geometry and
//! circuitry the way the editor's Outline does.

use crate::check::display_value;
use crate::{kind_of, name_matches, text_of, DesignKind, Files, ToolError};
use runmat_cad::evaluate::{AnalysisRequest, AnalysisResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// A run's input values (`set`), shared with the other tools that regenerate a design.
#[path = "run_set.rs"]
pub mod run_set;
// Nested here while lib.rs holds another lane's work in progress.
#[cfg(feature = "m-values")]
pub mod m_values;
mod surface;
pub use surface::Ray;

/// What to read. Parts: `faces`, `regions`, `published`, `inputs`, `features`, `bodies`,
/// `mass`, `surface` (named, with `ray`). Assemblies: `occurrences`, `mates`, `bom`, `mass`,
/// `interference` and `surface` (named; `changed` and `ray`). Circuits: `components`, `nets`,
/// `net_classes`, `bom`, `sheets`, `erc_power`, and with a board `rooms`, `footprints`, `layers`,
/// `unplaced` and `eco`. Drawings: `sheets`, `views`, `annotations`, `models` (the inputs of
/// the parts the views show).
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InspectInput {
    /// Workspace path of the design file.
    pub path: String,
    /// The sections to return; every section the file has when absent.
    #[serde(default)]
    pub sections: Option<Vec<String>>,
    /// Keep rows whose name, reference, path or designator matches (`*` globs; otherwise a
    /// case-insensitive substring).
    #[serde(default)]
    pub filter: Option<String>,
    /// Density for mass properties, kg/m³. Without it a part's mass uses 7850 (steel) and an
    /// assembly's uses each part's own material, as the Evaluate tab does; with it, every body
    /// takes this density.
    #[serde(default)]
    pub density: Option<f64>,
    /// Input values for this run, by input name (`{"thickness": "3.5 mm"}`): the part or
    /// assembly regenerates with them; the file is not changed. An unknown name is an
    /// `unknown_input` error listing the declared inputs.
    #[serde(default)]
    pub set: Option<BTreeMap<String, String>>,
    /// At most this many rows per section (default 200).
    #[serde(default)]
    pub limit: Option<usize>,
    /// Assemblies, with the `interference` section: the occurrences a change added or moved
    /// (names or paths; `*` globs). Only the pairs one of them is in are checked, so the answer
    /// lists the overlaps the change brings and leaves out those the assembly already had
    /// between other parts.
    #[serde(default)]
    pub changed: Option<Vec<String>>,
    /// The `surface` section's ray: where a line from a point along a direction meets the
    /// bodies, with the face and its normal there.
    #[serde(default)]
    pub ray: Option<Ray>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct InspectOutput {
    pub path: String,
    pub kind: DesignKind,
    /// Section name → rows.
    pub sections: BTreeMap<String, Value>,
    /// Section name → rows before `filter` and `limit`.
    pub totals: BTreeMap<String, usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

struct Sections<'a> {
    input: &'a InspectInput,
    out: InspectOutput,
}

impl Sections<'_> {
    /// The caller named this section (an expensive one isn't part of "every section").
    fn named(&self, name: &str) -> bool {
        self.input
            .sections
            .as_ref()
            .is_some_and(|s| s.iter().any(|x| x == name))
    }
    fn wants(&self, name: &str) -> bool {
        self.input
            .sections
            .as_ref()
            .is_none_or(|s| s.iter().any(|x| x == name))
    }
    /// Adds `rows`, keeping those whose `key` text matches the filter.
    fn add(&mut self, name: &str, rows: Vec<Value>, keys: &[&str]) {
        if !self.wants(name) {
            return;
        }
        self.out.totals.insert(name.to_owned(), rows.len());
        let limit = self.input.limit.unwrap_or(200);
        let kept: Vec<Value> = rows
            .into_iter()
            .filter(|r| {
                self.input.filter.is_none()
                    || keys.iter().any(|k| {
                        let v = &r[*k];
                        match v {
                            Value::String(s) => name_matches(self.input.filter.as_deref(), s),
                            Value::Array(a) => a.iter().any(|x| {
                                x.as_str()
                                    .is_some_and(|s| name_matches(self.input.filter.as_deref(), s))
                            }),
                            _ => false,
                        }
                    })
            })
            .take(limit)
            .collect();
        self.out
            .sections
            .insert(name.to_owned(), Value::Array(kept));
    }
    fn value(&mut self, name: &str, v: Value) {
        if self.wants(name) {
            self.out.sections.insert(name.to_owned(), v);
        }
    }
}

pub fn inspect(input: &InspectInput, files: &Files) -> Result<InspectOutput, ToolError> {
    match input.set.as_ref().filter(|set| !set.is_empty()) {
        Some(set) => inspect_files(input, &run_set::with_inputs(&input.path, files, set)?),
        None => inspect_files(input, files),
    }
}

fn inspect_files(input: &InspectInput, files: &Files) -> Result<InspectOutput, ToolError> {
    let kind = kind_of(&input.path)?;
    let text = text_of(files, &input.path)?;
    let mut s = Sections {
        input,
        out: InspectOutput {
            path: input.path.clone(),
            kind,
            sections: BTreeMap::new(),
            totals: BTreeMap::new(),
            notes: Vec::new(),
        },
    };
    match kind {
        DesignKind::Part => part(&mut s, &input.path, text)?,
        DesignKind::Assembly => assembly(&mut s, &input.path, text, files)?,
        DesignKind::Drawing => drawing(&mut s, &input.path, text, files),
        DesignKind::Circuit | DesignKind::Layout => circuit(&mut s, &input.path, text, files)?,
    }
    Ok(s.out)
}

fn names(v: &impl Serialize) -> Vec<String> {
    match serde_json::to_value(v).unwrap_or(Value::Null) {
        Value::Array(a) => a
            .iter()
            .map(|x| match x {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect(),
        Value::String(s) => vec![s],
        other => vec![other.to_string()],
    }
}

/// Area, centroid and mean normal of a face's triangles (metres).
fn face_geometry(positions: &[f32], triangles: &[u32]) -> (f64, [f64; 3], [f64; 3]) {
    let p = |i: u32| {
        let i = i as usize * 3;
        [
            positions[i] as f64,
            positions[i + 1] as f64,
            positions[i + 2] as f64,
        ]
    };
    let mut area = 0.0;
    let mut c = [0.0; 3];
    let mut n = [0.0; 3];
    for t in triangles.chunks_exact(3) {
        let (a, b, d) = (p(t[0]), p(t[1]), p(t[2]));
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
        let x = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let twice = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        area += twice / 2.0;
        for k in 0..3 {
            c[k] += (a[k] + b[k] + d[k]) / 3.0 * twice / 2.0;
            n[k] += x[k];
        }
    }
    if area > 0.0 {
        c = c.map(|v| v / area);
    }
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 0.0 {
        n = n.map(|v| v / l);
    }
    (area, c, n)
}

fn round(v: f64, places: i32) -> f64 {
    let f = 10f64.powi(places);
    (v * f).round() / f
}

fn mm3(p: [f64; 3]) -> [f64; 3] {
    p.map(|v| round(v * 1e3, 4))
}

fn mass_value(m: &runmat_cad::evaluate::MassProperties, density: f64) -> Value {
    json!({
        "density": density,
        "volume_mm3": round(m.volume * 1e9, 3),
        "mass_kg": round(m.volume * density, 6),
        "centroid_mm": mm3(m.centroid),
        // About the centroid, kg·mm².
        "inertia_kg_mm2": m.inertia.map(|row| row.map(|v| round(v * density * 1e6, 6))),
    })
}

fn part(s: &mut Sections, path: &str, text: &str) -> Result<(), ToolError> {
    let part = runmat_cad_wasm::runtime::regenerate_part(path, text).map_err(|d| {
        ToolError::new(
            "not_compiled",
            format!(
                "{path} has syntax errors (design.check lists them): {}",
                d.first().map(|d| d.message.as_str()).unwrap_or("")
            ),
        )
    })?;
    let mut faces = Vec::new();
    for body in &part.display.bodies {
        for face in &body.faces {
            let (area, centroid, normal) = face_geometry(&face.positions, &face.triangles);
            let reference = face.reference.clone().unwrap_or_else(|| {
                format!(
                    "feature.{}.face[{}:{}]",
                    face.features.first().map(String::as_str).unwrap_or("body"),
                    body.index,
                    face.index
                )
            });
            faces.push(json!({
                "reference": reference,
                "names": face.names,
                "body": body.index,
                "face": face.index,
                "features": face.features,
                "area_mm2": round(area * 1e6, 3),
                "centroid_mm": mm3(centroid),
                "normal": normal.map(|v| round(v, 4)),
            }));
        }
    }
    s.add("faces", faces, &["reference", "names"]);
    let regions = part
        .index
        .regions
        .iter()
        .map(|r| {
            json!({
                "name": r.name.to_string(),
                "kind": serde_json::to_value(r.kind).unwrap_or(Value::Null),
                "entities": names(&r.entities),
                "ok": r.ok,
            })
        })
        .collect();
    s.add("regions", regions, &["name", "entities"]);
    let published = part
        .index
        .publishes
        .iter()
        .map(|p| {
            json!({
                "name": p.name.to_string(),
                "auto": p.auto,
                "ok": p.ok,
                "entities": names(&p.entities),
            })
        })
        .collect();
    s.add("published", published, &["name", "entities"]);
    let inputs = part
        .index
        .inputs
        .iter()
        .map(|i| {
            json!({
                "name": i.name.to_string(),
                "kind": i.kind,
                "si": i.si,
                "display": i.si.map(|v| display_value(v, &i.kind)),
                "error": i.error,
            })
        })
        .collect();
    s.add("inputs", inputs, &["name"]);
    let features = part
        .index
        .features
        .iter()
        .map(|f| {
            json!({
                "id": f.id.0,
                "kind": f.kind,
                "status": serde_json::to_value(&f.status).unwrap_or(Value::Null),
                "references": f.references.iter().map(|r| json!({
                    "param": r.param, "display": r.display, "status": r.status,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    s.add("features", features, &["id", "kind"]);
    let bodies = part
        .index
        .bodies
        .iter()
        .map(|b| {
            json!({
                "names": names(&b.names),
                "faces": b.faces.len(),
                "edges": b.edge_count,
                "vertices": b.vertex_count,
            })
        })
        .collect();
    s.add("bodies", bodies, &["names"]);
    if s.wants("mass") {
        let density = s.input.density.unwrap_or(7850.0);
        let handles: Vec<_> = part.result.state.bodies.iter().map(|b| &b.handle).collect();
        let mass = if handles.is_empty() {
            Value::Null
        } else {
            match part
                .kernel
                .analyze(&AnalysisRequest::MassProperties { density: 1.0 }, &handles)
            {
                Ok(AnalysisResult::MassProperties(m)) => mass_value(&m, density),
                Ok(_) => Value::Null,
                Err(e) => {
                    s.out.notes.push(format!("mass properties: {e:?}"));
                    Value::Null
                }
            }
        };
        s.value("mass", mass);
    }
    if s.named("surface") {
        let value = surface::surface(s.input.ray.as_ref(), path, text, &Files::new(), None)?;
        s.value("surface", value);
    }
    Ok(())
}

fn assembly(s: &mut Sections, path: &str, text: &str, files: &Files) -> Result<(), ToolError> {
    let (index, problems) = runmat_cad_wasm::runtime::assembly_index(path, text, files)
        .map_err(|d| {
            ToolError::new(
                "not_compiled",
                format!(
                    "{path} has syntax errors (design.check lists them): {}",
                    d.first().map(|d| d.message.as_str()).unwrap_or("")
                ),
            )
        })?;
    for p in problems {
        s.out.notes.push(p.message);
    }
    let rows = |v: &Value| v.as_array().cloned().unwrap_or_default();
    s.add(
        "occurrences",
        rows(&index["occurrences"]),
        &["path", "file", "name"],
    );
    s.add("mates", rows(&index["mates"]), &["name", "a", "b", "kind"]);
    let bom = &index["bom"];
    let bom_rows = bom["rows"]
        .as_array()
        .or_else(|| bom.as_array())
        .cloned()
        .unwrap_or_default();
    s.add(
        "bom",
        bom_rows,
        &["item", "file", "part_number", "description"],
    );
    s.add("fasteners", rows(&index["fasteners"]), &["name"]);
    let occurrence_paths: Vec<String> = rows(&index["occurrences"])
        .iter()
        .filter_map(|o| o["path"].as_str().map(str::to_owned))
        .collect();
    // Interference: occurrences whose exact bodies share material, the Evaluate tab's check.
    // Only when asked for by name: it intersects every pair of bodies (or a change's pairs).
    if s.named("interference") {
        let mut values = serde_json::Map::new();
        if let Some(changed) = &s.input.changed {
            let paths = surface::occurrences_named(changed, &occurrence_paths)?;
            values.insert("changed".into(), json!(paths));
            s.out.notes.push(format!(
                "interference: only the pairs {} {} in were checked; overlaps the assembly already had between other parts aren't listed",
                paths.join(", "),
                if paths.len() == 1 { "is" } else { "are" }
            ));
        }
        let mut value = evaluated_with(path, text, files, "evaluate.interference", values).unwrap_or_else(|e| {
            s.out.notes.push(format!("interference: {e}"));
            Value::Null
        });
        overlap_volumes_in_mm3(&mut value);
        s.value("interference", value);
    }
    if s.named("surface") {
        let value = surface::surface(s.input.ray.as_ref(), path, text, files, Some(&occurrence_paths))?;
        s.value("surface", value);
    }
    if s.wants("mass") && s.input.density.is_none() {
        // Each part's own material (or density property), as the Evaluate tab's mass properties.
        let value = evaluated(path, text, files, "evaluate.mass-properties").unwrap_or_else(|e| {
            s.out.notes.push(format!("mass properties: {e}"));
            Value::Null
        });
        s.value("mass", value);
    } else if s.wants("mass") {
        // The assembly's mass: each occurrence's part mass at the given density.
        let density = s.input.density.unwrap_or(7850.0);
        let mut total = 0.0;
        let mut weighted = [0.0; 3];
        let mut counted = 0;
        let mut cache: BTreeMap<String, Option<runmat_cad::evaluate::MassProperties>> =
            BTreeMap::new();
        for occ in index["occurrences"].as_array().into_iter().flatten() {
            let Some(file) = occ["file"].as_str() else {
                continue;
            };
            if !file.ends_with(".part") {
                continue;
            }
            let at = runmat_cad_wasm::runtime::resolve_path(path, file, files);
            let props = cache
                .entry(at.clone())
                .or_insert_with(|| {
                    let t = files.get(&at)?;
                    let p = runmat_cad_wasm::runtime::regenerate_part(&at, t).ok()?;
                    let handles: Vec<_> = p.result.state.bodies.iter().map(|b| &b.handle).collect();
                    match p
                        .kernel
                        .analyze(&AnalysisRequest::MassProperties { density: 1.0 }, &handles)
                    {
                        Ok(AnalysisResult::MassProperties(m)) => Some(m),
                        _ => None,
                    }
                })
                .clone();
            if let Some(m) = props {
                let t: Vec<f64> = occ["transform"]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_f64).collect())
                    .unwrap_or_default();
                let c = if t.len() >= 12 {
                    let x = m.centroid;
                    [
                        t[0] * x[0] + t[1] * x[1] + t[2] * x[2] + t[9],
                        t[3] * x[0] + t[4] * x[1] + t[5] * x[2] + t[10],
                        t[6] * x[0] + t[7] * x[1] + t[8] * x[2] + t[11],
                    ]
                } else {
                    m.centroid
                };
                let mass = m.volume * density;
                total += mass;
                for k in 0..3 {
                    weighted[k] += c[k] * mass;
                }
                counted += 1;
            }
        }
        let centroid = if total > 0.0 {
            weighted.map(|v| v / total)
        } else {
            [0.0; 3]
        };
        s.value(
            "mass",
            json!({
                "density": density,
                "mass_kg": round(total, 6),
                "centroid_mm": mm3(centroid),
                "parts_counted": counted,
            }),
        );
    }
    Ok(())
}

/// An Evaluate command's result on an assembly: its one-line summary, the typed result and what
/// it couldn't evaluate.
fn evaluated(path: &str, text: &str, files: &Files, command: &str) -> Result<Value, String> {
    evaluated_with(path, text, files, command, Default::default())
}

fn evaluated_with(
    path: &str,
    text: &str,
    files: &Files,
    command: &str,
    values: serde_json::Map<String, Value>,
) -> Result<Value, String> {
    let request = runmat_cad_wasm::analyze::AnalyzeRequest {
        command: command.to_owned(),
        values,
    };
    let a = runmat_cad_wasm::analyze::analyze(path, text, files, &request)?;
    Ok(json!({ "summary": a.summary, "result": a.data, "problems": a.problems }))
}

fn drawing(s: &mut Sections, path: &str, text: &str, files: &Files) {
    let compiled: Value =
        serde_json::from_str(&runmat_cad_wasm::drawing::compile_json(path, text, files))
            .unwrap_or(Value::Null);
    let doc = &compiled["projection"]["data"]["document"];
    let rows = |v: &Value| v.as_array().cloned().unwrap_or_default();
    s.add("sheets", rows(&doc["sheets"]), &["name"]);
    s.add("views", rows(&doc["views"]), &["name"]);
    s.add("annotations", rows(&doc["annotations"]), &["id", "view"]);
    // The inputs of each part the views show, so a drawing's dimensions can be named by value.
    if s.wants("models") {
        let dir = path.rsplit_once('/').map_or("", |(d, _)| d);
        let mut models = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for view in rows(&doc["views"]) {
            let Some(file) = view["model"]["file"].as_str() else { continue };
            let model = if file.starts_with('/') { file.to_owned() } else { format!("{dir}/{file}") };
            if !model.ends_with(".part") || !seen.insert(model.clone()) {
                continue;
            }
            let input = InspectInput {
                path: model.clone(),
                sections: Some(vec!["inputs".into()]),
                filter: None,
                density: None,
                set: None,
                limit: None,
                changed: None,
                ray: None,
            };
            if let Ok(out) = inspect(&input, files) {
                models.push(json!({ "path": model, "inputs": out.sections.get("inputs").cloned().unwrap_or(Value::Null) }));
            }
        }
        s.value("models", Value::Array(models));
    }
}

fn circuit(s: &mut Sections, path: &str, text: &str, files: &Files) -> Result<(), ToolError> {
    let out = runmat_circuit::runtime::compile(path, text, files, &crate::circuit_values::circuit_options(path, text, files));
    let Some(index) = out.index else {
        return Err(ToolError::new(
            "not_compiled",
            format!(
                "{path} doesn't compile (design.check lists why): {}",
                out.diagnostics
                    .first()
                    .map(|d| d.message.as_str())
                    .unwrap_or("")
            ),
        ));
    };
    let rows = |v: &dyn erased::Rows| v.rows();
    s.add(
        "components",
        rows(&index.components),
        &["designator", "path", "part", "value"],
    );
    s.add("nets", rows(&index.nets), &["name", "pins", "class"]);
    s.add("net_classes", rows(&index.net_classes), &["name"]);
    s.add("bom", rows(&index.bom), &["part", "designators"]);
    s.add("sheets", rows(&index.hierarchy), &["path", "sheet"]);
    // The open work a guide reads: ERC errors that no power source drives, and the changes
    // the board hasn't taken from the schematic.
    let power_errors: Vec<Value> = index
        .erc
        .iter()
        .filter(|c| c.code == "power-not-driven" && c.severity == "error")
        .filter_map(|c| serde_json::to_value(c).ok())
        .collect();
    s.out.totals.insert("erc power errors".into(), power_errors.len());
    s.value("erc_power", Value::Array(power_errors));
    if let Some(board) = &index.board {
        s.out
            .totals
            .insert("eco differences".into(), index.eco.total as usize);
        // One row per change, by its id (`eco:<id>` selects it for `schematic.update-board`).
        let eco = serde_json::to_value(&index.eco).unwrap_or(Value::Null);
        let eco_rows: Vec<Value> = eco["groups"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|g| g["rows"].as_array().cloned().unwrap_or_default())
            .map(|r| {
                let mut row = json!({ "id": r["id"], "kind": r["kind"], "item": r["item"], "status": r["status"] });
                for key in ["from", "to", "reason", "requires"] {
                    if !r[key].is_null() {
                        row[key] = r[key].clone();
                    }
                }
                row
            })
            .collect();
        s.value(
            "eco",
            json!({ "total": eco["total"], "errors": eco["errors"], "warnings": eco["warnings"], "rows": eco_rows }),
        );
        let placed: std::collections::BTreeSet<&str> =
            board.footprints.iter().map(|f| f.reference.as_str()).collect();
        let unplaced: Vec<Value> = index
            .components
            .iter()
            .filter(|c| c.fitting == "fitted" && !placed.contains(c.designator.as_str()))
            .map(|c| json!({ "designator": c.designator, "path": c.path, "part": c.part }))
            .collect();
        s.out.totals.insert("unplaced components".into(), unplaced.len());
        s.value("unplaced", Value::Array(unplaced));
        s.add("rooms", rows(&board.rooms), &["name", "members"]);
        s.add(
            "footprints",
            rows(&board.footprints),
            &["reference", "designator", "footprint"],
        );
        s.add("layers", rows(&board.layers), &["name"]);
    } else if s.wants("rooms") {
        s.out
            .notes
            .push("the circuit has no board yet, so it has no rooms or footprints".into());
    }
    Ok(())
}

mod erased {
    use serde::Serialize;
    use serde_json::Value;

    pub trait Rows {
        fn rows(&self) -> Vec<Value>;
    }

    impl<T: Serialize> Rows for Vec<T> {
        fn rows(&self) -> Vec<Value> {
            self.iter()
                .filter_map(|r| serde_json::to_value(r).ok())
                .collect()
        }
    }
}

/// The interference check measures each overlap in m³; the tool states it in mm³
/// (`volume_mm3`, to 0.1 mm³), the unit the parts are drawn in.
fn overlap_volumes_in_mm3(value: &mut Value) {
    let at = if value.pointer("/result/overlaps").is_some() { "/result/overlaps" } else { "/overlaps" };
    let Some(Value::Array(overlaps)) = value.pointer_mut(at) else { return };
    for overlap in overlaps {
        let Some(row) = overlap.as_object_mut() else { continue };
        if let Some(m3) = row.remove("volume").and_then(|v| v.as_f64()) {
            row.insert("volume_mm3".into(), serde_json::json!((m3 * 1e10).round() / 10.0));
        }
    }
}

#[cfg(test)]
mod overlap_unit_tests {
    use super::*;
    #[test]
    fn overlap_volumes_read_in_cubic_millimetres() {
        let mut value = serde_json::json!({ "result": { "overlaps": [{ "a": "body", "b": "clamp", "volume": 1.6925e-7 }] } });
        overlap_volumes_in_mm3(&mut value);
        assert_eq!(value["result"]["overlaps"][0], serde_json::json!({ "a": "body", "b": "clamp", "volume_mm3": 169.3 }));
    }
}
