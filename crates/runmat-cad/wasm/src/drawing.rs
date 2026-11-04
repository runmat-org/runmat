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
//! - **Output.** `{ projection: { kind: "drawing", data }, models: { <file as written>:
//!   <display> }, diagnostics }`; a view whose model is missing or doesn't compile is reported
//!   as a warning and the sheet draws its frame without geometry.

use crate::runtime::{compile_assembly, compile_part, resolve_path, Diagnostic, Files};
use runmat_cad::assembly::bom::Bom;
use runmat_cad::assembly::model::RelPath;
use runmat_cad::drawing::index::{build_drawing_index, NoAnnotationValues};
use runmat_cad::drawing::{ModelRef, ModelReferences};
use runmat_cad::ids::PropertyName;
use runmat_cad::refs::name::EntityName;
use runmat_cad::refs::reference::Reference;
use runmat_cad::syntax::parse_drawing;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

/// What the drawn models are, as checking reads them.
#[derive(Default)]
pub struct CompiledModels {
    /// File as the drawing wrote it → every body and face name, serialized.
    names: BTreeMap<String, BTreeSet<String>>,
    /// File → property name → text.
    properties: BTreeMap<String, BTreeMap<String, String>>,
    /// File → occurrence paths (assemblies).
    occurrences: BTreeMap<String, BTreeSet<String>>,
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
    fn pmi(&self, _: &ModelRef) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
        (BTreeSet::new(), BTreeSet::new(), BTreeSet::new())
    }
    fn bom(&self, _: &ModelRef) -> Option<Bom> {
        None
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
    let Some(document) = parsed.document else {
        let diagnostics: Vec<Value> = parsed
            .diagnostics
            .iter()
            .map(|d| json!({ "path": path, "start": d.span.start, "end": d.span.end, "severity": "error", "message": d.message }))
            .collect();
        return json!({ "diagnostics": diagnostics }).to_string();
    };
    let mut models = CompiledModels::default();
    let mut displays = serde_json::Map::new();
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
                    models.names.insert(file.clone(), BTreeSet::new());
                    displays.insert(file.clone(), json!({ "kind": "assembly", "data": display }));
                }
                Err(_) => problems.push(warn(format!(
                    "{file} doesn't compile, so its views are empty"
                ))),
            }
            continue;
        }
        match compile_part(&at, source) {
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
                        let text = serde_json::to_value(material).ok().and_then(|v| match v {
                            Value::String(s) => Some(s),
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
                displays.insert(
                    file.clone(),
                    json!({ "kind": "part", "data": part.display }),
                );
            }
            Err(_) => problems.push(warn(format!(
                "{file} doesn't compile, so its views are empty"
            ))),
        }
    }
    let index = build_drawing_index(&document, &models, &NoAnnotationValues);
    json!({
        "projection": { "kind": "drawing", "data": index },
        "models": displays,
        "diagnostics": problems,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BRACKET: &str = include_str!("../../tests/corpus/documents/bracket.part");
    pub(crate) const DRAWING: &str = "runmat 0.7\nstd 1\n\ndrawing bracket {\n    format a3 = {\n        size = { size = iso, name = A3 },\n        title_block = [\n            { label = \"Part number\", source = { source = property, name = part_number } }\n        ]\n    }\n\n    sheet \"1\" = { format = a3, scale = [1, 1], projection = third_angle }\n\n    view front = standard(\n        orientation = front,\n        sheet = \"1\",\n        model = { file = \"bracket.part\" },\n        at = [100, 100],\n        display = hidden_lines\n    )\n    view top = projected(\n        from = front,\n        direction = up,\n        sheet = \"1\",\n        model = { file = \"bracket.part\" },\n        at = [100, 200],\n        display = hidden_lines\n    )\n}\n";

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
    fn syntax_errors_are_diagnostics() {
        let out: Value =
            serde_json::from_str(&compile_json("/p/x.drawing", "drawing x {", &Files::new()))
                .unwrap();
        assert!(out.get("projection").is_none());
        assert!(!out["diagnostics"].as_array().unwrap().is_empty());
    }
}
