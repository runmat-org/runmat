//! The CAD document runtime behind the wasm entry points, in plain Rust so it tests natively.

use runmat_cad::assembly::index::build_assembly_index;
use runmat_cad::assembly::model::{AssemblyDocument, DefinitionKind, DefinitionRef};
use runmat_cad::assembly::resolve::{resolve, ResolveContext, ResolveRequest};
use runmat_cad::assembly::source::{Aabb, ConfiguredSummary, DefinitionSummary, MemorySource};
use runmat_cad::display::{assembly_display, part_display, sketch_display, PartDisplay};
use runmat_cad::index::build_index;
use runmat_cad::model::PartDocument;
use runmat_cad::regen::kernel::TessellationOptions;
use runmat_cad::regen::{regenerate, FakeKernel, RegenCache, RegenContext, RegenRequest};
use runmat_cad::sketch_solver::ExplicitSketchSolver;
use runmat_cad::syntax::{parse_assembly, parse_part};
use runmat_cad::units::StubUnits;
use runmat_cad::value::NoDynamicValues;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

/// Workspace path → file text.
pub type Files = BTreeMap<String, String>;

/// A problem at a UTF-16 range of one file.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Diagnostic {
    pub path: String,
    pub start: usize,
    pub end: usize,
    /// `error`, `warning` or `info`.
    pub severity: String,
    pub message: String,
}

fn utf16(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    text[..at].encode_utf16().count()
}

fn syntax_diagnostics(
    path: &str,
    text: &str,
    diagnostics: &[runmat_design_syntax::FeaSyntaxDiagnostic],
) -> Vec<Diagnostic> {
    diagnostics
        .iter()
        .map(|d| Diagnostic {
            path: path.to_owned(),
            start: utf16(text, d.span.start),
            end: utf16(text, d.span.end),
            severity: "error".into(),
            message: d.message.clone(),
        })
        .collect()
}

pub fn parse_files(text: &str) -> Result<Files, String> {
    if text.trim().is_empty() {
        return Ok(Files::new());
    }
    serde_json::from_str(text).map_err(|e| format!("the files aren't valid: {e}"))
}

pub fn error_json(path: &str, message: &str) -> String {
    json!({ "diagnostics": [{ "path": path, "start": 0, "end": 0, "severity": "error", "message": message }] }).to_string()
}

fn folder(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(d, _)| d)
}

fn normalize(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    let joined = out.join("/");
    if path.starts_with('/') {
        format!("/{joined}")
    } else {
        joined
    }
}

/// The workspace path a reference written in `from` names: absolute as written, else against
/// the referencing file's folder; a project-relative path that only matches a file's tail
/// (`library/clip.part` for `/p/library/clip.part`) takes that file.
pub fn resolve_path(from: &str, reference: &str, files: &Files) -> String {
    if reference.starts_with('/') {
        return normalize(reference);
    }
    let local = normalize(&format!("{}/{reference}", folder(from)));
    if files.contains_key(&local) {
        return local;
    }
    let tail = format!("/{}", normalize(reference));
    files
        .keys()
        .find(|k| k.ends_with(&tail))
        .cloned()
        .unwrap_or(local)
}

fn is_assembly(path: &str) -> bool {
    path.ends_with(".assembly")
}

/// The definitions an assembly's occurrences name, in occurrence order.
fn definitions(doc: &AssemblyDocument) -> Vec<DefinitionRef> {
    let mut seen = BTreeSet::new();
    doc.occurrences
        .iter()
        .map(|o| o.source.definition.clone())
        .filter(|d| seen.insert(d.file.0.clone()))
        .collect()
}

/// The files `path` needs beyond itself: an assembly's parts and sub-assemblies.
pub fn dependencies(path: &str, text: &str) -> Vec<String> {
    if !is_assembly(path) {
        return Vec::new();
    }
    let parsed = parse_assembly(path, text);
    let Some(doc) = parsed.document else {
        return Vec::new();
    };
    definitions(&doc)
        .iter()
        .map(|d| resolve_path(path, d.file.as_str(), &Files::new()))
        .collect()
}

/// A part regenerated on the reference kernel: its projection and display.
pub struct CompiledPart {
    pub document: PartDocument,
    pub projection: Value,
    pub display: PartDisplay,
}

pub fn compile_part(path: &str, text: &str) -> Result<CompiledPart, Vec<Diagnostic>> {
    let parsed = parse_part(path, text);
    let Some(document) = parsed.document else {
        return Err(syntax_diagnostics(path, text, &parsed.diagnostics));
    };
    let kernel = FakeKernel::new();
    let ctx = RegenContext {
        kernel: &kernel,
        solver: &ExplicitSketchSolver,
        units: &StubUnits,
        dynamic: &NoDynamicValues,
        lock: None,
        catalog: None,
        policy: Default::default(),
        slots: None,
    };
    let result = regenerate(
        &document,
        &ctx,
        &RegenRequest::default(),
        &mut RegenCache::default(),
    );
    let index = build_index(&document, &result, None);
    let mut display = part_display(&result, &kernel, &TessellationOptions::default());
    display.sketches = sketch_display(&document, &result);
    let projection = json!({ "document": &document, "index": &index });
    Ok(CompiledPart {
        document,
        projection,
        display,
    })
}

fn bbox(display: &PartDisplay) -> Option<Aabb> {
    display.bounds.as_ref().map(|b| Aabb {
        min: b.min.map(f64::from),
        max: b.max.map(f64::from),
    })
}

/// An assembly resolved over the parts and sub-assemblies in `files`.
pub fn compile_assembly(
    path: &str,
    text: &str,
    files: &Files,
) -> Result<(Value, Value, Vec<Diagnostic>), Vec<Diagnostic>> {
    let parsed = parse_assembly(path, text);
    let Some(top) = parsed.document else {
        return Err(syntax_diagnostics(path, text, &parsed.diagnostics));
    };
    let mut source = MemorySource::default();
    let mut displays: BTreeMap<String, PartDisplay> = BTreeMap::new();
    let mut problems = Vec::new();
    let top_path = top.path.clone();
    // Breadth-first over sub-assemblies, each definition once.
    let mut queue: Vec<(String, AssemblyDocument)> = vec![(path.to_owned(), top)];
    let mut done: BTreeSet<String> = BTreeSet::new();
    while let Some((at, doc)) = queue.pop() {
        for def in definitions(&doc) {
            let key = def.file.0.clone();
            if !done.insert(key.clone()) {
                continue;
            }
            let file = resolve_path(&at, &key, files);
            let Some(text) = files.get(&file) else {
                problems.push(Diagnostic {
                    path: path.to_owned(),
                    start: 0,
                    end: 0,
                    severity: "warning".into(),
                    message: format!(
                        "{key} isn't open or in the workspace, so it draws as its box"
                    ),
                });
                continue;
            };
            match def.kind {
                DefinitionKind::Assembly => match parse_assembly(&key, text).document {
                    Some(sub) => {
                        queue.push((file.clone(), sub.clone()));
                        source.add_assembly(sub);
                    }
                    None => problems.push(Diagnostic {
                        path: file.clone(),
                        start: 0,
                        end: 0,
                        severity: "error".into(),
                        message: format!("{key} has errors"),
                    }),
                },
                _ => match compile_part(&key, text) {
                    Ok(part) => {
                        let mut summary = DefinitionSummary::new(def.clone());
                        summary.configurations = part
                            .document
                            .configurations
                            .iter()
                            .map(|c| c.name.clone())
                            .collect();
                        summary.digest = format!("text:{}", text.len());
                        let configured = ConfiguredSummary {
                            bbox: bbox(&part.display),
                            artifact: format!("text:{key}"),
                            ..Default::default()
                        };
                        source.add_part(summary, configured);
                        displays.insert(key.clone(), part.display);
                    }
                    Err(mut d) => problems.append(&mut d),
                },
            }
        }
    }
    let top_doc = parse_assembly(path, text).document.expect("parsed above");
    source.add_assembly(top_doc);
    let ctx = ResolveContext {
        source: &source,
        units: &StubUnits,
        dynamic: &NoDynamicValues,
        lock: None,
        catalog: None,
        tables: None,
        gauge: None,
    };
    let resolved = resolve(&top_path, &ResolveRequest::default(), &ctx);
    let index = build_assembly_index(&resolved, Some(&source));
    let display = assembly_display(&resolved, &source, &displays);
    Ok((
        serde_json::to_value(&index).unwrap_or(Value::Null),
        serde_json::to_value(&display).unwrap_or(Value::Null),
        problems,
    ))
}

/// `compile` as JSON.
pub fn compile_json(path: &str, text: &str, files: &Files) -> String {
    let out = if is_assembly(path) {
        match compile_assembly(path, text, files) {
            Ok((index, display, diagnostics)) => json!({
                "projection": { "kind": "assembly", "data": index },
                "display": { "kind": "assembly", "data": display },
                "diagnostics": diagnostics,
            }),
            Err(diagnostics) => json!({ "diagnostics": diagnostics }),
        }
    } else {
        match compile_part(path, text) {
            Ok(part) => json!({
                "projection": { "kind": "part", "data": part.projection },
                "display": { "kind": "part", "data": part.display },
                "diagnostics": [],
            }),
            Err(diagnostics) => json!({ "diagnostics": diagnostics }),
        }
    };
    out.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BRACKET: &str = include_str!("../../tests/corpus/documents/bracket.part");

    #[test]
    fn a_part_compiles_from_its_text_to_a_projection_and_a_mesh() {
        let out: Value =
            serde_json::from_str(&compile_json("/p/bracket.part", BRACKET, &Files::new())).unwrap();
        assert_eq!(out["projection"]["kind"], "part");
        assert!(out["projection"]["data"]["index"]["features"]
            .as_array()
            .is_some_and(|f| !f.is_empty()));
        let bodies = out["display"]["data"]["bodies"].as_array().unwrap();
        assert!(!bodies.is_empty(), "the bracket has a body");
        let faces = bodies[0]["faces"].as_array().unwrap();
        assert!(!faces.is_empty());
        assert!(out["display"]["data"]["bounds"].is_object());
    }

    #[test]
    fn syntax_errors_are_diagnostics_at_utf16_offsets() {
        let out: Value = serde_json::from_str(&compile_json(
            "/p/x.part",
            "runmat 0.7\nstd 1\n\npart x {\n  feature ±\n}\n",
            &Files::new(),
        ))
        .unwrap();
        assert!(out.get("projection").is_none());
        let d = &out["diagnostics"][0];
        assert_eq!(d["path"], "/p/x.part");
        assert!(d["start"].as_u64().unwrap() > 0);
    }

    #[test]
    fn an_assembly_draws_its_parts_from_their_text() {
        let assembly = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part bracket from \"bracket.part\" as a\n    use part bracket from \"bracket.part\" as b {\n        placement = { placement = offset, offset = { translation = [100 mm, 0 mm, 0 mm] } }\n    }\n}\n";
        let deps = dependencies("/p/rig.assembly", assembly);
        let files: Files = deps
            .iter()
            .map(|d| (d.clone(), BRACKET.to_owned()))
            .collect();
        let out: Value =
            serde_json::from_str(&compile_json("/p/rig.assembly", assembly, &files)).unwrap();
        if out.get("projection").is_none() {
            panic!("{}", out["diagnostics"]);
        }
        assert_eq!(out["projection"]["kind"], "assembly");
        let occ = out["display"]["data"]["occurrences"].as_array().unwrap();
        assert_eq!(occ.len(), 2, "{}", out["diagnostics"]);
        let protos = out["display"]["data"]["prototypes"].as_array().unwrap();
        assert_eq!(protos.len(), 1, "one prototype for both occurrences");
        assert_eq!(protos[0]["shape"], "mesh");
    }

    #[test]
    fn paths_resolve_against_the_referencing_file() {
        let files: Files = [("/p/library/clip.part".to_owned(), String::new())].into();
        assert_eq!(
            resolve_path("/p/rig.assembly", "./a.part", &files),
            "/p/a.part"
        );
        assert_eq!(
            resolve_path("/p/sub/rig.assembly", "../a.part", &files),
            "/p/a.part"
        );
        assert_eq!(
            resolve_path("/p/sub/rig.assembly", "library/clip.part", &files),
            "/p/library/clip.part"
        );
    }
}
