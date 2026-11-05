//! The CAD document runtime behind the wasm entry points, in plain Rust so it tests natively.

use runmat_cad::assembly::index::build_assembly_index;
use runmat_cad::assembly::model::{AssemblyDocument, DefinitionKind, DefinitionRef};
use runmat_cad::assembly::resolve::{resolve, ResolveContext, ResolveRequest};
use runmat_cad::assembly::source::{Aabb, ConfiguredSummary, DefinitionSummary, MemorySource};
use runmat_cad::display::{assembly_display_configured, part_display, sketch_display, PartDisplay};
use runmat_cad::index::build_index;
use runmat_cad::model::PartDocument;
use runmat_cad::regen::kernel::TessellationOptions;
use runmat_cad::regen::{regenerate, RegenCache, RegenContext, RegenRequest, RegenResult};
use runmat_cad::sketch_solver::ConstraintSketchSolver;
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
    if path.ends_with(".drawing") {
        return crate::drawing::dependencies(path, text);
    }
    if path.ends_with(".circuit") || path.ends_with(".circuit.layout") {
        // A circuit placed in an assembly draws its board: the layout, the circuit and its lock.
        return runmat_circuit::runtime::dependencies(path, text);
    }
    if !is_assembly(path) {
        // A part needs the files its imports read (STEP text the kernel is handed).
        let Some(doc) = parse_part(path, text).document else {
            return Vec::new();
        };
        return doc
            .features
            .iter()
            .filter_map(|f| match &f.kind {
                runmat_cad::model::FeatureKind::Import(i) => Some(resolve_path(path, &i.source, &Files::new())),
                _ => None,
            })
            .collect();
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

/// A part regenerated on the installed kernel: its projection and display.
pub struct CompiledPart {
    /// The regeneration's diagnostics (kernel and feature errors among them) at their spans.
    pub diagnostics: Vec<Diagnostic>,
    pub document: PartDocument,
    pub index: runmat_cad::index::PartIndex,
    pub projection: Value,
    pub display: PartDisplay,
}

/// A part regenerated on the installed kernel ([`crate::kernel`]) with everything a caller may
/// ask of it: the regeneration (bodies, values, diagnostics with their source spans attached), the kernel that
/// holds its bodies (for analyses such as mass properties), the projection and the display.
pub struct RegeneratedPart {
    pub document: PartDocument,
    pub spans: runmat_cad::diagnostics::SpanMap,
    pub result: RegenResult,
    pub kernel: Box<dyn crate::kernel::DesignKernel>,
    pub index: runmat_cad::index::PartIndex,
    pub display: PartDisplay,
}

/// Regenerates the part `path` from its text; syntax errors are the `Err` diagnostics.
pub fn regenerate_part(path: &str, text: &str) -> Result<RegeneratedPart, Vec<Diagnostic>> {
    regenerate_part_with(path, text, &TessellationOptions::default())
}

/// How parts tessellate inside an assembly's display: coarser than a part on its own (a chord
/// of 0.2 % of each body's size, at most 5 mm; 12° per segment), as large-assembly modes do,
/// so a 2,000-part assembly fits the scene budget. Normals stay exact, so surfaces shade
/// smoothly; a part opened on its own uses the default.
pub const ASSEMBLY_TESSELLATION: TessellationOptions = TessellationOptions {
    chord_tolerance: 5.0e-4,
    angle_tolerance: 12.0 * std::f64::consts::PI / 180.0,
    size_fraction: 2.0e-3,
    chord_max: None,
};

/// [`regenerate_part`] with the display tessellated under `tessellation`.
pub fn regenerate_part_with(
    path: &str,
    text: &str,
    tessellation: &TessellationOptions,
) -> Result<RegeneratedPart, Vec<Diagnostic>> {
    regenerate_part_in(path, text, None, tessellation)
}

/// [`regenerate_part_with`] in the named configuration `configuration` (the default when
/// `None`): its input values, suppressions and parameter overrides apply.
pub fn regenerate_part_in(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    tessellation: &TessellationOptions,
) -> Result<RegeneratedPart, Vec<Diagnostic>> {
    let parsed = parse_part(path, text);
    let Some(document) = parsed.document else {
        return Err(syntax_diagnostics(path, text, &parsed.diagnostics));
    };
    let kernel = crate::kernel::make();
    let ctx = RegenContext {
        kernel: kernel.as_ref(),
        solver: &ConstraintSketchSolver,
        units: &StubUnits,
        dynamic: &NoDynamicValues,
        lock: None,
        catalog: None,
        policy: Default::default(),
        slots: None,
    };
    let request = RegenRequest {
        configuration: configuration.map(runmat_cad::ids::ConfigurationName::new),
    };
    let mut result = regenerate(&document, &ctx, &request, &mut RegenCache::default());
    result.diagnostics.attach_spans(&parsed.spans);
    let index = build_index(&document, &result, None);
    let mut display = part_display(&result, kernel.as_ref(), tessellation);
    display.sketches = sketch_display(&document, &result);
    // The part draws in its first output's color, else that output's material appearance, the
    // CAD grey without either.
    let output = document.parts.first();
    let material = output
        .and_then(|p| p.material.as_ref())
        .or_else(|| document.parts.iter().find_map(|p| p.material.as_ref()));
    display.appearance = Some(runmat_cad::appearance::part_appearance(
        material.map(|m| m.0.as_str()),
        output.and_then(|p| p.color.as_deref()),
    ));
    Ok(RegeneratedPart {
        document,
        spans: parsed.spans,
        result,
        kernel,
        index,
        display,
    })
}

pub fn compile_part(path: &str, text: &str) -> Result<CompiledPart, Vec<Diagnostic>> {
    let part = regenerate_part(path, text)?;
    let projection = json!({ "document": &part.document, "index": &part.index });
    Ok(CompiledPart {
        diagnostics: regen_diagnostics(path, &part),
        document: part.document,
        index: part.index,
        projection,
        display: part.display,
    })
}

/// A regeneration's diagnostics as the compile output reports them: each at its source span
/// (the whole file when it has none), a cascade error (one caused by an earlier failure) as a
/// warning.
pub fn regen_diagnostics(path: &str, part: &RegeneratedPart) -> Vec<Diagnostic> {
    use runmat_cad::diagnostics::{Cause, Severity};
    part.result
        .diagnostics
        .diagnostics
        .iter()
        .map(|d| Diagnostic {
            path: path.to_owned(),
            start: d.span.as_ref().map_or(0, |s| s.start as usize),
            end: d.span.as_ref().map_or(0, |s| s.end as usize),
            severity: match (d.severity, &d.cause) {
                (Severity::Error, Cause::Cascade { .. }) => "warning",
                (Severity::Error, _) => "error",
                (Severity::Warning, _) => "warning",
                (Severity::Info, _) => "info",
            }
            .into(),
            message: d.message.clone(),
        })
        .collect()
}

/// The datum frames, planes, axes and points a part publishes, from its regeneration: what an
/// assembly attaches to (a skeleton's interface).
fn published_datums(
    part: &RegeneratedPart,
) -> BTreeMap<runmat_cad::ids::PublishName, runmat_cad::assembly::source::PublishedGeometry> {
    use runmat_cad::assembly::source::PublishedGeometry as G;
    use runmat_cad::model::datum::{DatumRef, DatumValue};
    use runmat_cad::model::items::PublishTarget;
    let mut out = BTreeMap::new();
    for p in &part.document.publishes {
        let PublishTarget::Datum {
            datum: DatumRef::Feature { feature },
        } = &p.target
        else {
            continue;
        };
        let Some(value) = part.result.state.datums.get(feature) else {
            continue;
        };
        let g = match value {
            DatumValue::Frame { frame } => G::Frame { frame: *frame },
            DatumValue::Plane { frame } => G::Plane { frame: *frame },
            DatumValue::Axis { point, direction } => G::Axis {
                point: *point,
                direction: *direction,
            },
            DatumValue::Point { at } => G::Point { at: *at },
        };
        out.insert(p.name.clone(), g);
    }
    out
}

fn bbox(display: &PartDisplay) -> Option<Aabb> {
    display.bounds.as_ref().map(|b| Aabb {
        min: b.min.map(f64::from),
        max: b.max.map(f64::from),
    })
}

/// An assembly resolved over the parts and sub-assemblies in `files`.
/// An assembly resolved from its text and `files`: every definition it uses, each part file
/// regenerated once (its document and display), the resolved occurrence tree, and what went
/// wrong on the way.
pub struct AssemblyResolution {
    pub source: MemorySource,
    pub displays: BTreeMap<String, PartDisplay>,
    pub documents: BTreeMap<String, PartDocument>,
    pub resolved: runmat_cad::assembly::resolve::ResolvedAssembly,
    pub problems: Vec<Diagnostic>,
}

/// Resolves the assembly `path` from its text; syntax errors are the `Err` diagnostics.
pub fn resolve_assembly(path: &str, text: &str, files: &Files) -> Result<AssemblyResolution, Vec<Diagnostic>> {
    resolve_assembly_in(path, text, files, None)
}

/// [`resolve_assembly`] in the top assembly's named configuration `configuration` (its input
/// values and suppressed children apply; the default when `None`).
pub fn resolve_assembly_in(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
) -> Result<AssemblyResolution, Vec<Diagnostic>> {
    let parsed = parse_assembly(path, text);
    let Some(top) = parsed.document else {
        return Err(syntax_diagnostics(path, text, &parsed.diagnostics));
    };
    if let Some(c) = configuration {
        if !top.configurations.iter().any(|k| k.name.as_str() == c) {
            return Err(vec![Diagnostic {
                path: path.to_owned(),
                start: 0,
                end: 0,
                severity: "error".into(),
                message: format!("{path} has no configuration `{c}`"),
            }]);
        }
    }
    let mut source = MemorySource::default();
    let mut displays: BTreeMap<String, PartDisplay> = BTreeMap::new();
    let mut documents: BTreeMap<String, PartDocument> = BTreeMap::new();
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
                DefinitionKind::Circuit => match crate::board::board_display(&file, files) {
                    Ok(display) => {
                        let configured = ConfiguredSummary {
                            bbox: bbox(&display),
                            artifact: format!("text:{key}"),
                            ..Default::default()
                        };
                        let mut summary = DefinitionSummary::new(def.clone());
                        summary.digest = format!("text:{}", text.len());
                        source.add_part(summary, configured);
                        displays.insert(key.clone(), display);
                    }
                    Err(message) => problems.push(Diagnostic {
                        path: path.to_owned(),
                        start: 0,
                        end: 0,
                        severity: "warning".into(),
                        message: format!("{key}: {message}, so it draws as its box"),
                    }),
                },
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
                _ => match regenerate_part_with(&key, text, &ASSEMBLY_TESSELLATION) {
                    Ok(part) => {
                        // A part that regenerates with errors (a feature the kernel refused)
                        // says so on the assembly, at the assembly's top.
                        for d in regen_diagnostics(&key, &part) {
                            if d.severity == "error" {
                                problems.push(Diagnostic {
                                    path: path.to_owned(),
                                    start: 0,
                                    end: 0,
                                    severity: "error".into(),
                                    message: format!("{key}: {}", d.message),
                                });
                            }
                        }
                        let mut summary = DefinitionSummary::new(def.clone());
                        summary.configurations = part
                            .document
                            .configurations
                            .iter()
                            .map(|c| c.name.clone())
                            .collect();
                        summary.inputs = part
                            .document
                            .inputs
                            .iter()
                            .map(|i| (i.name.clone(), i.kind.clone()))
                            .collect();
                        if let Some(output) = part.document.parts.first() {
                            summary.material = output.material.clone();
                            if summary.part_number.is_none() {
                                summary.part_number = output.part_number.clone();
                            }
                        }
                        summary.digest = format!("text:{}", text.len());
                        let configured = ConfiguredSummary {
                            bbox: bbox(&part.display),
                            artifact: format!("text:{key}"),
                            publishes: published_datums(&part),
                            ..Default::default()
                        };
                        source.add_part(summary, configured);
                        // Coarser tessellations for the canvas's level of detail.
                        let mut display = part.display;
                        display.levels = runmat_cad::display::display_levels(&part.result, part.kernel.as_ref(), &ASSEMBLY_TESSELLATION);
                        displays.insert(key.clone(), display);
                        documents.insert(key.clone(), part.document);
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
    let request = ResolveRequest {
        configuration: configuration.map(runmat_cad::ids::ConfigurationName::new),
        ..ResolveRequest::default()
    };
    let resolved = resolve(&top_path, &request, &ctx);
    // What resolution could not do (a placement whose frame is missing, a pattern that does
    // not expand) is shown on the assembly, not dropped.
    for d in &resolved.diagnostics.diagnostics {
        problems.push(Diagnostic {
            path: path.to_owned(),
            start: 0,
            end: 0,
            severity: match d.severity {
                runmat_cad::diagnostics::Severity::Error => "error",
                runmat_cad::diagnostics::Severity::Warning => "warning",
                runmat_cad::diagnostics::Severity::Info => "info",
            }
            .into(),
            message: d.message.clone(),
        });
    }
    Ok(AssemblyResolution {
        source,
        displays,
        documents,
        resolved,
        problems,
    })
}

/// Compiles the assembly `path`: its index (the projection), its display, and diagnostics.
pub fn compile_assembly(
    path: &str,
    text: &str,
    files: &Files,
) -> Result<(Value, Value, Vec<Diagnostic>), Vec<Diagnostic>> {
    let AssemblyResolution {
        source,
        displays,
        documents,
        resolved,
        problems,
    } = resolve_assembly(path, text, files)?;
    let index = build_assembly_index(&resolved, Some(&source));
    let configured = crate::configured::configured_displays(&resolved, &documents);
    let display = assembly_display_configured(&resolved, &source, &displays, &configured);
    Ok((
        serde_json::to_value(&index).unwrap_or(Value::Null),
        serde_json::to_value(&display).unwrap_or(Value::Null),
        problems,
    ))
}

/// `compile` as JSON.
pub fn compile_json(path: &str, text: &str, files: &Files) -> String {
    if path.ends_with(".drawing") {
        return crate::drawing::compile_json(path, text, files);
    }
    let out = if is_assembly(path) {
        match compile_assembly(path, text, files) {
            Ok((index, display, diagnostics)) => json!({
                "projection": { "kind": "assembly", "data": index },
                "display": { "kind": "assembly", "data": display },
                "diagnostics": diagnostics,
                "kernel": crate::kernel::name(),
            }),
            Err(diagnostics) => json!({ "diagnostics": diagnostics }),
        }
    } else {
        match compile_part(path, text) {
            Ok(part) => json!({
                "projection": { "kind": "part", "data": part.projection },
                "display": { "kind": "part", "data": part.display },
                "diagnostics": part.diagnostics,
                "kernel": crate::kernel::name(),
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
