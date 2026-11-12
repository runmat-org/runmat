//! The CAD document runtime behind the wasm entry points, in plain Rust so it tests natively.

pub mod assembly_stages;
pub mod part_cache;
mod prototype_reuse;
mod published_frames;
mod published_paths;
pub mod values_host;

/// Display meshes as bytes beside the JSON, for the hosts' framed compile.
pub use runmat_cad::display::wire as display_wire;

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
use runmat_cad::syntax::{parse_assembly_in, parse_part};
use runmat_cad::units::StubUnits;
use runmat_cad::value::NoDynamicValues;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

/// Workspace path → file text.
pub type Files = BTreeMap<String, String>;

/// A problem at a UTF-16 range of one file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

/// The folder scope of `files` with `path` (at `text`) among them: what designs named by
/// symbol resolve through ([`runmat_cad::modules`]).
pub fn module_scope(path: &str, text: &str, files: &Files) -> runmat_cad::modules::ModuleScope {
    runmat_cad::modules::ModuleScope::from_files(
        files
            .iter()
            .filter(|(p, _)| p.as_str() != path)
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .chain(std::iter::once((path, text))),
    )
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
        // A circuit placed in an assembly draws its board: the layout, the circuit and its lock,
        // and the STEP models its packages name (a board's STEP places them).
        let mut out = runmat_circuit::runtime::dependencies(path, text);
        out.extend(
            runmat_circuit::board3d::model_paths(path, text)
                .iter()
                .map(|m| resolve_path(path, m, &Files::new())),
        );
        return out;
    }
    if !is_assembly(path) {
        // A part needs the files its imports read (STEP text the kernel is handed).
        let Some(doc) = parse_part(path, text).document else {
            return Vec::new();
        };
        let mut out: Vec<String> = doc
            .features
            .iter()
            .filter_map(|f| match &f.kind {
                runmat_cad::model::FeatureKind::Import(i) => {
                    Some(resolve_path(path, &i.source, &Files::new()))
                }
                _ => None,
            })
            .collect();
        // Values that call `.m` functions: the folders those functions can be in.
        out.extend(runmat_cad::value::m_functions::call_folders(
            path, text, &doc,
        ));
        return out;
    }
    // The folders it can name designs in, and the design files it names by path.
    runmat_cad::modules::dependencies(path, text)
}

/// [`dependencies`] resolved among `files`, which hold the listed folders' files: each folder
/// listing becomes the design files declaring what `path` names there (an assembly's parts and
/// sub-assemblies, a drawing's models, a layout's circuit). Folders listed only for `.m`
/// functions drop out. A name that doesn't resolve adds nothing; the compile reports it.
pub fn dependency_files(path: &str, text: &str, files: &Files) -> Vec<String> {
    if path.ends_with(".drawing") {
        return crate::drawing::dependency_files(path, text, files);
    }
    if path.ends_with(".circuit") || path.ends_with(".circuit.layout") {
        return runmat_circuit::runtime::dependency_files(path, text, files);
    }
    let mut out: Vec<String> = dependencies(path, text)
        .into_iter()
        .filter(|d| !d.ends_with('/'))
        .collect();
    if is_assembly(path) {
        let scope = module_scope(path, text, files);
        if let Some(doc) = parse_assembly_in(path, text, &scope).document {
            out.extend(
                definitions(&doc)
                    .iter()
                    .map(|d| resolve_path(path, &d.file.0, files)),
            );
        }
    }
    let mut seen = BTreeSet::new();
    out.retain(|p| p != path && seen.insert(p.clone()));
    out
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
    /// The regeneration asked for a dynamic value (an `.m` function, another file's published
    /// value, a snapshot): it depends on more than the part's text and configuration.
    pub dynamic: bool,
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
    let request = RegenRequest {
        configuration: configuration.map(runmat_cad::ids::ConfigurationName::new),
    };
    // Values that call `.m` functions run through the host's runner (values_host).
    let used = std::cell::Cell::new(false);
    let mut result = values_host::with_evaluator(path, |dynamic| {
        let watched = part_cache::Watched::new(dynamic, &used);
        let ctx = RegenContext {
            kernel: kernel.as_ref(),
            solver: &ConstraintSketchSolver,
            units: &StubUnits,
            dynamic: &watched,
            lock: None,
            catalog: None,
            policy: Default::default(),
            slots: None,
        };
        regenerate(&document, &ctx, &request, &mut RegenCache::default())
    });
    result.diagnostics.attach_spans(&parsed.spans);
    // The part as the configuration makes it: its feature parameters, part number and
    // properties, for every reader of the regenerated part (title blocks, exports).
    let document = match &request.configuration {
        Some(name) => document.in_configuration(Some(name)),
        None => document,
    };
    let mut index = build_index(&document, &result, None);
    // The file behind each value's `.m` function, for the Properties panel and hover.
    values_host::resolve_let_files(path, &mut index.lets);
    let mut display = part_display(&result, kernel.as_ref(), tessellation);
    display.sketches = sketch_display(&document, &result);
    display.references = runmat_cad::display::references::reference_display(
        &document,
        &result,
        display.bounds.as_ref(),
    );
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
    runmat_cad::appearance::color_output_bodies(&document, &result, &mut display);
    Ok(RegeneratedPart {
        document,
        spans: parsed.spans,
        result,
        kernel,
        index,
        display,
        dynamic: used.get(),
    })
}

pub fn compile_part(path: &str, text: &str) -> Result<CompiledPart, Vec<Diagnostic>> {
    compile_part_in(path, text, None)
}

/// [`compile_part`] in the named configuration `configuration`: the projection's document is
/// the part as that configuration makes it, and its index names the configuration.
pub fn compile_part_in(
    path: &str,
    text: &str,
    configuration: Option<&str>,
) -> Result<CompiledPart, Vec<Diagnostic>> {
    let part = regenerate_part_in(path, text, configuration, &TessellationOptions::default())?;
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

/// The datum frames, planes, axes and points a part publishes, from its regeneration, its
/// published single faces and vertices as frames (what a mate's connectors anchor to), and its
/// published edge sets as paths: what an assembly attaches to (a skeleton's interface) and what
/// a cam or path mate follows.
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
    out.extend(published_frames::published_frames(part));
    out.extend(published_paths::published_paths(part));
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
    /// Each regenerated part file's digest ([`part_cache::digest`] of its cache key), by file.
    pub digests: BTreeMap<String, String>,
}

/// Resolves the assembly `path` from its text; syntax errors are the `Err` diagnostics.
pub fn resolve_assembly(
    path: &str,
    text: &str,
    files: &Files,
) -> Result<AssemblyResolution, Vec<Diagnostic>> {
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
    let scope = module_scope(path, text, files);
    let parsed = parse_assembly_in(path, text, &scope);
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
    let mut digests: BTreeMap<String, String> = BTreeMap::new();
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
                        // The parts list names the board by its layout's part number and
                        // laminate.
                        let part = crate::board::board_part(&file, files);
                        summary.part_number = part.part_number;
                        summary.material = part.material.map(runmat_cad::ids::MaterialRef::new);
                        source.add_part(summary, configured);
                        displays.insert(key.clone(), display);
                        if let Some(d) = crate::board::board_digest(&file, files) {
                            digests.insert(key.clone(), d);
                        }
                    }
                    Err(message) => problems.push(Diagnostic {
                        path: path.to_owned(),
                        start: 0,
                        end: 0,
                        severity: "warning".into(),
                        message: format!("{key}: {message}, so it draws as its box"),
                    }),
                },
                DefinitionKind::Assembly => match parse_assembly_in(&file, text, &scope).document {
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
                _ => {
                    let cache_key = part_cache::assembly_key(&key, text, &ASSEMBLY_TESSELLATION);
                    digests.insert(key.clone(), part_cache::digest(&cache_key));
                    // A probe leaves a part the cache doesn't hold for its caller to find.
                    if part_cache::skip(&cache_key) {
                        continue;
                    }
                    match part_cache::assembly_part(cache_key, || {
                        regenerate_part_with(&key, text, &ASSEMBLY_TESSELLATION).map(|part| {
                            let diagnostics = regen_diagnostics(&key, &part);
                            let publishes = published_datums(&part);
                            // Coarser tessellations for the canvas's level of detail.
                            let levels = runmat_cad::display::display_levels(
                                &part.result,
                                part.kernel.as_ref(),
                                &ASSEMBLY_TESSELLATION,
                            );
                            let mut display = part.display;
                            display.levels = levels;
                            let entry = part_cache::AssemblyPart {
                                diagnostics,
                                publishes,
                                display,
                                document: part.document,
                            };
                            (entry, part.dynamic)
                        })
                    }) {
                        Ok(part) => {
                            // A part that regenerates with errors (a feature the kernel refused)
                            // says so on the assembly, at the assembly's top.
                            for d in &part.diagnostics {
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
                            summary.configuration_part_numbers = part
                                .document
                                .all_configurations()
                                .into_iter()
                                .filter_map(|c| c.part_number.map(|pn| (c.name, pn)))
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
                                publishes: part.publishes.clone(),
                                ..Default::default()
                            };
                            source.add_part(summary, configured);
                            displays.insert(key.clone(), part.display.clone());
                            documents.insert(key.clone(), part.document.clone());
                        }
                        Err(mut d) => problems.append(&mut d),
                    }
                }
            }
        }
    }
    let top_doc = parse_assembly_in(path, text, &scope)
        .document
        .expect("parsed above");
    // Two designs of one kind and name in a folder, and folders that use each other.
    for problem in scope.problems() {
        problems.push(Diagnostic {
            path: path.to_owned(),
            start: 0,
            end: 0,
            severity: "error".into(),
            message: runmat_cad::modules::problem_text(&problem),
        });
    }
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
        digests,
    })
}

/// Compiles the assembly `path`: its index (the projection), its display, and diagnostics.
pub fn compile_assembly(
    path: &str,
    text: &str,
    files: &Files,
) -> Result<(Value, Value, Vec<Diagnostic>), Vec<Diagnostic>> {
    compile_assembly_known(path, text, files, &BTreeSet::new())
}

/// [`compile_assembly`] for a caller that holds the part displays whose digests are in `known`:
/// each prototype carries its part's `digest`, and a known one is sent `reused` without its
/// part display ([`prototype_reuse`]).
pub fn compile_assembly_known(
    path: &str,
    text: &str,
    files: &Files,
    known: &BTreeSet<String>,
) -> Result<(Value, Value, Vec<Diagnostic>), Vec<Diagnostic>> {
    compile_assembly_with(path, text, files, known, |_, _| ())
}

/// [`compile_assembly`] with the assembly's indented BOM (sub-assemblies as lines with their
/// contents under them), for a drawing's top-level and indented parts lists.
pub fn compile_assembly_indented(
    path: &str,
    text: &str,
    files: &Files,
) -> Result<
    (
        Value,
        Value,
        Vec<Diagnostic>,
        Vec<runmat_cad::assembly::bom::IndentedRow>,
    ),
    Vec<Diagnostic>,
> {
    let mut rows = Vec::new();
    let (index, display, problems) =
        compile_assembly_with(path, text, files, &BTreeSet::new(), |resolved, source| {
            rows = runmat_cad::assembly::bom::indented_bom(resolved, Some(source));
        })?;
    Ok((index, display, problems, rows))
}

/// [`compile_assembly_known`], handing the resolved tree and its source to `also` first.
fn compile_assembly_with(
    path: &str,
    text: &str,
    files: &Files,
    known: &BTreeSet<String>,
    also: impl FnOnce(&runmat_cad::assembly::resolve::ResolvedAssembly, &MemorySource),
) -> Result<(Value, Value, Vec<Diagnostic>), Vec<Diagnostic>> {
    let AssemblyResolution {
        source,
        displays,
        documents,
        resolved,
        problems,
        digests,
    } = resolve_assembly(path, text, files)?;
    also(&resolved, &source);
    let index = build_assembly_index(&resolved, Some(&source));
    let configured = crate::configured::configured_displays(&resolved, &documents);
    let mut display = assembly_display_configured(&resolved, &source, &displays, &configured);
    let prototype_digests = prototype_reuse::prototype_digests(&resolved, &documents, &digests);
    let reused = prototype_reuse::omit_known(&mut display, &prototype_digests, known);
    let mut display = serde_json::to_value(&display).unwrap_or(Value::Null);
    prototype_reuse::mark(&mut display, &prototype_digests, &reused);
    let mut index = serde_json::to_value(&index).unwrap_or(Value::Null);
    if let Some(row) = index.as_object_mut() {
        row.insert(
            "exploded".into(),
            crate::exploded::exploded_views(&resolved, &source),
        );
    }
    Ok((index, display, problems))
}

/// [`compile_assembly`]'s index and diagnostics without its display: for readers of the
/// assembly's structure (the design tools' check, inspect and change diff). A product's display
/// as JSON is the largest thing a compile makes, and the browser host's heap is bounded.
pub fn assembly_index(
    path: &str,
    text: &str,
    files: &Files,
) -> Result<(Value, Vec<Diagnostic>), Vec<Diagnostic>> {
    let AssemblyResolution {
        source,
        resolved,
        problems,
        ..
    } = resolve_assembly(path, text, files)?;
    let mut index =
        serde_json::to_value(build_assembly_index(&resolved, Some(&source))).unwrap_or(Value::Null);
    if let Some(row) = index.as_object_mut() {
        row.insert(
            "exploded".into(),
            crate::exploded::exploded_views(&resolved, &source),
        );
    }
    Ok((index, problems))
}

/// `compile` as JSON.
pub fn compile_json(path: &str, text: &str, files: &Files) -> String {
    compile_json_in(path, text, files, None)
}

/// The `options` the compile entry takes: `{ "configuration": "<name>" }` compiles a part in
/// that configuration (the Desktop's active configuration); empty or `"default"` is the part
/// as written.
pub fn compile_options(options: &str) -> Option<String> {
    let value: Value = serde_json::from_str(options).ok()?;
    value
        .get("configuration")
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty() && *c != "default")
        .map(str::to_owned)
}

/// [`compile_json`] with a part's active configuration.
pub fn compile_json_in(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
) -> String {
    compile_json_known(path, text, files, configuration, &BTreeSet::new())
}

/// [`compile_json_in`] for a caller that holds the part displays whose digests are in `known`
/// (an assembly sends those prototypes `reused`; see [`compile_assembly_known`]).
pub fn compile_json_known(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
    known: &BTreeSet<String>,
) -> String {
    if path.ends_with(".drawing") {
        return crate::drawing::compile_json(path, text, files);
    }
    let out = if is_assembly(path) {
        match compile_assembly_known(path, text, files, known) {
            Ok((mut index, display, diagnostics)) => {
                crate::repairs::label_candidates(&mut index["diagnostics"]);
                // Built from the values held: `json!` would serialize a copy of the assembly's
                // display, which runs to hundreds of thousands of values.
                let mut out = serde_json::Map::new();
                out.insert(
                    "projection".into(),
                    crate::drawing::tagged("assembly", index),
                );
                out.insert(
                    "display".into(),
                    crate::drawing::tagged("assembly", display),
                );
                out.insert("diagnostics".into(), json!(diagnostics));
                out.insert("kernel".into(), json!(crate::kernel::name()));
                return crate::drawing::json_text(&Value::Object(out));
            }
            Err(diagnostics) => json!({ "diagnostics": diagnostics }),
        }
    } else {
        match compile_part_in(path, text, configuration) {
            Ok(mut part) => {
                crate::repairs::label_candidates(&mut part.projection["index"]["diagnostics"]);
                json!({
                    "projection": { "kind": "part", "data": part.projection },
                    "display": { "kind": "part", "data": part.display },
                    "diagnostics": part.diagnostics,
                    "kernel": crate::kernel::name(),
                })
            }
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
        let assembly = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part bracket as a\n    use part bracket as b {\n        placement = { placement = offset, offset = { translation = [100 mm, 0 mm, 0 mm] } }\n    }\n}\n";
        // The parts are named in the assembly's folder: the host lists `/p/` and reads them.
        assert_eq!(dependencies("/p/rig.assembly", assembly), ["/p/"]);
        let files: Files = [("/p/bracket.part".to_owned(), BRACKET.to_owned())].into();
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

    /// Assemblies inside assemblies, through the files map (the Desktop's and the OCCT host's
    /// entry): a sub-assembly configured by input, two levels deep, in this folder and another.
    #[test]
    fn nested_assemblies_resolve_through_the_files_map() {
        let block = "runmat 0.7\nstd 1\n\npart block {\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = l1.end, end = [20 mm, 10 mm])\n        l3 = line(start = l2.end, end = [0 mm, 10 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature box = extrude(profile = sketch.s, distance = 5 mm)\n    output block = { body = feature.box.body }\n}\n";
        let unit = "runmat 0.7\nstd 1\n\nassembly unit {\n    input gap : length = 10 mm\n    use part block from \"../lib\" as a\n    use part block from \"../lib\" as b {\n        placement = { placement = offset, offset = { translation = [input.gap, 0 mm, 0 mm] } }\n    }\n}\n";
        let station = "runmat 0.7\nstd 1\n\nassembly station {\n    use assembly unit as u {\n        configuration = { select = inputs, values = { gap = 50 mm } }\n    }\n}\n";
        let top = "runmat 0.7\nstd 1\n\nassembly rig {\n    use assembly station from \"./sub\" as left\n    use assembly station from \"./sub\" as right {\n        placement = { placement = offset, offset = { translation = [0 mm, 200 mm, 0 mm] } }\n    }\n    use assembly unit from \"./sub\" as loose\n}\n";
        let files: Files = [
            ("/p/lib/block.part".to_owned(), block.to_owned()),
            ("/p/sub/unit.assembly".to_owned(), unit.to_owned()),
            ("/p/sub/station.assembly".to_owned(), station.to_owned()),
            ("/p/rig.assembly".to_owned(), top.to_owned()),
        ]
        .into();
        for (path, text) in &files {
            if !path.ends_with(".assembly") {
                continue;
            }
            let out: Value = serde_json::from_str(&compile_json(path, text, &files)).unwrap();
            let errors: Vec<&Value> = out["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|d| d["severity"] == "error")
                .collect();
            assert!(errors.is_empty(), "{path}: {errors:?}");
            let want = match path.as_str() {
                "/p/rig.assembly" => 6,
                "/p/sub/station.assembly" | "/p/sub/unit.assembly" => 2,
                _ => unreachable!(),
            };
            let occ = out["display"]["data"]["occurrences"].as_array().unwrap();
            assert_eq!(occ.len(), want, "{path}: {}", out["diagnostics"]);
        }
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
