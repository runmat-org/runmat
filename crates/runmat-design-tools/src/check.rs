//! `design.check`: regenerate a design file and report what the editor reports.
//!
//! Parts regenerate on the reference kernel; each diagnostic keeps the compiler's root-cause
//! structure: a root cause lists the features that failed because of it, and each cascade names
//! its root. Assemblies resolve their parts, mates and patterns; drawings compile against the
//! models their views show; circuits elaborate and run ERC (and a layout checks its rules).

use crate::{kind_of, text_of, DesignKind, Files, Locator, Range, ToolError};
use runmat_cad::diagnostics::{Cause, Diagnostic as CadDiagnostic, Severity, Subject};
use runmat_cad::index::FileState;
use runmat_cad::regen::FeatureStatus;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CheckInput {
    /// Workspace path of the design file.
    pub path: String,
}

/// One problem, as the editor shows it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DesignDiagnostic {
    /// The compiler's id for the diagnostic (root causes and cascades refer to it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u32>,
    /// `error`, `warning` or `info`.
    pub severity: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub message: String,
    /// `root` (fix this one) or `cascade` (failed because of `root`).
    pub cause: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<u32>,
    /// What it is about: `feature.boss`, `mate.m1`, `occurrence.wing/rib[2]`, `U3`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    /// For a root cause: the features that failed or were blocked because of it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed_features: Vec<String>,
    /// Where it is written.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<Range>,
    /// Typed repairs the compiler offers (one click in the editor).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repairs: Vec<Value>,
}

/// An input as regeneration resolved it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedInput {
    pub name: String,
    /// The quantity kind (`length`, `angle`, `count`, …).
    pub kind: String,
    /// The value in SI units (metres, radians), when it resolved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub si: Option<f64>,
    /// The value as an engineer reads it (`80 mm`, `30°`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A feature's state after regeneration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FeatureState {
    pub id: String,
    pub kind: String,
    /// `regenerated`, `suppressed`, `failed`, `blocked`, …
    pub status: String,
    /// The diagnostic that failed or blocked it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CheckOutput {
    pub path: String,
    pub kind: DesignKind,
    /// No errors.
    pub ok: bool,
    pub errors: usize,
    pub warnings: usize,
    /// One line: what was checked and what it found.
    pub summary: String,
    pub diagnostics: Vec<DesignDiagnostic>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<ResolvedInput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<FeatureState>,
    /// What the design holds: features, bodies, occurrences, mates, views, components, nets…
    pub counts: BTreeMap<String, usize>,
    /// What the file holds, as the guides choose by it: empty or written, and for a part its
    /// kind (sheet metal, a reference skeleton, runs, electrical, structural), from the part's
    /// index.
    pub file_state: FileState,
}

pub fn check(input: &CheckInput, files: &Files) -> Result<CheckOutput, ToolError> {
    let kind = kind_of(&input.path)?;
    let text = text_of(files, &input.path)?;
    let mut out = CheckOutput {
        path: input.path.clone(),
        kind,
        ok: true,
        errors: 0,
        warnings: 0,
        summary: String::new(),
        diagnostics: Vec::new(),
        inputs: Vec::new(),
        features: Vec::new(),
        counts: BTreeMap::new(),
        file_state: if text.trim().is_empty() {
            FileState::Empty
        } else {
            FileState::Written
        },
    };
    match kind {
        DesignKind::Part => check_part(&input.path, text, &mut out),
        DesignKind::Assembly => check_assembly(&input.path, text, files, &mut out),
        DesignKind::Drawing => check_drawing(&input.path, text, files, &mut out),
        DesignKind::Circuit | DesignKind::Layout => {
            check_circuit(&input.path, text, files, &mut out)
        }
    }
    out.errors = out
        .diagnostics
        .iter()
        .filter(|d| d.severity == "error")
        .count();
    out.warnings = out
        .diagnostics
        .iter()
        .filter(|d| d.severity == "warning")
        .count();
    out.ok = out.errors == 0;
    let what: Vec<String> = out
        .counts
        .iter()
        .filter(|(_, n)| **n > 0)
        .map(|(k, n)| format!("{n} {k}"))
        .collect();
    out.summary = format!(
        "{}: {} error(s), {} warning(s){}",
        input.path,
        out.errors,
        out.warnings,
        if what.is_empty() {
            String::new()
        } else {
            format!("; {}", what.join(", "))
        }
    );
    Ok(out)
}

/// A diagnostic from the CAD runtime's front end (UTF-16 offsets).
pub(crate) fn runtime_diagnostic(
    d: &runmat_cad_wasm::runtime::Diagnostic,
    files: &Files,
    fallback: &str,
) -> DesignDiagnostic {
    let text = files.get(&d.path).map(String::as_str).unwrap_or(fallback);
    DesignDiagnostic {
        id: None,
        severity: d.severity.clone(),
        code: None,
        message: d.message.clone(),
        cause: "root".into(),
        root: None,
        subject: None,
        failed_features: Vec::new(),
        range: Some(Locator::new(text).range_utf16(&d.path, d.start, d.end)),
        repairs: Vec::new(),
    }
}

fn severity(s: Severity) -> &'static str {
    match s {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Info => "info",
    }
}

fn subject_text(subject: &Subject) -> String {
    runmat_cad::diagnostics::Site {
        subject: subject.clone(),
        param: None,
    }
    .key()
}

/// CAD diagnostics with their root-cause structure; `failed` maps a root's id to the features
/// that failed because of it.
pub(crate) fn cad_diagnostics(
    path: &str,
    text: &str,
    diagnostics: &[CadDiagnostic],
    failed: &BTreeMap<u32, Vec<String>>,
) -> Vec<DesignDiagnostic> {
    let locator = Locator::new(text);
    diagnostics
        .iter()
        .map(|d| {
            let (cause, root) = match d.cause {
                Cause::Root => ("root", None),
                Cause::Cascade { root } => ("cascade", Some(root)),
            };
            // A cascade is a consequence: the editor shows it as a warning beside its root.
            let sev = match (d.severity, d.cause) {
                (Severity::Error, Cause::Cascade { .. }) => "warning",
                (s, _) => severity(s),
            };
            DesignDiagnostic {
                id: Some(d.id),
                severity: sev.into(),
                code: serde_json::to_value(d.code)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned)),
                message: d.message.clone(),
                cause: cause.into(),
                root,
                subject: Some(subject_text(&d.site.subject)),
                failed_features: if root.is_none() {
                    failed.get(&d.id).cloned().unwrap_or_default()
                } else {
                    Vec::new()
                },
                range: d.span.as_ref().map(|s| {
                    let file = s.file.as_str();
                    let file = if file.is_empty() { path } else { file };
                    locator.range(file, s.start as usize, s.end as usize)
                }),
                repairs: d
                    .repairs
                    .iter()
                    .filter_map(|r| serde_json::to_value(r).ok())
                    .collect(),
            }
        })
        .collect()
}

fn status_text(status: &FeatureStatus) -> (String, Option<u32>) {
    let (s, d) = match status {
        FeatureStatus::Regenerated => ("regenerated", None),
        FeatureStatus::CacheHit => ("regenerated", None),
        FeatureStatus::Frozen => ("frozen", None),
        FeatureStatus::Suppressed => ("suppressed", None),
        FeatureStatus::SuppressedInConfiguration => ("suppressed", None),
        FeatureStatus::RolledBack => ("rolled_back", None),
        FeatureStatus::Waiting { .. } => ("waiting", None),
        FeatureStatus::Failed { diagnostic } => ("failed", Some(*diagnostic)),
        FeatureStatus::Blocked { diagnostic, .. } => ("blocked", Some(*diagnostic)),
    };
    (s.to_owned(), d)
}

/// `80 mm`, `30°`, `0.5`: an SI value as the part's author reads it.
pub(crate) fn display_value(si: f64, kind: &str) -> String {
    let n = |v: f64| {
        let s = format!("{v:.6}");
        let s = s.trim_end_matches('0').trim_end_matches('.').to_owned();
        if s == "-0" {
            "0".to_owned()
        } else {
            s
        }
    };
    match kind {
        "length" | "distance" => format!("{} mm", n(si * 1e3)),
        "angle" => format!("{}°", n(si.to_degrees())),
        "area" => format!("{} mm²", n(si * 1e6)),
        "volume" => format!("{} mm³", n(si * 1e9)),
        "mass" => format!("{} kg", n(si)),
        _ => n(si),
    }
}

fn check_part(path: &str, text: &str, out: &mut CheckOutput) {
    let files = Files::new();
    let part = match runmat_cad_wasm::runtime::regenerate_part(path, text) {
        Ok(p) => p,
        Err(diagnostics) => {
            out.diagnostics = diagnostics
                .iter()
                .map(|d| runtime_diagnostic(d, &files, text))
                .collect();
            return;
        }
    };
    out.file_state = part.index.file_state;
    let mut failed: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for row in &part.index.features {
        let (status, diagnostic) = status_text(&row.status);
        if let Some(d) = diagnostic {
            // A blocked feature's diagnostic is the cascade; credit it to that cascade's root.
            let root = part
                .result
                .diagnostics
                .diagnostics
                .iter()
                .find(|x| x.id == d)
                .map(|x| match x.cause {
                    Cause::Root => x.id,
                    Cause::Cascade { root } => root,
                })
                .unwrap_or(d);
            failed.entry(root).or_default().push(row.id.0.clone());
        }
        out.features.push(FeatureState {
            id: row.id.0.clone(),
            kind: row.kind.clone(),
            status,
            diagnostic,
        });
    }
    out.diagnostics = cad_diagnostics(path, text, &part.result.diagnostics.diagnostics, &failed);
    out.inputs = part
        .index
        .inputs
        .iter()
        .map(|i| ResolvedInput {
            name: i.name.to_string(),
            kind: i.kind.clone(),
            si: i.si,
            display: i.si.map(|v| display_value(v, &i.kind)),
            error: i.error.clone(),
        })
        .collect();
    out.counts
        .insert("features".into(), part.index.features.len());
    out.counts.insert("bodies".into(), part.index.bodies.len());
    out.counts
        .insert("regions".into(), part.index.regions.len());
    out.counts
        .insert("published".into(), part.index.publishes.len());
}

fn check_assembly(path: &str, text: &str, files: &Files, out: &mut CheckOutput) {
    match runmat_cad_wasm::runtime::assembly_index(path, text, files) {
        Err(diagnostics) => {
            out.diagnostics = diagnostics
                .iter()
                .map(|d| runtime_diagnostic(d, files, text))
                .collect();
        }
        Ok((index, problems)) => {
            let scope = runmat_cad_wasm::runtime::module_scope(path, text, files);
            let parsed = runmat_cad::syntax::parse_assembly_in(path, text, &scope);
            let mut report = runmat_cad::diagnostics::DiagnosticReport {
                diagnostics: serde_json::from_value(index["diagnostics"].clone())
                    .unwrap_or_default(),
            };
            report.attach_spans(&parsed.spans);
            out.diagnostics = problems
                .iter()
                .map(|d| runtime_diagnostic(d, files, text))
                .collect();
            out.diagnostics.extend(cad_diagnostics(
                path,
                text,
                &report.diagnostics,
                &BTreeMap::new(),
            ));
            let len = |k: &str| index[k].as_array().map_or(0, Vec::len);
            out.counts.insert("occurrences".into(), len("occurrences"));
            out.counts.insert("mates".into(), len("mates"));
            out.counts.insert("prototypes".into(), len("prototypes"));
        }
    }
}

/// A drawing's check is the Desktop's Check drawing (`drawing.check`, runmat-cad-wasm
/// `analyze::drawing_check`): one set of rules, so the agent and the sheet's findings agree.
/// Its findings (references, models, title block, overlaps, scales and what the layout found)
/// come as diagnostics on the item each names. Syntax errors in the drawing itself come first,
/// with their ranges; a drawing that doesn't read is checked no further.
fn check_drawing(path: &str, text: &str, files: &Files, out: &mut CheckOutput) {
    let locator = Locator::new(text);
    let parsed = runmat_cad::syntax::parse_drawing_in(
        path,
        text,
        &runmat_cad_wasm::runtime::module_scope(path, text, files),
    );
    for d in parsed
        .diagnostics
        .iter()
        .filter(|_| parsed.document.is_none())
    {
        out.diagnostics.push(DesignDiagnostic {
            id: None,
            severity: "error".into(),
            code: None,
            message: d.message.clone(),
            cause: "root".into(),
            root: None,
            subject: None,
            failed_features: Vec::new(),
            range: Some(locator.range(path, d.span.start, d.span.end)),
            repairs: Vec::new(),
        });
    }
    let Some(document) = parsed.document else {
        return;
    };
    match runmat_cad_wasm::analyze::drawing_check::findings(
        path,
        text,
        files,
        &serde_json::Map::new(),
    ) {
        Ok((found, _)) => {
            for f in found {
                // The item's declaration: `view side =`, `annotation height =`, `sheet "1" =`.
                let declared = f.item.split_once(':').and_then(|(kind, name)| {
                    let quoted = format!("{kind} \"{name}\" =");
                    text.find(&format!("{kind} {name} ="))
                        .or_else(|| text.find(&quoted))
                });
                let range = declared.map(|start| {
                    let end = text[start..].find('\n').map_or(text.len(), |e| start + e);
                    locator.range(path, start, end)
                });
                out.diagnostics.push(DesignDiagnostic {
                    id: None,
                    severity: f.severity.to_owned(),
                    code: Some(format!(
                        "drawing.{}",
                        f.rule.to_lowercase().replace(' ', "_")
                    )),
                    message: format!("{}: {}", f.rule, f.message),
                    cause: "root".into(),
                    root: None,
                    subject: (!f.item.is_empty()).then(|| f.item.clone()),
                    failed_features: Vec::new(),
                    range,
                    repairs: Vec::new(),
                });
            }
        }
        Err(e) => out.diagnostics.push(DesignDiagnostic {
            id: None,
            severity: "error".into(),
            code: None,
            message: format!("the drawing check didn't run: {e}"),
            cause: "root".into(),
            root: None,
            subject: None,
            failed_features: Vec::new(),
            range: None,
            repairs: Vec::new(),
        }),
    }
    out.counts.insert("sheets".into(), document.sheets.len());
    out.counts.insert("views".into(), document.views.len());
    out.counts
        .insert("annotations".into(), document.annotations.len());
}

fn check_circuit(path: &str, text: &str, files: &Files, out: &mut CheckOutput) {
    let result = runmat_circuit::runtime::check(
        path,
        text,
        files,
        &crate::circuit_values::circuit_options(path, text, files),
    );
    for d in &result.diagnostics {
        let file_text = if d.path == path {
            text
        } else {
            files.get(&d.path).map(String::as_str).unwrap_or("")
        };
        out.diagnostics.push(DesignDiagnostic {
            id: None,
            severity: d.severity.clone(),
            code: None,
            message: d.message.clone(),
            cause: "root".into(),
            root: None,
            subject: None,
            failed_features: Vec::new(),
            range: Some(Locator::new(file_text).range_utf16(&d.path, d.start, d.end)),
            repairs: Vec::new(),
        });
    }
    if let Some(index) = &result.index {
        out.counts
            .insert("components".into(), index.components.len());
        out.counts.insert("nets".into(), index.nets.len());
        out.counts.insert("sheets".into(), index.hierarchy.len());
        out.counts.insert(
            "erc findings".into(),
            index.erc.iter().filter(|c| c.severity != "info").count(),
        );
        if let Some(board) = &index.board {
            out.counts
                .insert("footprints".into(), board.footprints.len());
            out.counts.insert("tracks".into(), board.tracks.len());
        }
    }
}
