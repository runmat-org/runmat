//! `design.propose_edit`: a change to a design as a staged change, the same kind the in-app
//! agent proposes. Nothing is written: the change carries the project calls that keep it
//! (`runmat.apply_patch` for changed files, `runmat.write_file` for new ones) and those that
//! undo it, the feature and dimension diff (parts and assemblies) or the schematic-to-board
//! change rows (circuits), and the diagnostics of every changed design file after the change.

use crate::check::{check, CheckInput, DesignDiagnostic};
use crate::{is_design_path, kind_of, text_of, DesignKind, Files, ToolError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProposeEditInput {
    /// Workspace path of the design file to change, or of a new design file to create (then
    /// give its whole `text`).
    pub path: String,
    /// The file's new source text. Give this or `command`.
    #[serde(default)]
    pub text: Option<String>,
    /// A catalog command id (`part.extrude`, `part.fillet`, `part.suppress`,
    /// `assembly.fix`, `board.route`, `schematic.update-board`, …). Give this or `text`.
    /// A blank id counts as no command: models fill every optional field and send `""` beside
    /// `replace`.
    #[serde(default, deserialize_with = "blank_as_none")]
    pub command: Option<String>,
    /// The command's values by field id, numbers in the field's unit (`{ "depth": 12 }` for
    /// 12 mm); selection fields take Outline ids or names.
    #[serde(default)]
    #[schemars(with = "Option<BTreeMap<String, Value>>")]
    pub values: Option<Map<String, Value>>,
    /// The Outline ids the command runs on (`feature:boss`, `component:R1`, `track:17`).
    #[serde(default)]
    pub selection: Option<Vec<String>>,
    /// Text replacements on the file's current source, applied in order: each `old` must occur
    /// exactly once. Give this, `text` or `command`; it saves resending a whole file for a
    /// change of a few lines.
    #[serde(default)]
    pub replace: Option<Vec<Replacement>>,
}

fn blank_as_none<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.filter(|text| !text.trim().is_empty()))
}

/// One replacement of [`ProposeEditInput::replace`].
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Replacement {
    /// Text that occurs exactly once in the file, with enough surrounding lines to be unique.
    pub old: String,
    /// What replaces it.
    pub new: String,
}

/// The file's text after `replace`, or why a replacement doesn't apply.
fn replaced(text: &str, replace: &[Replacement]) -> Result<String, ToolError> {
    let mut out = text.to_owned();
    for (n, r) in replace.iter().enumerate() {
        match out.matches(r.old.as_str()).count() {
            1 => out = out.replacen(r.old.as_str(), &r.new, 1),
            0 => return Err(ToolError::invalid(format!("replacement {} doesn't match the file's current text; read the file and copy `old` exactly", n + 1))),
            c => return Err(ToolError::invalid(format!("replacement {} matches {c} places; add surrounding lines to `old` so it matches once", n + 1))),
        }
    }
    Ok(out)
}

/// A text edit of the old text, in UTF-16 offsets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TextEdit {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// One file of a staged change.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StagedFile {
    pub path: String,
    /// The file is new.
    pub created: bool,
    /// The whole new text.
    pub text: String,
    /// The edits from the current text (empty for a new file).
    pub edits: Vec<TextEdit>,
}

/// A feature or occurrence the change adds, removes or changes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ItemChange {
    pub id: String,
    /// `added`, `removed` or `changed`.
    pub change: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// The parameters that changed (`depth`, `end`, `radius`), for `changed`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<String>,
    /// Regeneration status after the change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// An input whose value changes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DimensionChange {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ChangeDiff {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<ItemChange>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dimensions: Vec<DimensionChange>,
    /// Circuits: the schematic-to-board (ECO) rows the board would take after the change.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub eco_rows: Vec<Value>,
}

/// A project tool call (the server adds `projectId` and `baseRevision`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProjectCall {
    pub tool: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StagedChange {
    pub summary: String,
    pub files: Vec<StagedFile>,
    pub diff: ChangeDiff,
    /// The diagnostics of every changed design file with the change applied.
    pub diagnostics: Vec<DesignDiagnostic>,
    /// The change leaves no errors.
    pub ok: bool,
    /// Calls that keep the change, in order.
    pub keep: Vec<ProjectCall>,
    /// Calls that undo it after it is kept (pass the revision `keep` returned).
    pub undo: Vec<ProjectCall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// The smallest replacement that turns `old` into `new` whose old text occurs exactly once in
/// `old` (what `runmat.apply_patch` needs), growing by whole lines until it is unique.
pub fn unique_patch(old: &str, new: &str) -> (String, String) {
    let prefix = old
        .bytes()
        .zip(new.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    let room = old.len().min(new.len()) - prefix;
    let suffix = old
        .bytes()
        .rev()
        .zip(new.bytes().rev())
        .take(room)
        .take_while(|(a, b)| a == b)
        .count();
    let line_start = |at: usize| old[..at].rfind('\n').map_or(0, |i| i + 1);
    let line_end = |at: usize| {
        if at > 0 && old.as_bytes()[at - 1] == b'\n' {
            at
        } else {
            old[at..].find('\n').map_or(old.len(), |i| at + i + 1)
        }
    };
    let mut start = line_start(prefix);
    let mut end = line_end(old.len() - suffix);
    loop {
        let candidate = &old[start..end];
        if (!candidate.is_empty() && old.matches(candidate).count() == 1)
            || (start == 0 && end == old.len())
        {
            break;
        }
        if start > 0 {
            start = line_start(start - 1);
        }
        if end < old.len() {
            end = line_end(end + 1);
        }
    }
    let tail = old.len() - end;
    (
        old[start..end].to_owned(),
        new[start..new.len() - tail].to_owned(),
    )
}

fn calls_for(files: &[StagedFile], before: &Files) -> (Vec<ProjectCall>, Vec<ProjectCall>) {
    let mut keep_edits = Vec::new();
    let mut undo_edits = Vec::new();
    let mut keep = Vec::new();
    let mut undo = Vec::new();
    for f in files {
        match before.get(&f.path) {
            Some(old) if !f.created => {
                if old == &f.text {
                    continue;
                }
                let (o, n) = unique_patch(old, &f.text);
                keep_edits.push(json!({ "path": f.path, "oldText": o, "newText": n }));
                let (o, n) = unique_patch(&f.text, old);
                undo_edits.push(json!({ "path": f.path, "oldText": o, "newText": n }));
            }
            _ => {
                keep.push(ProjectCall {
                    tool: "runmat.write_file".into(),
                    arguments: json!({ "path": f.path, "contents": f.text }),
                });
                undo.push(ProjectCall {
                    tool: "runmat.delete_file".into(),
                    arguments: json!({ "path": f.path, "recursive": false }),
                });
            }
        }
    }
    if !keep_edits.is_empty() {
        keep.insert(
            0,
            ProjectCall {
                tool: "runmat.apply_patch".into(),
                arguments: json!({ "edits": keep_edits }),
            },
        );
        undo.insert(
            0,
            ProjectCall {
                tool: "runmat.apply_patch".into(),
                arguments: json!({ "edits": undo_edits }),
            },
        );
    }
    (keep, undo)
}

/// The feature and dimension diff of a part between two texts; a new part (`before` is `None`)
/// adds every feature and input.
fn part_diff(path: &str, before: Option<&str>, after: &str) -> ChangeDiff {
    let compile = |t: &str| runmat_cad_wasm::runtime::regenerate_part(path, t).ok();
    let Some(b) = compile(after) else {
        return ChangeDiff::default();
    };
    let a = match before {
        Some(t) => match compile(t) {
            Some(a) => Some(a),
            None => return ChangeDiff::default(),
        },
        None => None,
    };
    let features = |p: &runmat_cad_wasm::runtime::RegeneratedPart| -> BTreeMap<String, Value> {
        serde_json::to_value(&p.document)
            .ok()
            .and_then(|d| d["features"].as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|f| f["id"].as_str().map(|id| (id.to_owned(), f.clone())))
            .collect()
    };
    let status = |p: &runmat_cad_wasm::runtime::RegeneratedPart, id: &str| {
        p.index
            .features
            .iter()
            .find(|f| f.id.0 == id)
            .and_then(|f| serde_json::to_value(&f.status).ok())
            .map(|v| match v {
                Value::String(s) => s,
                Value::Object(o) => o.keys().next().cloned().unwrap_or_default(),
                other => other.to_string(),
            })
            .map(|s| s.to_ascii_lowercase())
    };
    let kind = |f: &Value| {
        f["kind"]["feature"]
            .as_str()
            .or_else(|| f["feature"].as_str())
            .map(str::to_owned)
    };
    let (fa, fb) = (a.as_ref().map(features).unwrap_or_default(), features(&b));
    let mut out = ChangeDiff::default();
    let ids: BTreeSet<&String> = fa.keys().chain(fb.keys()).collect();
    for id in ids {
        match (fa.get(id), fb.get(id)) {
            (None, Some(f)) => out.features.push(ItemChange {
                id: id.clone(),
                change: "added".into(),
                kind: kind(f),
                params: Vec::new(),
                status: status(&b, id),
            }),
            (Some(f), None) => out.features.push(ItemChange {
                id: id.clone(),
                change: "removed".into(),
                kind: kind(f),
                params: Vec::new(),
                status: None,
            }),
            (Some(x), Some(y)) if x != y => {
                let mut params = Vec::new();
                diff_keys("", x, y, &mut params);
                out.features.push(ItemChange {
                    id: id.clone(),
                    change: "changed".into(),
                    kind: kind(y),
                    params,
                    status: status(&b, id),
                });
            }
            _ => {}
        }
    }
    let inputs = |p: &runmat_cad_wasm::runtime::RegeneratedPart| -> BTreeMap<String, String> {
        p.index
            .inputs
            .iter()
            .map(|i| {
                (
                    i.name.to_string(),
                    i.si.map(|v| crate::check::display_value(v, &i.kind))
                        .unwrap_or_else(|| i.error.clone().unwrap_or_default()),
                )
            })
            .collect()
    };
    let (ia, ib) = (a.as_ref().map(inputs).unwrap_or_default(), inputs(&b));
    for name in ia.keys().chain(ib.keys()).collect::<BTreeSet<_>>() {
        let (x, y) = (ia.get(name), ib.get(name));
        if x != y {
            out.dimensions.push(DimensionChange {
                name: name.clone(),
                before: x.cloned(),
                after: y.cloned(),
            });
        }
    }
    out
}

/// The dotted keys at which two JSON values differ (`params.depth`, `end`).
fn diff_keys(at: &str, a: &Value, b: &Value, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys()).collect::<BTreeSet<_>>() {
                if k == "span" || k == "uuid" {
                    continue;
                }
                let next = if at.is_empty() {
                    k.clone()
                } else {
                    format!("{at}.{k}")
                };
                match (x.get(k), y.get(k)) {
                    (Some(p), Some(q)) => diff_keys(&next, p, q, out),
                    _ => out.push(next),
                }
            }
        }
        _ if a != b => out.push(if at.is_empty() {
            "value".into()
        } else {
            at.to_owned()
        }),
        _ => {}
    }
}

fn assembly_diff(before: &Files, after: &Files, path: &str) -> ChangeDiff {
    let index = |files: &Files| -> Option<Value> {
        let t = files.get(path)?;
        runmat_cad_wasm::runtime::assembly_index(path, t, files)
            .ok()
            .map(|(i, _)| i)
    };
    // A new assembly (nothing before) adds every occurrence and mate.
    let a = if before.contains_key(path) {
        match index(before) {
            Some(a) => a,
            None => return ChangeDiff::default(),
        }
    } else {
        Value::Null
    };
    let Some(b) = index(after) else {
        return ChangeDiff::default();
    };
    let rows = |v: &Value, key: &str, id: &str| -> BTreeMap<String, Value> {
        v[key]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|r| r[id].as_str().map(|k| (k.to_owned(), r.clone())))
            .collect()
    };
    let mut out = ChangeDiff::default();
    for (key, id, kind) in [
        ("occurrences", "path", "occurrence"),
        ("mates", "name", "mate"),
    ] {
        let (x, y) = (rows(&a, key, id), rows(&b, key, id));
        for k in x.keys().chain(y.keys()).collect::<BTreeSet<_>>() {
            let change = match (x.get(k), y.get(k)) {
                (None, Some(_)) => "added",
                (Some(_), None) => "removed",
                (Some(p), Some(q)) if p != q => "changed",
                _ => continue,
            };
            let mut params = Vec::new();
            if let (Some(p), Some(q)) = (x.get(k), y.get(k)) {
                diff_keys("", p, q, &mut params);
            }
            out.features.push(ItemChange {
                id: format!("{kind}.{k}"),
                change: change.into(),
                kind: Some(kind.into()),
                params,
                status: None,
            });
        }
    }
    out
}

fn circuit_rows(path: &str, files: &Files) -> Vec<Value> {
    let Some(text) = files.get(path) else {
        return Vec::new();
    };
    let out = runmat_circuit::runtime::compile(path, text, files, &crate::circuit_values::circuit_options(path, text, files));
    out.index
        .map(|i| {
            i.eco
                .groups
                .iter()
                .flat_map(|g| g.rows.iter())
                .filter_map(|r| serde_json::to_value(r).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Diagnostics of every changed design file with `after` applied.
fn diagnostics_after(changed: &[StagedFile], after: &Files) -> Vec<DesignDiagnostic> {
    let mut out = Vec::new();
    for f in changed.iter().filter(|f| is_design_path(&f.path)) {
        if f.path.ends_with(".circuit.lock") {
            continue;
        }
        if let Ok(c) = check(
            &CheckInput {
                path: f.path.clone(),
            },
            after,
        ) {
            out.extend(c.diagnostics);
        }
    }
    out
}

/// Builds the staged change from new file texts.
pub fn stage(
    summary: String,
    changed: Vec<(String, String)>,
    files: &Files,
    primary: &str,
) -> StagedChange {
    let staged: Vec<StagedFile> = changed
        .into_iter()
        .map(|(path, text)| {
            let created = !files.contains_key(&path);
            let edits = match files.get(&path) {
                Some(old) if old != &text => {
                    let (o, n) = unique_patch(old, &text);
                    let at = old.find(&o).unwrap_or(0);
                    let l = crate::Locator::new(old);
                    vec![TextEdit {
                        start: l.utf16(at),
                        end: l.utf16(at + o.len()),
                        text: n,
                    }]
                }
                _ => Vec::new(),
            };
            StagedFile {
                path,
                created,
                text,
                edits,
            }
        })
        .collect();
    let mut after = files.clone();
    for f in &staged {
        after.insert(f.path.clone(), f.text.clone());
    }
    let diff = match crate::DesignKind::of(primary) {
        Some(DesignKind::Part) => match (files.get(primary), after.get(primary)) {
            (a, Some(b)) => part_diff(primary, a.map(String::as_str), b),
            _ => ChangeDiff::default(),
        },
        Some(DesignKind::Assembly) => assembly_diff(files, &after, primary),
        Some(DesignKind::Circuit | DesignKind::Layout) => ChangeDiff {
            eco_rows: circuit_rows(primary, &after),
            ..ChangeDiff::default()
        },
        _ => ChangeDiff::default(),
    };
    let diagnostics = diagnostics_after(&staged, &after);
    let ok = !diagnostics.iter().any(|d| d.severity == "error");
    let (keep, undo) = calls_for(&staged, files);
    StagedChange {
        summary,
        files: staged,
        diff,
        diagnostics,
        ok,
        keep,
        undo,
        notes: Vec::new(),
    }
}

/// The catalog command `id`, or the error that names close matches.
fn known_command(id: &str, kind: DesignKind) -> Result<(), ToolError> {
    let catalog = if kind.is_circuit() {
        runmat_circuit::commands::catalog()
    } else {
        runmat_cad::commands::catalog()
    };
    if catalog.commands.iter().any(|c| c.id == id) {
        return Ok(());
    }
    let stem = id.split('.').next().unwrap_or(id);
    let near: Vec<&str> = catalog
        .commands
        .iter()
        .map(|c| c.id.as_str())
        .filter(|c| c.starts_with(stem))
        .take(12)
        .collect();
    Err(ToolError::invalid(format!(
        "{id} isn't a catalog command for this file{}",
        if near.is_empty() {
            String::new()
        } else {
            format!("; {stem} commands include {}", near.join(", "))
        }
    )))
}

pub fn propose(input: &ProposeEditInput, files: &Files) -> Result<StagedChange, ToolError> {
    let kind = kind_of(&input.path)?;
    if !files.contains_key(&input.path) {
        // A new design file: its whole text, staged as a file to create.
        return match (&input.text, &input.command, &input.replace) {
            (Some(new), None, None) => Ok(stage(
                format!("Create {} ({} bytes)", input.path, new.len()),
                vec![(input.path.clone(), new.clone())],
                files,
                &input.path,
            )),
            _ => Err(ToolError::new(
                "file_missing",
                format!("{} isn't in the workspace; to create it, give its whole `text`", input.path),
            )),
        };
    }
    let text = text_of(files, &input.path)?;
    let from_replace = match &input.replace {
        Some(replace) if input.text.is_none() && input.command.is_none() => Some(replaced(text, replace)?),
        Some(_) => {
            let other = if input.text.is_some() { "`text`".to_owned() } else { format!("`command` {:?}", input.command.as_deref().unwrap_or("")) };
            return Err(ToolError::invalid(format!(
                "`replace` came with {other}; give one of them: for a text change keep `replace` and leave `command` and `text` null"
            )));
        }
        None => None,
    };
    let text_input = from_replace.as_ref().or(input.text.as_ref());
    match (text_input, &input.command) {
        (Some(new), None) => {
            let summary = format!("Edit {} ({} → {} bytes)", input.path, text.len(), new.len());
            Ok(stage(
                summary,
                vec![(input.path.clone(), new.clone())],
                files,
                &input.path,
            ))
        }
        (None, Some(command)) => {
            known_command(command, kind)?;
            let values = input.values.clone().unwrap_or_default();
            let selection = input.selection.clone().unwrap_or_default();
            let (summary, changed): (String, Vec<(String, String)>) = if kind.is_circuit() {
                let request = runmat_circuit::runtime::EditRequest {
                    command: command.clone(),
                    values,
                    selection,
                    options: Default::default(),
                };
                let out = runmat_circuit::runtime::edit(&input.path, text, files, &request)
                    .map_err(|e| ToolError::new("edit_failed", e.message))?;
                (
                    out.summary,
                    out.files.into_iter().map(|f| (f.path, f.text)).collect(),
                )
            } else if kind.is_model() {
                let request = runmat_cad_wasm::edit::EditRequest {
                    command: command.clone(),
                    values,
                    selection,
                };
                let out = runmat_cad_wasm::edit::edit(&input.path, text, files, &request)
                    .map_err(|e| ToolError::new("edit_failed", e.to_string()))?;
                (
                    out.summary,
                    out.files.into_iter().map(|f| (f.path, f.text)).collect(),
                )
            } else {
                return Err(ToolError::new(
                    "unsupported",
                    "drawings change through their source text: pass `text`",
                ));
            };
            Ok(stage(summary, changed, files, &input.path))
        }
        _ => Err(ToolError::invalid(
            "give one of `text` (the new source), `replace` (replacements on the current source) or `command` (a catalog command id)",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_patch_is_whole_lines_and_unique() {
        let old = "a\nx = 1\nb\nx = 1\nc\n";
        let new = "a\nx = 1\nb\nx = 2\nc\n";
        let (o, n) = unique_patch(old, new);
        assert_eq!(old.matches(&o).count(), 1, "{o:?}");
        assert_eq!(old.replacen(&o, &n, 1), new);
    }

    #[test]
    fn replacements_apply_once_each_and_refuse_ambiguity() {
        let r = |old: &str, new: &str| Replacement { old: old.into(), new: new.into() };
        assert_eq!(replaced("a\nb\nc\n", &[r("b\n", "b\nx\n")]).unwrap(), "a\nb\nx\nc\n");
        assert!(replaced("a\na\n", &[r("a", "b")]).unwrap_err().message.contains("matches 2 places"));
        assert!(replaced("a\n", &[r("z", "b")]).unwrap_err().message.contains("doesn't match"));
    }

    #[test]
    fn a_blank_command_beside_replace_counts_as_no_command() {
        let input: ProposeEditInput = serde_json::from_value(json!({
            "path": "/a.circuit",
            "command": "",
            "replace": [{ "old": "b", "new": "c" }],
        }))
        .unwrap();
        assert!(input.command.is_none());
        let input: ProposeEditInput =
            serde_json::from_value(json!({ "path": "/a.circuit", "command": "board.route" })).unwrap();
        assert_eq!(input.command.as_deref(), Some("board.route"));
    }

    #[test]
    fn a_command_beside_replace_is_refused_naming_both() {
        let input: ProposeEditInput = serde_json::from_value(json!({
            "path": "/a.circuit",
            "command": "x",
            "replace": [{ "old": "b", "new": "c" }],
        }))
        .unwrap();
        let files: Files = [("/a.circuit".to_string(), "b\n".to_string())].into_iter().collect();
        let message = propose(&input, &files).err().expect("refused").message;
        assert!(message.contains("`replace` came with `command` \"x\""), "{message}");
        assert!(message.contains("leave `command` and `text` null"), "{message}");
    }

    #[test]
    fn an_insertion_is_anchored_on_a_line() {
        let old = "part p {\n}\n";
        let new = "part p {\n    input w = 3 mm\n}\n";
        let (o, n) = unique_patch(old, new);
        assert!(!o.is_empty());
        assert_eq!(old.replacen(&o, &n, 1), new);
    }
}
