//! Catalog commands on parts and assemblies, run from the canvas: each becomes a structured
//! model edit (`runmat_cad::syntax::edit`) written back as the smallest text change.
//!
//! The request is `{ command, values, selection }`: the catalog command ID, the running
//! command's values as the Desktop holds them (numbers in the field's unit, value edits,
//! selection lists of Outline IDs) and the selection it started from. The reply is
//! `{ ok: true, files: [{ path, text, edits, reprinted }], summary }` or
//! `{ ok: false, message, diagnostics }`, the same shape the circuit runtime returns.
//!
//! Parts: `part.extrude` (blind, mid-plane and through-all ends in one or two directions, add,
//! cut, intersect or a new body, sketch-plane or offset start, draft and thin walls),
//! `part.delete`, `part.suppress`, `part.unsuppress` (in the active configuration when
//! `values.configuration` names one), `part.move-up` and `part.move-down`; Properties and
//! configuration commands ([`configuration`]).
//! Sketches: `part.sketch` and the sketch tools ([`sketch`]).
//! Assemblies: delete, suppress, fix, insert, move, pattern and mate ([`assembly`]). Every
//! other part command that adds a feature is built by [`crate::features`] from
//! its values and picks ([`crate::picks`]); a command or option the model lacks says so.

mod assembly;
mod configuration;
mod sketch;

use crate::runtime::Files;
use runmat_cad::model::{Feature, PartDocument};
use runmat_cad::syntax::edit::{edit_part, reorder_feature, EditFailure, SourceChange};
use runmat_cad::syntax::parse_part;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

#[derive(Clone, Debug, Default, Deserialize)]
pub struct EditRequest {
    pub command: String,
    #[serde(default)]
    pub values: Map<String, Value>,
    #[serde(default)]
    pub selection: Vec<String>,
}

/// An edit in UTF-16 offsets of the old text.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Utf16Edit {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FileChange {
    pub path: String,
    pub text: String,
    pub edits: Vec<Utf16Edit>,
    pub reprinted: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EditOutput {
    pub files: Vec<FileChange>,
    pub summary: String,
}

fn utf16(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    text[..at].encode_utf16().count()
}

fn file_change(path: &str, old: &str, change: SourceChange) -> FileChange {
    let mut edits: Vec<Utf16Edit> = change
        .edits
        .iter()
        .map(|e| Utf16Edit {
            start: utf16(old, e.span.start),
            end: utf16(old, e.span.end),
            text: e.text.clone(),
        })
        .collect();
    edits.sort_by_key(|e| (e.start, e.end));
    FileChange {
        path: path.to_owned(),
        text: change.source,
        edits,
        reprinted: change.reprinted,
    }
}

pub(crate) fn model(message: impl Into<String>) -> EditFailure {
    EditFailure::Model(message.into())
}

/// The feature an Outline ID names (`feature:body`, `sketch:s1`).
fn feature_of(id: &str) -> Option<&str> {
    id.strip_prefix("feature:")
        .or_else(|| id.strip_prefix("sketch:"))
}

/// The first free `{stem}{n}` among the part's features.
fn fresh_id(document: &PartDocument, stem: &str) -> String {
    (1..)
        .map(|n| format!("{stem}{n}"))
        .find(|id| !document.features.iter().any(|f| f.id.0 == *id))
        .expect("an unused name")
}

fn selected_features(request: &EditRequest) -> Vec<String> {
    request
        .selection
        .iter()
        .filter_map(|s| feature_of(s).map(str::to_owned))
        .collect()
}

fn part_edit(
    path: &str,
    text: &str,
    request: &EditRequest,
) -> Result<(SourceChange, String), EditFailure> {
    if let Some(result) = configuration::edit(path, text, request) {
        return result;
    }
    let selected = selected_features(request);
    let need_one = || -> Result<(), EditFailure> {
        if selected.is_empty() {
            Err(model("select a feature first"))
        } else {
            Ok(())
        }
    };
    match request.command.as_str() {
        "part.delete" => {
            need_one()?;
            let change = edit_part(text, path, |document| {
                for id in &selected {
                    let at = document
                        .features
                        .iter()
                        .position(|f| f.id.0 == *id)
                        .ok_or_else(|| model(format!("the part has no feature `{id}`")))?;
                    document.features.remove(at);
                }
                Ok(())
            })?;
            Ok((change, format!("Deleted {}", selected.join(", "))))
        }
        "part.suppress" | "part.unsuppress" => {
            need_one()?;
            let on = request.command == "part.suppress";
            // In the active configuration only, when one is active.
            let configuration = request
                .values
                .get("configuration")
                .and_then(Value::as_str);
            let change = runmat_cad::syntax::configuration_edit::suppress_features(
                text,
                path,
                &selected,
                configuration,
                on,
            )?;
            let verb = if on { "Suppressed" } else { "Unsuppressed" };
            Ok((change, format!("{verb} {}", selected.join(", "))))
        }
        "part.move-up" | "part.move-down" => {
            need_one()?;
            let id = &selected[0];
            let parsed = parse_part(path, text);
            let document = parsed
                .document
                .ok_or(EditFailure::Invalid(parsed.diagnostics))?;
            let at = document
                .features
                .iter()
                .position(|f| f.id.0 == *id)
                .ok_or_else(|| model(format!("the part has no feature `{id}`")))?;
            let names: Vec<&str> = document.features.iter().map(|f| f.id.0.as_str()).collect();
            // After the feature two up (or at the front), or after the next one.
            let after = if request.command == "part.move-up" {
                if at == 0 {
                    return Err(model(format!("`{id}` is first already")));
                }
                at.checked_sub(2).map(|k| names[k])
            } else {
                Some(
                    *names
                        .get(at + 1)
                        .ok_or_else(|| model(format!("`{id}` is last already")))?,
                )
            };
            let change = reorder_feature(text, path, id, after)?;
            Ok((change, format!("Moved {id}")))
        }
        "repair.repair" => crate::repairs::repair_part(path, text, &request.values),
        command => {
            // A command that adds a feature: its values and picks as the feature's model.
            let parsed = parse_part(path, text);
            let document = parsed
                .document
                .ok_or(EditFailure::Invalid(parsed.diagnostics))?;
            let picks = crate::picks::Picks::new(&document, path, text);
            if let Some(result) = sketch::sketch_edit(path, text, command, &request.values, &picks)
            {
                return result;
            }
            let lowered = match crate::direct_edits::direct_kind(command, &request.values, &picks)? {
                Some(kind) => Some(kind),
                None => match crate::surface_edits::surface_kind(command, &request.values, &picks)? {
                    Some(kind) => Some(kind),
                    None => match crate::weldment_edits::weldment_kind(command, &request.values, &picks)? {
                        Some(kind) => Some(kind),
                        None => crate::features::feature_kind(command, &request.values, &picks)?,
                    },
                },
            };
            let Some((stem, kind)) = lowered else {
                return Err(model(format!(
                    "`{command}` edits parts through the source editor in this build"
                )));
            };
            let kind: runmat_cad::model::FeatureKind =
                serde_json::from_value(kind).map_err(|e| model(format!("the {stem}: {e}")))?;
            let mut added = String::new();
            let change = edit_part(text, path, |document| {
                added = fresh_id(document, stem);
                document.features.push(Feature::new(&added, kind.clone()));
                Ok(())
            })?;
            Ok((change, format!("Added {added}")))
        }
    }
}

fn text_value<'a>(values: &'a Map<String, Value>, id: &str) -> &'a str {
    values.get(id).and_then(Value::as_str).unwrap_or("")
}

/// Runs a catalog command on the part or assembly `path`.
pub fn edit(
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
) -> Result<EditOutput, EditFailure> {
    let (change, summary) = if path.ends_with(".drawing") {
        match request.command.as_str() {
            "drawing.auto" => crate::drawing::auto::auto(path, text, files, &request.values)?,
            other => match crate::drawing_views::view_edit(path, text, files, other, &request.values, &request.selection) {
                Some(result) => result?,
                None => match crate::drawing_edits::drawing_edit(path, text, files, other, &request.values, &request.selection) {
                    // Views, annotations and tables; tolerances and datums also change the part.
                    Some(result) => {
                        let (changes, summary) = result?;
                        return Ok(EditOutput {
                            files: changes.into_iter().map(|(p, old, change)| file_change(&p, &old, change)).collect(),
                            summary,
                        });
                    }
                    None => {
                        return Err(model(format!(
                            "`{other}` edits drawings through the source editor in this build"
                        )))
                    }
                },
            },
        }
    } else if path.ends_with(".assembly") {
        let (change, summary, others) = assembly::edit(path, text, files, request)?;
        let mut out = vec![file_change(path, text, change)];
        out.extend(others.into_iter().map(|(p, old, change)| file_change(&p, &old, change)));
        return Ok(EditOutput { files: out, summary });
    } else if request.command == "repair.repair" {
        // Pins and publishes can change other files.
        let (changes, summary) = crate::repairs::repair_part_files(path, text, files, &request.values)?;
        return Ok(EditOutput {
            files: changes.into_iter().map(|(p, old, change)| file_change(&p, &old, change)).collect(),
            summary,
        });
    } else {
        part_edit(path, text, request)?
    };
    Ok(EditOutput {
        files: vec![file_change(path, text, change)],
        summary,
    })
}

/// `edit` as JSON.
pub fn edit_json(path: &str, text: &str, files: &Files, request: &str) -> String {
    let request: EditRequest = match serde_json::from_str(request) {
        Ok(r) => r,
        Err(e) => return json!({ "ok": false, "message": format!("the edit request isn't valid: {e}"), "diagnostics": [] }).to_string(),
    };
    match edit(path, text, files, &request) {
        Ok(out) => json!({ "ok": true, "files": out.files, "summary": out.summary }).to_string(),
        Err(failure) => {
            let diagnostics: Vec<Value> = match &failure {
                EditFailure::Invalid(d) => d
                    .iter()
                    .map(|d| json!({ "path": path, "start": utf16(text, d.span.start), "end": utf16(text, d.span.end), "severity": "error", "message": d.message }))
                    .collect(),
                _ => Vec::new(),
            };
            json!({ "ok": false, "message": failure.to_string(), "diagnostics": diagnostics })
                .to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::compile_assembly;

    const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n}\n";

    fn run(path: &str, text: &str, request: Value) -> Value {
        serde_json::from_str(&edit_json(path, text, &Files::new(), &request.to_string())).unwrap()
    }

    #[test]
    fn extrude_from_the_canvas_writes_one_statement_and_compiles() {
        let out = run(
            "/p/plate.part",
            PLATE,
            json!({ "command": "part.extrude", "values": { "profile": ["feature:s1"], "op": "new", "end": "blind", "distance": 12.5 } }),
        );
        assert_eq!(out["ok"], true, "{out}");
        let file = &out["files"][0];
        let new = file["text"].as_str().unwrap();
        assert!(new.contains("feature extrude1 = extrude("), "{new}");
        assert!(new.contains("12.5 mm"), "{new}");
        assert_eq!(file["reprinted"], false);
        // A minimal edit: one insertion, the sketch untouched.
        let edits = file["edits"].as_array().unwrap();
        assert_eq!(edits.len(), 1, "{edits:?}");
        assert_eq!(edits[0]["start"], edits[0]["end"]);
        assert!(new.starts_with(&PLATE[..PLATE.rfind('}').unwrap() - 1]));
        // The new text compiles to a body.
        let compiled: Value = serde_json::from_str(&crate::runtime::compile_json(
            "/p/plate.part",
            new,
            &Files::new(),
        ))
        .unwrap();
        assert!(!compiled["display"]["data"]["bodies"]
            .as_array()
            .unwrap()
            .is_empty());
        assert!(new.contains("feature extrude1"));
        // A second extrude takes the next name; a value edit carries its expression.
        let again = run(
            "/p/plate.part",
            new,
            json!({ "command": "part.extrude", "values": { "profile": ["feature:s1"], "op": "add", "end": "mid_plane", "distance": { "kind": "value", "value": { "form": "quantity", "value": 3, "unit": "mm" }, "text": "3 mm", "scope": { "scope": "all" } } } }),
        );
        assert_eq!(again["ok"], true, "{again}");
        assert!(again["files"][0]["text"]
            .as_str()
            .unwrap()
            .contains("feature extrude2"));
    }

    #[test]
    fn suppress_delete_and_reorder_edit_the_history() {
        let with = run(
            "/p/plate.part",
            PLATE,
            json!({ "command": "part.extrude", "values": { "profile": ["feature:s1"], "distance": 5 } }),
        );
        let text = with["files"][0]["text"].as_str().unwrap().to_owned();
        let suppressed = run(
            "/p/plate.part",
            &text,
            json!({ "command": "part.suppress", "selection": ["feature:extrude1"] }),
        );
        assert_eq!(suppressed["ok"], true, "{suppressed}");
        assert_ne!(suppressed["files"][0]["text"].as_str().unwrap(), text);
        let up = run(
            "/p/plate.part",
            &text,
            json!({ "command": "part.move-up", "selection": ["feature:extrude1"] }),
        );
        assert_eq!(up["ok"], true, "{up}");
        let moved = up["files"][0]["text"].as_str().unwrap();
        assert!(moved.find("feature extrude1").unwrap() < moved.find("sketch s1").unwrap());
        let deleted = run(
            "/p/plate.part",
            &text,
            json!({ "command": "part.delete", "selection": ["feature:extrude1"] }),
        );
        assert_eq!(deleted["ok"], true, "{deleted}");
        assert!(!deleted["files"][0]["text"]
            .as_str()
            .unwrap()
            .contains("extrude1"));
    }

    #[test]
    fn edits_it_cant_make_say_why() {
        let none = run(
            "/p/plate.part",
            PLATE,
            json!({ "command": "part.extrude", "values": { "distance": 5 } }),
        );
        assert_eq!(none["ok"], false);
        assert!(none["message"].as_str().unwrap().contains("sketch"));
        let face = run(
            "/p/plate.part",
            PLATE,
            json!({ "command": "part.extrude", "values": { "profile": ["feature:s1"], "end": "up_to_face" } }),
        );
        assert_eq!(face["ok"], false);
        // An end that names geometry asks for the pick.
        assert!(
            face["message"]
                .as_str()
                .unwrap()
                .contains("pick what the extrude goes up to"),
            "{face}"
        );
        let bad = run(
            "/p/plate.part",
            "part {",
            json!({ "command": "part.delete", "selection": ["feature:x"] }),
        );
        assert_eq!(bad["ok"], false);
    }

    #[test]
    fn assembly_components_delete_and_suppress_by_their_outline_ids() {
        let assembly = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as a\n    use part plate as b {\n        placement = { placement = offset, offset = { translation = [100 mm, 0 mm, 0 mm] } }\n    }\n}\n";
        let files: Files = [("/p/plate.part".to_owned(), PLATE.to_owned())].into();
        let (index, _, _) = compile_assembly("/p/rig.assembly", assembly, &files).unwrap();
        let b = index["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["path"] == "b")
            .unwrap()["uuid"]
            .as_str()
            .unwrap()
            .to_owned();
        let req = |command: &str| {
            json!({ "command": command, "selection": [format!("occurrence:{b}")] }).to_string()
        };
        let out: Value = serde_json::from_str(&edit_json(
            "/p/rig.assembly",
            assembly,
            &files,
            &req("assembly.delete"),
        ))
        .unwrap();
        assert_eq!(out["ok"], true, "{out}");
        let text = out["files"][0]["text"].as_str().unwrap();
        assert!(text.contains(" as a") && !text.contains(" as b"), "{text}");
        let out: Value = serde_json::from_str(&edit_json(
            "/p/rig.assembly",
            assembly,
            &files,
            &req("assembly.suppress"),
        ))
        .unwrap();
        assert_eq!(out["ok"], true, "{out}");
        assert!(
            out["files"][0]["text"]
                .as_str()
                .unwrap()
                .contains("suppressed"),
            "{out}"
        );
    }
}
