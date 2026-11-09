//! `manage.split-to-files`: each picked body (every body when none is picked) becomes its own
//! `.part` beside this one, and an assembly places them where they sit.
//!
//! A new part holds this part's history and ends with a `body_op` that deletes the other bodies,
//! so it regenerates the one body in place; its only output is the body's (this file's output
//! for it when there is one, with its part number, material and color, else one named after the
//! body's feature). Regions and publishes stay in this file, since they may name faces of the
//! other bodies. The assembly (`<part>_assembly.assembly`) uses each new part once, fixed at its
//! origin, so the bodies keep their places. This file is left as it is. A name that is taken is
//! refused, never overwritten.

use super::super::{model, EditRequest};
use super::{all_bodies, body_name, new_output, part_stem};
use crate::features::picks_of;
use crate::picks::Picks;
use crate::runtime::Files;
use crate::xref_repairs::FileEdit;
use runmat_cad::assembly::model::{AssemblyDocument, DefinitionRef, Occurrence};
use runmat_cad::model::{Feature, FeatureKind, PartDocument};
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use runmat_cad::syntax::print_assembly;
use runmat_design_syntax::edit::TextEdit;
use runmat_types::Span;
use serde_json::{json, Value};

fn created(path: String, text: String) -> FileEdit {
    let change = SourceChange {
        edits: vec![TextEdit::replace(Span { start: 0, end: 0 }, text.clone())],
        source: text,
        reprinted: true,
    };
    (path, String::new(), change)
}

pub(crate) fn split(
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
    document: &PartDocument,
) -> Result<(Vec<FileEdit>, String), EditFailure> {
    let picks = Picks::new(document, path, text);
    let bodies = all_bodies(&picks)?;
    if bodies.is_empty() {
        return Err(model("the part has no bodies to split yet"));
    }
    let mut chosen = picks_of(&request.values, "bodies");
    if chosen.is_empty() {
        chosen = request
            .selection
            .iter()
            .filter(|s| s.starts_with("body:"))
            .cloned()
            .collect();
    }
    let indices: Vec<usize> = if chosen.is_empty() {
        (0..bodies.len()).collect()
    } else {
        chosen
            .iter()
            .map(|id| {
                let reference = serde_json::from_value(picks.reference(id)?)
                    .map_err(|e| model(format!("`{id}`: {e}")))?;
                bodies
                    .iter()
                    .position(|(b, _)| *b == reference)
                    .ok_or_else(|| model(format!("`{id}` isn't one of the part's bodies")))
            })
            .collect::<Result<_, _>>()?
    };
    let stem = part_stem(path);
    let folder = &path[..path.rfind('/').map_or(0, |i| i + 1)];
    let taken = |p: &str| files.contains_key(p) || p == path;
    let mut out: Vec<FileEdit> = Vec::new();
    let mut parts: Vec<(String, String, String)> = Vec::new();
    for &at in &indices {
        let (body, written) = &bodies[at];
        let mut output = document
            .parts
            .iter()
            .find(|p| &p.body == body)
            .cloned()
            .unwrap_or_else(|| new_output(&body_name(written, at), body.clone()));
        let file = format!("{stem}_{}.part", output.name);
        let new_path = format!("{folder}{file}");
        if taken(&new_path) || parts.iter().any(|(f, _, _)| *f == file) {
            return Err(model(format!(
                "`{file}` already exists; rename it or the body first"
            )));
        }
        // The bodies' own material lines follow them; the others' stay behind.
        let others: Vec<Value> = bodies
            .iter()
            .filter(|(b, _)| b != body)
            .map(|(b, _)| serde_json::to_value(b))
            .collect::<Result<_, _>>()
            .map_err(|e| model(e.to_string()))?;
        let keep: Option<FeatureKind> = if others.is_empty() {
            None
        } else {
            Some(
                serde_json::from_value(
                    json!({ "feature": "body_op", "op": "delete", "bodies": others }),
                )
                .map_err(|e| model(format!("the split: {e}")))?,
            )
        };
        if output.material.is_none() {
            output.material = document
                .materials
                .iter()
                .find(|m| &m.body == body)
                .map(|m| m.material.clone());
        }
        // The part is named after its file.
        let renamed = renamed_declaration(text, &stem, &file[..file.len() - ".part".len()])?;
        let change = edit_part(&renamed, &new_path, |copy| {
            copy.parts = vec![output.clone()];
            copy.materials.clear();
            copy.regions.clear();
            copy.publishes.clear();
            if let Some(kind) = &keep {
                let id = (1..)
                    .map(|n| {
                        if n == 1 {
                            "split".to_owned()
                        } else {
                            format!("split{n}")
                        }
                    })
                    .find(|id| !copy.features.iter().any(|f| f.id.0 == *id))
                    .expect("an unused name");
                copy.features.push(Feature::new(&id, kind.clone()));
            }
            Ok(())
        })?;
        out.push(created(new_path.clone(), change.source));
        parts.push((file, new_path, output.name.0.clone()));
    }
    let wants_assembly = request
        .values
        .get("assembly")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let mut summary = format!(
        "Split into {}",
        parts
            .iter()
            .map(|(f, _, _)| f.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if wants_assembly {
        let file = format!("{stem}_assembly.assembly");
        let assembly_path = format!("{folder}{file}");
        if taken(&assembly_path) {
            return Err(model(format!("`{file}` already exists; rename it first")));
        }
        let mut assembly = AssemblyDocument::new(&assembly_path, &format!("{stem}_assembly"));
        // The definition's identity is its workspace path; the text names it by symbol.
        for (_, part_path, name) in &parts {
            let mut occurrence = Occurrence::new(name, DefinitionRef::part(part_path, None));
            occurrence.fixed = true;
            assembly.occurrences.push(occurrence);
        }
        let printed = print_assembly(&assembly, &[]).map_err(EditFailure::Print)?;
        out.push(created(assembly_path, printed));
        summary.push_str(&format!(" and {file}"));
    }
    Ok((out, summary))
}

/// `text` with its `part <stem>` declaration named `name`.
fn renamed_declaration(text: &str, stem: &str, name: &str) -> Result<String, EditFailure> {
    let head = format!("part {stem}");
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        if let Some(rest) = line.strip_prefix(&head) {
            if rest.starts_with(|c: char| c.is_whitespace() || c == '{') {
                return Ok(format!(
                    "{}part {name}{}",
                    &text[..at],
                    &text[at + head.len()..]
                ));
            }
        }
        at += line.len();
    }
    Err(model(format!("the file has no `part {stem}` declaration")))
}
