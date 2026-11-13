//! Occurrences placed without their model: a definition whose file isn't in the workspace, or a
//! circuit board whose layout doesn't compile to a board. The assembly says so twice: a warning
//! that names the occurrences, what they draw instead and why, and a `problem` on each of their
//! index rows, which the Desktop's Outline marks.

use super::Diagnostic;
use runmat_cad::assembly::model::{AssemblyDocument, DefinitionRef};
use serde_json::Value;
use std::collections::BTreeMap;

/// The warning for definition `def`, placed by `doc`'s occurrences, built without its model for
/// `reason`; `boxed` when the occurrences draw the definition's box, else they draw nothing.
/// For example: "`esc` draws as its box: …/esc.circuit.layout has no board: package `X` is not
/// defined in this layout".
pub(super) fn diagnostic(
    path: &str,
    doc: &AssemblyDocument,
    def: &DefinitionRef,
    reason: &str,
    boxed: bool,
) -> Diagnostic {
    let names: Vec<String> = doc
        .occurrences
        .iter()
        .filter(|o| o.source.definition.file == def.file)
        .map(|o| format!("`{}`", o.name))
        .collect();
    let who = if names.is_empty() {
        format!("`{}`", def.file.0)
    } else {
        names.join(", ")
    };
    let draws = match (boxed, names.len() > 1) {
        (true, false) => "draws as its box",
        (true, true) => "draw as their box",
        (false, false) => "draws nothing",
        (false, true) => "draw nothing",
    };
    Diagnostic {
        path: path.to_owned(),
        start: 0,
        end: 0,
        severity: "warning".into(),
        message: format!("{who} {draws}: {reason}"),
    }
}

/// Puts each unbuilt definition's reason on the index rows of its occurrences (`problem`).
pub(super) fn mark_rows(index: &mut Value, unbuilt: &BTreeMap<String, String>) {
    if unbuilt.is_empty() {
        return;
    }
    let Some(rows) = index["occurrences"].as_array_mut() else {
        return;
    };
    for row in rows {
        let reason = row["definition"]["file"]
            .as_str()
            .and_then(|file| unbuilt.get(file))
            .cloned();
        if let (Some(reason), Some(row)) = (reason, row.as_object_mut()) {
            row.insert("problem".into(), Value::String(reason));
        }
    }
}
