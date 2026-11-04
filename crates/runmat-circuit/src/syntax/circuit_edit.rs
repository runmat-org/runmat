//! Structured edits on `.circuit` sources.
//!
//! Back-annotation changes a design in place: a re-designated component, a part on another
//! package, swapped pins in a sheet's connections. [`diff_circuit`] writes those changes as
//! the smallest text change: each changed `component` or `connect` statement is replaced in
//! place, and every other statement keeps its text and comments.
//!
//! The result is checked: the edited source must read back as the new design (compared by
//! printed text). If it would not, or the edit changed anything other than components and
//! connections, the whole file is reprinted and [`SourceChange::reprinted`] says so.

use super::board_edit::SourceChange;
use super::{component_statement, connection_statement, parse_circuit, print_circuit};
use crate::ir::Design;
use runmat_design_syntax::edit::{apply_edits, TextEdit};
use runmat_design_syntax::json::INDENT;
use runmat_types::Span;

/// The minimal edits turning `source` (read as `old`) into a source for `new`.
pub fn diff_circuit(
    path: &str,
    source: &str,
    old: &Design,
    new: &Design,
) -> Result<SourceChange, String> {
    let new_text = print_circuit(new).map_err(|e| e.0)?;
    let reprint = || SourceChange {
        edits: vec![TextEdit::replace(
            Span {
                start: 0,
                end: source.len(),
            },
            new_text.clone(),
        )],
        source: new_text.clone(),
        reprinted: true,
    };
    let mut edits = Vec::new();
    let same_shape = old.sheets.len() == new.sheets.len()
        && old.sheets.iter().zip(&new.sheets).all(|((a, x), (b, y))| {
            a == b
                && x.components.len() == y.components.len()
                && x.connections.len() == y.connections.len()
        });
    if !same_shape {
        return Ok(reprint());
    }
    let mut replace = |span: (u32, u32), text: String| {
        let (start, end) = (span.0 as usize, span.1 as usize);
        if start >= end || end > source.len() {
            return false;
        }
        let line_start = source[..start].rfind('\n').map_or(0, |i| i + 1);
        let indent = &source[line_start..start];
        let indent = if indent.trim().is_empty() {
            indent.to_owned()
        } else {
            INDENT.repeat(2)
        };
        edits.push(TextEdit::replace(
            Span { start, end },
            text.replace('\n', &format!("\n{indent}")),
        ));
        true
    };
    for (name, sheet) in &new.sheets {
        let before = &old.sheets[name];
        for (a, b) in before.components.iter().zip(&sheet.components) {
            let (ta, tb) = (
                component_statement(a).map_err(|e| e.0)?,
                component_statement(b).map_err(|e| e.0)?,
            );
            if ta != tb && !replace((a.source.start, a.source.end), tb) {
                return Ok(reprint());
            }
        }
        for (a, b) in before.connections.iter().zip(&sheet.connections) {
            let (ta, tb) = (
                connection_statement(a).map_err(|e| e.0)?,
                connection_statement(b).map_err(|e| e.0)?,
            );
            if ta != tb && !replace((a.source.start, a.source.end), tb) {
                return Ok(reprint());
            }
        }
    }
    let Ok(edited) = apply_edits(source, &edits) else {
        return Ok(reprint());
    };
    let reads_back = parse_circuit(path, &edited)
        .design
        .and_then(|d| print_circuit(&d).ok())
        .is_some_and(|t| t == new_text);
    if !reads_back {
        return Ok(reprint());
    }
    Ok(SourceChange {
        edits,
        source: edited,
        reprinted: false,
    })
}
