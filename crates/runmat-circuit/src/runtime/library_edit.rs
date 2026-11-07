//! `library.place`: places a catalog part (a part search hit) on a schematic sheet.
//!
//! The edit embeds the part in the circuit file as a `part "<mpn>" = { … }` declaration, unless
//! the file already declares it or the built-in pool has that MPN, and adds
//! `component <designator> = { part = "<mpn>" }` to the sheet. The designator is the next free
//! one of the part's prefix (`U7`) unless the request names one. Both are the smallest text
//! change: the declaration goes before the first sheet, the component after the sheet's last
//! component (or at the top of an empty sheet).

use super::edit::{EditError, EditOutput, EditRequest};
use super::{file_change, Project};
use crate::library::catalog::{prefix_for, CatalogPart};
use crate::syntax::board_edit::SourceChange;
use crate::syntax::{parse_circuit, part_decl, words};
use runmat_design_syntax::ast::{Headed, Item, Statement};
use runmat_design_syntax::edit::{apply_edits, TextEdit};
use runmat_design_syntax::json::{quote, INDENT};
use runmat_design_syntax::{parse_source, Dialect};
use runmat_types::Span;

struct SheetAt {
    name: String,
    /// Just after the block's opening brace.
    open: usize,
    /// The block's closing brace.
    close: usize,
    /// The end of each component statement.
    component_ends: Vec<usize>,
}

struct CircuitAt {
    /// Just after the `circuit` block's opening brace.
    open: usize,
    /// The first sheet statement's start.
    first_sheet: Option<usize>,
    /// The end of the last `part` declaration.
    last_part: Option<usize>,
    sheets: Vec<SheetAt>,
}

fn circuit_at(text: &str) -> Option<CircuitAt> {
    let syntax = parse_source(Dialect::Circuit, text);
    for item in syntax.file().items() {
        let Item::Declaration(d) = item else { continue };
        if d.keyword() != words::CIRCUIT {
            continue;
        }
        let block = d.block()?;
        let open = text[block.span().start..].find('{').map(|i| block.span().start + i + 1)?;
        let mut out = CircuitAt { open, first_sheet: None, last_part: None, sheets: vec![] };
        for statement in block.statements() {
            match &statement {
                Statement::Block(sheet) if sheet.keyword() == words::SHEET => {
                    out.first_sheet.get_or_insert(statement.span().start);
                    let Some(body) = sheet.block() else { continue };
                    let open = text[body.span().start..].find('{').map(|i| body.span().start + i + 1).unwrap_or(body.span().start);
                    let component_ends = body
                        .statements()
                        .iter()
                        .filter_map(|s| match s {
                            Statement::Value(v) if v.keyword() == words::COMPONENT => Some(s.span().end),
                            _ => None,
                        })
                        .collect();
                    out.sheets.push(SheetAt {
                        name: sheet.name().map(|n| n.text()).unwrap_or_default(),
                        open,
                        close: body.span().end.saturating_sub(1),
                        component_ends,
                    });
                }
                Statement::Value(v) if v.keyword() == words::PART => out.last_part = Some(statement.span().end),
                _ => {}
            }
        }
        return Some(out);
    }
    None
}

fn line_start(text: &str, at: usize) -> usize {
    text[..at].rfind('\n').map_or(0, |i| i + 1)
}

fn indent_at(text: &str, at: usize) -> String {
    let start = line_start(text, at);
    text[start..].chars().take_while(|c| *c == ' ' || *c == '\t').collect()
}

/// The next free designator `<prefix><n>` across the whole design.
fn next_designator(project: &Project, prefix: &str) -> String {
    let used: std::collections::BTreeSet<String> = project
        .design
        .sheets
        .values()
        .flat_map(|s| s.components.iter().map(|c| c.name.clone()))
        .collect();
    (1..).map(|n| format!("{prefix}{n}")).find(|d| !used.contains(d)).expect("a free designator")
}

pub(super) fn place(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let v = &request.values;
    let part: CatalogPart = serde_json::from_value(v.get("part").cloned().ok_or_else(|| EditError::new("choose a part to place"))?)
        .map_err(|e| EditError::new(format!("the part record doesn't read: {e}")))?;
    part.check().map_err(EditError::new)?;
    let text = &project.circuit_text;
    let at = circuit_at(text).ok_or_else(|| EditError::new(format!("{} has no `circuit` block", project.circuit_path)))?;
    let wanted = super::schematic_edit::target_sheet(project, request)?;
    let sheet = at
        .sheets
        .iter()
        .find(|s| s.name == wanted)
        .ok_or_else(|| EditError::new(format!("{} has no sheet `{wanted}`", project.circuit_path)))?;
    let prefix = match project.pool.part(&part.mpn).and_then(|p| project.pool.entities.get(&p.entity)) {
        Some(e) if !project.own.parts.contains_key(&part.mpn) && part.prefix.is_empty() => e.prefix.clone(),
        _ if !part.prefix.is_empty() => part.prefix.clone(),
        _ => prefix_for(&part.category).to_string(),
    };
    let designator = match v.get("designator").and_then(|d| d.as_str()).map(str::trim) {
        Some(d) if !d.is_empty() && !d.ends_with('?') => d.to_string(),
        _ => next_designator(project, &prefix),
    };
    if project.design.sheets.values().any(|s| s.components.iter().any(|c| c.name == designator)) {
        return Err(EditError::new(format!("{designator} is already in the design")));
    }
    let mut edits = Vec::new();
    // Embed the record unless the file declares that MPN or the built-in pool has it.
    let declared = project.own.parts.get(&part.mpn);
    let built_in = declared.is_none() && project.pool.part(&part.mpn).is_some();
    if declared.is_some_and(|d| d != &part) {
        return Err(EditError::new(format!("{} declares a different part `{}`; edit that declaration", project.circuit_path, part.mpn)));
    }
    if declared.is_none() && !built_in {
        let statement = part_decl::statement(&part).map_err(|e| EditError::new(e.0))?;
        let (offset, indent, before_sheet) = match (at.last_part, at.first_sheet) {
            (Some(end), _) => (end, indent_at(text, end), false),
            (None, Some(start)) => (line_start(text, start), indent_at(text, start), true),
            (None, None) => (at.open, indent_at(text, at.open) + INDENT, false),
        };
        let body = statement.replace('\n', &format!("\n{indent}"));
        let piece = if before_sheet { format!("{indent}{body}\n\n") } else { format!("\n{indent}{body}") };
        edits.push(TextEdit::replace(Span { start: offset, end: offset }, piece));
    }
    let line = format!("{} {} = {{ part = {} }}", words::COMPONENT, designator, quote(&part.mpn));
    let edit = match sheet.component_ends.last() {
        Some(&end) => TextEdit::replace(Span { start: end, end }, format!("\n{}{line}", indent_at(text, end))),
        None => {
            let indent = indent_at(text, sheet.close) + INDENT;
            if text[sheet.open..sheet.close].trim().is_empty() {
                TextEdit::replace(Span { start: sheet.open, end: sheet.close }, format!("\n{indent}{line}\n{}", indent_at(text, sheet.close)))
            } else {
                TextEdit::replace(Span { start: sheet.open, end: sheet.open }, format!("\n{indent}{line}"))
            }
        }
    };
    edits.push(edit);
    let source = apply_edits(text, &edits).map_err(|e| EditError::new(format!("{e:?}")))?;
    let read = parse_circuit(&project.circuit_path, &source);
    if let Some(d) = read.diagnostics.first() {
        return Err(EditError::new(format!("placing {designator} would not read back: {}", d.message)));
    }
    Ok(EditOutput {
        summary: format!("placed {designator} ({}) on sheet {}", part.mpn, sheet.name),
        files: vec![file_change(&project.circuit_path, text, SourceChange { edits, source, reprinted: false })],
    })
}
