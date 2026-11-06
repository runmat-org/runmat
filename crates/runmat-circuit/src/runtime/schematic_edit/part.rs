//! Parts from the canvas: place one, and paste copies.
//!
//! - **Place part** (`schematic.place-part`): adds `component <designator> = { part = "<mpn>" }`
//!   to the sheet, with the value when one is given, and records the symbol's place, quarter
//!   turn and mirror in the lock. A part record (a library search hit) is embedded in the file
//!   first, as `library.place` does. The designator is the next free one of the part's prefix
//!   unless the request names a free one.
//! - **Paste** (`schematic.paste`): copies components (their whole statements, so part, value
//!   and every other property come along) under the next free designators of their prefixes,
//!   each placed at its original's place plus the offset, with its turn and mirror. The copies
//!   join no net: a copied part's wiring belongs to the original's nets.
//!
//! Both write the `.circuit` change and the lock change together, and the edited file must
//! compile.

use super::super::edit::{lock_change, EditError, EditOutput, EditRequest};
use super::super::{file_change, load_with, Files, Project};
use crate::ids::SchPlace;
use crate::ir::Endpoint;
use crate::library::catalog::{prefix_for, CatalogPart};
use crate::syntax::board_edit::SourceChange;
use crate::syntax::{parse_circuit, part_decl, words};
use runmat_design_syntax::edit::{apply_edits, TextEdit};
use runmat_design_syntax::json::{quote, INDENT};
use runmat_types::Span;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// The schematic grid recorded places snap to (nm).
const GRID_NM: i64 = 1_270_000;

fn snap(v: i64) -> i64 {
    (v as f64 / GRID_NM as f64).round() as i64 * GRID_NM
}

fn mm_nm(v: f64) -> i64 {
    (v * 1.0e6).round() as i64
}

/// Every designator the design uses (all sheets, the file's and the ones it brings in).
fn used(project: &Project) -> BTreeSet<String> {
    project.design.sheets.values().flat_map(|s| s.components.iter().map(|c| c.name.clone())).collect()
}

fn next_free(used: &BTreeSet<String>, prefix: &str) -> String {
    (1..).map(|n| format!("{prefix}{n}")).find(|d| !used.contains(d)).expect("a free designator")
}

/// A designator's prefix: its leading letters (`R12` → `R`, `U?` → `U`).
fn prefix_of(designator: &str) -> String {
    designator.chars().take_while(|c| !c.is_ascii_digit() && *c != '?').collect()
}

/// A clicked page point: `point:x,y` in page millimetres, y up (as the canvas names it).
fn point(v: Option<&Value>) -> Option<(f64, f64)> {
    let s = match v? {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().find_map(|v| v.as_str().map(str::to_owned))?,
        _ => return None,
    };
    let (x, y) = s.strip_prefix("point:")?.split_once(',')?;
    Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
}

/// Where a sheet's components end, and its braces.
struct SheetText {
    open: usize,
    close: usize,
    component_ends: Vec<usize>,
    /// Each component statement by name: its span.
    components: Vec<(String, usize, usize)>,
    /// Just after the last `part` declaration, else the first sheet's line start.
    part_anchor: (usize, bool),
}

fn sheet_text(project: &Project, name: &str) -> Result<SheetText, EditError> {
    let text = &project.circuit_text;
    let blocks = super::blocks(text).ok_or_else(|| EditError::new(format!("{} has no `circuit` block", project.circuit_path)))?;
    let first_sheet = blocks.sheets.first().map(|s| super::line_start(text, s.open));
    let block = blocks
        .sheets
        .into_iter()
        .find(|s| s.name == name)
        .ok_or_else(|| EditError::new(format!("{} has no sheet `{name}`", project.circuit_path)))?;
    let components: Vec<(String, usize, usize)> = block
        .statements
        .iter()
        .filter(|s| s.word == words::COMPONENT)
        .map(|s| {
            let name = text[s.start..s.end].split_whitespace().nth(1).unwrap_or_default().to_owned();
            (name, s.start, s.end)
        })
        .collect();
    // A new part declaration goes after the file's last one, else before the first sheet.
    let last_part = parse_last_part(text);
    let part_anchor = match (last_part, first_sheet) {
        (Some(end), _) => (end, false),
        (None, Some(start)) => (start, true),
        (None, None) => (block.open, false),
    };
    Ok(SheetText { open: block.open, close: block.close, component_ends: components.iter().map(|c| c.2).collect(), components, part_anchor })
}

fn parse_last_part(text: &str) -> Option<usize> {
    use runmat_design_syntax::ast::{Headed, Item, Statement};
    use runmat_design_syntax::{parse_source, Dialect};
    let syntax = parse_source(Dialect::Circuit, text);
    for item in syntax.file().items() {
        let Item::Declaration(d) = item else { continue };
        if d.keyword() != words::CIRCUIT {
            continue;
        }
        let mut last = None;
        for s in d.block()?.statements() {
            if let Statement::Value(v) = &s {
                if v.keyword() == words::PART {
                    last = Some(s.span().end);
                }
            }
        }
        return last;
    }
    None
}

/// Inserts component `line` after the sheet's last component (or at the top of the sheet).
fn component_edit(text: &str, sheet: &SheetText, line: &str) -> TextEdit {
    match sheet.component_ends.last() {
        Some(&end) => TextEdit::replace(Span { start: end, end }, format!("\n{}{line}", super::indent_at(text, end))),
        None => {
            let outer = super::indent_at(text, sheet.close);
            let indent = format!("{outer}{INDENT}");
            if text[sheet.open..sheet.close].trim().is_empty() {
                TextEdit::replace(Span { start: sheet.open, end: sheet.close }, format!("\n{indent}{line}\n{outer}"))
            } else {
                TextEdit::replace(Span { start: sheet.open, end: sheet.open }, format!("\n{indent}{line}"))
            }
        }
    }
}

/// The sheet a request targets: `sheet`, a `sheet:` selection, or the root.
fn target_sheet(project: &Project, request: &EditRequest) -> String {
    request
        .values
        .get("sheet")
        .and_then(|s| s.as_str())
        .map(str::to_owned)
        .or_else(|| request.selection.iter().find_map(|s| s.strip_prefix("sheet:").map(|p| p.rsplit('/').next().unwrap_or(p).to_owned())))
        .filter(|s| !s.is_empty() && !s.starts_with('/'))
        .unwrap_or_else(|| project.own.root.clone())
}

/// Applies `edits`, compiles the result, records `places` (keys by designator, set on every
/// unit of that designator) in the lock, and returns both changes.
fn finish(
    project: &Project,
    files: &Files,
    request: &EditRequest,
    edits: Vec<TextEdit>,
    places: Vec<(String, String, Placed)>,
    summary: String,
) -> Result<EditOutput, EditError> {
    let path = &project.circuit_path;
    let source = apply_edits(&project.circuit_text, &edits).map_err(|e| EditError::new(format!("{e:?}")))?;
    let read = parse_circuit(path, &source);
    if let Some(d) = read.diagnostics.first() {
        return Err(EditError::new(format!("the edit would not read back: {}", d.message)));
    }
    let next = load_with(path, &source, files, &request.options, false).map_err(|diagnostics| {
        let first = diagnostics.first().map(|d| d.message.clone()).unwrap_or_default();
        EditError { message: format!("the edit wasn't made: {first}"), diagnostics }
    })?;
    let drawn = crate::schematic::layout(&next.netlist);
    let mut recorded = crate::schematic::places(&drawn);
    for (sheet, designator, placed) in &places {
        // A part's units are `<sheet>/<designator>` or `<sheet>/<designator>:<unit>`.
        let own = format!("{sheet}/{designator}");
        let page = drawn.sheets.iter().find(|p| p.sheet == *sheet).map(|p| p.height_mm).unwrap_or(0.0);
        let keys: Vec<String> = recorded.keys().filter(|k| **k == own || k.starts_with(&format!("{own}:"))).cloned().collect();
        if keys.is_empty() {
            return Err(EditError::new(format!("{designator} isn't drawn on sheet {sheet}")));
        }
        for (k, key) in keys.iter().enumerate() {
            let p = recorded.get_mut(key).expect("listed");
            let (x, y) = match placed.at {
                At::PageMm(x, y) => (mm_nm(x), mm_nm(page - y)),
                At::Nm(x, y) => (x, y),
            };
            // A split part's units go side by side from the point, 25.4 mm apart.
            *p = SchPlace { x: snap(x + k as i64 * 20 * GRID_NM), y: snap(y), rotation: placed.rotation, mirror: placed.mirror, page: p.page };
        }
    }
    let mut lock = next.lock.clone();
    lock.places = recorded;
    let mut files_out = vec![file_change(path, &project.circuit_text, SourceChange { edits, source, reprinted: false })];
    files_out.extend(lock_change(&next, &lock, true));
    Ok(EditOutput { summary, files: files_out })
}

/// Where a placed symbol goes.
#[derive(Clone, Copy)]
struct Placed {
    at: At,
    rotation: u16,
    mirror: bool,
}

#[derive(Clone, Copy)]
enum At {
    /// A canvas point: page millimetres, y up.
    PageMm(f64, f64),
    /// A lock place: paper nanometres from the top left, y down.
    Nm(i64, i64),
}

fn quarter_turn(request: &EditRequest) -> u16 {
    let r = request.values.get("rotation").and_then(Value::as_f64).unwrap_or(0.0);
    ((r / 90.0).round() as i64).rem_euclid(4) as u16 * 90
}

fn flag(request: &EditRequest, id: &str) -> bool {
    match request.values.get(id) {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s == "true",
        _ => false,
    }
}

/// `schematic.place-part` (see the module docs).
pub(super) fn place_part(project: &Project, files: &Files, request: &EditRequest) -> Result<EditOutput, EditError> {
    let v = &request.values;
    let text = &project.circuit_text;
    let sheet_name = target_sheet(project, request);
    let sheet = sheet_text(project, &sheet_name)?;
    let mut edits = Vec::new();
    // The part: a record to embed, or an MPN the file or the pool knows.
    let (mpn, prefix) = match v.get("part") {
        Some(record @ Value::Object(_)) => {
            let part: CatalogPart = serde_json::from_value(record.clone()).map_err(|e| EditError::new(format!("the part record doesn't read: {e}")))?;
            part.check().map_err(EditError::new)?;
            match project.own.parts.get(&part.mpn) {
                Some(d) if d != &part => {
                    return Err(EditError::new(format!("{} declares a different part `{}`; edit that declaration", project.circuit_path, part.mpn)))
                }
                Some(_) => {}
                None if project.pool.part(&part.mpn).is_some() => {}
                None => {
                    let statement = part_decl::statement(&part).map_err(|e| EditError::new(e.0))?;
                    let (at, before_sheet) = sheet.part_anchor;
                    let indent = super::indent_at(text, at);
                    let body = statement.replace('\n', &format!("\n{indent}"));
                    let piece = if before_sheet { format!("{indent}{body}\n\n") } else { format!("\n{indent}{body}") };
                    edits.push(TextEdit::replace(Span { start: at, end: at }, piece));
                }
            }
            let prefix = if part.prefix.is_empty() { prefix_for(&part.category).to_string() } else { part.prefix.clone() };
            (part.mpn, prefix)
        }
        Some(Value::String(mpn)) if !mpn.trim().is_empty() => {
            let mpn = mpn.trim().to_owned();
            let prefix = if let Some(own) = project.own.parts.get(&mpn) {
                if own.prefix.is_empty() { prefix_for(&own.category).to_string() } else { own.prefix.clone() }
            } else if let Some(p) = project.pool.part(&mpn) {
                project.pool.entities.get(&p.entity).map(|e| e.prefix.clone()).unwrap_or_default()
            } else {
                return Err(EditError::new(format!("no part `{mpn}` in the file or the library; search the library to place a new part")));
            };
            (mpn, prefix)
        }
        _ => return Err(EditError::new("choose a part to place")),
    };
    let used = used(project);
    let asked = v.get("designator").and_then(Value::as_str).map(str::trim).unwrap_or_default();
    let designator = if asked.is_empty() || asked.ends_with('?') {
        // `R?` asks for the next R; a bare `?` or nothing takes the part's prefix.
        let p = prefix_of(asked);
        next_free(&used, if p.is_empty() { &prefix } else { &p })
    } else if used.contains(asked) {
        return Err(EditError::new(format!("{asked} is already in the design")));
    } else {
        asked.to_owned()
    };
    let mut fields = vec![format!("part = {}", quote(&mpn))];
    if let Some(value) = v.get("value").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()) {
        fields.push(format!("value = {value}"));
    }
    let line = format!("{} {designator} = {{ {} }}", words::COMPONENT, fields.join(", "));
    edits.push(component_edit(text, &sheet, &line));
    let placed = point(v.get("at")).map(|(x, y)| Placed { at: At::PageMm(x, y), rotation: quarter_turn(request), mirror: flag(request, "mirror") });
    let places = placed.map(|p| vec![(sheet_name.clone(), designator.clone(), p)]).unwrap_or_default();
    let mut out = finish(project, files, request, edits, places, format!("placed {designator} ({mpn}) on sheet {sheet_name}"))?;
    if placed.is_none() {
        // With no point, the layout finds the part a free place; the lock records it there.
        out.summary.push_str(" where the layout found room");
    }
    Ok(out)
}

/// `schematic.text`: a note on the sheet (`text note1 = "…"` after the sheet's last
/// statement), at the clicked point or, with none, above the drawing.
pub(super) fn place_text(project: &Project, files: &Files, request: &EditRequest) -> Result<EditOutput, EditError> {
    let v = &request.values;
    let words_in = v.get("text").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).ok_or_else(|| EditError::new("type the text to place"))?;
    let text = &project.circuit_text;
    let sheet_name = target_sheet(project, request);
    let sheet = project.own.sheets.get(&sheet_name).ok_or_else(|| EditError::new(format!("sheet `{sheet_name}` comes from another file; open that file to edit it")))?;
    let used: BTreeSet<&str> = sheet.texts.iter().map(|t| t.name.as_str()).collect();
    let name = (1..).map(|n| format!("note{n}")).find(|n| !used.contains(n.as_str())).expect("a free name");
    let block = super::blocks(text).and_then(|b| b.sheets.into_iter().find(|s| s.name == sheet_name)).ok_or_else(|| EditError::new(format!("{} has no sheet `{sheet_name}`", project.circuit_path)))?;
    let line = format!("{} {name} = {}", words::TEXT, quote(words_in));
    let edit = match block.statements.last() {
        Some(last) => TextEdit::replace(Span { start: last.end, end: last.end }, format!("\n{}{line}", super::indent_at(text, last.end))),
        None => {
            let outer = super::indent_at(text, block.close);
            TextEdit::replace(Span { start: block.open, end: block.close }, format!("\n{outer}{INDENT}{line}\n{outer}"))
        }
    };
    let places = point(v.get("at"))
        .map(|(x, y)| vec![(sheet_name.clone(), format!("text:{name}"), Placed { at: At::PageMm(x, y), rotation: 0, mirror: false })])
        .unwrap_or_default();
    finish(project, files, request, vec![edit], places, format!("placed {name} on sheet {sheet_name}"))
}

/// `schematic.paste` (see the module docs).
pub(super) fn paste(project: &Project, files: &Files, request: &EditRequest) -> Result<EditOutput, EditError> {
    let v = &request.values;
    let text = &project.circuit_text;
    let mut ids: Vec<String> = match v.get("items") {
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect(),
        Some(Value::String(s)) => s.split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect(),
        _ => Vec::new(),
    };
    if ids.is_empty() {
        ids = request.selection.clone();
    }
    let designators: Vec<String> = ids.iter().filter_map(|id| id.strip_prefix("component:").map(str::to_owned)).collect();
    if designators.is_empty() {
        return Err(EditError::new("copy one or more parts first"));
    }
    let num = |k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(0.0);
    // y up on the canvas; the lock counts y down.
    let (dx, dy) = (mm_nm(num("dx")), -mm_nm(num("dy")));
    let drawn = crate::schematic::layout(&project.netlist);
    let recorded = crate::schematic::places(&drawn);
    let mut used = used(project);
    let mut by_sheet: Vec<(String, SheetText, Vec<String>)> = Vec::new();
    let mut places = Vec::new();
    let mut names = Vec::new();
    let mut copies: BTreeMap<usize, String> = BTreeMap::new();
    for d in &designators {
        let (_, c) = project.netlist.component(d).ok_or_else(|| EditError::new(format!("{d} isn't in the design")))?;
        let sheet = c.sheet.clone();
        let st = sheet_text(project, &sheet)?;
        let (_, start, end) = st
            .components
            .iter()
            .find(|(n, _, _)| n == c.path.leaf())
            .cloned()
            .ok_or_else(|| EditError::new(format!("{d} comes from another file; open that file to copy it")))?;
        let statement = &text[start..end];
        let new = next_free(&used, &prefix_of(c.path.leaf()));
        used.insert(new.clone());
        if let Some((ci, _)) = project.netlist.component(d) {
            copies.insert(ci, new.clone());
        }
        // `component R1 = { … }` → `component R3 = { … }`: the name is the second word.
        let rest = statement.trim_start().strip_prefix(words::COMPONENT).unwrap_or(statement).trim_start();
        let after = rest.find(|ch: char| ch.is_whitespace() || ch == '=').unwrap_or(rest.len());
        let line = format!("{} {new}{}", words::COMPONENT, &rest[after..]);
        let key = format!("{sheet}/{}", c.path.leaf());
        let from = recorded.get(&key).or_else(|| recorded.iter().find(|(k, _)| k.starts_with(&format!("{key}:"))).map(|(_, p)| p));
        if let Some(p) = from {
            places.push((sheet.clone(), new.clone(), Placed { at: At::Nm(p.x + dx, p.y + dy), rotation: p.rotation, mirror: p.mirror }));
        }
        names.push(format!("{d} as {new}"));
        match by_sheet.iter_mut().find(|(s, _, _)| *s == sheet) {
            Some((_, _, lines)) => lines.push(line),
            None => by_sheet.push((sheet, st, vec![line])),
        }
    }
    let wires = group_wires(project, &copies)?;
    let mut kept = 0;
    let mut edits = Vec::new();
    for (sheet, st, lines) in &mut by_sheet {
        for (s, line) in &wires {
            if s == sheet {
                lines.push(line.clone());
                kept += 1;
            }
        }
        let indent = st.component_ends.last().map(|&e| super::indent_at(text, e)).unwrap_or_default();
        edits.push(component_edit(text, st, &lines.join(&format!("\n{indent}"))));
    }
    let mut summary = format!("pasted {}", names.join(", "));
    if kept > 0 {
        summary += &format!(" with {kept} connection{}", if kept == 1 { "" } else { "s" });
    }
    finish(project, files, request, edits, places, summary)
}

/// The connections a pasted group keeps, as `(sheet, connect statement)`:
/// - a net that joins two or more copied pins on one sheet joins their copies on a new net of
///   their own, so the group arrives wired as it was;
/// - a global net (a rail such as GND or +3V3) takes the copies of its copied pins too, as a
///   pasted power port keeps its rail;
/// - any other net (one that leaves the group) stays with the originals, and the copy's pin
///   is left open.
fn group_wires(project: &Project, copies: &BTreeMap<usize, String>) -> Result<Vec<(String, String)>, EditError> {
    let nl = &project.netlist;
    let end = |ci: usize, pi: usize, name: &str| {
        let c = &nl.components[ci];
        let p = &c.pins[pi];
        let pin = if c.pins.iter().filter(|q| q.pin == p.pin).count() > 1 { format!("{}.{}", p.gate, p.pin) } else { p.pin.clone() };
        crate::syntax::endpoint(&Endpoint::pin(name, &pin)).map_err(|e| EditError::new(e.0))
    };
    let mut out = Vec::new();
    for net in &nl.nets {
        let mut by_sheet: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
        for &(ci, pi) in &net.pins {
            if copies.contains_key(&ci) {
                by_sheet.entry(nl.components[ci].sheet.as_str()).or_default().push((ci, pi));
            }
        }
        for (sheet, pins) in by_sheet {
            let mut ends = Vec::new();
            if net.global {
                let (ci, pi) = pins[0];
                ends.push(end(ci, pi, nl.components[ci].path.leaf())?);
            } else if pins.len() < 2 {
                continue;
            }
            for &(ci, pi) in &pins {
                ends.push(end(ci, pi, &copies[&ci])?);
            }
            out.push((sheet.to_owned(), format!("connect = [{}]", ends.join(", "))));
        }
    }
    Ok(out)
}
