//! `schematic.annotate`: records every component's designator in the circuit's lock, so the
//! numbering stays put as the design changes. With `reset`, the designators the compiler
//! numbers (parts in repeated channels and `R?`-style names) are renumbered per prefix in
//! sheet order: across then down (rows top to bottom, left to right in a row) or down then
//! across. Designators written in the source keep their names.

use super::edit::{lock_change, EditError, EditOutput, EditRequest};
use super::*;

pub(super) fn run(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let values = &request.values;
    let reset = values.get("reset").and_then(Value::as_bool) == Some(true);
    let down = values.get("order").and_then(Value::as_str) == Some("down");
    let mut lock = project.lock.clone();
    let mut renumbered = 0usize;
    if reset {
        let places = places(&project.netlist);
        // Numbered designators by prefix, and the components that hold them in sheet order.
        let mut groups: BTreeMap<String, Vec<(PlaceKey, usize)>> = BTreeMap::new();
        for (i, c) in project.netlist.components.iter().enumerate() {
            if !numbered(c) {
                continue;
            }
            let key = places
                .get(c.path.to_string().as_str())
                .map(|(sheet, x, y)| place_key(sheet, *x, *y, down))
                .unwrap_or_else(|| (String::new(), i64::MAX, i64::MAX));
            groups
                .entry(prefix(&c.designator))
                .or_default()
                .push((key, i));
        }
        for (_, mut comps) in groups {
            let mut names: Vec<String> = comps
                .iter()
                .map(|(_, i)| project.netlist.components[*i].designator.clone())
                .collect();
            names.sort_by_key(|d| number(d));
            comps.sort();
            for ((_, i), name) in comps.iter().zip(names) {
                let c = &project.netlist.components[*i];
                if c.designator != name {
                    renumbered += 1;
                }
                if let Some(e) = lock.entries.get_mut(&c.path) {
                    e.designator = Some(name);
                }
            }
        }
    }
    let total = project.netlist.components.len();
    let summary = if reset {
        format!("Annotated {total} component(s); renumbered {renumbered}")
    } else {
        format!("Annotated {total} component(s)")
    };
    Ok(EditOutput {
        summary,
        files: lock_change(project, &lock, true).into_iter().collect(),
    })
}

/// Sort key: sheet, then row and column (or column and row) on a 2.54 mm grid.
type PlaceKey = (String, i64, i64);

fn place_key(sheet: &str, x: f64, y: f64, down: bool) -> PlaceKey {
    let (gx, gy) = ((x / 2.54).round() as i64, (y / 2.54).round() as i64);
    if down {
        (sheet.to_owned(), gx, gy)
    } else {
        (sheet.to_owned(), gy, gx)
    }
}

/// Each component's sheet and position on it (its first unit).
fn places(netlist: &Netlist) -> BTreeMap<String, (String, f64, f64)> {
    let mut out = BTreeMap::new();
    for sheet in crate::schematic::layout(netlist).sheets {
        for s in sheet.symbols {
            out.entry(s.path)
                .or_insert((sheet.path.clone(), s.x_mm, s.y_mm));
        }
    }
    out
}

/// Whether the compiler numbers this component's designator (the source doesn't name it).
fn numbered(c: &crate::elab::CompInst) -> bool {
    let leaf = c.path.leaf().to_string();
    c.channel.is_some()
        || c.path.0.iter().any(|s| s.ends_with(']'))
        || leaf.contains('?')
        || leaf != c.designator
}

fn prefix(designator: &str) -> String {
    designator
        .chars()
        .take_while(|ch| !ch.is_ascii_digit())
        .collect()
}

fn number(designator: &str) -> u64 {
    designator
        .trim_start_matches(|ch: char| !ch.is_ascii_digit())
        .parse()
        .unwrap_or(0)
}
