//! The right-click commands on a circuit's components and nets, in the schematic and on the
//! board alike:
//!
//! - `circuit.edit-value` (`values.value`): the component's value, written in its `component`
//!   statement; the same handler as Properties' Value field ([`super::set_property`]).
//! - `circuit.rename-net` (`values.name`): the declared net and every connection that names it,
//!   keeping the old name for Update board; the same handler as Properties' Name field.
//! - `circuit.delete`: the selected items.
//!   - A component goes from its sheet in the `.circuit`, with its pins in `connect` and
//!     no-ERC statements (a `connect` left with one end goes too), and its footprint from the
//!     board, so the schematic and the board agree at once. Update board does the rest of the
//!     board's clean-up (copper left on a net that no longer exists).
//!   - Tracks, vias, pours, keepouts, rooms and design rules go from the `.circuit.layout`.

use super::edit::{failure, EditError, EditOutput, EditRequest};
use super::set_property::{circuit_change, own_component};
use super::*;
use crate::ir::Endpoint;
use crate::syntax::board_edit::{self, BoardEdit};

pub(super) const COMMANDS: &[&str] =
    &["circuit.edit-value", "circuit.rename-net", "circuit.delete"];

pub(super) fn run(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    match request.command.as_str() {
        "circuit.edit-value" => as_property(project, request, "component", "value", "Value"),
        "circuit.rename-net" => as_property(project, request, "net", "name", "Name"),
        _ => delete(project, request),
    }
}

/// Runs the command as Properties' `field` on the one selected item of `kind`.
fn as_property(
    project: &Project,
    request: &EditRequest,
    kind: &str,
    field: &str,
    label: &str,
) -> Result<EditOutput, EditError> {
    let noun = if kind == "net" { "net" } else { "component" };
    let [id] = request.selection.as_slice() else {
        return Err(EditError::new(format!("select one {noun}")));
    };
    if !id.starts_with(&format!("{kind}:")) {
        return Err(EditError::new(format!("select one {noun}")));
    }
    let value = request
        .values
        .get(field)
        .filter(|v| v.as_str().is_none_or(|s| !s.trim().is_empty()))
        .cloned()
        .ok_or_else(|| EditError::new(format!("{label} can't be empty")))?;
    let property = EditRequest {
        command: "circuit.set-property".into(),
        values: [(field.to_owned(), value)].into_iter().collect(),
        selection: request.selection.clone(),
        options: request.options.clone(),
    };
    super::set_property::run(project, &property)
}

fn delete(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    if request.selection.is_empty() {
        return Err(EditError::new("select the items to delete"));
    }
    let mut components = Vec::new();
    let mut board_items = Vec::new();
    let mut rooms = Vec::new();
    let mut rules = Vec::new();
    for id in &request.selection {
        let (kind, key) = id.split_once(':').unwrap_or(("", id.as_str()));
        match kind {
            "component" => components.push(key.to_owned()),
            "track" | "via" | "zone" | "pour" | "keepout" => board_items.push(id.clone()),
            "room" => rooms.push(key.to_owned()),
            "rule" => rules.push(key.to_owned()),
            _ => {
                return Err(EditError::new(format!(
                    "`{id}` can't be deleted here; select components, tracks, vias, pours, keepouts, rooms or rules"
                )))
            }
        }
    }
    let mut files = Vec::new();
    let mut said = Vec::new();
    if !components.is_empty() {
        let (change, names) = delete_components(project, &components)?;
        files.extend(change.files);
        said.push(names);
    }
    // The board: the components' footprints, and the board items.
    let footprints: Vec<String> = match &project.layout {
        Some(layout) => {
            let board = layout.parsed.board.as_ref().expect("read");
            components
                .iter()
                .filter(|c| board.footprint_by_ref(c).is_some())
                .cloned()
                .collect()
        }
        None => Vec::new(),
    };
    if !footprints.is_empty() || !board_items.is_empty() || !rooms.is_empty() || !rules.is_empty() {
        let layout = project.layout_or_err()?;
        let board = layout.parsed.board.as_ref().expect("read");
        let mut items = footprints;
        for id in &board_items {
            items.push(super::edit::site_of(board, &layout.parsed.sites, id)?);
        }
        let sites = &layout.parsed.sites;
        let (change, ()) = board_edit::edit_parsed(&layout.path, &layout.text, &layout.parsed, |b| {
            for room in &rooms {
                let before = b.rooms.len();
                b.rooms.retain(|r| &r.name != room);
                if b.rooms.len() == before {
                    return Err(format!("the board has no room `{room}`"));
                }
            }
            for rule in &rules {
                if rule == "limits" {
                    return Err("the board's limits are its fabrication minimums; set them with Board house".into());
                }
                let before = b.rules.len();
                b.rules.retain(|r| &r.id != rule);
                if b.rules.len() == before {
                    return Err(format!("the board has no rule `{rule}`"));
                }
            }
            if !items.is_empty() {
                board_edit::apply_board_edit(b, sites, &BoardEdit::Delete { items: items.clone() })?;
            }
            Ok(())
        })
        .map_err(failure)?;
        if !board_items.is_empty() {
            said.push(plural(board_items.len(), "board item", "board items"));
        }
        if !rooms.is_empty() {
            said.push(format!("room {}", rooms.join(", ")));
        }
        if !rules.is_empty() {
            said.push(format!("rule {}", rules.join(", ")));
        }
        files.push(file_change(&layout.path, &layout.text, change));
    }
    Ok(EditOutput {
        report: None,
        summary: format!("Deleted {}", said.join(" and ")),
        files,
    })
}

fn plural(n: usize, one: &str, many: &str) -> String {
    if n == 1 {
        format!("1 {one}")
    } else {
        format!("{n} {many}")
    }
}

/// The `.circuit` without `designators`: their statements, their pins in the sheet's `connect`
/// and no-ERC statements. A component drawn in a repeated sheet goes from every channel, so it
/// is refused unless every channel's copy is selected.
fn delete_components(
    project: &Project,
    designators: &[String],
) -> Result<(EditOutput, String), EditError> {
    let mut own = project.own.clone();
    for designator in designators {
        let (sheet, local) = own_component(project, designator)?;
        let drawn = project
            .netlist
            .components
            .iter()
            .filter(|c| {
                c.path.leaf() == local.as_str()
                    && super::schematic_edit::sheet_def_of(
                        project,
                        &c.path.parent().unwrap_or_else(crate::ids::NamePath::root),
                    )
                    .is_ok_and(|s| s == sheet)
            })
            .map(|c| c.designator.clone())
            .collect::<Vec<_>>();
        if let Some(other) = drawn.iter().find(|d| !designators.contains(d)) {
            return Err(EditError::new(format!(
                "{designator} is drawn in sheet `{sheet}` with {other} in another channel; delete them together"
            )));
        }
        let def = own.sheets.get_mut(&sheet).expect("found by own_component");
        if !def.components.iter().any(|c| c.name == local) {
            // Another channel's copy of a component already removed.
            continue;
        }
        def.components.retain(|c| c.name != local);
        let ours =
            |e: &Endpoint| matches!(e, Endpoint::Pin { component, .. } if *component == local);
        for connection in &mut def.connections {
            connection.ends.retain(|e| !ours(e));
        }
        def.connections.retain(|c| c.ends.len() >= 2);
        def.no_erc.retain(|e| !ours(e));
    }
    let names = designators.join(", ");
    let change = circuit_change(project, &own, format!("Deleted {names}"))?;
    Ok((change, names))
}
