//! `circuit.set-property`: one field of the selected item's Properties, written where the item is
//! defined, as the smallest text change.
//!
//! - A component's designator, value, part, package (Footprint) and fitting in the active
//!   variant are its `component` statement in the `.circuit` ([`diff_circuit`]). A new
//!   designator renames the statement, records the old name in `renamed_from` (so Update board
//!   renames the footprint) and follows into the sheet's connections.
//! - A component's X, Y, rotation, side and lock are its footprint in the `.circuit.layout`:
//!   Move by the difference, Flip, Lock.
//! - A net's class is its `net` statement in the `.circuit`.
//! - A track's width, layer and lock, a via's drill, diameter, span, tenting and lock, and a
//!   pour's net, layer, clearance, pad connection, priority, widths and lock are those items in
//!   the layout ([`board_edit::edit_board`]).
//!
//! The Desktop sends the field's id and its new value (`values = { <field>: <value> }`) with the
//! item's Outline id as the selection. Lengths are millimetres as numbers, or text with a unit
//! (`0.25 mm`, `10 mil`); angles are degrees as numbers or text (`90 deg`).

use crate::layer_set::CopperIndex;
use super::edit::{failure, EditError, EditOutput, EditRequest};
use super::*;
use crate::geom::{Coord, NM_PER_MIL, NM_PER_MM};
use crate::ir::{ComponentDecl, Endpoint, Fitting, PartRef};
use crate::layout::{copper_name, PadConnect};

pub(super) fn run(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let [id] = request.selection.as_slice() else {
        return Err(EditError::new("select one item to set its properties"));
    };
    let (field, value) = one_value(&request.values)?;
    let (kind, name) = id.split_once(':').unwrap_or(("", id.as_str()));
    match (kind, field) {
        ("component", "designator" | "value" | "part" | "footprint" | "fitting") => {
            component_field(project, request, name, field, value)
        }
        ("component", "x" | "y" | "rotation" | "side" | "locked") => {
            footprint_field(project, name, field, value)
        }
        ("net", "class") => net_class(project, name, value),
        ("net", "name") => net_name(project, name, value),
        ("track" | "via" | "zone" | "pour", _) => board_item_field(project, kind, name, field, value),
        ("component" | "net" | "net-class" | "netclass" | "pad" | "rule", _) => Err(EditError::new(
            format!("{field} is shown, not set, on this item"),
        )),
        _ => Err(EditError::new(format!("`{id}` has no properties to set here"))),
    }
}

/// The one field the request sets, and its value.
fn one_value(values: &serde_json::Map<String, Value>) -> Result<(&str, &Value), EditError> {
    let mut fields = values.iter();
    match (fields.next(), fields.next()) {
        (Some((field, value)), None) => Ok((field.as_str(), value)),
        (None, _) => Err(EditError::new("set a value first")),
        _ => Err(EditError::new("set one property at a time")),
    }
}

/* ------------------------------------------------------------------------------------------------
 * Values
 * --------------------------------------------------------------------------------------------- */

fn text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.trim().to_owned()).filter(|s| !s.is_empty()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn required(field: &str, value: &Value) -> Result<String, EditError> {
    text(value).ok_or_else(|| EditError::new(format!("{field} can't be empty")))
}

/// A number with an optional unit: `(0.25, Some("mm"))`.
fn number_and_unit(value: &Value) -> Option<(f64, Option<String>)> {
    match value {
        Value::Number(n) => n.as_f64().map(|v| (v, None)),
        Value::String(s) => {
            let s = s.trim();
            let split = s
                .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'))
                .unwrap_or(s.len());
            let (n, unit) = s.split_at(split);
            let unit = unit.trim();
            n.trim().parse().ok().map(|v| (v, (!unit.is_empty()).then(|| unit.to_owned())))
        }
        _ => None,
    }
}

/// A length in board units (nm): millimetres when no unit is written.
pub(super) fn length(field: &str, value: &Value) -> Result<Coord, EditError> {
    let bad = || EditError::new(format!("{field} takes a length such as 0.25 mm"));
    let (v, unit) = number_and_unit(value).ok_or_else(bad)?;
    let per = match unit.as_deref() {
        None | Some("mm") => NM_PER_MM as f64,
        Some("um" | "µm") => NM_PER_MM as f64 / 1000.0,
        Some("mil" | "mils") => NM_PER_MIL as f64,
        Some("in") => NM_PER_MIL as f64 * 1000.0,
        Some("cm") => NM_PER_MM as f64 * 10.0,
        Some(_) => return Err(bad()),
    };
    Ok((v * per).round() as Coord)
}

/// An angle in degrees: degrees when no unit is written.
pub(super) fn angle(field: &str, value: &Value) -> Result<f64, EditError> {
    let bad = || EditError::new(format!("{field} takes an angle such as 90 deg"));
    let (v, unit) = number_and_unit(value).ok_or_else(bad)?;
    match unit.as_deref() {
        None | Some("deg" | "°") => Ok(v),
        Some("rad") => Ok(v.to_degrees()),
        Some(_) => Err(bad()),
    }
}

fn boolean(field: &str, value: &Value) -> Result<bool, EditError> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "true" || s == "false" => Ok(s == "true"),
        _ => Err(EditError::new(format!("{field} is on or off"))),
    }
}

fn count(field: &str, value: &Value) -> Result<i32, EditError> {
    number_and_unit(value)
        .filter(|(v, unit)| unit.is_none() && v.fract() == 0.0)
        .map(|(v, _)| v as i32)
        .ok_or_else(|| EditError::new(format!("{field} takes a whole number")))
}

/// The component's value as the circuit language reads it (`100 nF ± 10 %`, `4.7 kohm`),
/// through the language's own reader: a one-component circuit holding the text.
fn read_value(text: &str) -> Result<crate::ir::Param, EditError> {
    let source = format!(
        "runmat 0.7\nstd 1\n\ncircuit probe {{\n    root = main\n    sheet main {{\n        component X1 = {{ part = \"P\", value = {text} }}\n    }}\n}}\n"
    );
    let parsed = parse_circuit("/value.circuit", &source);
    parsed
        .design
        .and_then(|d| d.sheets.get("main").and_then(|s| s.components.first()).and_then(|c| c.params.get("value").cloned()))
        .ok_or_else(|| {
            EditError::new(format!(
                "`{text}` isn't a value the circuit reads; write a quantity such as 100 nF ± 10 %"
            ))
        })
}

/* ------------------------------------------------------------------------------------------------
 * The .circuit: component and net statements
 * --------------------------------------------------------------------------------------------- */

/// The file's own sheet and the local name of the component with this designator.
fn own_component(project: &Project, designator: &str) -> Result<(String, String), EditError> {
    let local = project
        .netlist
        .components
        .iter()
        .find(|c| c.designator == designator)
        .map(|c| c.path.leaf().to_string())
        .unwrap_or_else(|| designator.to_owned());
    let in_own: Vec<&String> = project
        .own
        .sheets
        .iter()
        .filter(|(_, s)| s.components.iter().any(|c| c.name == local))
        .map(|(name, _)| name)
        .collect();
    match in_own.as_slice() {
        [sheet] => Ok(((*sheet).clone(), local)),
        [] if project.design.sheets.values().any(|s| s.components.iter().any(|c| c.name == local)) => Err(
            EditError::new(format!("{designator} comes from a used file; set its properties there")),
        ),
        [] => Err(EditError::new(format!("{designator} isn't in {}", project.circuit_path))),
        _ => Err(EditError::new(format!(
            "{designator} is drawn by more than one sheet; set it in the sheet's source"
        ))),
    }
}

fn component_field(
    project: &Project,
    request: &EditRequest,
    designator: &str,
    field: &str,
    value: &Value,
) -> Result<EditOutput, EditError> {
    let (sheet, local) = own_component(project, designator)?;
    let mut own = project.own.clone();
    let def = own.sheets.get_mut(&sheet).expect("found above");
    let at = def.components.iter().position(|c| c.name == local).expect("found above");
    let summary = match field {
        "designator" => {
            let new = required("Designator", value)?;
            if new == designator {
                return Ok(nothing(designator));
            }
            if !new.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
                || !new.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return Err(EditError::new(format!(
                    "`{new}` isn't a designator; use letters then a number, such as C25"
                )));
            }
            if project.netlist.components.iter().any(|c| c.designator == new)
                || def.components.iter().any(|c| c.name == new)
            {
                return Err(EditError::new(format!("{new} is already in the design")));
            }
            rename_component(def, at, &local, &new);
            format!("renamed {designator} to {new}")
        }
        "value" => {
            let written = required("Value", value)?;
            def.components[at].params.insert("value".into(), read_value(&written)?);
            format!("{designator} value {written}")
        }
        "part" => {
            let mpn = required("Part", value)?;
            if project.own.parts.get(&mpn).is_none() && project.pool.part(&mpn).is_none() {
                return Err(EditError::new(format!(
                    "no part `{mpn}` in the file or the library; search the library for it"
                )));
            }
            def.components[at].part = PartRef::Part(mpn.clone());
            format!("{designator} part {mpn}")
        }
        "footprint" => {
            let package = required("Footprint", value)?;
            match &mut def.components[at].part {
                PartRef::Entity { package: p, .. } => *p = Some(package.clone()),
                PartRef::Part(mpn) => {
                    return Err(EditError::new(format!(
                        "{designator}'s footprint comes from its part {mpn}; choose another part to change it"
                    )))
                }
            }
            format!("{designator} footprint {package}")
        }
        "fitting" => {
            let variant = request.options.variant.clone().ok_or_else(|| {
                EditError::new("choose the variant first; fitting is set per variant")
            })?;
            let fitting = match text(value).as_deref() {
                Some("fitted") => Fitting::Fitted,
                Some("not_fitted") => Fitting::NotFitted,
                Some("alternate") => {
                    return Err(EditError::new(format!(
                        "an alternate names its part; set it in the {variant} variant's overrides"
                    )))
                }
                _ => return Err(EditError::new("fitting is fitted, not fitted or alternate")),
            };
            def.components[at].variants.insert(variant.clone(), fitting);
            format!("{designator} {} in {variant}", text(value).unwrap_or_default().replace('_', " "))
        }
        _ => unreachable!("dispatched above"),
    };
    circuit_change(project, &own, summary)
}

/// Renames component `at` of `def` from `old` to `new`, keeping the name the board knows in
/// `renamed_from`, and moves the sheet's connections to the new name.
fn rename_component(def: &mut crate::ir::SheetDef, at: usize, old: &str, new: &str) {
    let component: &mut ComponentDecl = &mut def.components[at];
    component.renamed_from = Some(component.renamed_from.clone().unwrap_or_else(|| old.to_owned()));
    component.name = new.to_owned();
    for connection in &mut def.connections {
        for end in &mut connection.ends {
            if let Endpoint::Pin { component, .. } = end {
                if component == old {
                    *component = new.to_owned();
                }
            }
        }
    }
}

fn net_class(project: &Project, net: &str, value: &Value) -> Result<EditOutput, EditError> {
    let class = text(value);
    if let Some(class) = &class {
        if !project.design.net_classes.contains_key(class) {
            return Err(EditError::new(format!(
                "no net class `{class}`; add it to the circuit's net classes first"
            )));
        }
    }
    let mut own = project.own.clone();
    let decl = own
        .sheets
        .values_mut()
        .flat_map(|s| s.nets.iter_mut())
        .find(|n| n.name == net)
        .ok_or_else(|| {
            EditError::new(format!(
                "{net} has no `net` statement in {}; declare it to give it a class",
                project.circuit_path
            ))
        })?;
    if decl.class == class {
        return Ok(nothing(net));
    }
    decl.class = class.clone();
    circuit_change(
        project,
        &own,
        match class {
            Some(c) => format!("{net} in net class {c}"),
            None => format!("{net} in the default net class"),
        },
    )
}

/// Renames a declared net: its `net` statement (keeping the old name in `renamed_from`, so
/// Update board renames the board's net) and every connection in the file that names it. A
/// global net (a rail) joins by name across the design and its power ports, so it is renamed
/// in its source, not here.
fn net_name(project: &Project, net: &str, value: &Value) -> Result<EditOutput, EditError> {
    let new = required("Name", value)?;
    if new == net {
        return Ok(nothing(net));
    }
    let mut own = project.own.clone();
    let taken = own.sheets.values().flat_map(|s| s.nets.iter()).any(|n| n.name == new);
    let decl = own
        .sheets
        .values_mut()
        .flat_map(|s| s.nets.iter_mut())
        .find(|n| n.name == net)
        .ok_or_else(|| {
            EditError::new(format!(
                "{net} has no `net` statement in {}; declare it to rename it",
                project.circuit_path
            ))
        })?;
    if decl.global {
        return Err(EditError::new(format!(
            "{net} is a global rail, named by its power ports too; rename it in the source"
        )));
    }
    if taken {
        return Err(EditError::new(format!("a net {new} is already in the design")));
    }
    decl.renamed_from = Some(decl.renamed_from.clone().unwrap_or_else(|| net.to_owned()));
    decl.name = new.clone();
    for sheet in own.sheets.values_mut() {
        for connection in &mut sheet.connections {
            for end in &mut connection.ends {
                if matches!(end, Endpoint::Net(n) if n == net) {
                    *end = Endpoint::Net(new.clone());
                }
            }
        }
    }
    circuit_change(project, &own, format!("renamed net {net} to {new}"))
}

/// The `.circuit` rewritten for `own`, the file's own design after the edit.
fn circuit_change(project: &Project, own: &Design, summary: String) -> Result<EditOutput, EditError> {
    let change = diff_circuit(&project.circuit_path, &project.circuit_text, &project.own, own)?;
    Ok(EditOutput {
        summary,
        files: vec![file_change(&project.circuit_path, &project.circuit_text, change)],
    })
}

fn nothing(item: &str) -> EditOutput {
    EditOutput {
        summary: format!("{item} already has that value"),
        files: Vec::new(),
    }
}

/* ------------------------------------------------------------------------------------------------
 * The .circuit.layout: footprints and board items
 * --------------------------------------------------------------------------------------------- */

fn footprint_field(project: &Project, designator: &str, field: &str, value: &Value) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let board = project.board().expect("read");
    let footprint = board
        .footprint_by_ref(designator)
        .ok_or_else(|| EditError::new(format!("{designator} is not on the board; run Update board first")))?;
    let items = vec![designator.to_owned()];
    let edit = match field {
        "x" | "y" => {
            let to = length(if field == "x" { "X" } else { "Y" }, value)?;
            let (dx, dy) = if field == "x" { (to - footprint.at.x, 0) } else { (0, to - footprint.at.y) };
            if dx == 0 && dy == 0 {
                return Ok(nothing(designator));
            }
            BoardEdit::Move { items, dx, dy, rotation: 0.0 }
        }
        "rotation" => {
            let turn = angle("Rotation", value)? - footprint.rotation;
            if turn.rem_euclid(360.0).abs() < 1e-9 {
                return Ok(nothing(designator));
            }
            BoardEdit::Move { items, dx: 0, dy: 0, rotation: turn }
        }
        "side" => {
            let side = match text(value).as_deref() {
                Some("top") => Side::Top,
                Some("bottom") => Side::Bottom,
                _ => return Err(EditError::new("side is top or bottom")),
            };
            if side == footprint.side {
                return Ok(nothing(designator));
            }
            BoardEdit::Flip { items }
        }
        "locked" => {
            let locked = boolean("Locked", value)?;
            if locked == footprint.locked {
                return Ok(nothing(designator));
            }
            BoardEdit::Lock { items, locked }
        }
        _ => unreachable!("dispatched above"),
    };
    let change = board_edit::board_command(&layout.path, &layout.text, &edit).map_err(failure)?;
    Ok(EditOutput {
        summary: format!("{} on {designator}", edit.command()),
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}

/// The copper layer index of a layer name (`F.Cu`, `Top`, `Inner 1`, `B.Cu`).
fn copper_index(board: &Board, name: &str) -> Result<CopperIndex, EditError> {
    let named = crate::drawn::copper_layer(board, name).map_err(EditError::new)?;
    let n = board.copper_count();
    let names = board.stackup.copper_names();
    (0..n)
        .find(|i| copper_name(*i, n) == named || names.get(*i as usize) == Some(&named))
        .ok_or_else(|| EditError::new(format!("`{name}` is not a copper layer of this board")))
}

fn board_item_field(project: &Project, kind: &str, id: &str, field: &str, value: &Value) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let n: u64 = id
        .parse()
        .map_err(|_| EditError::new(format!("`{kind}:{id}` is not a board item")))?;
    let item = ItemId(n);
    let (change, summary) = board_edit::edit_board(&layout.path, &layout.text, |b| set_board_item(b, kind, item, field, value).map_err(|e| e.message))
        .map_err(failure)?;
    Ok(EditOutput {
        summary,
        files: if change.edits.is_empty() { Vec::new() } else { vec![file_change(&layout.path, &layout.text, change)] },
    })
}

fn set_board_item(board: &mut Board, kind: &str, item: ItemId, field: &str, value: &Value) -> Result<String, EditError> {
    let copper = board.copper_count();
    let missing = || EditError::new(format!("{kind} {} is not on the board", item.0));
    match kind {
        "track" => {
            let layer = if field == "layer" { Some(copper_index(board, &required("Layer", value)?)?) } else { None };
            let t = board.tracks.get_mut(&item).ok_or_else(missing)?;
            match field {
                "width" => t.width = positive("Width", length("Width", value)?)?,
                "layer" => t.layer = layer.expect("read above"),
                "locked" => t.locked = boolean("Locked", value)?,
                _ => return Err(EditError::new(format!("{field} is shown, not set, on a track"))),
            }
        }
        "via" => {
            let v = board.vias.get_mut(&item).ok_or_else(missing)?;
            match field {
                "drill" => v.drill = positive("Drill", length("Drill", value)?)?,
                "diameter" => v.diameter = positive("Diameter", length("Diameter", value)?)?,
                "tented" => v.tented = boolean("Tented", value)?,
                "locked" => v.locked = boolean("Locked", value)?,
                "span" => {
                    let last = copper.saturating_sub(1);
                    let span = text(value);
                    if copper < 4 && span.as_deref() != Some("through") {
                        return Err(EditError::new("blind, buried and micro vias need a board with inner layers"));
                    }
                    let (from, to, micro) = match span.as_deref() {
                        Some("through") => (0, last, false),
                        Some("micro") => (0, 1.min(last), true),
                        // A blind via starts on an outer layer and stops inside; a buried via
                        // stays inside. An existing span of that kind is kept.
                        Some("blind") if (v.from == 0) != (v.to == last) && !v.micro => (v.from, v.to, false),
                        Some("blind") => (0, 1.min(last), false),
                        Some("buried") if v.from > 0 && v.to < last => (v.from, v.to, false),
                        Some("buried") => (1, last - 1, false),
                        _ => return Err(EditError::new("span is through, blind, buried or micro")),
                    };
                    (v.from, v.to, v.micro) = (from, to, micro);
                }
                _ => return Err(EditError::new(format!("{field} is shown, not set, on a via"))),
            }
            if v.drill >= v.diameter {
                return Err(EditError::new("the via's drill must be smaller than its diameter"));
            }
        }
        "zone" | "pour" => {
            let net = if field == "net" { Some(text(value).map(|name| board.add_net(&name, None))) } else { None };
            let layer = if field == "layer" { Some(copper_index(board, &required("Layer", value)?)?) } else { None };
            let z = board.zones.get_mut(&item).ok_or_else(missing)?;
            match field {
                "net" => z.net = net.expect("read above"),
                "layer" => z.layers = vec![layer.expect("read above")],
                "clearance" => z.clearance = Some(length("Clearance", value)?),
                "priority" => z.priority = count("Priority", value)?,
                "min_width" => z.min_width = positive("Minimum width", length("Minimum width", value)?)?,
                "thermal_gap" => z.thermal_gap = length("Relief gap", value)?,
                "spoke_width" => z.spoke_width = positive("Spoke width", length("Spoke width", value)?)?,
                "locked" => z.locked = boolean("Locked", value)?,
                "connect" => {
                    z.connect = match text(value).as_deref() {
                        Some("relief") => PadConnect::Thermal,
                        Some("relief_tht") => PadConnect::ThermalThtOnly,
                        Some("solid") => PadConnect::Solid,
                        Some("none") => PadConnect::None,
                        _ => return Err(EditError::new("pad connection is thermal relief, thermal relief through-hole only, solid or none")),
                    }
                }
                _ => return Err(EditError::new(format!("{field} is shown, not set, on a pour"))),
            }
            // The fill follows the pour's new settings at the next compile.
            z.fill.clear();
        }
        _ => return Err(EditError::new(format!("`{kind}` has no properties to set here"))),
    }
    Ok(format!("set {field} on {kind} {}", item.0))
}

fn positive(field: &str, v: Coord) -> Result<Coord, EditError> {
    if v > 0 {
        Ok(v)
    } else {
        Err(EditError::new(format!("{field} must be more than zero")))
    }
}
