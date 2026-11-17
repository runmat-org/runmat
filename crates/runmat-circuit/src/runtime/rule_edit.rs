//! Design rules and net classes:
//!
//! - `board.rules`: a `rule` statement in the `.circuit.layout`, added, or replaced when its
//!   name is taken (a Rules row selected edits that rule). `values.kind` picks the property
//!   the value sets; `values.applies` is the selector (`:net-class(power)`, `track:net(USB*)`);
//!   `values.value` is the value as written (`0.2 mm`, `8 mil`, `50 ohm`; a via takes
//!   `drill/diameter`); `values.priority` breaks ties. Clearances are pair rules between the
//!   selected items and everything else.
//! - `circuit.net-class`: the selected nets join a class in the `.circuit` (`default` or empty
//!   takes them out of their class). A class the circuit doesn't declare yet is declared, empty,
//!   so its rules can be written next; a net the sheet doesn't declare gets a `net` statement.

use super::edit::{EditError, EditOutput, EditRequest};
use super::set_property::{circuit_change, length, text};
use super::*;
use crate::board_rules::{props, RuleSpec, RuleValue};
use crate::ir::{Endpoint, NetClass, NetDecl};

use crate::board_rules::RULE_KINDS as KINDS;

fn word<'v>(request: &'v EditRequest, field: &str) -> Option<&'v str> {
    request
        .values
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// The rule `board.rules` writes.
pub(super) fn rule(board: &Board, request: &EditRequest) -> Result<BoardEdit, EditError> {
    let kind = word(request, "kind").unwrap_or("clearance");
    let property = KINDS
        .iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, p)| *p)
        .ok_or_else(|| {
            EditError::new(format!(
                "`{kind}` isn't a rule kind; expected clearance, width, via, diff_pair, length, impedance, plane, mask or silk"
            ))
        })?;
    let selector = word(request, "applies").unwrap_or("*").to_owned();
    let written = word(request, "value")
        .ok_or_else(|| EditError::new("give the rule's value, such as 0.2 mm"))?;
    let editing = request
        .selection
        .iter()
        .find_map(|id| id.strip_prefix("rule:"))
        .map(str::to_owned);
    let id = match word(request, "name").map(str::to_owned).or(editing) {
        Some(id) => {
            if !id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return Err(EditError::new(format!(
                    "`{id}` isn't a rule name; use letters, digits, - and _"
                )));
            }
            id
        }
        None => (1..)
            .map(|n| format!("{kind}_{n}"))
            .find(|id| !board.rules.iter().any(|r| &r.id == id))
            .expect("a free name"),
    };
    let mut spec = if crate::board_rules::is_pair_property(property) {
        RuleSpec::pair(&id, &selector, "*")
    } else {
        RuleSpec::single(&id, &selector)
    };
    spec.priority = match request.values.get("priority") {
        Some(v) => super::edit::number(&request.values, "priority")
            .filter(|p| p.fract() == 0.0)
            .ok_or_else(|| EditError::new(format!("Priority takes a whole number, not {v}")))?
            as i32,
        None => 1,
    };
    match kind {
        "impedance" => {
            let ohms = written
                .trim_end_matches(['Ω', ' '])
                .trim_end_matches("ohms")
                .trim_end_matches("ohm")
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|z| *z > 0.0)
                .ok_or_else(|| EditError::new("Impedance takes ohms, such as 50 ohm"))?;
            spec = spec.set(property, RuleValue::Num(ohms));
        }
        "via" if written.contains('/') => {
            let (drill, diameter) = written.split_once('/').expect("checked");
            let drill = length("Drill", &Value::String(drill.trim().into()))?;
            let diameter = length("Diameter", &Value::String(diameter.trim().into()))?;
            if drill >= diameter {
                return Err(EditError::new("a via's drill is smaller than its diameter"));
            }
            spec = spec
                .set(props::VIA_DRILL, RuleValue::Len(drill))
                .set(props::VIA_DIAMETER, RuleValue::Len(diameter));
        }
        _ => {
            let v = length("Value", &Value::String(written.into()))?;
            if v < 0 && property != props::MASK_EXPANSION {
                return Err(EditError::new("the value can't be negative"));
            }
            spec = spec.set(property, RuleValue::Len(v));
        }
    }
    // The selector checks the way the layout's reader checks it.
    let (_, errors) =
        crate::board_rules::BoardRules::new(&board.limits, std::slice::from_ref(&spec));
    if let Some((_, e)) = errors.first() {
        return Err(EditError::new(format!(
            "`{selector}` isn't a selector: {e}"
        )));
    }
    Ok(BoardEdit::Rule(spec))
}

/// `circuit.net-class` on the selected nets.
pub(super) fn net_class(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let nets: Vec<&str> = request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("net:"))
        .collect();
    if nets.is_empty() || nets.len() != request.selection.len() {
        return Err(EditError::new("select the nets to put in a class"));
    }
    let class = request
        .values
        .get("class")
        .and_then(text)
        .filter(|c: &String| c != "default");
    let mut own = project.own.clone();
    if let Some(c) = &class {
        if !c
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
        {
            return Err(EditError::new(format!(
                "`{c}` isn't a class name; use letters, digits, - and _"
            )));
        }
        if !project.design.net_classes.contains_key(c) {
            own.net_classes.insert(
                c.clone(),
                NetClass {
                    name: c.clone(),
                    params: Default::default(),
                    source: Default::default(),
                },
            );
        }
    }
    let mut changed = Vec::new();
    for net in &nets {
        // A net in a repeated sheet is named by its path (`/ch[0]/INF`); its sheet declares it
        // by its local name, so every channel's copy joins the class.
        let local = net.rsplit('/').next().unwrap_or(net);
        let declared = own
            .sheets
            .values_mut()
            .flat_map(|s| s.nets.iter_mut())
            .find(|n| n.name == local);
        match declared {
            Some(decl) => {
                if decl.class != class {
                    decl.class = class.clone();
                    changed.push(*net);
                }
            }
            None => {
                let sheet = own
                    .sheets
                    .values_mut()
                    .find(|s| {
                        s.connections.iter().any(|c| {
                            c.ends
                                .iter()
                                .any(|e| matches!(e, Endpoint::Net(n) if n == local))
                        })
                    })
                    .ok_or_else(|| {
                        EditError::new(format!(
                            "{net} isn't drawn in {}; set its class where it is drawn",
                            project.circuit_path
                        ))
                    })?;
                if class.is_some() {
                    sheet.nets.push(NetDecl {
                        name: local.to_owned(),
                        class: class.clone(),
                        global: false,
                        params: Default::default(),
                        renamed_from: None,
                        source: Default::default(),
                    });
                    changed.push(*net);
                }
            }
        }
    }
    if changed.is_empty() {
        return Ok(EditOutput {
            summary: "the nets are in that class already".into(),
            files: Vec::new(),
            report: None,
        });
    }
    circuit_change(
        project,
        &own,
        match &class {
            Some(c) => format!("{} in net class {c}", changed.join(", ")),
            None => format!("{} in the default net class", changed.join(", ")),
        },
    )
}
