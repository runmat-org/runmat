//! Wires, disconnects, net labels, power ports and ports: each joins (or leaves) a net by
//! writing its sheet's `connect` statements.

use super::{Editor, End};
use crate::ids::{NamePath, WireMode};
use crate::schematic::hand::pin_ref;
use crate::ir::{Endpoint, NetDecl, Param, ParamValue, PortDecl, PortDir, PortKind, SheetDef};
use crate::syntax::{net_statement, port_statement};
use super::super::edit::EditError;

/// Where an end is drawn and the sheet that holds its statements.
pub(super) struct Place<'a> {
    pub instance: NamePath,
    pub sheet: &'a SheetDef,
}

pub(super) fn place<'a>(e: &Editor<'a>, ends: &[&End]) -> Result<Place<'a>, EditError> {
    // A pin decides the page; a wire's net follows it.
    let pins: Vec<NamePath> = ends
        .iter()
        .filter(|end| matches!(end, End::Pin { .. }))
        .map(|end| e.instance_of(end))
        .collect();
    if let [first, rest @ ..] = pins.as_slice() {
        if rest.iter().any(|p| p != first) {
            return Err(EditError::new(
                "the pins are on different sheets; join sheets with a port or a power port",
            ));
        }
    }
    let instance = pins.first().cloned().unwrap_or_else(|| e.instance_of(ends[0]));
    let name = e.sheet_def_of(&instance)?;
    Ok(Place { sheet: e.own_sheet(&name)?, instance })
}

/// A name for net `n` as its sheet writes it (`GND`, `SENSE`), when it has one.
pub(super) fn written_name(e: &Editor, instance: &NamePath, n: usize) -> Option<String> {
    let net = &e.project.netlist.nets[n];
    if net.global {
        return Some(net.name.clone());
    }
    (net.path.parent().as_ref() == Some(instance)).then(|| net.path.leaf().to_owned())
}

/// The endpoint that joins `end` to another group: the pin itself, or for a wire one of its
/// net's endpoints on this sheet (or its written name).
pub(super) fn endpoint_of(e: &Editor, at: &Place, end: &End) -> Result<Endpoint, EditError> {
    match end {
        End::Pin { comp, pin } => Ok(e.pin_endpoint(*comp, *pin)),
        End::Net(n) => {
            if let Some(&i) = e.statements_of(at.sheet, &at.instance, *n).first() {
                if let Some(ep) = at.sheet.connections[i].ends.first() {
                    return Ok(ep.clone());
                }
            }
            written_name(e, &at.instance, *n).map(|name| Endpoint::Net(name)).ok_or_else(|| {
                EditError::new(format!("{} isn't on this sheet", e.project.netlist.nets[*n].name))
            })
        }
    }
}

/// The name net `n` carries on this sheet: a global rail's, or a `net.` or `port.` endpoint
/// of a statement that reaches it.
fn name_on(e: &Editor, at: &Place, n: usize) -> Option<String> {
    let net = &e.project.netlist.nets[n];
    if net.global {
        return Some(net.name.clone());
    }
    e.statements_of(at.sheet, &at.instance, n).iter().find_map(|&i| {
        at.sheet.connections[i].ends.iter().find_map(|ep| match ep {
            Endpoint::Net(name) | Endpoint::Port(name) => Some(name.clone()),
            _ => None,
        })
    })
}

fn net_name(e: &Editor, end: &End) -> String {
    e.net_of(end).map(|n| e.project.netlist.nets[n].name.clone()).unwrap_or_default()
}

fn describe(e: &Editor, end: &End) -> String {
    match end {
        End::Pin { comp, pin } => {
            let c = &e.project.netlist.components[*comp];
            format!("{}.{}", c.designator, c.pins[*pin].pin)
        }
        End::Net(n) => e.project.netlist.nets[*n].name.clone(),
    }
}

/// Adds `adds` to `end`'s group: into the first statement that reaches its net; for an end on
/// no net yet, into the statement that already holds one of the names in `adds`; else as a new
/// `connect` statement with the end.
fn attach(e: &mut Editor, at: &Place, end: &End, mut adds: Vec<Endpoint>) -> Result<(), EditError> {
    let mut statements = e.net_of(end).map(|n| e.statements_of(at.sheet, &at.instance, n)).unwrap_or_default();
    if statements.is_empty() {
        let named = at.sheet.connections.iter().position(|c| {
            adds.iter().any(|a| matches!(a, Endpoint::Net(_) | Endpoint::Port(_)) && c.ends.contains(a))
        });
        if let Some(i) = named {
            statements.push(i);
            adds.push(endpoint_of(e, at, end)?);
        }
    }
    if let Some(&i) = statements.first() {
        let connection = &at.sheet.connections[i];
        let mut ends = connection.ends.clone();
        let mut names = 0;
        for add in adds {
            if ends.contains(&add) {
                continue;
            }
            // Names go first, as a sheet writes them (`connect = [net.VIN, U1.VIN, …]`).
            if matches!(add, Endpoint::Net(_) | Endpoint::Port(_)) {
                ends.insert(names, add);
                names += 1;
            } else {
                ends.push(add);
            }
        }
        if ends == connection.ends {
            return Ok(());
        }
        return e.rewrite_connection(connection, ends);
    }
    let mut ends = adds;
    ends.push(endpoint_of(e, at, end)?);
    new_connection(e, at, ends)
}

fn new_connection(e: &mut Editor, at: &Place, ends: Vec<Endpoint>) -> Result<(), EditError> {
    let texts: Result<Vec<String>, EditError> = ends.iter().map(Editor::endpoint_text).collect();
    let statement = format!("connect = [{}]", texts?.join(", "));
    let block = e.block(&at.sheet.name)?;
    e.insert(&block, &["connect", "instance", "harness", "bus", "net", "component", "port"], &statement);
    Ok(())
}

/// `schematic.place-wire`: joins `from` and `to` (pins or wires) in one net.
pub(super) fn place_wire(e: &mut Editor) -> Result<String, EditError> {
    let from = e.pick("from", 0).ok_or_else(|| EditError::new("click the pin or wire the wire starts at"))?;
    let to = e.pick("to", 1).ok_or_else(|| EditError::new("click the pin or wire the wire ends at"))?;
    let (a, b) = (e.resolve(&from)?, e.resolve(&to)?);
    if a == b {
        return Err(EditError::new("a wire joins two different pins"));
    }
    let at = place(e, &[&a, &b])?;
    // A wire drawn by hand between two pins keeps its path: its waypoints go in the lock.
    if let (End::Pin { comp: ca, pin: pa }, End::Pin { comp: cb, pin: pb }) = (&a, &b) {
        let points = e.points("path");
        if !points.is_empty() {
            let nl = &e.project.netlist;
            let mode = e.value_str("mode").and_then(|m| WireMode::from_word(&m)).unwrap_or_default();
            e.wire = Some(super::WireRequest {
                sheet: at.sheet.name.clone(),
                a: pin_ref(&nl.components[*ca], *pa),
                b: pin_ref(&nl.components[*cb], *pb),
                points,
                mode,
            });
        }
    }
    let (na, nb) = (e.net_of(&a), e.net_of(&b));
    if na.is_some() && na == nb {
        if e.wire.is_some() {
            return Ok(format!("Drew the wire from {} to {} by hand", describe(e, &a), describe(e, &b)));
        }
        return Ok(format!("{} and {} are already on {}", describe(e, &a), describe(e, &b), net_name(e, &a)));
    }
    // Two named nets don't join by a wire: the drawing would carry two names for one net.
    if let (Some(x), Some(y)) = (na.and_then(|n| name_on(e, &at, n)), nb.and_then(|n| name_on(e, &at, n))) {
        if x != y {
            return Err(EditError::new(format!(
                "the wire would join {x} and {y}; rename one with a net label to join them"
            )));
        }
    }
    let named = e.value_str("net").map(Endpoint::Net);
    let reach = |n: Option<usize>| n.map(|n| e.statements_of(at.sheet, &at.instance, n)).unwrap_or_default();
    let (sa, sb) = (reach(na), reach(nb));
    // Into the group that already has a statement here, else the one on a net; two free pins
    // get a statement of their own.
    let keep = if !sa.is_empty() {
        Some((&a, &b))
    } else if !sb.is_empty() {
        Some((&b, &a))
    } else if na.is_some() {
        Some((&a, &b))
    } else if nb.is_some() {
        Some((&b, &a))
    } else {
        None
    };
    if let Some((keep, join)) = keep {
        let mut adds: Vec<Endpoint> = named.into_iter().collect();
        adds.push(endpoint_of(e, &at, join)?);
        attach(e, &at, keep, adds)?;
    } else {
        let mut ends = Vec::new();
        ends.extend(named);
        ends.push(endpoint_of(e, &at, &a)?);
        ends.push(endpoint_of(e, &at, &b)?);
        new_connection(e, &at, ends)?;
    }
    Ok(format!("Wired {} to {}", describe(e, &a), describe(e, &b)))
}

/// `schematic.disconnect`: takes the picked pin out of every `connect` statement of its sheet.
pub(super) fn disconnect(e: &mut Editor) -> Result<String, EditError> {
    let id = e.pick("pin", 0).ok_or_else(|| EditError::new("pick the pin to disconnect"))?;
    let end = e.resolve(&id)?;
    let End::Pin { comp, pin } = end else {
        return Err(EditError::new("pick a pin to disconnect"));
    };
    let at = place(e, &[&end])?;
    let c = &e.project.netlist.components[comp];
    let leaf = c.path.leaf().to_owned();
    let mut changed = 0;
    for connection in &at.sheet.connections {
        let ends: Vec<Endpoint> = connection
            .ends
            .iter()
            .filter(|ep| match ep {
                Endpoint::Pin { component, pin: p } if *component == leaf => {
                    !crate::elab::find_pins(&c.pins, p).map(|ix| ix.contains(&pin)).unwrap_or(false)
                }
                _ => true,
            })
            .cloned()
            .collect();
        if ends.len() == connection.ends.len() {
            continue;
        }
        changed += 1;
        // A statement left with no pin joins nothing.
        let pins = ends.iter().filter(|ep| !matches!(ep, Endpoint::Net(_) | Endpoint::Port(_))).count();
        e.rewrite_connection(connection, if pins == 0 { Vec::new() } else { ends })?;
    }
    if changed == 0 {
        return Ok(format!("{} isn't wired on this sheet", describe(e, &end)));
    }
    Ok(format!("Disconnected {}", describe(e, &end)))
}

/// Gives the net of `end` the name `name`: a wire already named elsewhere on the sheet joins it;
/// a group with a name is renamed (its `net` declaration keeps the old name as `renamed_from`);
/// an unnamed group takes the name.
fn name_net(e: &mut Editor, at: &Place, end: &End, name: &str) -> Result<(), EditError> {
    let new = Endpoint::Net(name.to_owned());
    let current = e.net_of(end);
    let statements = current.map(|n| e.statements_of(at.sheet, &at.instance, n)).unwrap_or_default();
    // A net already called `name` here: join it.
    let target = e.project.netlist.nets.iter().position(|n| {
        n.name == name || n.path == at.instance.child(name) || n.aliases.iter().any(|a| *a == at.instance.child(name).to_string())
    });
    if target.is_some() && target != current {
        return attach(e, at, end, vec![new]);
    }
    let old = statements.iter().find_map(|&i| {
        at.sheet.connections[i].ends.iter().find_map(|ep| match ep {
            Endpoint::Net(n) if n != name => Some(n.clone()),
            _ => None,
        })
    });
    let Some(old) = old else {
        return attach(e, at, end, vec![new]);
    };
    for connection in &at.sheet.connections {
        if !connection.ends.contains(&Endpoint::Net(old.clone())) {
            continue;
        }
        let ends = connection
            .ends
            .iter()
            .map(|ep| if *ep == Endpoint::Net(old.clone()) { new.clone() } else { ep.clone() })
            .collect();
        e.rewrite_connection(connection, ends)?;
    }
    if let Some(decl) = at.sheet.nets.iter().find(|n| n.name == old) {
        let renamed = NetDecl { name: name.to_owned(), renamed_from: Some(old.clone()), ..decl.clone() };
        let text = net_statement(&renamed).map_err(|e| EditError::new(e.0))?;
        e.replace(decl.source.start as usize, decl.source.end as usize, text);
    }
    Ok(())
}

/// `schematic.net-label`: names the picked wire's or pin's net.
pub(super) fn net_label(e: &mut Editor) -> Result<String, EditError> {
    let name = e.value_str("name").ok_or_else(|| EditError::new("type the net's name"))?;
    let on = e.pick("on", 0).ok_or_else(|| EditError::new("click the wire or pin the label names"))?;
    let end = e.resolve(&on)?;
    let at = place(e, &[&end])?;
    name_net(e, &at, &end, &name)?;
    Ok(format!("Labelled {} {name}", describe(e, &end)))
}

/// `schematic.power-port`: joins the picked pin or wire to a global rail, declaring the rail
/// on this sheet when no sheet does.
pub(super) fn power_port(e: &mut Editor) -> Result<String, EditError> {
    let rail = e.value_str("net").ok_or_else(|| EditError::new("choose the rail (GND, +3V3 …)"))?;
    let on = e.pick("on", 0).ok_or_else(|| EditError::new("click the pin or wire the power port goes on"))?;
    let end = e.resolve(&on)?;
    let at = place(e, &[&end])?;
    // A rail that enters the board (at a connector) is marked driven with a power flag.
    let flag = e.request.values.get("power_flag").and_then(|v| v.as_bool()) == Some(true);
    let with_flag = |mut d: NetDecl| {
        if flag {
            d.params.insert(
                "power_flag".into(),
                Param { value: ParamValue::Bool(true), source: Default::default(), call: None, written: None },
            );
        }
        d
    };
    let own = e.project.own.sheets.values().find_map(|s| s.nets.iter().find(|n| n.name == rail && n.global));
    let declared = e.project.design.sheets.values().any(|s| s.nets.iter().any(|n| n.name == rail && n.global));
    let flagged = |d: &NetDecl| d.params.contains_key("power_flag");
    if let Some(decl) = own.filter(|d| flag && !flagged(d)) {
        let text = net_statement(&with_flag(decl.clone())).map_err(|e| EditError::new(e.0))?;
        e.replace(decl.source.start as usize, decl.source.end as usize, text);
    } else if !declared {
        if let Some(local) = at.sheet.nets.iter().find(|n| n.name == rail) {
            let text = net_statement(&with_flag(NetDecl { global: true, ..local.clone() })).map_err(|e| EditError::new(e.0))?;
            e.replace(local.source.start as usize, local.source.end as usize, text);
        } else {
            let decl = with_flag(NetDecl {
                name: rail.clone(),
                class: None,
                global: true,
                params: Default::default(),
                renamed_from: None,
                source: Default::default(),
            });
            let text = net_statement(&decl).map_err(|e| EditError::new(e.0))?;
            let block = e.block(&at.sheet.name)?;
            e.insert(&block, &["net", "component", "port"], &text);
        }
    }
    attach(e, &at, &end, vec![Endpoint::Net(rail.clone())])?;
    Ok(format!("Put {} on {rail}", describe(e, &end)))
}

fn direction(word: &str) -> PortDir {
    match word {
        "input" | "in" => PortDir::In,
        "output" | "out" => PortDir::Out,
        "bidirectional" | "inout" => PortDir::InOut,
        "power" => PortDir::Power,
        _ => PortDir::Passive,
    }
}

/// Declares port `name` on `sheet` (a no-op when it has one).
pub(super) fn declare_port(e: &mut Editor, sheet: &SheetDef, name: &str, dir: PortDir) -> Result<bool, EditError> {
    if sheet.ports.iter().any(|p| p.name == name) {
        return Ok(false);
    }
    let decl = PortDecl { name: name.to_owned(), dir, kind: PortKind::Signal, source: Default::default() };
    let text = port_statement(&decl).map_err(|e| EditError::new(e.0))?;
    let block = e.block(&sheet.name)?;
    e.insert(&block, &["port"], &text);
    Ok(true)
}

/// `schematic.port`: declares a port of the sheet, joined to the picked pin or wire if any.
pub(super) fn port(e: &mut Editor) -> Result<String, EditError> {
    let name = e.value_str("name").ok_or_else(|| EditError::new("type the port's name"))?;
    let dir = direction(e.value_str("direction").as_deref().unwrap_or("passive"));
    match e.pick("on", 0) {
        Some(on) => {
            let end = e.resolve(&on)?;
            let at = place(e, &[&end])?;
            declare_port(e, at.sheet, &name, dir)?;
            attach(e, &at, &end, vec![Endpoint::Port(name.clone())])?;
            Ok(format!("Port {name} on {}", describe(e, &end)))
        }
        None => {
            let sheet_name = e.value_str("sheet").unwrap_or_else(|| e.project.design.root.clone());
            let sheet = e.own_sheet(&sheet_name)?;
            declare_port(e, sheet, &name, dir)?;
            Ok(format!("Port {name} on sheet {sheet_name}"))
        }
    }
}
