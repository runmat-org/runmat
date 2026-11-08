//! Hierarchy and directives: sheet symbols, sheet entries, repeated channels, no-ERC marks,
//! net classes and differential pairs on nets, buses and harnesses.

use super::super::edit::EditError;
use super::wire::{declare_port, endpoint_of, place, written_name};
use super::{Editor, End};
use crate::ids::NamePath;
use crate::ir::{Binding, BlanketDecl, BusDecl, DiffPairDecl, HarnessDecl, NetDecl, PortDir, SheetDef, SheetInstance};
use crate::syntax::{
    blanket_statement, bus_statement, diff_pair_statement, harness_statement, instance_statement, net_statement, no_erc_statement,
};

/// A sheet instance path from a pick (`sheet:/pwr`, `sheet:/ch[0]`, `sheet:/ch[0]#2`).
fn instance_path(id: &str) -> NamePath {
    super::page_instance(id)
}

/// The instance name a path ends in, without a channel index (`ch[0]` → `ch`).
fn instance_name(path: &NamePath) -> String {
    let leaf = path.leaf();
    leaf.split('[').next().unwrap_or(leaf).to_owned()
}

/// The sheet the command works on: the `parent` field, a picked page, else the root sheet.
fn page_sheet<'a>(e: &Editor<'a>) -> Result<&'a SheetDef, EditError> {
    let page = e
        .value_str("parent")
        .or_else(|| e.request.selection.iter().find(|s| s.starts_with("sheet:")).cloned());
    let name = match page {
        Some(id) => e.sheet_def_of(&instance_path(&id))?,
        None => e.project.design.root.clone(),
    };
    e.own_sheet(&name)
}

/// The picked sheet symbol: its instance statement and the sheet holding it.
fn picked_instance<'a>(e: &Editor<'a>) -> Result<(&'a SheetDef, &'a SheetInstance, NamePath), EditError> {
    let id = e.pick_sheet().ok_or_else(|| EditError::new("pick a sheet symbol"))?;
    let path = instance_path(&id);
    let parent = path.parent().ok_or_else(|| EditError::new("the root sheet has no sheet symbol"))?;
    let sheet = e.own_sheet(&e.sheet_def_of(&parent)?)?;
    let name = instance_name(&path);
    let instance = sheet
        .instances
        .iter()
        .find(|i| i.name == name)
        .ok_or_else(|| EditError::new(format!("sheet `{}` has no instance `{name}`", sheet.name)))?;
    Ok((sheet, instance, path))
}

impl Editor<'_> {
    fn pick_sheet(&self) -> Option<String> {
        self.value_str("symbol")
            .or_else(|| self.request.selection.iter().find(|s| s.starts_with("sheet:")).cloned())
    }
}

/// `schematic.sheet-symbol`: an instance of a sheet (made empty when the file has none of
/// that name) on the current page, repeated as channels when `channels` is above one.
pub(super) fn sheet_symbol(e: &mut Editor) -> Result<String, EditError> {
    let name = e.value_str("name").ok_or_else(|| EditError::new("name the instance"))?;
    let child = e.value_str("sheet").unwrap_or_else(|| name.clone());
    let channels = e.value_u32("channels").unwrap_or(1).max(1);
    let parent = page_sheet(e)?;
    if parent.instances.iter().any(|i| i.name == name) || parent.components.iter().any(|c| c.name == name) {
        return Err(EditError::new(format!("sheet `{}` already has `{name}`", parent.name)));
    }
    if child == parent.name {
        return Err(EditError::new("a sheet can't hold an instance of itself"));
    }
    if !e.project.design.sheets.contains_key(&child) {
        let blocks = super::blocks(e.text()).ok_or_else(|| EditError::new("the file has no `circuit` block"))?;
        let at = blocks.after_sheets;
        let indent = blocks.sheets.last().map(|s| super::indent_at(e.text(), s.close)).unwrap_or_default();
        e.replace(at, at, format!("\n\n{indent}sheet {child} {{\n{indent}}}"));
    }
    let instance = SheetInstance {
        name: name.clone(),
        sheet: child.clone(),
        repeat: (channels > 1).then_some(channels),
        bindings: Vec::new(),
        renamed_from: None,
        designators: None,
        source: Default::default(),
    };
    let text = instance_statement(&instance).map_err(|e| EditError::new(e.0))?;
    let block = e.block(&parent.name)?;
    e.insert(&block, &["instance", "harness", "bus", "net", "component", "port"], &text);
    Ok(format!("Placed sheet symbol {name} ({child}) on {}", parent.name))
}

/// `schematic.sheet-entry`: a port on the picked symbol's sheet, which shows as an entry on
/// every symbol of that sheet. With `to` (a pin or wire on the symbol's own sheet) the entry is
/// bound there (`bind = { PORT = U1.VIN }`); with `bus`, channel `i` of a repeated symbol binds
/// to member `offset + i` of that bus (`bind = { PORT = { bus = D, offset = 0 } }`). A new
/// binding of the same port replaces the old one.
pub(super) fn sheet_entry(e: &mut Editor) -> Result<String, EditError> {
    let port = e.value_str("port").ok_or_else(|| EditError::new("name the entry's port"))?;
    let (parent, instance, path) = picked_instance(e)?;
    let child = e.own_sheet(&instance.sheet)?;
    let added = declare_port(e, child, &port, PortDir::Passive)?;
    let binding = match (e.pick("to", 0), e.value_str("bus")) {
        (Some(_), Some(_)) => return Err(EditError::new("bind the entry to a pin or wire, or to a bus, not both")),
        (Some(to), None) => {
            let end = e.resolve(&to)?;
            let at = place(e, &[&end])?;
            if Some(&at.instance) != path.parent().as_ref() {
                return Err(EditError::new(format!("pick a pin or wire on sheet {}, where the symbol is", parent.name)));
            }
            Some(Binding::Shared(endpoint_of(e, &at, &end)?))
        }
        (None, Some(bus)) => {
            if !parent.buses.iter().any(|b| b.name == bus) {
                return Err(EditError::new(format!("sheet {} has no bus `{bus}`", parent.name)));
            }
            Some(Binding::BusIndexed { bus, offset: e.value_u32("offset").unwrap_or(0) })
        }
        (None, None) => None,
    };
    let Some(binding) = binding else {
        return Ok(if added { format!("Added entry {port} (a port of sheet {})", child.name) } else { format!("{} already has port {port}", child.name) });
    };
    let shown = match &binding {
        Binding::Shared(end) => crate::syntax::endpoint(end).map_err(|e| EditError::new(e.0))?,
        Binding::BusIndexed { bus, offset } => format!("bus {bus} from member {offset}"),
        Binding::NetIndexed { prefix, offset } => format!("{prefix}{offset}…"),
    };
    let mut bound = instance.clone();
    bound.bindings.retain(|(p, _)| *p != port);
    bound.bindings.push((port.clone(), binding));
    let text = instance_statement(&bound).map_err(|e| EditError::new(e.0))?;
    e.replace(instance.source.start as usize, instance.source.end as usize, text);
    Ok(format!("Entry {port} of {} bound to {shown}", instance.name))
}

/// `schematic.repeat-channel`: the picked symbol's instance repeated `channels` times.
pub(super) fn repeat_channel(e: &mut Editor) -> Result<String, EditError> {
    let channels = e.value_u32("channels").ok_or_else(|| EditError::new("give the number of channels"))?;
    if channels < 1 {
        return Err(EditError::new("a sheet repeats at least once"));
    }
    let (_, instance, _) = picked_instance(e)?;
    let designators = match e.value_str("designators") {
        Some(t) if channels > 1 => {
            crate::elab::check_channel_designators(&t).map_err(EditError::new)?;
            Some(t)
        }
        _ => None,
    };
    let named = designators.as_ref().map(|t| format!(", parts named {t}")).unwrap_or_default();
    let repeated = SheetInstance { repeat: (channels > 1).then_some(channels), designators, ..instance.clone() };
    let text = instance_statement(&repeated).map_err(|e| EditError::new(e.0))?;
    e.replace(instance.source.start as usize, instance.source.end as usize, text);
    Ok(format!("{} repeats as {channels} channel(s){named}", instance.name))
}

/// `schematic.no-erc`: marks the picked pin so ERC doesn't report it.
pub(super) fn no_erc(e: &mut Editor) -> Result<String, EditError> {
    let id = e.pick("pin", 0).ok_or_else(|| EditError::new("pick the pin to mark"))?;
    let end = e.resolve(&id)?;
    let End::Pin { comp, pin } = end else {
        return Err(EditError::new("No-ERC marks a pin"));
    };
    let at = place(e, &[&end])?;
    let ep = e.pin_endpoint(comp, pin);
    let designator = e.project.netlist.components[comp].designator.clone();
    if at.sheet.no_erc.contains(&ep) || e.project.netlist.no_erc.contains(&(comp, pin)) {
        return Ok(format!("{designator} pin is already marked"));
    }
    let block = e.block(&at.sheet.name)?;
    let statements: Vec<_> = block.statements.iter().filter(|s| s.word == "no_erc").cloned().collect();
    match statements.as_slice() {
        // One list: it takes the pin.
        [one] => {
            let mut ends = at.sheet.no_erc.clone();
            ends.push(ep);
            let text = no_erc_statement(&ends).map_err(|e| EditError::new(e.0))?;
            e.replace(one.start, one.end, text);
        }
        _ => {
            let text = no_erc_statement(&[ep]).map_err(|e| EditError::new(e.0))?;
            e.insert(&block, &["no_erc", "connect", "instance", "net", "component"], &text);
        }
    }
    Ok(format!("Marked {designator} No-ERC"))
}

/// The picked nets, each with the sheet that writes it and its written name.
fn picked_nets<'a>(e: &Editor<'a>) -> Result<Vec<(&'a SheetDef, String)>, EditError> {
    let mut ids = e.value_strs("nets");
    if ids.is_empty() {
        ids = e.request.selection.iter().filter(|s| s.starts_with("net:")).cloned().collect();
    }
    if ids.is_empty() {
        return Err(EditError::new("pick the nets"));
    }
    ids.iter()
        .map(|id| {
            let end = e.resolve(id)?;
            let n = e.net_of(&end).ok_or_else(|| EditError::new(format!("{id} is on no net")))?;
            let at = place(e, &[&end])?;
            let name = written_name(e, &at.instance, n).ok_or_else(|| {
                EditError::new(format!(
                    "{} has no name of its own; name it with a net label first",
                    e.project.netlist.nets[n].name
                ))
            })?;
            // A global rail is declared on whichever sheet declares it.
            let sheet = match e.project.own.sheets.values().find(|s| s.nets.iter().any(|d| d.name == name)) {
                Some(s) if e.project.netlist.nets[n].global => s,
                _ => at.sheet,
            };
            Ok((sheet, name))
        })
        .collect()
}

/// `schematic.directive`: a net class or parameters on the picked nets, a differential pair of
/// two, or a blanket over them all.
pub(super) fn directive(e: &mut Editor) -> Result<String, EditError> {
    let kind = e.value_str("type").unwrap_or_else(|| "net_class".into());
    let nets = picked_nets(e)?;
    match kind.as_str() {
        "net_class" => {
            let class = e.value_str("value").ok_or_else(|| EditError::new("name the net class"))?;
            if !e.project.design.net_classes.contains_key(&class) {
                let known: Vec<&str> = e.project.design.net_classes.keys().map(String::as_str).collect();
                return Err(EditError::new(format!("the circuit has no net class `{class}` (it has {})", if known.is_empty() { "none".to_owned() } else { known.join(", ") })));
            }
            for (sheet, name) in &nets {
                update_net(e, sheet, name, |d| d.class = Some(class.clone()))?;
            }
            Ok(format!("{} net(s) in class {class}", nets.len()))
        }
        "parameters" => {
            let value = e.value_str("value").ok_or_else(|| EditError::new("write the parameters, such as `max_length = 50 mm`"))?;
            let params = net_parameters(&value)?;
            let keys: Vec<String> = params.keys().cloned().collect();
            for (sheet, name) in &nets {
                update_net(e, sheet, name, |d| d.params.extend(params.clone()))?;
            }
            Ok(format!("{} on {} net(s)", keys.join(", "), nets.len()))
        }
        "diff_pair" => {
            let [(sheet, p), (other, n)] = nets.as_slice() else {
                return Err(EditError::new("a differential pair takes two nets, positive first"));
            };
            if sheet.name != other.name {
                return Err(EditError::new("a differential pair's nets are on one sheet"));
            }
            let name = e.value_str("value").filter(|v| !v.contains(' ')).unwrap_or_else(|| pair_name(p, n));
            let pair = DiffPairDecl { name: name.clone(), p: p.clone(), n: n.clone(), class: None, source: Default::default() };
            let text = diff_pair_statement(&pair).map_err(|e| EditError::new(e.0))?;
            let block = e.block(&sheet.name)?;
            e.insert(&block, &["diff_pair", "no_erc", "connect", "net"], &text);
            Ok(format!("Paired {p} and {n} as {name}"))
        }
        "blanket" => {
            let value = e.value_str("value").ok_or_else(|| EditError::new("write the class (`power`) or entries such as `class = power, max_length = 50 mm`"))?;
            let Some((sheet, _)) = nets.first() else { return Err(EditError::new("pick the nets the blanket covers")) };
            if let Some((other, _)) = nets.iter().find(|(s, _)| s.name != sheet.name) {
                return Err(EditError::new(format!("a blanket's nets are on one sheet (`{}` and `{}`)", sheet.name, other.name)));
            }
            let (class, params) = blanket_entries(&value)?;
            if let Some(class) = &class {
                if !e.project.design.net_classes.contains_key(class) {
                    let known: Vec<&str> = e.project.design.net_classes.keys().map(String::as_str).collect();
                    return Err(EditError::new(format!("the circuit has no net class `{class}` (it has {})", if known.is_empty() { "none".to_owned() } else { known.join(", ") })));
                }
            }
            let name = (1..).map(|i| format!("blanket{i}")).find(|n| !sheet.blankets.iter().any(|b| &b.name == n)).unwrap_or_default();
            let mut covered: Vec<String> = nets.iter().map(|(_, n)| n.clone()).collect();
            covered.dedup();
            let count = covered.len();
            let blanket = BlanketDecl { name: name.clone(), nets: covered, class, params, source: Default::default() };
            let text = blanket_statement(&blanket).map_err(|e| EditError::new(e.0))?;
            let block = e.block(&sheet.name)?;
            e.insert(&block, &["blanket", "diff_pair", "no_erc", "connect", "net"], &text);
            Ok(format!("Blanket {name} over {count} net(s)"))
        }
        other => Err(EditError::new(format!("the `{other}` directive isn't one of net class, parameters, differential pair or blanket"))),
    }
}

/// A blanket's value: a class name alone (`power`), or entries (`class = power, max_length =
/// 50 mm`) read by the `.circuit` parser itself.
fn blanket_entries(value: &str) -> Result<(Option<String>, std::collections::BTreeMap<String, crate::ir::Param>), EditError> {
    let value = value.trim();
    if !value.contains('=') {
        return Ok((Some(value.to_owned()), Default::default()));
    }
    let probe = format!("runmat 0.7\nstd 1\n\ncircuit probe {{\n    root = s\n\n    sheet s {{\n        blanket b = {{ nets = [N], {value} }}\n    }}\n}}\n");
    let read = crate::syntax::parse_circuit("blanket", &probe);
    let bad = || EditError::new(format!("`{value}` isn't a blanket's entries; write `class = power, max_length = 50 mm`"));
    if !read.diagnostics.is_empty() {
        return Err(bad());
    }
    let design = read.design.ok_or_else(bad)?;
    let b = design.sheets.get("s").and_then(|s| s.blankets.first()).ok_or_else(bad)?;
    if b.nets != ["N"] {
        return Err(EditError::new("pick the nets on the canvas; the value holds the class and parameters"));
    }
    Ok((b.class.clone(), b.params.clone()))
}

/// Rewrites net `name`'s declaration on `sheet` (or adds one) after `change`.
fn update_net(e: &mut Editor, sheet: &SheetDef, name: &str, change: impl FnOnce(&mut NetDecl)) -> Result<(), EditError> {
    let decl = sheet.nets.iter().find(|d| d.name == name);
    let mut next = decl.cloned().unwrap_or(NetDecl {
        name: name.to_owned(),
        class: None,
        global: false,
        params: Default::default(),
        renamed_from: None,
        source: Default::default(),
    });
    change(&mut next);
    let text = net_statement(&next).map_err(|e| EditError::new(e.0))?;
    match decl {
        Some(d) => e.replace(d.source.start as usize, d.source.end as usize, text),
        None => {
            let block = e.block(&sheet.name)?;
            e.insert(&block, &["net", "component", "port"], &text);
        }
    }
    Ok(())
}

/// A parameter set as a net statement's entries take it (`max_length = 50 mm, impedance =
/// 90 ohm`), read by the `.circuit` parser itself so the directive writes only what reads back.
fn net_parameters(value: &str) -> Result<std::collections::BTreeMap<String, crate::ir::Param>, EditError> {
    const RESERVED: [&str; 3] = ["class", "global", "renamed_from"];
    let probe = format!("runmat 0.7\nstd 1\n\ncircuit probe {{\n    root = s\n\n    sheet s {{\n        net N = {{ {value} }}\n    }}\n}}\n");
    let read = crate::syntax::parse_circuit("parameters", &probe);
    let bad = || EditError::new(format!("`{value}` isn't a parameter list; write entries such as `max_length = 50 mm, impedance = 90 ohm`"));
    let design = read.design.ok_or_else(bad)?;
    let decl = design.sheets.get("s").and_then(|s| s.nets.first()).ok_or_else(bad)?;
    if let Some(k) = RESERVED.iter().find(|k| value.split(',').any(|entry| entry.split('=').next().map(str::trim) == Some(**k))) {
        return Err(EditError::new(format!("`{k}` isn't a parameter; use the net class directive or a net label")));
    }
    if decl.params.is_empty() {
        return Err(bad());
    }
    Ok(decl.params.clone())
}

/// `USB_P` and `USB_N` pair as `USB`.
fn pair_name(p: &str, n: &str) -> String {
    let common: String = p.chars().zip(n.chars()).take_while(|(a, b)| a == b).map(|(a, _)| a).collect();
    let common = common.trim_end_matches(['_', '-']).to_owned();
    if common.is_empty() { format!("{p}_{n}") } else { common }
}

/// `D[0..7]` → (`D`, `D0` … `D7`); `DATA` alone names a bus of no members yet.
fn expand_bus(text: &str) -> Result<(String, Vec<String>), EditError> {
    let text = text.trim();
    let Some((name, rest)) = text.split_once('[') else {
        return Err(EditError::new(format!("write the bus as NAME[first..last] (`{text}`)")));
    };
    let range = rest.strip_suffix(']').ok_or_else(|| EditError::new(format!("`{text}` has no closing `]`")))?;
    let (a, b) = range.split_once("..").ok_or_else(|| EditError::new(format!("write the range as first..last (`{text}`)")))?;
    let (a, b): (u32, u32) = (
        a.trim().parse().map_err(|_| EditError::new(format!("`{a}` isn't a number")))?,
        b.trim().parse().map_err(|_| EditError::new(format!("`{b}` isn't a number")))?,
    );
    let members = if a <= b { (a..=b).collect::<Vec<_>>() } else { (b..=a).rev().collect() };
    if members.len() > 1024 {
        return Err(EditError::new("a bus holds at most 1024 members"));
    }
    Ok((name.trim().to_owned(), members.iter().map(|i| format!("{}{i}", name.trim())).collect()))
}

/// `schematic.place-bus`: declares bus `D[0..7]` (members `D0` … `D7`) on the page's sheet.
pub(super) fn bus(e: &mut Editor) -> Result<String, EditError> {
    let written = e.value_str("name").ok_or_else(|| EditError::new("name the bus, as D[0..7]"))?;
    let (name, members) = expand_bus(&written)?;
    let sheet = page_sheet(e)?;
    if sheet.buses.iter().any(|b| b.name == name) {
        return Err(EditError::new(format!("sheet `{}` already has bus {name}", sheet.name)));
    }
    let count = members.len();
    let text = bus_statement(&BusDecl { name: name.clone(), members, source: Default::default() }).map_err(|e| EditError::new(e.0))?;
    let block = e.block(&sheet.name)?;
    e.insert(&block, &["bus", "net", "component", "port"], &text);
    Ok(format!("Bus {name} with {count} member(s)"))
}

/// `schematic.place-harness`: declares a harness of one of the circuit's interfaces.
pub(super) fn harness(e: &mut Editor) -> Result<String, EditError> {
    let name = e.value_str("name").ok_or_else(|| EditError::new("name the harness"))?;
    let interface = e.value_str("interface").ok_or_else(|| EditError::new("choose the interface"))?;
    if !e.project.design.interfaces.contains_key(&interface) {
        return Err(EditError::new(format!("the circuit declares no interface `{interface}`")));
    }
    let sheet = page_sheet(e)?;
    if sheet.harnesses.iter().any(|h| h.name == name) {
        return Err(EditError::new(format!("sheet `{}` already has harness {name}", sheet.name)));
    }
    let text = harness_statement(&HarnessDecl { name: name.clone(), interface: interface.clone(), source: Default::default() })
        .map_err(|e| EditError::new(e.0))?;
    let block = e.block(&sheet.name)?;
    e.insert(&block, &["harness", "bus", "net", "component", "port"], &text);
    Ok(format!("Harness {name} ({interface})"))
}
