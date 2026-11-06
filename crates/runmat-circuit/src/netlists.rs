//! Netlist exports of an elaborated design (its fitted components and their nets).
//!
//! - [`kicad`]: the KiCad netlist (`.net`, s-expression `(export (version "E") …)`): each
//!   component with its value, footprint, part number and fields, and each net with one `node`
//!   per pad (reference, pad number, pin name, pin type).
//! - [`edif`]: EDIF 2.0.0 (`.edn`), netlist view: a `COMPONENTS` library with one cell per
//!   part and package (its ports are the pads), and a `DESIGN` library whose top cell
//!   instantiates every component and joins pads into nets. Names that are not EDIF
//!   identifiers are written as `(rename <id> "<name>")`.
//!
//! The SPICE deck comes from `runmat-circuit-spice` and the IPC-D-356A test netlist from
//! [`crate::fab::ipc356`]. Output is deterministic.

use crate::elab::{CompInst, Netlist};
use crate::kicad::sexpr::{write, SExpr};
use crate::library::Pool;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

fn pin_type(t: &crate::library::PinType) -> &'static str {
    use crate::library::PinType as P;
    match t {
        P::Input => "input",
        P::Output => "output",
        P::Bidirectional => "bidirectional",
        P::TriState => "tri_state",
        P::Passive => "passive",
        P::PowerIn => "power_in",
        P::PowerOut => "power_out",
        P::OpenCollector => "open_collector",
        P::OpenEmitter => "open_emitter",
        P::NotConnected => "no_connect",
        _ => "unspecified",
    }
}

fn package_name(c: &CompInst, pool: &Pool) -> String {
    c.package
        .and_then(|u| pool.packages.get(&u))
        .map(|p| p.name.clone())
        .unwrap_or_default()
}

fn mpn(c: &CompInst, pool: &Pool) -> (String, String) {
    c.part
        .and_then(|u| pool.parts.get(&u))
        .map(|p| (p.mpn.clone(), p.manufacturer.clone()))
        .unwrap_or_default()
}

fn fitted(nl: &Netlist) -> Vec<(usize, &CompInst)> {
    let mut v: Vec<(usize, &CompInst)> = nl
        .components
        .iter()
        .enumerate()
        .filter(|(_, c)| c.fitted())
        .collect();
    v.sort_by_key(|(_, c)| crate::fab::natural_key(&c.designator));
    v
}

/// The KiCad netlist of the fitted components.
pub fn kicad(nl: &Netlist, pool: &Pool) -> String {
    let s = SExpr::str;
    let mut comps = Vec::new();
    for (_, c) in fitted(nl) {
        let (part, maker) = mpn(c, pool);
        let mut items = vec![
            SExpr::list("ref", vec![s(&c.designator)]),
            SExpr::list("value", vec![s(&crate::fab::assembly::component_value(c))]),
            SExpr::list("footprint", vec![s(&package_name(c, pool))]),
        ];
        let mut fields = Vec::new();
        if !part.is_empty() {
            fields.push(SExpr::list(
                "field",
                vec![SExpr::list("name", vec![s("MPN")]), s(&part)],
            ));
        }
        if !maker.is_empty() {
            fields.push(SExpr::list(
                "field",
                vec![SExpr::list("name", vec![s("Manufacturer")]), s(&maker)],
            ));
        }
        for (k, v) in &c.properties {
            if k != "Value" {
                fields.push(SExpr::list(
                    "field",
                    vec![SExpr::list("name", vec![s(k)]), s(v)],
                ));
            }
        }
        if !fields.is_empty() {
            items.push(SExpr::list("fields", fields));
        }
        items.push(SExpr::list(
            "sheetpath",
            vec![
                SExpr::list("names", vec![s(&format!("/{}", c.sheet))]),
                SExpr::list("tstamps", vec![s(&format!("/{}", c.sheet))]),
            ],
        ));
        comps.push(SExpr::list("comp", items));
    }
    let mut nets = Vec::new();
    let mut names: Vec<(usize, &crate::elab::NetInst)> = nl.nets.iter().enumerate().collect();
    names.sort_by(|a, b| a.1.name.cmp(&b.1.name));
    let mut code = 0;
    for (_, n) in names {
        let mut nodes = Vec::new();
        for &(ci, pi) in &n.pins {
            let c = &nl.components[ci];
            if !c.fitted() {
                continue;
            }
            let p = &c.pins[pi];
            for pad in &p.pads {
                nodes.push(SExpr::list(
                    "node",
                    vec![
                        SExpr::list("ref", vec![s(&c.designator)]),
                        SExpr::list("pin", vec![s(pad)]),
                        SExpr::list("pinfunction", vec![s(&p.pin)]),
                        SExpr::list("pintype", vec![s(pin_type(&p.kind))]),
                    ],
                ));
            }
        }
        if nodes.is_empty() {
            continue;
        }
        code += 1;
        let mut items = vec![
            SExpr::list("code", vec![s(&code.to_string())]),
            SExpr::list("name", vec![s(&n.name)]),
        ];
        if let Some(class) = &n.class {
            items.push(SExpr::list("class", vec![s(class)]));
        }
        items.extend(nodes);
        nets.push(SExpr::list("net", items));
    }
    let doc = SExpr::list(
        "export",
        vec![
            SExpr::list("version", vec![s("E")]),
            SExpr::list(
                "design",
                vec![
                    SExpr::list("source", vec![s(&format!("{}.circuit", nl.design))]),
                    SExpr::list(
                        "tool",
                        vec![s(&format!(
                            "RunMat runmat-circuit {}",
                            env!("CARGO_PKG_VERSION")
                        ))],
                    ),
                ],
            ),
            SExpr::list("components", comps),
            SExpr::list("nets", nets),
        ],
    );
    write(&doc)
}

/// An EDIF identifier for `name`: letters, digits and underscores, starting with a letter;
/// with the original as a `rename` when they differ. `used` keeps identifiers unique.
fn edif_name(name: &str, used: &mut BTreeSet<String>) -> String {
    let mut id: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if !id.starts_with(|c: char| c.is_ascii_alphabetic()) {
        id.insert(0, 'N');
    }
    let base = id.clone();
    let mut k = 1;
    while !used.insert(id.clone()) {
        k += 1;
        id = format!("{base}_{k}");
    }
    if id == name {
        id
    } else {
        format!("(rename {id} \"{}\")", edif_string(name))
    }
}

/// An EDIF string: printable ASCII, with the micro and ohm signs spelled `u` and `Ohm` and
/// any other character as `?`.
fn edif_string(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{b5}' | '\u{3bc}' => "u".to_owned(),
            '\u{3a9}' | '\u{2126}' => "Ohm".to_owned(),
            '"' => "'".to_owned(),
            c if (' '..='~').contains(&c) => c.to_string(),
            _ => "?".to_owned(),
        })
        .collect()
}

/// The identifier part of what [`edif_name`] returned.
fn edif_id(n: &str) -> &str {
    n.strip_prefix("(rename ")
        .and_then(|r| r.split(' ').next())
        .unwrap_or(n)
}

/// The EDIF 2.0.0 netlist of the fitted components.
pub fn edif(nl: &Netlist, pool: &Pool) -> String {
    let comps = fitted(nl);
    // One cell per (part number or value, package); its ports are the pads its components use.
    let mut cells: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    let mut cell_of: BTreeMap<usize, (String, String)> = BTreeMap::new();
    for (i, c) in &comps {
        let (part, _) = mpn(c, pool);
        let key = (
            if part.is_empty() {
                crate::fab::assembly::component_value(c)
            } else {
                part
            },
            package_name(c, pool),
        );
        let pads = cells.entry(key.clone()).or_default();
        for p in &c.pins {
            pads.extend(p.pads.iter().cloned());
        }
        cell_of.insert(*i, key);
    }
    let mut out = String::new();
    let design = {
        let mut u = BTreeSet::new();
        edif_name(&nl.design, &mut u)
    };
    let _ = writeln!(out, "(edif {design}");
    out.push_str("  (edifVersion 2 0 0)\n  (edifLevel 0)\n  (keywordMap (keywordLevel 0))\n");
    let _ = writeln!(
        out,
        "  (status (written (timeStamp 2000 1 1 0 0 0) (program \"RunMat runmat-circuit\" (version \"{}\"))))",
        env!("CARGO_PKG_VERSION")
    );
    out.push_str("  (library COMPONENTS\n    (edifLevel 0)\n    (technology (numberDefinition))\n");
    let mut cell_ids: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut used_cells = BTreeSet::new();
    let mut port_ids: BTreeMap<(String, String), BTreeMap<String, String>> = BTreeMap::new();
    for (key, pads) in &cells {
        let name = edif_name(&format!("{}_{}", key.0, key.1), &mut used_cells);
        let _ = writeln!(out, "    (cell {name} (cellType GENERIC)\n      (view netlist (viewType NETLIST)\n        (interface");
        let mut used_ports = BTreeSet::new();
        let mut ids = BTreeMap::new();
        let mut sorted: Vec<&String> = pads.iter().collect();
        sorted.sort_by_key(|p| crate::fab::natural_key(p));
        for pad in sorted {
            let pn = edif_name(&format!("P{pad}"), &mut used_ports);
            let _ = writeln!(out, "          (port {pn} (direction INOUT))");
            ids.insert(pad.clone(), edif_id(&pn).to_owned());
        }
        out.push_str("        )))\n");
        cell_ids.insert(key.clone(), edif_id(&name).to_owned());
        port_ids.insert(key.clone(), ids);
    }
    out.push_str(
        "  )\n  (library DESIGN\n    (edifLevel 0)\n    (technology (numberDefinition))\n",
    );
    let top = edif_id(&design).to_owned();
    let _ = writeln!(out, "    (cell {top} (cellType GENERIC)\n      (view netlist (viewType NETLIST)\n        (interface)\n        (contents");
    let mut used_inst = BTreeSet::new();
    let mut inst_ids: BTreeMap<usize, String> = BTreeMap::new();
    for (i, c) in &comps {
        let name = edif_name(&c.designator, &mut used_inst);
        let cell = &cell_ids[&cell_of[i]];
        let _ = writeln!(
            out,
            "          (instance {name} (viewRef netlist (cellRef {cell} (libraryRef COMPONENTS)))\n            (property VALUE (string \"{}\")))",
            edif_string(&crate::fab::assembly::component_value(c))
        );
        inst_ids.insert(*i, edif_id(&name).to_owned());
    }
    let mut used_nets = BTreeSet::new();
    let mut nets: Vec<&crate::elab::NetInst> = nl.nets.iter().collect();
    nets.sort_by(|a, b| a.name.cmp(&b.name));
    for n in nets {
        let mut refs = Vec::new();
        for &(ci, pi) in &n.pins {
            let Some(inst) = inst_ids.get(&ci) else {
                continue;
            };
            let ports = &port_ids[&cell_of[&ci]];
            for pad in &nl.components[ci].pins[pi].pads {
                refs.push(format!("(portRef {} (instanceRef {inst}))", ports[pad]));
            }
        }
        if refs.is_empty() {
            continue;
        }
        let name = edif_name(&n.name, &mut used_nets);
        let _ = writeln!(
            out,
            "          (net {name} (joined\n            {}))",
            refs.join("\n            ")
        );
    }
    out.push_str("        )))\n  )\n");
    let _ = writeln!(
        out,
        "  (design {top} (cellRef {top} (libraryRef DESIGN)))\n)"
    );
    out
}
