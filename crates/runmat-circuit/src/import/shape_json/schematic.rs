//! Shape-string schematics (`docType` 1):
//!
//! - each placed part (`LIB`) becomes a component: its designator and value from its `T~P` and
//!   `T~N` texts, its part from the part-number attribute (else its package and value), each
//!   pin (`P`) a part pin named as the symbol shows it, on the pad its number names;
//! - wires (`W`) join what touches them: a pin's end, another wire's end, a net label (`N`) or a
//!   net flag (`F`, a power or ground symbol, whose net is global); the label or flag names the
//!   net.

use super::{attributes, Document, Fields};
use crate::import::eagle::library::{find_attr, param_key};
use crate::import::{file_safe, Report};
use crate::ir::{ComponentDecl, Connection, Design, Endpoint, NetDecl, PartRef};
use crate::library::catalog::{CatalogPart, CatalogPin};
use crate::library::Package;
use std::collections::BTreeMap;

const MPN_KEYS: &[&str] = &["Manufacturer Part", "BOM_Manufacturer Part", "MPN"];
const MAKER_KEYS: &[&str] = &["Manufacturer", "BOM_Manufacturer"];

fn pin_kind(electric: &str) -> &'static str {
    match electric {
        "1" => "input",
        "2" => "output",
        "3" => "bidirectional",
        "4" => "power_in",
        _ => "passive",
    }
}

/// Union-find over the schematic's connection points and wires.
struct Joins(Vec<usize>);

impl Joins {
    fn add(&mut self) -> usize {
        self.0.push(self.0.len());
        self.0.len() - 1
    }
    fn root(&mut self, mut i: usize) -> usize {
        while self.0[i] != i {
            self.0[i] = self.0[self.0[i]];
            i = self.0[i];
        }
        i
    }
    fn join(&mut self, a: usize, b: usize) {
        let (a, b) = (self.root(a), self.root(b));
        self.0[a] = b;
    }
}

/// `p` lies on the segment `a`–`b` (canvas units, within a hundredth).
fn on_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> bool {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    if l2 < 1e-12 {
        return (p.0 - a.0).hypot(p.1 - a.1) < 0.01;
    }
    let t = ((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2;
    if !(-1e-9..=1.0 + 1e-9).contains(&t) {
        return false;
    }
    (a.0 + t * dx - p.0).hypot(a.1 + t * dy - p.1) < 0.01
}

struct Pin {
    designator: String,
    name: String,
    at: (f64, f64),
}

/// Reads a schematic document. `packages` are the board's footprints by key and `footprints`
/// its designator → key (both empty without a board); `footprint_dir` is where the footprint
/// files go, relative to the `.circuit`.
pub fn read(doc: &Document, name: &str, footprint_dir: &str, packages: &BTreeMap<String, Package>, footprints: &BTreeMap<String, String>, report: &mut Report) -> Result<Design, String> {
    let mut design = Design::new(name);
    let mut components = Vec::new();
    let mut pins: Vec<Pin> = Vec::new();
    let mut wires: Vec<Vec<(f64, f64)>> = Vec::new();
    // (point, net name, global)
    let mut names: Vec<((f64, f64), String, bool)> = Vec::new();
    let mut unnamed = 0;
    for s in &doc.shapes {
        let mut items = s.split("#@$");
        let head = items.next().unwrap_or("");
        let groups: Vec<&str> = head.split("^^").collect();
        let f = Fields::of(groups[0]);
        match f.kind() {
            "LIB" => {
                let attrs = attributes(f.str(3));
                let subs: Vec<&str> = items.collect();
                let text = |mark: &str| subs.iter().map(|t| Fields::of(t)).find(|t| t.kind() == "T" && t.str(1) == mark).map(|t| t.str(12).trim().to_owned()).filter(|t| !t.is_empty());
                let designator = text("P").unwrap_or_else(|| {
                    unnamed += 1;
                    format!("U{}", 900 + unnamed)
                });
                let value = text("N").unwrap_or_default();
                let package = attrs.get("package").cloned().unwrap_or_default();
                let mut part_pins: Vec<CatalogPin> = Vec::new();
                for sub in &subs {
                    let g: Vec<&str> = sub.split("^^").collect();
                    let p = Fields::of(g[0]);
                    if p.kind() != "P" {
                        continue;
                    }
                    let dot = g.get(1).map(|d| Fields::of(d)).map(|d| (d.num(0), d.num(1))).unwrap_or((p.num(4), p.num(5)));
                    let pin_name = g.get(3).map(|d| Fields::of(d).str(4).trim().to_owned()).unwrap_or_default();
                    let number = g.get(4).map(|d| Fields::of(d).str(4).trim().to_owned()).filter(|n| !n.is_empty()).unwrap_or_else(|| p.str(3).to_owned());
                    let mut pin = if pin_name.is_empty() { number.clone() } else { pin_name };
                    if part_pins.iter().any(|q| q.name == pin) {
                        pin = format!("{pin}_{number}");
                    }
                    part_pins.push(CatalogPin { name: pin.clone(), pads: vec![number], kind: pin_kind(p.str(2)).into() });
                    pins.push(Pin { designator: designator.clone(), name: pin, at: dot });
                }
                if part_pins.is_empty() {
                    report.not_carried(format!("part {designator}: a symbol without pins"));
                    continue;
                }
                // The board's footprint for this designator, else one named for the package.
                let key = footprints.get(&designator).cloned().unwrap_or_else(|| file_safe(&package));
                if !packages.contains_key(&key) {
                    report.not_carried(format!("part {designator}: footprint {package} (footprints come from the board)"));
                }
                let fallback = if value.is_empty() { package.clone() } else { format!("{package}-{value}") };
                let mpn = find_attr(&attrs, MPN_KEYS).unwrap_or_else(|| fallback.clone());
                if mpn == fallback {
                    report.approximated("parts without a part number are named from their package and value");
                }
                let mut params = BTreeMap::new();
                for (k, v) in &attrs {
                    if k == "package" || v.trim().is_empty() || MPN_KEYS.iter().chain(MAKER_KEYS).any(|m| k.eq_ignore_ascii_case(m)) {
                        continue;
                    }
                    params.insert(param_key(k), v.clone());
                }
                let part = CatalogPart {
                    mpn: mpn.clone(),
                    manufacturer: find_attr(&attrs, MAKER_KEYS).unwrap_or_default(),
                    description: String::new(),
                    category: String::new(),
                    prefix: designator.trim_end_matches(|c: char| !c.is_ascii_alphabetic()).to_owned(),
                    package: format!("{footprint_dir}/{key}.kicad_mod"),
                    pins: part_pins,
                    params,
                    datasheet: None,
                    spice: None,
                };
                design.parts.entry(mpn.clone()).or_insert(part);
                let mut c = ComponentDecl::new(&designator, PartRef::Part(mpn));
                if !value.is_empty() {
                    c.properties.insert("Value".into(), value);
                }
                components.push(c);
            }
            "W" => wires.push(super::pairs(f.str(1))),
            "N" => names.push(((f.num(1), f.num(2)), f.str(5).trim().to_owned(), false)),
            "F" => {
                let at = groups.get(1).map(|g| Fields::of(g)).map(|g| (g.num(0), g.num(1))).unwrap_or((f.num(2), f.num(3)));
                let net = groups.get(2).map(|g| Fields::of(g).str(0).trim().to_owned()).unwrap_or_default();
                names.push((at, net, true));
            }
            "J" | "T" | "PL" | "R" | "E" | "A" | "PG" | "PT" | "I" | "" => {}
            "B" | "BE" => report.approximated("busses are carried as their member nets where labels name them"),
            other => report.not_carried(format!("{other} items")),
        }
    }

    // Wires join at shared points and where an end lands on another wire.
    let mut joins = Joins(Vec::new());
    let wire_ids: Vec<usize> = wires.iter().map(|_| joins.add()).collect();
    let touches = |p: (f64, f64), w: &[(f64, f64)]| w.windows(2).any(|s| on_segment(p, s[0], s[1])) || w.iter().any(|q| (q.0 - p.0).hypot(q.1 - p.1) < 0.01);
    for (i, a) in wires.iter().enumerate() {
        for (j, b) in wires.iter().enumerate().skip(i + 1) {
            if a.iter().any(|p| touches(*p, b)) || b.iter().any(|p| touches(*p, a)) {
                joins.join(wire_ids[i], wire_ids[j]);
            }
        }
    }
    let node_of = |joins: &mut Joins, p: (f64, f64)| -> usize {
        let n = joins.add();
        for (i, w) in wires.iter().enumerate() {
            if touches(p, w) {
                joins.join(n, wire_ids[i]);
            }
        }
        n
    };
    let pin_nodes: Vec<usize> = pins.iter().map(|p| node_of(&mut joins, p.at)).collect();
    let name_nodes: Vec<usize> = names.iter().map(|(p, _, _)| node_of(&mut joins, *p)).collect();
    // Pins on the same point join without a wire too.
    for i in 0..pins.len() {
        for j in i + 1..pins.len() {
            if (pins[i].at.0 - pins[j].at.0).hypot(pins[i].at.1 - pins[j].at.1) < 0.01 {
                joins.join(pin_nodes[i], pin_nodes[j]);
            }
        }
    }
    // Same name, same net (labels and flags join by name across the sheet).
    let mut by_name: BTreeMap<&str, usize> = BTreeMap::new();
    for (k, (_, n, _)) in names.iter().enumerate() {
        if n.is_empty() {
            continue;
        }
        match by_name.get(n.as_str()) {
            Some(first) => joins.join(*first, name_nodes[k]),
            None => {
                by_name.insert(n, name_nodes[k]);
            }
        }
    }

    let mut groups: BTreeMap<usize, (Option<String>, bool, Vec<Endpoint>)> = BTreeMap::new();
    for (k, (_, n, global)) in names.iter().enumerate() {
        let g = groups.entry(joins.root(name_nodes[k])).or_insert((None, false, vec![]));
        if !n.is_empty() && (g.0.is_none() || *global) {
            g.0 = Some(n.clone());
        }
        g.1 |= *global;
    }
    for (k, p) in pins.iter().enumerate() {
        let g = groups.entry(joins.root(pin_nodes[k])).or_insert((None, false, vec![]));
        g.2.push(Endpoint::pin(&p.designator, &p.name));
    }
    let main = design.sheet_mut("main");
    main.components = components;
    let mut auto = 0;
    let mut net_count = 0;
    for (_, (net, global, ends)) in groups {
        if ends.is_empty() || (net.is_none() && ends.len() < 2) {
            continue;
        }
        net_count += 1;
        let net = net.unwrap_or_else(|| {
            auto += 1;
            format!("N{auto}")
        });
        if global {
            main.nets.push(NetDecl { name: net.clone(), class: None, global: true, params: Default::default(), renamed_from: None, source: Default::default() });
        }
        let mut all = vec![Endpoint::Net(net)];
        all.extend(ends);
        main.connections.push(Connection { ends: all, source: Default::default() });
    }
    report.count("components", design.sheet_mut("main").components.len());
    report.count("parts", design.parts.len());
    report.count("nets", net_count);
    Ok(design)
}
