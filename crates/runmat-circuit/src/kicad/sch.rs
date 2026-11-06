//! `.kicad_sch` hierarchies and `.kicad_sym` libraries.
//!
//! Import builds a [`Design`] and a [`Pool`]:
//!
//! - Each schematic file becomes a [`SheetDef`] named by its file; a `(sheet …)` node becomes
//!   a [`SheetInstance`] (a file used by several sheets is one definition, several instances).
//! - `lib_symbols` (and `.kicad_sym` libraries) become units and entities: one gate per unit,
//!   pins named by pin *number* (the pin name is kept as an alternate), KiCad electrical
//!   types mapped to [`PinType`]. Unit-0 pins (common to all units) go to the first gate.
//! - Symbol instances with the same reference are one component; the `Footprint` property
//!   becomes the package of a [`PartRef::Entity`]. Power symbols are not components: their
//!   pin joins a global net named by their value.
//! - Connectivity is geometric: wire and bus segments join their end points and any pin,
//!   label, junction, sheet pin or wire end lying on them. Each group becomes one connection;
//!   local labels name it (same name, same net within a sheet — which also realises bus
//!   members), global labels and power symbols make it global, hierarchical labels make it a
//!   port of the sheet, sheet pins bind it to the child instance's port. Bus labels such as
//!   `D[0..7]` expand member-wise. `no_connect` markers become no-ERC markers.
//! - Per-instance references from `(instances (project (path …)))` are returned as a
//!   [`LockSection`] so elaboration keeps KiCad's annotation, including reused sheets.
//!
//! Writing is lossless: the parsed tree is written back token for token;
//! [`set_symbol_property`] edits a property in place (back-annotation).
//!
//! Symbol transforms: library coordinates are y-up; a placed symbol is mirrored
//! (`(mirror x)` flips y, `(mirror y)` flips x) and then rotated counter-clockwise by its
//! angle, then translated.

use super::at_of;
use super::sexpr::{self, SExpr};
use crate::geom::{Point, Transform};
use crate::ids::{LockEntry, LockSection, NamePath};
use crate::ir::*;
use crate::library::{make, Entity, Gate, PinType, Pool, Unit, UnitPin};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
pub struct LibPin {
    pub number: String,
    pub name: String,
    pub kind: PinType,
    /// Pin tip in library coordinates (y up), nm.
    pub at: Point,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LibSymbol {
    pub name: String,
    pub prefix: String,
    pub power: bool,
    /// Pins by unit number (0 = common to all units).
    pub units: BTreeMap<u32, Vec<LibPin>>,
}

fn pin_type(s: &str) -> PinType {
    match s {
        "input" => PinType::Input,
        "output" => PinType::Output,
        "bidirectional" => PinType::Bidirectional,
        "tri_state" => PinType::TriState,
        "passive" => PinType::Passive,
        "power_in" => PinType::PowerIn,
        "power_out" => PinType::PowerOut,
        "open_collector" => PinType::OpenCollector,
        "open_emitter" => PinType::OpenEmitter,
        "no_connect" => PinType::NotConnected,
        "free" => PinType::Free,
        _ => PinType::Unspecified,
    }
}

/// Parses one `(symbol "Lib:Name" …)` library definition.
pub fn parse_lib_symbol(node: &SExpr) -> LibSymbol {
    let name = node.arg(0).unwrap_or("").to_string();
    let mut prefix = "U".to_string();
    for p in node.children("property") {
        if p.arg(0) == Some("Reference") {
            prefix = p
                .arg(1)
                .unwrap_or("U")
                .trim_end_matches(['?', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                .to_string();
        }
    }
    let mut units: BTreeMap<u32, Vec<LibPin>> = BTreeMap::new();
    let mut add_pins = |unit: u32, n: &SExpr| {
        for pin in n.children("pin") {
            // In a library pin, the position is the connection point; y is up.
            let (x, y) = (
                pin.child("at").and_then(|a| a.arg_f(0)).unwrap_or(0.0),
                pin.child("at").and_then(|a| a.arg_f(1)).unwrap_or(0.0),
            );
            units.entry(unit).or_default().push(LibPin {
                number: pin.child_arg("number").unwrap_or("").to_string(),
                name: pin.child_arg("name").unwrap_or("").to_string(),
                kind: pin_type(pin.arg(0).unwrap_or("")),
                at: Point::new((x * 1e6).round() as i64, (y * 1e6).round() as i64),
            });
        }
    };
    add_pins(1, node);
    for sub in node.children("symbol") {
        let sname = sub.arg(0).unwrap_or("");
        let parts: Vec<&str> = sname.rsplitn(3, '_').collect();
        let unit = parts
            .get(1)
            .and_then(|u| u.parse::<u32>().ok())
            .unwrap_or(1);
        add_pins(unit, sub);
    }
    // Drop the empty default entry for symbols whose pins live in sub-symbols.
    if units.get(&1).is_some_and(|v| v.is_empty()) {
        units.remove(&1);
    }
    LibSymbol {
        name,
        prefix,
        power: node.child("power").is_some(),
        units,
    }
}

/// Reads a `.kicad_sym` library.
pub fn read_symbol_lib(text: &str) -> Result<Vec<LibSymbol>, String> {
    let e = sexpr::parse(text).map_err(|e| e.to_string())?;
    if e.head() != Some("kicad_symbol_lib") {
        return Err("not a kicad_symbol_lib".into());
    }
    Ok(e.children("symbol").map(parse_lib_symbol).collect())
}

/// Registers a library symbol as units and an entity. Returns the entity name.
pub fn lib_into_pool(sym: &LibSymbol, pool: &mut Pool) -> String {
    if pool.entity(&sym.name).is_some() {
        return sym.name.clone();
    }
    let common = sym.units.get(&0).cloned().unwrap_or_default();
    let mut gates = Vec::new();
    let unit_numbers: Vec<u32> = sym.units.keys().copied().filter(|u| *u > 0).collect();
    let unit_numbers = if unit_numbers.is_empty() {
        vec![1]
    } else {
        unit_numbers
    };
    for (k, u) in unit_numbers.iter().enumerate() {
        let mut pins: Vec<LibPin> = sym.units.get(u).cloned().unwrap_or_default();
        if k == 0 {
            pins.extend(common.iter().cloned());
        }
        let uname = format!("{}:{u}", sym.name);
        let uid = pool.add_unit(Unit {
            uuid: make::uuid("unit", &uname),
            name: uname,
            pins: pins
                .iter()
                .map(|p| UnitPin {
                    name: p.number.clone(),
                    kind: p.kind,
                    swap_group: 0,
                    alternates: vec![p.name.clone()],
                })
                .collect(),
        });
        gates.push(Gate {
            name: unit_letter(*u),
            unit: uid,
            swap_group: if unit_numbers.len() > 1 { 1 } else { 0 },
        });
    }
    pool.add_entity(Entity {
        uuid: make::uuid("entity", &sym.name),
        name: sym.name.clone(),
        prefix: if sym.prefix.is_empty() {
            "U".into()
        } else {
            sym.prefix.clone()
        },
        gates,
        tags: vec![],
    });
    sym.name.clone()
}

/// KiCad unit number to a gate name: 1 → A, 2 → B, …
pub fn unit_letter(u: u32) -> String {
    let u = u.max(1) - 1;
    if u < 26 {
        ((b'A' + u as u8) as char).to_string()
    } else {
        format!("U{}", u + 1)
    }
}

fn symbol_transform(node: &SExpr) -> Transform {
    let (at, rot) = at_of(node.child("at"));
    let m = node.child_arg("mirror");
    // Library y-up → schematic y-down → RunMat y-up: the two flips cancel, so library
    // coordinates are used as they are. `(mirror x)` flips y, `(mirror y)` flips x.
    match m {
        Some("y") => Transform::new(at, rot, true),
        Some("x") => Transform::new(at, crate::geom::normalize_deg(rot + 180.0), true),
        _ => Transform::new(at, rot, false),
    }
}

#[derive(Clone, Debug, Default)]
pub struct SchImport {
    pub design: Design,
    pub pool: Pool,
    /// KiCad per-instance designators, keyed by RunMat path.
    pub lock: LockSection,
    pub warnings: Vec<String>,
    /// (KiCad path, file, local reference, instance reference) gathered while reading.
    lock_paths: Vec<(String, String, String, String)>,
    /// Sheet node UUID → sheet name.
    sheet_uuid: BTreeMap<String, String>,
}

#[derive(Default)]
struct Points {
    pts: Vec<Point>,
    parent: Vec<usize>,
    index: BTreeMap<Point, usize>,
}

impl Points {
    fn id(&mut self, p: Point) -> usize {
        if let Some(i) = self.index.get(&p) {
            return *i;
        }
        let i = self.pts.len();
        self.pts.push(p);
        self.parent.push(i);
        self.index.insert(p, i);
        i
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (x, y) = (self.find(a), self.find(b));
        if x != y {
            self.parent[x.max(y)] = x.min(y);
        }
    }
}

enum Attach {
    Pin { component: String, pin: String },
    Power(String),
    Label(String),
    Global(String),
    Hier(String),
    SheetPin { instance: String, port: String },
    NoConnect,
}

fn read_sheet(
    file: &str,
    tree: &SExpr,
    imp: &mut SchImport,
    sheet_files: &mut Vec<(String, String)>,
) -> SheetDef {
    let mut sheet = SheetDef {
        name: file.to_string(),
        ..Default::default()
    };
    let mut libs: BTreeMap<String, LibSymbol> = BTreeMap::new();
    if let Some(ls) = tree.child("lib_symbols") {
        for s in ls.children("symbol") {
            let l = parse_lib_symbol(s);
            if !l.power {
                lib_into_pool(&l, &mut imp.pool);
            }
            libs.insert(l.name.clone(), l);
        }
    }
    let mut pts = Points::default();
    let mut attach: Vec<(Point, Attach)> = Vec::new();
    let mut segments: Vec<(Point, Point)> = Vec::new();
    let mut comps: BTreeMap<String, ComponentDecl> = BTreeMap::new();
    for node in tree.items().iter().skip(1) {
        match node.head() {
            Some("symbol") => {
                let lib_id = node.child_arg("lib_id").unwrap_or("").to_string();
                let Some(lib) = libs.get(&lib_id) else {
                    imp.warnings
                        .push(format!("{file}: symbol {lib_id} missing from lib_symbols"));
                    continue;
                };
                let unit = node.child_f("unit").unwrap_or(1.0) as u32;
                let t = symbol_transform(node);
                let prop = |k: &str| {
                    node.children("property")
                        .find(|p| p.arg(0) == Some(k))
                        .and_then(|p| p.arg(1))
                        .unwrap_or("")
                        .to_string()
                };
                let reference = prop("Reference");
                let mut pins: Vec<&LibPin> = lib
                    .units
                    .get(&unit)
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();
                pins.extend(lib.units.get(&0).into_iter().flatten());
                if lib.power {
                    let net = prop("Value");
                    for p in pins {
                        attach.push((t.apply(p.at), Attach::Power(net.clone())));
                    }
                    continue;
                }
                for p in pins {
                    attach.push((
                        t.apply(p.at),
                        Attach::Pin {
                            component: reference.clone(),
                            pin: p.number.clone(),
                        },
                    ));
                }
                if let Some(path) = node.child("instances") {
                    for proj in path.children("project") {
                        for pth in proj.children("path") {
                            let r = pth.child_arg("reference").unwrap_or(&reference).to_string();
                            imp.lock_paths.push((
                                pth.arg(0).unwrap_or("").to_string(),
                                file.to_string(),
                                reference.clone(),
                                r,
                            ));
                        }
                    }
                }
                let footprint = prop("Footprint");
                let c = comps.entry(reference.clone()).or_insert_with(|| {
                    let mut c = ComponentDecl::new(
                        &reference,
                        PartRef::Entity {
                            entity: lib_id.clone(),
                            package: (!footprint.is_empty()).then(|| footprint.clone()),
                        },
                    );
                    c.properties.insert("Value".into(), prop("Value"));
                    for p in node.children("property") {
                        let k = p.arg(0).unwrap_or("");
                        if !matches!(k, "Reference" | "Value" | "Footprint") {
                            c.properties
                                .insert(k.to_string(), p.arg(1).unwrap_or("").to_string());
                        }
                    }
                    c
                });
                if node.child_arg("dnp") == Some("yes") {
                    c.variants.insert("default".into(), Fitting::NotFitted);
                }
            }
            Some("wire") | Some("bus") => {
                let p = super::pts_of(node.child("pts"));
                for w in p.windows(2) {
                    segments.push((w[0], w[1]));
                }
            }
            Some("junction") => {
                let (p, _) = at_of(node.child("at"));
                pts.id(p);
            }
            Some("no_connect") => attach.push((at_of(node.child("at")).0, Attach::NoConnect)),
            Some("label") => attach.push((
                at_of(node.child("at")).0,
                Attach::Label(node.arg(0).unwrap_or("").into()),
            )),
            Some("global_label") => attach.push((
                at_of(node.child("at")).0,
                Attach::Global(node.arg(0).unwrap_or("").into()),
            )),
            Some("hierarchical_label") => {
                let name = node.arg(0).unwrap_or("").to_string();
                let dir = match node.child_arg("shape") {
                    Some("input") => PortDir::In,
                    Some("output") => PortDir::Out,
                    Some("bidirectional") | Some("tri_state") => PortDir::InOut,
                    _ => PortDir::Passive,
                };
                let members = expand_bus(&name);
                if !sheet.ports.iter().any(|p| p.name == name) {
                    sheet.ports.push(PortDecl {
                        name: name.clone(),
                        dir,
                        kind: if members.len() > 1 {
                            PortKind::Bus(members)
                        } else {
                            PortKind::Signal
                        },
                        source: Default::default(),
                    });
                }
                attach.push((at_of(node.child("at")).0, Attach::Hier(name)));
            }
            Some("sheet") => {
                let prop = |k: &str| {
                    node.children("property")
                        .find(|p| p.arg(0) == Some(k))
                        .and_then(|p| p.arg(1))
                        .unwrap_or("")
                        .to_string()
                };
                let name = prop("Sheetname");
                let file_ref = prop("Sheetfile");
                sheet_files.push((
                    file_ref.clone(),
                    node.child_arg("uuid").unwrap_or("").to_string(),
                ));
                imp.sheet_uuid.insert(
                    node.child_arg("uuid").unwrap_or("").to_string(),
                    name.clone(),
                );
                sheet.instances.push(SheetInstance {
                    name: name.clone(),
                    sheet: file_ref,
                    repeat: None,
                    bindings: vec![],
                    renamed_from: None,
                    source: Default::default(),
                });
                for pin in node.children("pin") {
                    attach.push((
                        at_of(pin.child("at")).0,
                        Attach::SheetPin {
                            instance: name.clone(),
                            port: pin.arg(0).unwrap_or("").to_string(),
                        },
                    ));
                }
            }
            _ => {}
        }
    }
    // Geometry: segment ends, then anything lying on a segment.
    for (a, b) in &segments {
        let (i, j) = (pts.id(*a), pts.id(*b));
        pts.union(i, j);
    }
    let attach_ids: Vec<usize> = attach.iter().map(|(p, _)| pts.id(*p)).collect();
    let all: Vec<Point> = pts.pts.clone();
    for (a, b) in &segments {
        let ia = pts.id(*a);
        for (k, p) in all.iter().enumerate() {
            if crate::geom::orient(*a, *b, *p) == 0
                && p.x >= a.x.min(b.x)
                && p.x <= a.x.max(b.x)
                && p.y >= a.y.min(b.y)
                && p.y <= a.y.max(b.y)
            {
                pts.union(ia, k);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (k, id) in attach_ids.iter().enumerate() {
        let r = pts.find(*id);
        groups.entry(r).or_default().push(k);
    }
    let mut globals: BTreeSet<String> = BTreeSet::new();
    // Labels on a wire or pin: a label touching nothing names no net, so it pairs with nothing.
    let mut connected_labels: BTreeSet<String> = BTreeSet::new();
    for (g, (_, members)) in groups.iter().enumerate() {
        let mut ends = Vec::new();
        let mut name: Option<String> = None;
        let mut nc = false;
        for &k in members {
            match &attach[k].1 {
                Attach::Label(n) => {
                    if name.is_none() {
                        name = Some(n.clone());
                    }
                    ends.push(Endpoint::Net(n.clone()));
                }
                Attach::Global(n) | Attach::Power(n) => {
                    globals.insert(n.clone());
                    ends.push(Endpoint::Net(n.clone()));
                }
                Attach::Hier(n) => {
                    // A hierarchical label names the net and is the sheet's port.
                    if name.is_none() {
                        name = Some(n.clone());
                    }
                    ends.push(Endpoint::Net(n.clone()));
                    ends.push(Endpoint::Port(n.clone()));
                }
                Attach::SheetPin { instance, port } => ends.push(Endpoint::InstancePort {
                    instance: instance.clone(),
                    port: port.clone(),
                    channel: None,
                }),
                Attach::Pin { component, pin } => ends.push(Endpoint::pin(component, pin)),
                Attach::NoConnect => nc = true,
            }
        }
        if nc {
            for e in &ends {
                if let Endpoint::Pin { .. } = e {
                    sheet.no_erc.push(e.clone());
                }
            }
        }
        let pin_count = ends
            .iter()
            .filter(|e| matches!(e, Endpoint::Pin { .. }))
            .count();
        if ends.len() < 2 {
            continue;
        }
        for &k in members {
            if let Attach::Label(n) | Attach::Global(n) = &attach[k].1 {
                connected_labels.insert(n.clone());
            }
        }
        if name.is_none() && pin_count == ends.len() {
            // An unnamed group of pins only: a local net with an automatic name.
            ends.insert(0, Endpoint::Net(format!("N${g}")));
        }
        sheet.connections.push(Connection {
            ends,
            source: Default::default(),
        });
    }
    for gname in globals {
        sheet.nets.push(NetDecl {
            name: gname,
            class: None,
            global: true,
            params: Default::default(),
            renamed_from: None,
            source: Default::default(),
        });
    }
    sheet.components = comps.into_values().collect();
    // Differential pairs by KiCad's naming convention: `X_P`/`X_N` or `X+`/`X-`, both sides
    // labels of nets: a global label reaches its namesakes on other sheets, a local label
    // touching no wire or pin names no net (KiCad's netlist has none for it).
    let mut labels = connected_labels;
    labels.extend(attach.iter().filter_map(|(_, a)| match a {
        Attach::Global(n) => Some(n.clone()),
        _ => None,
    }));
    for l in &labels {
        for (p, n) in [("_P", "_N"), ("+", "-")] {
            if let Some(base) = l.strip_suffix(p) {
                let other = format!("{base}{n}");
                if labels.contains(&other) {
                    sheet.diff_pairs.push(DiffPairDecl {
                        name: base.to_string(),
                        p: l.clone(),
                        n: other,
                        class: None,
                        source: Default::default(),
                    });
                }
            }
        }
    }
    sheet
}

/// Imports a hierarchy: `files` maps file names to contents; `root` is the root file.
pub fn read_hierarchy(files: &BTreeMap<String, String>, root: &str) -> Result<SchImport, String> {
    let mut imp = SchImport::default();
    let mut design = Design::new(root.trim_end_matches(".kicad_sch"));
    design.root = root.to_string();
    let mut todo = vec![root.to_string()];
    let mut root_uuid = String::new();
    while let Some(f) = todo.pop() {
        if design.sheets.contains_key(&f) {
            continue;
        }
        let text = files
            .get(&f)
            .ok_or_else(|| format!("missing schematic file `{f}`"))?;
        let tree = sexpr::parse(text).map_err(|e| format!("{f}: {e}"))?;
        if tree.head() != Some("kicad_sch") {
            return Err(format!("{f}: not a kicad_sch file"));
        }
        if f == root {
            root_uuid = tree.child_arg("uuid").unwrap_or("").to_string();
        }
        let mut subs = Vec::new();
        let sheet = read_sheet(&f, &tree, &mut imp, &mut subs);
        design.sheets.insert(f.clone(), sheet);
        for (sf, _) in subs {
            todo.push(sf);
        }
    }
    // Per-instance references → lock entries keyed by RunMat path.
    let mut lock = LockSection::new();
    for (kpath, _file, local, reference) in std::mem::take(&mut imp.lock_paths) {
        let mut np = NamePath::root();
        for seg in kpath.split('/').filter(|s| !s.is_empty()) {
            if seg == root_uuid {
                continue;
            }
            if let Some(name) = imp.sheet_uuid.get(seg) {
                np = np.child(name.clone());
            }
        }
        let path = np.child(local);
        lock.entries.insert(
            path.clone(),
            LockEntry {
                uuid: crate::ids::Uuid(crate::ids::fnv128(
                    format!("kicad|{kpath}|{reference}").as_bytes(),
                ))
                .with_version(8),
                kind: "component".into(),
                designator: Some(reference),
            },
        );
    }
    imp.lock = lock;
    imp.design = design;
    Ok(imp)
}

/// Edits a symbol property in place (`Value`, `Footprint`, …) for the symbol with this
/// reference. Returns whether anything changed.
pub fn set_symbol_property(tree: &mut SExpr, reference: &str, key: &str, value: &str) -> bool {
    let mut changed = false;
    if let Some(items) = tree.items_mut() {
        for node in items.iter_mut() {
            if node.head() != Some("symbol") {
                continue;
            }
            let is = node
                .children("property")
                .any(|p| p.arg(0) == Some("Reference") && p.arg(1) == Some(reference));
            if !is {
                continue;
            }
            if let Some(list) = node.items_mut() {
                for p in list.iter_mut() {
                    if p.head() == Some("property") && p.arg(0) == Some(key) {
                        if let Some(v) = p.items_mut() {
                            if v.len() > 2 {
                                v[2] = SExpr::str(value);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }
    changed
}
