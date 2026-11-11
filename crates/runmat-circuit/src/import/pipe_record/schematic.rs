//! Pipe-record schematics: the numbered records of one or more sheets.
//!
//! - Each part (`RECORD=1`) becomes a component: its designator from its designator record
//!   (`34`), its value from its `Value` parameter (`41`, else `Comment`), its footprint from
//!   its current PCB implementation (`45`, owned through an implementation list `44`). Each pin
//!   (`2`) of the part's shown unit (`OWNERPARTID` = the part's `CURRENTPARTID`) becomes a part
//!   pin named as the symbol shows it, on the pad its designator names. Parts placed once per
//!   unit join under their designator.
//! - A pin connects at its outer end: its location plus its length, in the direction the low
//!   two bits of `PINCONGLOMERATE` give (0 right, 1 up, 2 left, 3 down).
//! - Wires (`27`) join what touches them: a pin's end, another wire, a net label (`25`), a
//!   power port (`17`, whose net is global) or a port (`18`, which joins sheets by name). Labels
//!   join by name across the sheets given.

use super::Record;
use crate::import::shape_json::schematic::{on_segment, Joins};
use crate::import::xml_project::library::{find_attr, param_key};
use crate::import::{file_safe, Report};
use crate::ir::{ComponentDecl, Connection, Design, Endpoint, NetDecl, PartRef};
use crate::library::catalog::{PartRecord, PartPin};
use crate::library::Package;
use std::collections::BTreeMap;

const MPN_KEYS: &[&str] = &[
    "Manufacturer Part Number",
    "Manufacturer Part Number 1",
    "MPN",
    "Part Number",
];
const MAKER_KEYS: &[&str] = &["Manufacturer", "Manufacturer 1", "Manufacturer Name"];

fn pin_kind(electrical: &str) -> &'static str {
    match electrical.trim() {
        "0" => "input",
        "1" => "bidirectional",
        "2" => "output",
        "3" => "open_collector",
        "5" => "tri_state",
        "6" => "open_emitter",
        "7" => "power_in",
        _ => "passive",
    }
}

/// What the pins and connection items of all sheets give.
struct Pin {
    designator: String,
    name: String,
    at: (f64, f64),
}

/// One placed part, gathered over the units placed under its designator.
struct Placed {
    designator: String,
    libref: String,
    value: String,
    footprint: String,
    params: BTreeMap<String, String>,
    pins: Vec<PartPin>,
}

fn point(r: &Record, x: &str, y: &str) -> (f64, f64) {
    (r.sch(x), r.sch(y))
}

/// The component record that owns record `i` (itself for a part), through owner chains.
fn owner_part(body: &[&Record], mut i: usize) -> Option<usize> {
    for _ in 0..8 {
        let r = body.get(i)?;
        if r.kind().trim() == "1" {
            return Some(i);
        }
        i = r.index("OWNERINDEX")?;
    }
    None
}

/// Reads the schematic sheets (`(file name, records)`) into a design named `name`.
/// `packages` are the board's footprints by key and `footprints` its designator → key (both
/// empty without a board); `footprint_dir` is where footprint files go, beside the `.circuit`.
pub fn read(
    sheets: &[(String, &Vec<Record>)],
    name: &str,
    footprint_dir: &str,
    packages: &BTreeMap<String, Package>,
    footprints: &BTreeMap<String, String>,
    report: &mut Report,
) -> Result<Design, String> {
    let mut design = Design::new(name);
    let mut placed: Vec<Placed> = Vec::new();
    let mut by_designator: BTreeMap<String, usize> = BTreeMap::new();
    let mut pins: Vec<Pin> = Vec::new();
    // Each sheet's wires and names, joined per sheet; names then join by name across sheets.
    let mut wires: Vec<(usize, Vec<(f64, f64)>)> = Vec::new();
    // (sheet, point, net name, global)
    let mut names: Vec<(usize, (f64, f64), String, bool)> = Vec::new();
    let mut pin_sheet: Vec<usize> = Vec::new();
    let mut unnamed = 0;
    if sheets.len() > 1 {
        report.approximated(
            "sheets join by net-label, port and power-port names across the sheets given",
        );
    }
    for (sheet, (_, recs)) in sheets.iter().enumerate() {
        let body: Vec<&Record> = recs.iter().filter(|r| r.has("RECORD")).collect();
        // Per part record: designator, parameters, current footprint.
        let mut designator: BTreeMap<usize, String> = BTreeMap::new();
        let mut params: BTreeMap<usize, BTreeMap<String, String>> = BTreeMap::new();
        let mut footprint: BTreeMap<usize, String> = BTreeMap::new();
        for r in &body {
            let owner = || r.index("OWNERINDEX").and_then(|o| owner_part(&body, o));
            match r.kind().trim() {
                "34" => {
                    if let Some(o) = owner() {
                        designator.insert(o, r.str("TEXT").trim().to_owned());
                    }
                }
                "41" => {
                    if let Some(o) = owner() {
                        params.entry(o).or_default().insert(
                            r.str("NAME").trim().to_owned(),
                            r.str("TEXT").trim().to_owned(),
                        );
                    }
                }
                "45" if r.str("MODELTYPE").trim().eq_ignore_ascii_case("PCBLIB") => {
                    if let Some(o) = owner() {
                        if r.flag("ISCURRENT") || !footprint.contains_key(&o) {
                            footprint.insert(o, r.str("MODELNAME").trim().to_owned());
                        }
                    }
                }
                _ => {}
            }
        }
        for (i, r) in body.iter().enumerate() {
            match r.kind().trim() {
                "1" => {
                    let des = designator.get(&i).cloned().filter(|d| !d.is_empty() && !d.ends_with('?')).unwrap_or_else(|| {
                        unnamed += 1;
                        format!("U{}", 900 + unnamed)
                    });
                    let part_id = r.num("CURRENTPARTID");
                    let mode = r.num("DISPLAYMODE");
                    let mut unit_pins = Vec::new();
                    for p in &body {
                        if p.kind().trim() != "2" || p.index("OWNERINDEX") != Some(i) {
                            continue;
                        }
                        let owner_part = p.num("OWNERPARTID");
                        if (owner_part > 0.0 && owner_part != part_id) || p.num("OWNERPARTDISPLAYMODE") != mode {
                            continue;
                        }
                        let (x, y) = point(p, "LOCATION.X", "LOCATION.Y");
                        let len = p.sch("PINLENGTH");
                        let (dx, dy) = match p.num("PINCONGLOMERATE") as i64 & 3 {
                            0 => (1.0, 0.0),
                            1 => (0.0, 1.0),
                            2 => (-1.0, 0.0),
                            _ => (0.0, -1.0),
                        };
                        let number = p.str("DESIGNATOR").trim().to_owned();
                        let pin_name = p.str("NAME").trim().to_owned();
                        unit_pins.push((number, pin_name, pin_kind(p.str("ELECTRICAL")), (x + dx * len, y + dy * len)));
                    }
                    let k = *by_designator.entry(des.clone()).or_insert_with(|| {
                        let ps = params.get(&i).cloned().unwrap_or_default();
                        let value = ps.get("Value").cloned().filter(|v| !v.is_empty() && !v.starts_with('=')).or_else(|| ps.get("Comment").cloned().filter(|v| !v.is_empty() && !v.starts_with('='))).unwrap_or_default();
                        placed.push(Placed { designator: des.clone(), libref: r.str("LIBREFERENCE").trim().to_owned(), value, footprint: footprint.get(&i).cloned().unwrap_or_default(), params: ps, pins: vec![] });
                        placed.len() - 1
                    });
                    report.approximated("part symbols are drawn from their pins, not from the file's symbol graphics");
                    for (number, pin_name, kind, at) in unit_pins {
                        let pl = &mut placed[k];
                        let mut pin = if pin_name.is_empty() { number.clone() } else { pin_name };
                        if let Some(q) = pl.pins.iter_mut().find(|q| q.name == pin) {
                            if !q.pads.contains(&number) {
                                // One signal on several pads (GND on 1 and 5): one pin, both pads.
                                q.pads.push(number.clone());
                            }
                        } else {
                            if pl.pins.iter().any(|q| q.pads.contains(&number)) {
                                pin = format!("{pin}_{number}");
                            }
                            pl.pins.push(PartPin { name: pin.clone(), pads: vec![number], kind: kind.into() });
                        }
                        pins.push(Pin { designator: des.clone(), name: pin, at });
                        pin_sheet.push(sheet);
                    }
                }
                "27" => {
                    let n = r.num("LOCATIONCOUNT") as usize;
                    let pts: Vec<(f64, f64)> = (1..=n).map(|k| point(r, &format!("X{k}"), &format!("Y{k}"))).collect();
                    if pts.len() >= 2 {
                        wires.push((sheet, pts));
                    }
                }
                "25" => names.push((sheet, point(r, "LOCATION.X", "LOCATION.Y"), r.str("TEXT").trim().to_owned(), false)),
                "17" => names.push((sheet, point(r, "LOCATION.X", "LOCATION.Y"), r.str("TEXT").trim().to_owned(), true)),
                "18" => {
                    let at = point(r, "LOCATION.X", "LOCATION.Y");
                    let w = r.sch("WIDTH");
                    let vertical = (4.0..=7.0).contains(&r.num("STYLE"));
                    let end = if vertical { (at.0, at.1 - w) } else { (at.0 + w, at.1) };
                    let n = r.str("NAME").trim().to_owned();
                    names.push((sheet, at, n.clone(), false));
                    names.push((sheet, end, n, false));
                }
                "26" | "37" => report.approximated("busses are carried as their member nets where labels name them"),
                "15" | "16" => report.not_carried("sheet symbols and their entries (give each sheet's file; ports join them by name)"),
                "22" => report.not_carried("no-ERC markers"),
                // Annotations, graphics, junctions (wires join without them), the sheet itself
                // and the records read above.
                _ => {}
            }
        }
    }

    // Wires join at shared points and where an end lands on another wire, on one sheet.
    let mut joins = Joins(Vec::new());
    let wire_ids: Vec<usize> = wires.iter().map(|_| joins.add()).collect();
    let touches = |p: (f64, f64), w: &[(f64, f64)]| {
        w.windows(2).any(|s| on_segment(p, s[0], s[1]))
            || w.iter().any(|q| (q.0 - p.0).hypot(q.1 - p.1) < 0.01)
    };
    for (i, (sa, a)) in wires.iter().enumerate() {
        for (j, (sb, b)) in wires.iter().enumerate().skip(i + 1) {
            if sa == sb && (a.iter().any(|p| touches(*p, b)) || b.iter().any(|p| touches(*p, a))) {
                joins.join(wire_ids[i], wire_ids[j]);
            }
        }
    }
    let node_of = |joins: &mut Joins, sheet: usize, p: (f64, f64)| -> usize {
        let n = joins.add();
        for (i, (s, w)) in wires.iter().enumerate() {
            if *s == sheet && touches(p, w) {
                joins.join(n, wire_ids[i]);
            }
        }
        n
    };
    let pin_nodes: Vec<usize> = pins
        .iter()
        .zip(&pin_sheet)
        .map(|(p, s)| node_of(&mut joins, *s, p.at))
        .collect();
    let name_nodes: Vec<usize> = names
        .iter()
        .map(|(s, p, _, _)| node_of(&mut joins, *s, *p))
        .collect();
    for i in 0..pins.len() {
        for j in i + 1..pins.len() {
            if pin_sheet[i] == pin_sheet[j]
                && (pins[i].at.0 - pins[j].at.0).hypot(pins[i].at.1 - pins[j].at.1) < 0.01
            {
                joins.join(pin_nodes[i], pin_nodes[j]);
            }
        }
    }
    let mut by_name: BTreeMap<&str, usize> = BTreeMap::new();
    for (k, (_, _, n, _)) in names.iter().enumerate() {
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

    let mut components = Vec::new();
    for pl in placed {
        if pl.pins.is_empty() {
            report.not_carried(format!("part {}: a symbol without pins", pl.designator));
            continue;
        }
        let key = footprints
            .get(&pl.designator)
            .cloned()
            .unwrap_or_else(|| file_safe(&pl.footprint));
        if !packages.contains_key(&key) {
            report.not_carried(format!(
                "part {}: footprint {} (footprints come from the board)",
                pl.designator, pl.footprint
            ));
        }
        let fallback = match (pl.libref.is_empty(), pl.value.is_empty()) {
            (false, false) => format!("{}-{}", pl.libref, pl.value),
            (false, true) => pl.libref.clone(),
            _ => format!("{}-{}", pl.footprint, pl.value),
        };
        let mpn = find_attr(&pl.params, MPN_KEYS).unwrap_or_else(|| fallback.clone());
        if mpn == fallback {
            report.approximated(
                "parts without a part number are named from their library reference and value",
            );
        }
        let mut params = BTreeMap::new();
        for (k, v) in &pl.params {
            if v.trim().is_empty()
                || v.starts_with('=')
                || k == "Comment"
                || k == "Value"
                || MPN_KEYS
                    .iter()
                    .chain(MAKER_KEYS)
                    .any(|m| k.eq_ignore_ascii_case(m))
            {
                continue;
            }
            params.insert(param_key(k), v.clone());
        }
        let part = PartRecord {
            mpn: mpn.clone(),
            manufacturer: find_attr(&pl.params, MAKER_KEYS).unwrap_or_default(),
            description: String::new(),
            category: String::new(),
            prefix: pl
                .designator
                .trim_end_matches(|c: char| !c.is_ascii_alphabetic())
                .to_owned(),
            package: format!("{footprint_dir}/{key}.kicad_mod"),
            pins: pl.pins,
            params,
            datasheet: None,
            spice: None,
            ..Default::default()
        };
        design.parts.entry(mpn.clone()).or_insert(part);
        let mut c = ComponentDecl::new(&pl.designator, PartRef::Part(mpn));
        if !pl.value.is_empty() {
            c.properties.insert("Value".into(), pl.value);
        }
        components.push(c);
    }

    let mut groups: BTreeMap<usize, (Option<String>, bool, Vec<Endpoint>)> = BTreeMap::new();
    for (k, (_, _, n, global)) in names.iter().enumerate() {
        let g = groups
            .entry(joins.root(name_nodes[k]))
            .or_insert((None, false, vec![]));
        if !n.is_empty() && (g.0.is_none() || *global) {
            g.0 = Some(n.clone());
        }
        g.1 |= *global;
    }
    let kept: std::collections::BTreeSet<&str> =
        components.iter().map(|c| c.name.as_str()).collect();
    for (k, p) in pins.iter().enumerate() {
        if !kept.contains(p.designator.as_str()) {
            continue;
        }
        let g = groups
            .entry(joins.root(pin_nodes[k]))
            .or_insert((None, false, vec![]));
        let end = Endpoint::pin(&p.designator, &p.name);
        if !g.2.contains(&end) {
            g.2.push(end);
        }
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
            main.nets.push(NetDecl {
                name: net.clone(),
                class: None,
                global: true,
                params: Default::default(),
                renamed_from: None,
                source: Default::default(),
            });
        }
        let mut all = vec![Endpoint::Net(net)];
        all.extend(ends);
        main.connections.push(Connection {
            ends: all,
            source: Default::default(),
        });
    }
    report.count("components", design.sheet_mut("main").components.len());
    report.count("parts", design.parts.len());
    report.count("nets", net_count);
    Ok(design)
}
