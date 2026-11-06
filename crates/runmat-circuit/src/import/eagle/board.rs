//! EAGLE XML boards (`.brd`), from the format's public DTD:
//!
//! - the outline from the dimension layer (20): its wires chained into rings, the largest the
//!   board and the rest cutouts;
//! - copper layers 1 (top) to 16 (bottom), with the inner layers the board uses between them;
//! - elements as footprints from the board's own libraries, placed with their rotation, and
//!   mirrored ones on the bottom;
//! - signals as nets: their wires as tracks, vias, polygons as zones, and each `contactref` as
//!   the pad's net;
//! - the design rules' clearance, track, drill, annular ring and copper-to-edge as the board's
//!   limits.

use super::library::{arc_mid, layer, len, package, point, rotation, DIMENSION, MILLING};
use super::schematic::{footprint_key, libraries, library_of};
use crate::geom::poly::{Polygon, Ring};
use crate::geom::{mm, Coord, Point};
use crate::import::{xml::Element, Report};
use crate::layout::{Board, FabLimits, Footprint, Side, Stackup, Track, TrackGeom, Via, Zone};
use std::collections::{BTreeMap, BTreeSet};

/// A design-rule length (`8mil`, `0.2mm`, `0.01inch`, `200mic`) in board units.
fn rule_length(v: &str) -> Option<Coord> {
    let v = v.trim();
    let split = v.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(v.len());
    let n: f64 = v[..split].trim().parse().ok()?;
    let to_mm = match &v[split..] {
        "mil" => 0.0254,
        "inch" => 25.4,
        "mic" => 0.001,
        "mm" | "" => 1.0,
        _ => return None,
    };
    Some(mm(n * to_mm))
}

/// The copper layers the board uses, top to bottom (EAGLE numbers 1 … 16).
fn copper_layers(board: &Element) -> Vec<u32> {
    let mut used: BTreeSet<u32> = [1, 16].into_iter().collect();
    let mut walk = |e: &Element| {
        let l = layer(e);
        if (1..=16).contains(&l) {
            used.insert(l);
        }
    };
    for s in board.list("signals", "signal") {
        for item in &s.children {
            walk(item);
        }
    }
    let mut layers: Vec<u32> = used.into_iter().collect();
    // RunMat stacks come in pairs: an odd count gets one more inner layer, unused.
    if layers.len() % 2 == 1 {
        let spare = (2..16).find(|l| !layers.contains(l)).unwrap_or(2);
        layers.push(spare);
        layers.sort();
    }
    layers
}

/// The ring a dimension-layer item draws: a circle's, or none for a wire (wires chain).
fn wire_points(w: &Element) -> Vec<Point> {
    let (a, b) = (point(w, "x1", "y1"), point(w, "x2", "y2"));
    match w.num("curve").filter(|c| c.abs() > 1e-9) {
        Some(curve) => {
            let mid = arc_mid(a, b, curve);
            match crate::geom::shape::arc_from_three(a, mid, b) {
                Some((c, sweep)) => crate::geom::shape::arc_points(c, a, sweep, crate::geom::ARC_TOL),
                None => vec![a, b],
            }
        }
        None => vec![a, b],
    }
}

/// The board outline from the dimension (and milling) layer of `plain`.
fn outline(plain: Option<&Element>, report: &mut Report) -> Option<Polygon> {
    let mut segments = Vec::new();
    let mut rings: Vec<Ring> = Vec::new();
    for item in plain.map(|p| p.children.as_slice()).unwrap_or_default() {
        if !matches!(layer(item), DIMENSION | MILLING) {
            continue;
        }
        match item.name.as_str() {
            "wire" => {
                for w in wire_points(item).windows(2) {
                    segments.push((w[0], w[1]));
                }
            }
            "circle" => rings.push(crate::geom::poly::circle_ring(point(item, "x", "y"), len(item, "radius"), crate::geom::ARC_TOL)),
            "rectangle" => rings.push(crate::geom::poly::rect_ring(crate::geom::Rect::new(point(item, "x1", "y1"), point(item, "x2", "y2")))),
            other => report.not_carried(format!("a <{other}> on the dimension layer")),
        }
    }
    crate::kicad::pcb::outline_from(&segments, rings)
}

/// The board's limits from its design rules (the rest keep RunMat's defaults).
fn limits(rules: Option<&Element>) -> FabLimits {
    let mut out = FabLimits::default();
    let Some(rules) = rules else { return out };
    let params: BTreeMap<&str, &str> = rules.all("param").filter_map(|p| Some((p.attr("name")?, p.attr("value")?))).collect();
    // A rule may list a value per layer (`10mil 10mil …`): the first applies.
    let get = |k: &str| params.get(k).and_then(|v| v.split_whitespace().next()).and_then(rule_length).filter(|v| *v > 0);
    if let Some(v) = get("mdWireWire") {
        out.min_clearance = v;
    }
    if let Some(v) = get("msWidth") {
        out.min_track = v;
    }
    if let Some(v) = get("msDrill") {
        out.min_drill = v;
    }
    if let Some(v) = get("rlMinViaOuter") {
        out.min_annular = v;
    }
    if let Some(v) = get("mdCopperDimension") {
        out.min_copper_to_edge = v;
    }
    out
}

/// Reads the `<board>` of `drawing` into a board named `name`.
pub fn read(drawing: &Element, name: &str, report: &mut Report) -> Result<Board, String> {
    let brd = drawing.child("board").ok_or("the file has no <board>")?;
    let copper = copper_layers(brd);
    let index = |eagle: u32| copper.iter().position(|l| *l == eagle).map(|i| i as u8);
    let stackup = Stackup::standard(copper.len() as u8, mm(1.6), 1.0, 0.5);
    let mut board = Board::new(name, Polygon::default(), stackup);
    match outline(brd.child("plain"), report) {
        Some(o) => board.outline = o,
        None => report.not_carried("the outline: the dimension layer has no closed ring"),
    }
    board.limits = limits(brd.child("designrules"));

    // Nets first, so pads and tracks can name them.
    let mut pad_nets: BTreeMap<String, BTreeMap<String, crate::layout::NetId>> = BTreeMap::new();
    for s in brd.list("signals", "signal") {
        let net = board.add_net(s.attr("name").unwrap_or(""), None);
        for c in s.all("contactref") {
            pad_nets.entry(c.attr("element").unwrap_or("").to_owned()).or_default().insert(c.attr("pad").unwrap_or("").to_owned(), net);
        }
    }

    let libs = libraries(brd);
    let mut packages = BTreeMap::new();
    for e in brd.list("elements", "element") {
        let reference = e.attr("name").unwrap_or("").to_owned();
        let pkg_name = e.attr("package").unwrap_or("");
        let Some(lib) = library_of(&libs, e) else {
            report.not_carried(format!("element {reference}: its library isn't in the file"));
            continue;
        };
        let key = footprint_key(lib.attr("name").unwrap_or(""), pkg_name);
        if !packages.contains_key(&key) {
            let Some(p) = lib.list("packages", "package").find(|p| p.attr("name") == Some(pkg_name)) else {
                report.not_carried(format!("element {reference}: package {pkg_name} isn't in its library"));
                continue;
            };
            packages.insert(key.clone(), package(&key, p, report));
        }
        // The file rotates a mirrored element, then mirrors it in x; RunMat mirrors, then
        // rotates, so the same placement turns the other way.
        let (deg, mirror) = rotation(e.attr("rot"));
        let deg = if mirror { (360.0 - deg).rem_euclid(360.0) } else { deg };
        let id = board.alloc();
        board.footprints.insert(
            id,
            Footprint {
                id,
                reference: reference.clone(),
                component: None,
                value: e.attr("value").unwrap_or("").to_owned(),
                package: packages[&key].clone(),
                at: point(e, "x", "y"),
                rotation: deg,
                side: if mirror { Side::Bottom } else { Side::Top },
                locked: e.attr("locked") == Some("yes"),
                pad_nets: pad_nets.remove(&reference).unwrap_or_default(),
                fitted: e.attr("populate") != Some("no"),
                properties: Default::default(),
            },
        );
    }

    let (mut tracks, mut vias, mut zones) = (0, 0, 0);
    for s in brd.list("signals", "signal") {
        let net = Some(board.add_net(s.attr("name").unwrap_or(""), None));
        for item in &s.children {
            match item.name.as_str() {
                "wire" => {
                    let Some(l) = index(layer(item)) else { continue };
                    let (a, b) = (point(item, "x1", "y1"), point(item, "x2", "y2"));
                    let geom = match item.num("curve").filter(|c| c.abs() > 1e-9) {
                        Some(curve) => TrackGeom::Arc { start: a, mid: arc_mid(a, b, curve), end: b },
                        None => TrackGeom::Segment { a, b },
                    };
                    let id = board.alloc();
                    board.tracks.insert(id, Track { id, net, layer: l, width: len(item, "width"), geom, locked: false });
                    tracks += 1;
                }
                "via" => {
                    let (from, to) = match item.attr("extent").and_then(|x| x.split_once('-')) {
                        Some((a, b)) => (a.parse().ok().and_then(index), b.parse().ok().and_then(index)),
                        None => (Some(0), index(16)),
                    };
                    let drill = len(item, "drill");
                    let diameter = match len(item, "diameter") {
                        0 => drill + 2 * board.limits.min_annular.max(mm(0.1)),
                        d => d,
                    };
                    let id = board.alloc();
                    let (from, to) = (from.unwrap_or(0), to.unwrap_or(copper.len() as u8 - 1));
                    board.vias.insert(id, Via { id, net, at: point(item, "x", "y"), diameter, drill, from: from.min(to), to: from.max(to), micro: false, tented: true, locked: false });
                    vias += 1;
                }
                "polygon" => {
                    let Some(l) = index(layer(item)) else { continue };
                    let ring: Ring = item.all("vertex").map(|v| point(v, "x", "y")).collect();
                    if item.all("vertex").any(|v| v.num("curve").is_some_and(|c| c.abs() > 1e-9)) {
                        report.approximated("curved polygon edges are carried as straight ones");
                    }
                    if ring.len() < 3 {
                        continue;
                    }
                    let id = board.alloc();
                    let mut z = Zone::new(id, s.attr("name").unwrap_or(""), net, vec![l], Polygon::from_ring(ring));
                    if let Some(iso) = item.num("isolate").filter(|v| *v > 0.0) {
                        z.clearance = Some(mm(iso));
                    }
                    if let Some(w) = item.num("width").filter(|v| *v > 0.0) {
                        z.min_width = mm(w);
                    }
                    z.priority = item.num("rank").map(|r| 7 - r as i32).unwrap_or(0);
                    board.zones.insert(id, z);
                    zones += 1;
                }
                "contactref" => {}
                other => report.not_carried(format!("signal {}: <{other}>", s.attr("name").unwrap_or(""))),
            }
        }
    }
    if let Some(plain) = brd.child("plain") {
        let others = plain.children.iter().filter(|i| i.name != "hole" && !matches!(layer(i), DIMENSION | MILLING)).count();
        if others > 0 {
            report.not_carried(format!("{others} board drawings and texts outside the outline layer"));
        }
        let holes = plain.all("hole").count();
        if holes > 0 {
            report.not_carried(format!("{holes} board holes outside a package"));
        }
    }
    report.count("footprints placed", board.footprints.len());
    report.count("tracks", tracks);
    report.count("vias", vias);
    report.count("zones", zones);
    report.count("copper layers", copper.len());
    Ok(board)
}
