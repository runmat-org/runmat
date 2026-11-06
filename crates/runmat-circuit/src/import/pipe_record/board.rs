//! Pipe-record boards: the named records of a board document.
//!
//! - The outline from the `Board` record's vertices (`VX0`, `VY0`, `KIND0` …; an arc vertex
//!   gives `VCX`, `VCY`, `R`, `SA`, `EA`), else from tracks and arcs on the keep-out layer.
//! - Copper layers `TOP`, `MID1`… (those the board uses) and `BOTTOM`.
//! - Each `Component` becomes a footprint: the pads and overlay drawings that name it
//!   (`COMPONENT=<index>`), taken back to its own frame from where the file places them,
//!   mirrored on the bottom layer. Footprints with the same pattern and pads share a file.
//! - Tracks, arcs, vias, polygons and fills carry their nets; `Class` records give the nets'
//!   classes; `Clearance`, `Width` and `HoleSize` rules give the board limits (scope `All`) or
//!   a net class's parameters (scope `InNetClass('<name>')`).

use super::Record;
use crate::geom::poly::{Polygon, Ring};
use crate::geom::{Coord, Point, Transform};
use crate::import::{file_safe, Report};
use crate::layout::{Board, BoardGraphic, BoardText, Footprint, IslandPolicy, Layer, NetClassRules, NetId, PadConnect, Side, Stackup, Track, TrackGeom, Via, Zone};
use crate::library::{CopperSpan, Drill, Graphic, PadDef, PadShape, Package};
use std::collections::{BTreeMap, BTreeSet};

/// What a board import makes.
pub struct BoardRead {
    pub board: Board,
    /// Footprint key → package, written as `<key>.kicad_mod`.
    pub packages: BTreeMap<String, Package>,
    /// Designator → footprint key.
    pub footprints: BTreeMap<String, String>,
}

/// File coordinates (board units, y up) to RunMat ones, relative to the outline's corner.
#[derive(Clone, Copy)]
struct Frame {
    origin: (Coord, Coord),
}

impl Frame {
    fn point(&self, r: &Record, x: &str, y: &str) -> Point {
        Point::new(r.len(x) - self.origin.0, r.len(y) - self.origin.1)
    }
    fn at(&self, x: Coord, y: Coord) -> Point {
        Point::new(x - self.origin.0, y - self.origin.1)
    }
}

/// From board coordinates to a footprint's own (the inverse of its placement).
struct Local {
    at: Point,
    rotation: f64,
    mirror: bool,
}

impl Local {
    fn point(&self, p: Point) -> Point {
        let q = Transform::rotation(-self.rotation).apply(p - self.at);
        if self.mirror {
            Point::new(-q.x, q.y)
        } else {
            q
        }
    }
    fn angle(&self, world: f64) -> f64 {
        let a = world - self.rotation;
        (if self.mirror { 180.0 - a } else { a }).rem_euclid(360.0)
    }
}

fn layer(r: &Record) -> String {
    r.str("LAYER").trim().to_ascii_uppercase()
}

/// Inner copper layer number (`MID3` → 3).
fn mid(l: &str) -> Option<u32> {
    l.strip_prefix("MID").and_then(|n| n.parse().ok())
}

/// The copper layers the board uses, top to bottom, as the file names them.
///
/// The Board record chains its layer stack (`LAYER1NEXT=39`, `LAYER39NEXT=40` …, by layer
/// number) from the top copper to the bottom; its signal (`MIDn`) and plane (`PLANEn`) layers
/// are the inner copper, in that order. Without a chain the inner layers are the `MIDn` the
/// items use, in number order.
fn copper_layers(records: &[Record]) -> Vec<String> {
    if let Some(chain) = records.iter().find(|r| r.kind().eq_ignore_ascii_case("Board")).and_then(layer_chain) {
        return chain;
    }
    let mut inner = BTreeSet::new();
    for r in records {
        for key in ["LAYER", "STARTLAYER", "ENDLAYER"] {
            if let Some(n) = mid(&r.str(key).trim().to_ascii_uppercase()) {
                inner.insert(n);
            }
        }
    }
    // RunMat stacks come in pairs: an odd inner count gets one more, unused.
    if inner.len() % 2 == 1 {
        let spare = (1..).find(|n| !inner.contains(n)).unwrap_or(1);
        inner.insert(spare);
    }
    let mut out = vec!["TOP".to_owned()];
    out.extend(inner.iter().map(|n| format!("MID{n}")));
    out.push("BOTTOM".into());
    out
}

/// The Board record's copper stack from `LAYER1NEXT`, when the chain reaches the bottom.
fn layer_chain(board: &Record) -> Option<Vec<String>> {
    let mut out = vec!["TOP".to_owned()];
    let mut at = 1u8;
    for _ in 0..80 {
        let next = board.str(&format!("LAYER{at}NEXT")).trim().parse::<u8>().ok()?;
        if next == 32 {
            out.push("BOTTOM".into());
            return Some(out);
        }
        if matches!(next, 2..=31 | 39..=54) {
            out.push(super::primitives::layer_name(next));
        }
        at = next;
    }
    None
}

/// Plane layer number (`PLANE2` → 2).
fn plane(l: &str) -> Option<u32> {
    l.strip_prefix("PLANE").and_then(|n| n.parse().ok())
}

/// A point at `angle` degrees on the circle about `c` of radius `r` (board units).
fn on_circle(c: (Coord, Coord), r: Coord, angle: f64) -> (Coord, Coord) {
    let a = angle.to_radians();
    (c.0 + (r as f64 * a.cos()).round() as Coord, c.1 + (r as f64 * a.sin()).round() as Coord)
}

/// An `Arc` record's start, mid and end (file coordinates): counter-clockwise from
/// `STARTANGLE` to `ENDANGLE`.
fn arc_points(r: &Record) -> ((Coord, Coord), (Coord, Coord), (Coord, Coord)) {
    let c = (r.len("LOCATION.X"), r.len("LOCATION.Y"));
    let rad = r.len("RADIUS");
    let (a0, mut a1) = (r.num("STARTANGLE"), r.num("ENDANGLE"));
    if a1 <= a0 {
        a1 += 360.0;
    }
    (on_circle(c, rad, a0), on_circle(c, rad, (a0 + a1) / 2.0), on_circle(c, rad, a1))
}

/// A record's vertices (`VX0`, `VY0` …) as file points, arcs flattened.
fn vertices(r: &Record) -> Vec<(Coord, Coord)> {
    let mut out = Vec::new();
    for i in 0.. {
        let (kx, ky) = (format!("VX{i}"), format!("VY{i}"));
        if !r.has(&kx) {
            break;
        }
        let p = (r.len(&kx), r.len(&ky));
        if r.num(&format!("KIND{i}")) as i64 == 1 && r.has(&format!("VCX{i}")) {
            let c = (r.len(&format!("VCX{i}")), r.len(&format!("VCY{i}")));
            let rad = r.len(&format!("R{i}"));
            let (a0, mut a1) = (r.num(&format!("SA{i}")), r.num(&format!("EA{i}")));
            if a1 <= a0 {
                a1 += 360.0;
            }
            let steps = (((a1 - a0) / 10.0).ceil() as usize).max(2);
            out.extend((0..=steps).map(|k| on_circle(c, rad, a0 + (a1 - a0) * k as f64 / steps as f64)));
        } else {
            out.push(p);
        }
    }
    out
}

fn pad_shape(r: &Record, w: Coord, h: Coord, report: &mut Report) -> PadShape {
    match r.str("SHAPE").trim().to_ascii_uppercase().as_str() {
        "ROUND" | "CIRCLE" if w == h => PadShape::Circle { d: w },
        "ROUND" | "CIRCLE" => PadShape::Oval { w, h },
        // The corner radius is in percent of half the smaller side.
        "ROUNDEDRECTANGLE" if r.has("CORNERRADIUS") => PadShape::RoundRect { w, h, ratio: (r.num("CORNERRADIUS") / 200.0).clamp(0.0, 0.5) },
        "ROUNDEDRECTANGLE" => PadShape::RoundRect { w, h, ratio: 0.25 },
        "OCTAGONAL" => {
            let (x, y, c) = (w / 2, h / 2, w.min(h) / 4);
            let points = vec![Point::new(-x + c, -y), Point::new(x - c, -y), Point::new(x, -y + c), Point::new(x, y - c), Point::new(x - c, y), Point::new(-x + c, y), Point::new(-x, y - c), Point::new(-x, -y + c)];
            PadShape::Custom { points, holes: vec![] }
        }
        "RECTANGLE" | "RECT" | "" => PadShape::Rect { w, h },
        other => {
            report.approximated(format!("pads of shape {other} are carried as rectangles"));
            PadShape::Rect { w, h }
        }
    }
}

/// A `Pad` record of a placed component, in the component's frame.
///
/// A slot (`SLOTSIZE`, its length, at `SLOTROTATION` in the pad's frame) runs along the
/// pad's x axis; a slot across it turns the pad a quarter turn with its sides swapped.
fn pad(r: &Record, frame: &Frame, local: &Local, side: Side, report: &mut Report) -> PadDef {
    let mut w = if r.has("XSIZE") { r.len("XSIZE") } else { r.len("TOPXSIZE") };
    let mut h = if r.has("YSIZE") { r.len("YSIZE") } else { r.len("TOPYSIZE") };
    let mut turn = 0.0;
    let slot = Some(r.len("SLOTSIZE")).filter(|s| *s > r.len("HOLESIZE"));
    if slot.is_some() {
        match r.num("SLOTROTATION").rem_euclid(180.0) {
            a if a == 0.0 => {}
            a if a == 90.0 => {
                (w, h, turn) = (h, w, 90.0);
            }
            _ => report.approximated("slots at an angle to their pad are carried along the pad's x axis"),
        }
    }
    let shape = pad_shape(r, w, h, report);
    let center = local.point(frame.point(r, "X", "Y"));
    let number = r.str("NAME").trim();
    let hole = r.len("HOLESIZE");
    let l = layer(r);
    let mut p = if l == "MULTILAYER" && hole > 0 {
        let plated = !r.has("PLATED") || r.flag("PLATED");
        if !plated && number.is_empty() {
            PadDef::npth("", center, hole)
        } else {
            let mut p = PadDef::tht(number, center, shape, hole);
            if let Some(d) = p.drill.as_mut() {
                *d = Drill { diameter: hole, slot, plated };
            }
            p
        }
    } else {
        let mut p = PadDef::smd(number, center, shape);
        let own = if side == Side::Top { "TOP" } else { "BOTTOM" };
        p.copper = if l == own { CopperSpan::Front } else { CopperSpan::Back };
        if hole > 0 {
            report.approximated("a drilled pad on one copper layer is carried without its hole");
        }
        p
    };
    p.rotation = local.angle(r.num("ROTATION") + turn);
    p
}

/// The rule scope's net class (`InNetClass('Power')` → `Power`).
fn net_class_scope(expr: &str) -> Option<String> {
    let e = expr.trim();
    let inner = e.strip_prefix("InNetClass(")?.strip_suffix(')')?;
    Some(inner.trim().trim_matches('\'').trim_matches('"').to_owned())
}

/// Reads a board document's records into a board named `name`.
pub fn read(records: &[Record], name: &str, report: &mut Report) -> Result<BoardRead, String> {
    let copper = copper_layers(records);
    let index = |l: &str| copper.iter().position(|c| c == l).map(|i| i as u8);
    // The outline: the Board record's vertices, else the keep-out layer's tracks and arcs.
    let board_rec = records.iter().find(|r| r.kind().eq_ignore_ascii_case("Board"));
    let outline_pts = board_rec.map(vertices).unwrap_or_default();
    let keepout = |r: &Record| layer(r) == "KEEPOUT" && r.index("COMPONENT").is_none();
    let mut extent: Vec<(Coord, Coord)> = outline_pts.clone();
    if extent.is_empty() {
        for r in records.iter().filter(|r| keepout(r)) {
            match r.kind() {
                "Track" => extent.extend([(r.len("X1"), r.len("Y1")), (r.len("X2"), r.len("Y2"))]),
                "Arc" => {
                    let (a, m, b) = arc_points(r);
                    extent.extend([a, m, b]);
                }
                _ => {}
            }
        }
    }
    if extent.is_empty() {
        extent = records.iter().filter(|r| r.kind() == "Component").map(|r| (r.len("X"), r.len("Y"))).collect();
    }
    let frame = Frame { origin: (extent.iter().map(|p| p.0).min().unwrap_or(0), extent.iter().map(|p| p.1).min().unwrap_or(0)) };
    let mut board = Board::new(name, Polygon::default(), Stackup::standard(copper.len() as u8, crate::geom::mm(1.6), 1.0, 0.5));

    // Nets by index, with their classes.
    let net_names: Vec<String> = records.iter().filter(|r| r.kind() == "Net").map(|r| r.str("NAME").trim().to_owned()).collect();
    let mut class_of: BTreeMap<String, String> = BTreeMap::new();
    for r in records.iter().filter(|r| r.kind() == "Class" && r.num("KIND") as i64 == 0) {
        let class = r.str("NAME").trim();
        if class.is_empty() || class.eq_ignore_ascii_case("All Nets") {
            continue;
        }
        for i in 0.. {
            let Some(m) = r.0.get(&format!("M{i}")) else { break };
            class_of.insert(m.trim().to_owned(), class.to_owned());
        }
    }
    let mut ids: BTreeMap<usize, NetId> = BTreeMap::new();
    for (i, n) in net_names.iter().enumerate() {
        if !n.is_empty() {
            ids.entry(i).or_insert_with(|| board.add_net(n, class_of.get(n).map(String::as_str)));
        }
    }
    let mut net = |board: &mut Board, r: &Record| -> Option<NetId> {
        let i = r.index("NET")?;
        let n = net_names.get(i)?;
        (!n.is_empty()).then(|| *ids.entry(i).or_insert_with(|| board.add_net(n, class_of.get(n).map(String::as_str))))
    };

    // Rules.
    for r in records.iter().filter(|r| r.kind() == "Rule") {
        let kind = r.str("RULEKIND").trim().to_owned();
        let scope = r.str("SCOPE1EXPRESSION").trim().to_owned();
        let value = match kind.as_str() {
            "Clearance" => Some(("clearance", r.len("GAP"))),
            "Width" => Some(("track_width", r.len("MINLIMIT"))),
            "HoleSize" => Some(("drill", r.len("MINLIMIT"))),
            _ => None,
        };
        let Some((key, v)) = value.filter(|(_, v)| *v > 0) else {
            report.not_carried(format!("{kind} rules"));
            continue;
        };
        if scope.is_empty() || scope.eq_ignore_ascii_case("All") {
            match key {
                "clearance" => board.limits.min_clearance = v,
                "track_width" => board.limits.min_track = v,
                _ => board.limits.min_drill = v,
            }
        } else if let Some(class) = net_class_scope(&scope).filter(|_| key != "drill") {
            let c = board.net_class_rules.entry(class).or_insert_with(|| NetClassRules { decls: vec![], source: Default::default(), order: Default::default() });
            c.decls.push((key.to_owned(), crate::board_rules::RuleValue::Len(v)));
        } else {
            report.not_carried(format!("{kind} rules scoped to {scope}"));
        }
    }

    // Components, then the items that name them.
    let comps: Vec<&Record> = records.iter().filter(|r| r.kind() == "Component").collect();
    struct Place {
        reference: String,
        value: String,
        pattern: String,
        at: Point,
        rotation: f64,
        side: Side,
        locked: bool,
        local: Local,
        pkg: Package,
        pad_nets: BTreeMap<String, NetId>,
    }
    let mut places: Vec<Place> = Vec::new();
    let mut unnamed = 0;
    for c in &comps {
        let side = if layer(c) == "BOTTOM" { Side::Bottom } else { Side::Top };
        let at = frame.point(c, "X", "Y");
        let rotation = c.num("ROTATION").rem_euclid(360.0);
        let reference = Some(c.str("SOURCEDESIGNATOR").trim().to_owned()).filter(|d| !d.is_empty()).unwrap_or_else(|| {
            unnamed += 1;
            format!("U{}", 900 + unnamed)
        });
        let pkg = Package { name: String::new(), uuid: crate::library::make::uuid("package", ""), description: String::new(), pads: vec![], courtyard: vec![], silk: vec![], fab: vec![], model: None, body: None, smd: false };
        places.push(Place { reference, value: c.str("COMMENT").trim().to_owned(), pattern: c.str("PATTERN").trim().to_owned(), at, rotation, side, locked: c.flag("LOCKED"), local: Local { at, rotation, mirror: side == Side::Bottom }, pkg, pad_nets: BTreeMap::new() });
    }
    let silk_side = |l: &str| match l {
        "TOPOVERLAY" => Some(Side::Top),
        "BOTTOMOVERLAY" => Some(Side::Bottom),
        _ => None,
    };
    let (mut segments, mut tracks, mut vias, mut zones, mut texts) = (Vec::new(), 0, 0, 0, 0);
    let mut cutouts: Vec<Ring> = Vec::new();
    if !outline_pts.is_empty() {
        let pts: Vec<Point> = outline_pts.iter().map(|p| frame.at(p.0, p.1)).collect();
        segments.extend(pts.windows(2).map(|w| (w[0], w[1])));
        if let (Some(a), Some(b)) = (pts.last(), pts.first()) {
            segments.push((*a, *b));
        }
    }
    for r in records {
        let l = layer(r);
        let owner = r.index("COMPONENT").filter(|i| *i < places.len());
        match r.kind() {
            "Pad" => {
                let Some(i) = owner else {
                    report.not_carried("free pads outside a component");
                    continue;
                };
                let side = places[i].side;
                let p = pad(r, &frame, &places[i].local, side, report);
                if let Some(n) = net(&mut board, r) {
                    places[i].pad_nets.insert(p.name.clone(), n);
                }
                places[i].pkg.pads.push(p);
            }
            "Track" | "Arc" => {
                let geom = if r.kind() == "Track" {
                    let (a, b) = (frame.point(r, "X1", "Y1"), frame.point(r, "X2", "Y2"));
                    TrackGeom::Segment { a, b }
                } else {
                    let (a, m, b) = arc_points(r);
                    TrackGeom::Arc { start: frame.at(a.0, a.1), mid: frame.at(m.0, m.1), end: frame.at(b.0, b.1) }
                };
                let width = r.len("WIDTH");
                if let Some(i) = owner {
                    // A component's own drawing: silkscreen in its frame.
                    if silk_side(&l).is_some() {
                        let lp = |p: Point| places[i].local.point(p);
                        let g = match geom {
                            TrackGeom::Segment { a, b } => Graphic::Line { points: vec![lp(a), lp(b)], width },
                            TrackGeom::Arc { start, mid, end } => Graphic::Arc { start: lp(start), mid: lp(mid), end: lp(end), width },
                        };
                        places[i].pkg.silk.push(g);
                    } else {
                        report.not_carried(format!("component drawings on layer {l}"));
                    }
                } else if keepout(r) && outline_pts.is_empty() {
                    match geom {
                        TrackGeom::Segment { a, b } => segments.push((a, b)),
                        TrackGeom::Arc { start, mid, end } => match crate::geom::shape::arc_from_three(start, mid, end) {
                            Some((c, sw)) => segments.extend(crate::geom::shape::arc_points(c, start, sw, crate::geom::ARC_TOL).windows(2).map(|w| (w[0], w[1]))),
                            None => segments.push((start, end)),
                        },
                    }
                } else if plane(&l).is_some() {
                    report.approximated("plane split lines are left out: each plane is one pour of its net over the board");
                } else if let Some(li) = index(&l) {
                    let n = net(&mut board, r);
                    let id = board.alloc();
                    board.tracks.insert(id, Track { id, net: n, layer: li, width, geom, locked: false });
                    tracks += 1;
                } else if let Some(side) = silk_side(&l) {
                    let graphic = match geom {
                        TrackGeom::Segment { a, b } => Graphic::Line { points: vec![a, b], width },
                        TrackGeom::Arc { start, mid, end } => Graphic::Arc { start, mid, end, width },
                    };
                    let id = board.alloc();
                    board.graphics.insert(id, BoardGraphic { id, layer: Layer::Silk(side), graphic });
                } else {
                    report.not_carried(format!("board drawings on layer {l}"));
                }
            }
            "Via" => {
                let id = board.alloc();
                let n = net(&mut board, r);
                let last = copper.len() as u8 - 1;
                let from = index(&r.str("STARTLAYER").trim().to_ascii_uppercase()).unwrap_or(0);
                let to = index(&r.str("ENDLAYER").trim().to_ascii_uppercase()).unwrap_or(last);
                board.vias.insert(id, Via { id, net: n, at: frame.point(r, "X", "Y"), diameter: r.len("DIAMETER"), drill: r.len("HOLESIZE"), from: from.min(to), to: from.max(to), micro: false, tented: true, locked: false });
                vias += 1;
            }
            "Polygon" | "Fill" => {
                let Some(li) = index(&l) else {
                    report.not_carried(format!("{} records on layer {l}", r.kind().to_lowercase()));
                    continue;
                };
                let ring: Vec<Point> = if r.kind() == "Fill" {
                    let (a, b) = (frame.point(r, "X1", "Y1"), frame.point(r, "X2", "Y2"));
                    if r.num("ROTATION").rem_euclid(360.0) != 0.0 {
                        report.approximated("rotated fills are carried unrotated");
                    }
                    vec![a, Point::new(b.x, a.y), b, Point::new(a.x, b.y)]
                } else {
                    vertices(r).into_iter().map(|p| frame.at(p.0, p.1)).collect()
                };
                if ring.len() < 3 {
                    continue;
                }
                let n = net(&mut board, r);
                let id = board.alloc();
                let zone_name = n.and_then(|n| board.nets.get(n.0 as usize)).map(|n| n.name.clone()).unwrap_or_default();
                board.zones.insert(id, Zone::new(id, &zone_name, n, vec![li], Polygon::from_ring(Ring::from(ring))));
                zones += 1;
            }
            "Text" => {
                if owner.is_some() {
                    continue;
                }
                let Some(side) = silk_side(&l) else {
                    report.not_carried(format!("board texts on layer {l}"));
                    continue;
                };
                let id = board.alloc();
                board.texts.insert(id, BoardText { id, layer: Layer::Silk(side), at: frame.point(r, "X", "Y"), text: r.str("TEXT").to_owned(), size: r.len("HEIGHT"), thickness: r.len("WIDTH"), angle: r.num("ROTATION"), mirrored: side == Side::Bottom || r.flag("MIRROR") });
                texts += 1;
                report.approximated("board texts are drawn in RunMat's stroke font from their start point");
            }
            "Region" => {
                let ring: Vec<Point> = vertices(r).into_iter().map(|p| frame.at(p.0, p.1)).collect();
                if r.index("POLYGON").is_some() || ring.len() < 3 {
                    // A pour's own fill: the pour is carried and filled again.
                } else if r.flag("ISBOARDCUTOUT") {
                    cutouts.push(Ring::from(ring));
                } else if owner.is_some() {
                    report.not_carried(format!("component regions on layer {l}"));
                } else if let (Some(li), 0) = (index(&l), r.num("KIND") as i64) {
                    let Some(n) = net(&mut board, r) else {
                        report.not_carried("copper regions without a net");
                        continue;
                    };
                    if r.num("HOLES") > 0.0 {
                        report.approximated("holes in copper regions are left out");
                    }
                    let id = board.alloc();
                    let zone_name = board.nets.get(n.0 as usize).map(|n| n.name.clone()).unwrap_or_default();
                    // Solid copper: joined to its pads whole, kept whatever its size, and
                    // filled before the layer's pours.
                    let mut z = Zone::new(id, &zone_name, Some(n), vec![li], Polygon::from_ring(Ring::from(ring)));
                    (z.connect, z.islands, z.priority) = (PadConnect::Solid, IslandPolicy::Keep, 1);
                    board.zones.insert(id, z);
                    zones += 1;
                    report.approximated("solid copper regions are carried as pours of their net");
                } else if index(&l).is_some() {
                    report.not_carried("pour cutouts");
                } else {
                    report.not_carried(format!("regions on layer {l}"));
                }
            }
            "Dimension" => report.not_carried("dimensions"),
            "ComponentBody" => report.not_carried("3D bodies of components"),
            "Board" | "Net" | "Class" | "Rule" | "Component" | "" => {}
            other => report.not_carried(format!("{other} records")),
        }
    }

    let mut packages: BTreeMap<String, Package> = BTreeMap::new();
    let mut footprints = BTreeMap::new();
    for mut pl in places {
        pl.pkg.smd = !pl.pkg.pads.is_empty() && pl.pkg.pads.iter().all(|p| p.kind == crate::library::PadKind::Smd);
        let base = file_safe(if pl.pattern.is_empty() { "package" } else { &pl.pattern });
        let mut key = base.clone();
        let mut n = 1;
        while packages.get(&key).is_some_and(|p| p.pads != pl.pkg.pads) {
            n += 1;
            key = format!("{base}_{n}");
        }
        pl.pkg.name = key.clone();
        pl.pkg.uuid = crate::library::make::uuid("package", &key);
        let pkg = packages.entry(key.clone()).or_insert(pl.pkg).clone();
        footprints.insert(pl.reference.clone(), key);
        let id = board.alloc();
        board.footprints.insert(id, Footprint { id, reference: pl.reference, component: None, value: pl.value, package: pkg, at: pl.at, rotation: pl.rotation, side: pl.side, locked: pl.locked, pad_nets: pl.pad_nets, fitted: true, properties: Default::default() });
    }
    if !cutouts.is_empty() {
        report.count("board cutouts", cutouts.len());
    }
    match crate::kicad::pcb::outline_from(&segments, cutouts) {
        Some(o) => board.outline = o,
        None => report.not_carried("the outline: the board has no closed outline"),
    }
    // Each plane layer is a pour of its net (`PLANE1NETNAME=GND`) over the board, below the
    // layer's other pours.
    let mut planes = 0;
    for (li, l) in copper.iter().enumerate() {
        let Some(k) = plane(l) else { continue };
        let net_name = board_rec.map(|b| b.str(&format!("PLANE{k}NETNAME")).trim().to_owned()).unwrap_or_default();
        let Some(i) = net_names.iter().position(|n| *n == net_name).filter(|_| !net_name.is_empty()) else {
            report.not_carried(format!("plane {l} without a net"));
            continue;
        };
        let n = net(&mut board, &Record([("NET".to_owned(), i.to_string())].into_iter().collect()));
        let id = board.alloc();
        let mut z = Zone::new(id, &net_name, n, vec![li as u8], board.outline.clone());
        z.priority = -1;
        board.zones.insert(id, z);
        planes += 1;
    }
    report.count("footprints placed", board.footprints.len());
    report.count("tracks", tracks);
    report.count("vias", vias);
    report.count("zones", zones);
    if planes > 0 {
        report.count("planes", planes);
    }
    report.count("board texts", texts);
    report.count("copper layers", copper.len());
    report.count("net classes", board.net_class_rules.len());
    Ok(BoardRead { board, packages, footprints })
}
