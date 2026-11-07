//! Shape-string boards (`docType` 3):
//!
//! - the outline from the outline layer (10): its tracks and arcs chained into rings, circles
//!   and rectangles as rings; the largest ring is the board and the rest cutouts;
//! - copper layers 1 (top) and 2 (bottom), with the inner layers (21 on) the board uses;
//! - each placed part (`LIB`) becomes a footprint: its pads, silkscreen and assembly drawing,
//!   taken back to the part's own frame from where the file places them, on the bottom when its
//!   pads are on the bottom copper;
//! - tracks, arcs, vias and copper areas carry their nets; the default design rules give the
//!   board's clearance, track and drill limits;
//! - solid copper regions become zones of their net; a board hole outside a part becomes a
//!   footprint of one unplated hole (`H901` …), on the board only.

use crate::layer_set::CopperIndex;
use super::{arc_mid, attributes, length, pairs, path, Document, Fields, Frame, PathStep};
use crate::geom::poly::{Polygon, Ring};
use crate::geom::{Point, Transform};
use crate::import::{file_safe, Report};
use crate::layout::{Board, BoardGraphic, BoardText, Dimension, Footprint, Layer, NetId, Side, Stackup, Track, TrackGeom, Via, Zone};
use crate::library::{CopperSpan, Drill, Graphic, PadDef, PadShape, Package};
use std::collections::{BTreeMap, BTreeSet};

/// Layer numbers of the format.
const TOP: u32 = 1;
const BOTTOM: u32 = 2;
const TOP_SILK: u32 = 3;
const BOTTOM_SILK: u32 = 4;
const OUTLINE: u32 = 10;
const MULTI: u32 = 11;
const TOP_ASSEMBLY: u32 = 13;
const BOTTOM_ASSEMBLY: u32 = 14;
const FIRST_INNER: u32 = 21;

/// What a board import makes.
pub struct BoardRead {
    pub board: Board,
    /// Footprint key → package, written as `<key>.kicad_mod`.
    pub packages: BTreeMap<String, Package>,
    /// Designator → footprint key.
    pub footprints: BTreeMap<String, String>,
}

fn layer_of(f: &Fields, i: usize) -> u32 {
    f.num(i) as u32
}

/// Copper layers used, top to bottom, as the format numbers them.
fn copper_layers(shapes: &[String]) -> Vec<u32> {
    let mut inner = BTreeSet::new();
    for s in shapes.iter().flat_map(|s| s.split("#@$")) {
        let f = Fields::of(s);
        let l = match f.kind() {
            "TRACK" | "ARC" | "COPPERAREA" => layer_of(&f, 2),
            "SOLIDREGION" => layer_of(&f, 1),
            _ => 0,
        };
        if l >= FIRST_INNER {
            inner.insert(l);
        }
    }
    let mut layers = vec![TOP];
    layers.extend(inner.iter().copied());
    // RunMat stacks come in pairs: an odd inner count gets one more, unused.
    if inner.len() % 2 == 1 {
        layers.push((FIRST_INNER..).find(|l| !inner.contains(l)).unwrap_or(FIRST_INNER));
        layers[1..].sort();
    }
    layers.push(BOTTOM);
    layers
}

/// The points an outline item passes through (canvas units), for the frame.
fn outline_points(shapes: &[String]) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for s in shapes {
        let f = Fields::of(s);
        match f.kind() {
            "TRACK" if layer_of(&f, 2) == OUTLINE => out.extend(pairs(f.str(4))),
            "ARC" if layer_of(&f, 2) == OUTLINE => {
                for step in path(f.str(4)).unwrap_or_default() {
                    match step {
                        PathStep::Move(x, y) | PathStep::Line(x, y) => out.push((x, y)),
                        PathStep::Arc { to, r, .. } => out.extend([to, (to.0 - r, to.1 + r)]),
                        PathStep::Close => {}
                    }
                }
            }
            "CIRCLE" if layer_of(&f, 5) == OUTLINE => out.push((f.num(1) - f.num(3), f.num(2) + f.num(3))),
            "RECT" if layer_of(&f, 5) == OUTLINE => out.extend([(f.num(1), f.num(2)), (f.num(1) + f.num(3), f.num(2) + f.num(4))]),
            _ => {}
        }
    }
    out
}

/// A path as points in RunMat coordinates, arcs flattened; `None` with curves the import
/// doesn't read.
fn path_points(frame: &Frame, d: &str) -> Option<Vec<Vec<Point>>> {
    let mut rings: Vec<Vec<Point>> = Vec::new();
    let mut last = (0.0, 0.0);
    for step in path(d)? {
        match step {
            PathStep::Move(x, y) => {
                rings.push(vec![frame.point(x, y)]);
                last = (x, y);
            }
            PathStep::Line(x, y) => {
                rings.last_mut()?.push(frame.point(x, y));
                last = (x, y);
            }
            PathStep::Arc { r, large, sweep, to } => {
                let ring = rings.last_mut()?;
                match arc_mid(last, to, r, large, sweep) {
                    Some(m) => {
                        let (a, mid, b) = (frame.point(last.0, last.1), frame.point(m.0, m.1), frame.point(to.0, to.1));
                        match crate::geom::shape::arc_from_three(a, mid, b) {
                            Some((c, sw)) => ring.extend(crate::geom::shape::arc_points(c, a, sw, crate::geom::ARC_TOL).into_iter().skip(1)),
                            None => ring.push(b),
                        }
                    }
                    None => ring.push(frame.point(to.0, to.1)),
                }
                last = to;
            }
            PathStep::Close => {}
        }
    }
    Some(rings)
}

/// An `ARC` item's start, mid and end (RunMat coordinates), from its `M … A …` path.
fn arc_of(frame: &Frame, d: &str) -> Option<(Point, Point, Point)> {
    let steps = path(d)?;
    let PathStep::Move(x, y) = steps.first()? else { return None };
    let PathStep::Arc { r, large, sweep, to } = steps.get(1)? else { return None };
    let m = arc_mid((*x, *y), *to, *r, *large, *sweep)?;
    Some((frame.point(*x, *y), frame.point(m.0, m.1), frame.point(to.0, to.1)))
}

/// A `DIMENSION~layer~path~id` item as a board dimension: the path draws the dimension line
/// first (one move and one line between the measured points), then its arrows and its text.
/// Answers whether it was carried.
fn dimension(f: &Fields, frame: &Frame, board: &mut Board, report: &mut Report) -> bool {
    let first = path(f.str(2)).and_then(|steps| match steps.as_slice() {
        [PathStep::Move(x0, y0), PathStep::Line(x1, y1), ..] => Some((frame.point(*x0, *y0), frame.point(*x1, *y1))),
        _ => None,
    });
    let Some((a, b)) = first.filter(|(a, b)| a != b) else {
        report.not_carried("dimensions whose path doesn't start with their line");
        return false;
    };
    let id = board.alloc();
    board.dimensions.insert(id, Dimension { id, layer: Layer::User(0), a, b, offset: 0, precision: 2 });
    report.approximated("dimensions are carried as their measured line on the drawing layer; RunMat draws their arrows and text");
    true
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

/// A pad item of a placed part, in the part's frame.
fn pad(f: &Fields, frame: &Frame, local: &Local, side: Side, report: &mut Report) -> PadDef {
    let (w, h) = (length(f.num(4)), length(f.num(5)));
    let center = local.point(frame.point(f.num(2), f.num(3)));
    let number = f.str(8);
    let shape = match f.str(1) {
        "ELLIPSE" if w == h => PadShape::Circle { d: w },
        "ELLIPSE" | "OVAL" => PadShape::Oval { w, h },
        "POLYGON" => {
            let points: Vec<Point> = pairs(f.str(10)).into_iter().map(|(x, y)| local.point(frame.point(x, y)) - center).collect();
            if points.len() >= 3 {
                PadShape::Custom { points, holes: vec![] }
            } else {
                PadShape::Rect { w, h }
            }
        }
        _ => PadShape::Rect { w, h },
    };
    let polygon = matches!(shape, PadShape::Custom { .. });
    let hole = 2.0 * f.num(9);
    let layer = layer_of(f, 6);
    let mut p = if layer == MULTI && hole > 0.0 {
        let plated = f.str(15) != "N";
        if !plated && number.is_empty() {
            PadDef::npth("", center, length(hole))
        } else {
            let mut p = PadDef::tht(number, center, shape, length(hole));
            let slot = f.num(13);
            if let Some(d) = p.drill.as_mut() {
                *d = Drill { diameter: length(hole), slot: (slot > hole).then(|| length(slot)), plated };
            }
            p
        }
    } else {
        let mut p = PadDef::smd(number, center, shape);
        let own = if side == Side::Top { TOP } else { BOTTOM };
        p.copper = if layer == own { CopperSpan::Front } else { CopperSpan::Back };
        p
    };
    if !polygon {
        p.rotation = local.angle(f.num(11));
    }
    if layer != MULTI && hole > 0.0 {
        report.approximated("a drilled pad on one copper layer is carried without its hole");
    }
    p
}

/// A drawing item of a placed part (track, arc, circle) as a graphic in the part's frame.
fn graphic(f: &Fields, frame: &Frame, local: &Local) -> Option<Graphic> {
    let lp = |x: f64, y: f64| local.point(frame.point(x, y));
    Some(match f.kind() {
        "TRACK" => Graphic::Line { points: pairs(f.str(4)).into_iter().map(|(x, y)| lp(x, y)).collect(), width: length(f.num(1)) },
        "ARC" => {
            let (a, m, b) = arc_of(frame, f.str(4))?;
            Graphic::Arc { start: local.point(a), mid: local.point(m), end: local.point(b), width: length(f.num(1)) }
        }
        "CIRCLE" => Graphic::Circle { center: lp(f.num(1), f.num(2)), radius: length(f.num(3)), width: length(f.num(4)), fill: false },
        _ => return None,
    })
}

/// The drawing layer of a part's item (`CIRCLE` names it in field 5, the others in field 2).
fn item_layer(f: &Fields) -> u32 {
    if f.kind() == "CIRCLE" {
        layer_of(f, 5)
    } else {
        layer_of(f, 2)
    }
}

/// Reads a board document into a board named `name`.
pub fn read(doc: &Document, name: &str, report: &mut Report) -> Result<BoardRead, String> {
    let shapes = &doc.shapes;
    let copper = copper_layers(shapes);
    let index = |l: u32| copper.iter().position(|c| *c == l).map(|i| i as CopperIndex);
    let mut outline = outline_points(shapes);
    if outline.is_empty() {
        outline = shapes.iter().filter_map(|s| s.split("#@$").next()).map(Fields::of).filter(|f| f.kind() == "TRACK").flat_map(|f| pairs(f.str(4))).collect();
    }
    let frame = Frame::around(outline);
    let mut board = Board::new(name, Polygon::default(), Stackup::standard(copper.len() as CopperIndex, crate::geom::mm(1.6), 1.0, 0.5));
    let rule = |k: &str| doc.json["DRCRULE"]["Default"][k].as_f64().filter(|v| *v > 0.0).map(length);
    if let Some(v) = rule("clearance") {
        board.limits.min_clearance = v;
    }
    if let Some(v) = rule("trackWidth") {
        board.limits.min_track = v;
    }
    if let Some(v) = rule("viaHoleD") {
        board.limits.min_drill = v;
    }

    let mut nets: BTreeMap<String, NetId> = BTreeMap::new();
    let mut net = |board: &mut Board, n: &str| -> Option<NetId> {
        let n = n.trim();
        (!n.is_empty()).then(|| *nets.entry(n.to_owned()).or_insert_with(|| board.add_net(n, None)))
    };
    let mut packages: BTreeMap<String, Package> = BTreeMap::new();
    let mut footprints = BTreeMap::new();
    let (mut segments, mut rings) = (Vec::new(), Vec::new());
    let (mut tracks, mut vias, mut zones, mut texts, mut dimensions) = (0, 0, 0, 0, 0);
    let mut unnamed = 0;
    let mut holes = 0;
    for s in shapes {
        let mut items = s.split("#@$");
        let f = Fields::of(items.next().unwrap_or(""));
        match f.kind() {
            "LIB" => {
                let subs: Vec<Fields> = items.map(Fields::of).collect();
                let attrs = attributes(f.str(3));
                let pkg_name = attrs.get("package").cloned().unwrap_or_else(|| "package".into());
                let bottom = subs.iter().any(|p| p.kind() == "PAD" && layer_of(p, 6) == BOTTOM) || subs.iter().any(|g| item_layer(g) == BOTTOM_SILK);
                let side = if bottom { Side::Bottom } else { Side::Top };
                let at = frame.point(f.num(1), f.num(2));
                let local = Local { at, rotation: f.num(4), mirror: bottom };
                let text = |mark: &str| subs.iter().find(|t| t.kind() == "TEXT" && t.str(1) == mark).map(|t| t.str(10).to_owned());
                let reference = text("P").filter(|r| !r.is_empty()).unwrap_or_else(|| {
                    unnamed += 1;
                    format!("U{}", 900 + unnamed)
                });
                let (silk_layer, fab_layer) = if bottom { (BOTTOM_SILK, BOTTOM_ASSEMBLY) } else { (TOP_SILK, TOP_ASSEMBLY) };
                let mut pkg = Package { name: String::new(), uuid: crate::library::make::uuid("package", ""), description: String::new(), pads: vec![], courtyard: vec![], silk: vec![], fab: vec![], model: None, body: None, smd: false };
                let mut pad_nets = BTreeMap::new();
                for sub in &subs {
                    match sub.kind() {
                        "PAD" => {
                            let p = pad(sub, &frame, &local, side, report);
                            if let Some(n) = net(&mut board, sub.str(7)) {
                                pad_nets.insert(p.name.clone(), n);
                            }
                            pkg.pads.push(p);
                        }
                        "HOLE" => pkg.pads.push(PadDef::npth("", local.point(frame.point(sub.num(1), sub.num(2))), length(sub.num(3)))),
                        "TRACK" | "ARC" | "CIRCLE" => {
                            let l = item_layer(sub);
                            if let Some(g) = graphic(sub, &frame, &local) {
                                if l == silk_layer {
                                    pkg.silk.push(g);
                                } else if l == fab_layer {
                                    pkg.fab.push(g);
                                }
                            }
                        }
                        "TEXT" => {}
                        // A dimension drawn with a part measures the board: it is a board dimension.
                        "DIMENSION" => {
                            if dimension(sub, &frame, &mut board, report) {
                                dimensions += 1;
                            }
                        }
                        "SVGNODE" => report.not_carried("3D model outlines of placed parts"),
                        other => report.not_carried(format!("{other} items in a placed part")),
                    }
                }
                pkg.smd = !pkg.pads.is_empty() && pkg.pads.iter().all(|p| p.kind == crate::library::PadKind::Smd);
                // One file per distinct package: an instance whose pads differ gets its own.
                let base = file_safe(&pkg_name);
                let mut key = base.clone();
                let mut n = 1;
                while packages.get(&key).is_some_and(|p| p.pads != pkg.pads) {
                    n += 1;
                    key = format!("{base}_{n}");
                }
                pkg.name = key.clone();
                pkg.uuid = crate::library::make::uuid("package", &key);
                let pkg = packages.entry(key.clone()).or_insert(pkg).clone();
                footprints.insert(reference.clone(), key);
                let id = board.alloc();
                board.footprints.insert(
                    id,
                    Footprint {
                        id,
                        reference,
                        component: None,
                        value: text("N").unwrap_or_default(),
                        package: pkg,
                        at,
                        rotation: f.num(4).rem_euclid(360.0),
                        side,
                        locked: f.str(7) == "1",
                        pad_nets,
                        fitted: true,
                        properties: Default::default(),
                    },
                );
            }
            "TRACK" => {
                let l = layer_of(&f, 2);
                let pts: Vec<Point> = pairs(f.str(4)).into_iter().map(|(x, y)| frame.point(x, y)).collect();
                if l == OUTLINE {
                    segments.extend(pts.windows(2).map(|w| (w[0], w[1])));
                } else if let Some(li) = index(l) {
                    let n = net(&mut board, f.str(3));
                    for w in pts.windows(2) {
                        let id = board.alloc();
                        board.tracks.insert(id, Track { id, net: n, layer: li, width: length(f.num(1)), geom: TrackGeom::Segment { a: w[0], b: w[1] }, locked: false });
                        tracks += 1;
                    }
                } else if l == TOP_SILK || l == BOTTOM_SILK {
                    let id = board.alloc();
                    let side = if l == TOP_SILK { Side::Top } else { Side::Bottom };
                    board.graphics.insert(id, BoardGraphic { id, layer: Layer::Silk(side), graphic: Graphic::Line { points: pts, width: length(f.num(1)) } });
                } else {
                    report.not_carried(format!("board drawings on layer {l}"));
                }
            }
            "ARC" => {
                let l = layer_of(&f, 2);
                let Some((a, m, b)) = arc_of(&frame, f.str(4)) else {
                    report.not_carried("arcs whose path isn't one move and one arc");
                    continue;
                };
                if l == OUTLINE {
                    match crate::geom::shape::arc_from_three(a, m, b) {
                        Some((c, sw)) => segments.extend(crate::geom::shape::arc_points(c, a, sw, crate::geom::ARC_TOL).windows(2).map(|w| (w[0], w[1]))),
                        None => segments.push((a, b)),
                    }
                } else if let Some(li) = index(l) {
                    let id = board.alloc();
                    let n = net(&mut board, f.str(3));
                    board.tracks.insert(id, Track { id, net: n, layer: li, width: length(f.num(1)), geom: TrackGeom::Arc { start: a, mid: m, end: b }, locked: false });
                    tracks += 1;
                } else {
                    report.not_carried(format!("board arcs on layer {l}"));
                }
            }
            "CIRCLE" if layer_of(&f, 5) == OUTLINE => rings.push(crate::geom::poly::circle_ring(frame.point(f.num(1), f.num(2)), length(f.num(3)), crate::geom::ARC_TOL)),
            "RECT" if layer_of(&f, 5) == OUTLINE => {
                let (a, b) = (frame.point(f.num(1), f.num(2)), frame.point(f.num(1) + f.num(3), f.num(2) + f.num(4)));
                rings.push(crate::geom::poly::rect_ring(crate::geom::Rect::new(a, b)));
            }
            "VIA" => {
                let id = board.alloc();
                let n = net(&mut board, f.str(4));
                let last = copper.len() as CopperIndex - 1;
                board.vias.insert(id, Via { id, net: n, at: frame.point(f.num(1), f.num(2)), diameter: length(f.num(3)), drill: length(2.0 * f.num(5)), from: 0, to: last, micro: false, tented: true, locked: false });
                vias += 1;
            }
            "COPPERAREA" => {
                let Some(li) = index(layer_of(&f, 2)) else { continue };
                let Some(ring) = path_points(&frame, f.str(4)).and_then(|r| r.into_iter().max_by_key(|r| r.len())) else {
                    report.not_carried("copper areas whose outline uses curves other than arcs");
                    continue;
                };
                if ring.len() < 3 {
                    continue;
                }
                let n = net(&mut board, f.str(3));
                let id = board.alloc();
                let mut z = Zone::new(id, f.str(3), n, vec![li], Polygon::from_ring(Ring::from(ring)));
                if f.num(5) > 0.0 {
                    z.clearance = Some(length(f.num(5)));
                }
                board.zones.insert(id, z);
                zones += 1;
            }
            "TEXT" => {
                let l = layer_of(&f, 7);
                if l != TOP_SILK && l != BOTTOM_SILK {
                    report.not_carried(format!("board texts on layer {l}"));
                    continue;
                }
                let id = board.alloc();
                let side = if l == TOP_SILK { Side::Top } else { Side::Bottom };
                board.texts.insert(
                    id,
                    BoardText { id, layer: Layer::Silk(side), at: frame.point(f.num(2), f.num(3)), text: f.str(10).to_owned(), size: length(f.num(9)), thickness: length(f.num(4)), angle: f.num(5), mirrored: side == Side::Bottom },
                );
                texts += 1;
                report.approximated("board texts are drawn in RunMat's stroke font from their start point");
            }
            "HOLE" => {
                // A board hole: a footprint of one unplated hole, on the board only.
                let d = length(f.num(3));
                let key = file_safe(&format!("hole_{:.3}mm", crate::geom::to_mm(d)));
                let pkg = Package { name: key.clone(), uuid: crate::library::make::uuid("package", &key), description: String::new(), pads: vec![PadDef::npth("", Point::ORIGIN, d)], courtyard: vec![], silk: vec![], fab: vec![], model: None, body: None, smd: false };
                let pkg = packages.entry(key).or_insert(pkg).clone();
                holes += 1;
                let id = board.alloc();
                board.footprints.insert(id, Footprint { id, reference: format!("H{}", 900 + holes), component: None, value: String::new(), package: pkg, at: frame.point(f.num(1), f.num(2)), rotation: 0.0, side: Side::Top, locked: true, pad_nets: BTreeMap::new(), fitted: true, properties: Default::default() });
            }
            "SOLIDREGION" => {
                // A solid copper region is a zone of its net, filled where nothing else is.
                let (Some(li), "solid") = (index(layer_of(&f, 1)), f.str(4)) else {
                    report.not_carried(format!("solid regions of kind {} on layer {}", f.str(4), layer_of(&f, 1)));
                    continue;
                };
                let Some(ring) = path_points(&frame, f.str(3)).and_then(|r| r.into_iter().max_by_key(|r| r.len())).filter(|r| r.len() >= 3) else {
                    report.not_carried("solid regions whose outline uses curves other than arcs");
                    continue;
                };
                let n = net(&mut board, f.str(2));
                let id = board.alloc();
                board.zones.insert(id, Zone::new(id, f.str(2), n, vec![li], Polygon::from_ring(Ring::from(ring))));
                zones += 1;
                report.approximated("solid copper regions are carried as zones of their net");
            }
            "DIMENSION" => {
                if dimension(&f, &frame, &mut board, report) {
                    dimensions += 1;
                }
            }
            "" => {}
            other => report.not_carried(format!("{other} items")),
        }
    }
    match crate::kicad::pcb::outline_from(&segments, rings) {
        Some(o) => board.outline = o,
        None => report.not_carried("the outline: the outline layer has no closed ring"),
    }
    report.count("footprints placed", board.footprints.len() - holes);
    report.count("board holes", holes);
    report.count("tracks", tracks);
    if dimensions > 0 {
        report.count("dimensions", dimensions);
    }
    report.count("vias", vias);
    report.count("zones", zones);
    report.count("board texts", texts);
    report.count("copper layers", copper.len());
    Ok(BoardRead { board, packages, footprints })
}
