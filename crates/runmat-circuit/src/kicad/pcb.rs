//! `.kicad_pcb` ⇄ [`Board`].
//!
//! [`read`] maps the documented top-level items: `general` (thickness), `layers`,
//! `setup`/`stackup`, `net`, `footprint`, `segment`, `arc`, `via` (through, `blind`,
//! `micro`), `zone` (copper pours with priority, clearance, minimum thickness, thermal
//! settings, island removal, filled polygons; keepout zones become [`Keepout`]s), and the
//! `Edge.Cuts` graphics that form the outline and cutouts, board text (`gr_text`) and
//! graphics (`gr_line`, `gr_rect`, `gr_circle`, `gr_arc`, `gr_poly`) on the layers RunMat
//! models, aligned dimensions, fiducials (one-pad footprints from a `Fiducial` library entry),
//! and the board-level [`SYNC_PROPERTY`] holding the sync record (KiCad 8 and 9 read a
//! top-level `(property "name" "value")` as a board property and write it back). Design rules
//! travel in the project's `.kicad_dru` ([`super::dru`]). Everything else (other dimension
//! types, groups, plot settings, generator data, unknown future tokens) stays in the tree.
//!
//! [`KicadPcb::write`] walks the original tree: each mapped item that is unchanged is written
//! verbatim; an edited item is regenerated *onto its original node*, replacing only the
//! children we model; deleted items are dropped; new items are appended. A file read and
//! written without edits is token-identical.

use super::footprint;
use super::sexpr::{self, SExpr};
use super::{at_of, copper_index, copper_layer, len, mm_of, pts_node, uuid_node, xy};
use crate::geom::poly::{area2, PolySet, Polygon, Ring};
use crate::geom::{mm, Coord, Point};
use crate::layout::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Footprint(ItemId),
    Track(ItemId),
    Via(ItemId),
    Zone(ItemId),
    Keepout(ItemId),
    Text(ItemId),
    Graphic(ItemId),
    Dimension(ItemId),
    Fiducial(ItemId),
    Edge,
    Net,
    Setup,
    Sync,
}

/// The board-level property that carries the board's sync record (the `.circuit.layout`
/// `sync { }` section, as text), so a board exported and re-imported keeps its ECO baseline,
/// comparator settings and footprint links.
pub const SYNC_PROPERTY: &str = "RUNMAT_SYNC";

/// The section in `text` (the `sync { … }` statement), or why it does not read.
fn read_sync_property(text: &str) -> Result<crate::syntax::layout::SyncSection, String> {
    let parsed = crate::syntax::layout::parse_layout("", &format!("layout kicad {{\n{text}\n}}\n"));
    match (parsed.sync, parsed.diagnostics.first()) {
        (Some(section), None) => Ok(section),
        (_, Some(d)) => Err(d.message.clone()),
        (None, None) => Ok(Default::default()),
    }
}

fn sync_property(board: &Board) -> Option<SExpr> {
    let section = crate::syntax::layout::SyncSection::of(board);
    (section != Default::default()).then(|| {
        SExpr::list(
            "property",
            vec![
                SExpr::str(SYNC_PROPERTY),
                SExpr::str(&crate::syntax::layout::print_sync(&section, 0)),
            ],
        )
    })
}

/// A board read from (or destined for) a KiCad file.
#[derive(Clone, Debug)]
pub struct KicadPcb {
    pub tree: SExpr,
    pub board: Board,
    origin: BTreeMap<usize, Key>,
    snapshot: Board,
    net_codes: Codes,
    pub warnings: Vec<String>,
}

fn layer_names(n: u8) -> Vec<(u8, String)> {
    (0..n).map(|i| (i, copper_layer(i, n))).collect()
}

fn read_stackup(setup: Option<&SExpr>, n_copper: u8, thickness: Coord) -> Stackup {
    let Some(st) = setup.and_then(|s| s.child("stackup")) else {
        return Stackup::standard(n_copper.max(2), thickness, 1.0, 0.5);
    };
    let mut layers = Vec::new();
    for l in st.children("layer") {
        let name = l.arg(0).unwrap_or("").to_string();
        let ty = l.child_arg("type").unwrap_or("");
        let t = mm_of(l.child("thickness")).unwrap_or(0);
        let mut m = match ty {
            "copper" => Material::copper(),
            t if t.contains("Mask") => Material::solder_mask(),
            _ => Material::fr4(),
        };
        if let Some(mat) = l.child_arg("material") {
            m.name = mat.to_string();
        }
        if let Some(e) = l.child_f("epsilon_r") {
            m.dk = e;
        }
        if let Some(d) = l.child_f("loss_tangent") {
            m.df = d;
        }
        let kind = match ty {
            "copper" => StackKind::Copper {
                weight_oz: t as f64 / 34_790.0,
            },
            "core" => StackKind::Dielectric { core: true },
            "prepreg" => StackKind::Dielectric { core: false },
            t if t.contains("Mask") => StackKind::SolderMask,
            t if t.contains("Silk") => StackKind::Silkscreen,
            t if t.contains("Paste") => StackKind::Paste,
            _ => StackKind::Dielectric { core: true },
        };
        if matches!(kind, StackKind::Silkscreen | StackKind::Paste) {
            continue;
        }
        layers.push(StackLayer {
            name,
            kind,
            thickness: t,
            material: m,
        });
    }
    Stackup {
        name: "rigid".into(),
        layers,
        finish: st.child_arg("copper_finish").unwrap_or("None").to_string(),
    }
}

fn write_stackup(s: &Stackup) -> SExpr {
    let mut items = Vec::new();
    for l in &s.layers {
        let mut v = vec![SExpr::str(&l.name)];
        let ty = match l.kind {
            StackKind::Copper { .. } => "copper".to_string(),
            StackKind::Dielectric { core: true } => "core".into(),
            StackKind::Dielectric { core: false } => "prepreg".into(),
            StackKind::SolderMask => if l.name.starts_with('B') {
                "Bottom Solder Mask"
            } else {
                "Top Solder Mask"
            }
            .into(),
            StackKind::Silkscreen => "Top Silk Screen".into(),
            StackKind::Paste => "Top Solder Paste".into(),
        };
        v.push(SExpr::list("type", vec![SExpr::str(&ty)]));
        v.push(SExpr::list("thickness", vec![len(l.thickness)]));
        if let StackKind::Dielectric { .. } = l.kind {
            v.push(SExpr::list("material", vec![SExpr::str(&l.material.name)]));
            v.push(SExpr::list("epsilon_r", vec![SExpr::num(l.material.dk)]));
            v.push(SExpr::list("loss_tangent", vec![SExpr::num(l.material.df)]));
        }
        items.push(SExpr::list("layer", v));
    }
    items.push(SExpr::list("copper_finish", vec![SExpr::str(&s.finish)]));
    items.push(SExpr::list(
        "dielectric_constraints",
        vec![SExpr::sym("no")],
    ));
    SExpr::list("stackup", items)
}

/// Chains Edge.Cuts segments into rings: the largest is the outline, the rest are cutouts.
fn outline_from(segments: &[(Point, Point)], rings: Vec<Ring>) -> Option<Polygon> {
    let mut rings = rings;
    let mut used = vec![false; segments.len()];
    let tol = 1000;
    for s in 0..segments.len() {
        if used[s] {
            continue;
        }
        used[s] = true;
        let mut ring = vec![segments[s].0, segments[s].1];
        loop {
            let end = *ring.last().unwrap();
            if end.dist(ring[0]) <= tol as f64 && ring.len() > 2 {
                ring.pop();
                break;
            }
            let next = (0..segments.len()).find(|&k| {
                !used[k]
                    && (segments[k].0.dist(end) <= tol as f64
                        || segments[k].1.dist(end) <= tol as f64)
            });
            match next {
                Some(k) => {
                    used[k] = true;
                    let (a, b) = segments[k];
                    ring.push(if a.dist(end) <= tol as f64 { b } else { a });
                }
                None => break,
            }
        }
        if ring.len() >= 3 {
            rings.push(ring);
        }
    }
    rings.sort_by_key(|r| std::cmp::Reverse(area2(r).abs()));
    let mut it = rings.into_iter();
    let mut outer = it.next()?;
    if area2(&outer) < 0 {
        outer.reverse();
    }
    let holes = it
        .map(|mut h| {
            if area2(&h) > 0 {
                h.reverse();
            }
            h
        })
        .collect();
    Some(Polygon { outer, holes })
}

fn net_of(node: &SExpr, codes: &BTreeMap<String, NetId>) -> Option<NetId> {
    let n = node.child("net")?;
    let key = n.arg(1).or(n.arg(0))?;
    codes
        .get(key)
        .copied()
        .or_else(|| codes.get(n.arg(0)?).copied())
}

/// Reads a board.
pub fn read(text: &str) -> Result<KicadPcb, String> {
    let tree = sexpr::parse(text).map_err(|e| e.to_string())?;
    if tree.head() != Some("kicad_pcb") {
        return Err("not a kicad_pcb file".into());
    }
    let mut warnings = Vec::new();
    // Copper layer count from the layer table (signal/power/mixed/jumper roles).
    let n_copper = tree
        .child("layers")
        .map(|l| {
            l.items()
                .iter()
                .skip(1)
                .filter(|x| x.arg(0).is_some_and(|n| n.ends_with(".Cu")))
                .count() as u8
        })
        .unwrap_or(2)
        .max(2);
    let thickness = tree
        .child("general")
        .and_then(|g| mm_of(g.child("thickness")))
        .unwrap_or(mm(1.6));
    let stackup = read_stackup(tree.child("setup"), n_copper, thickness);
    let mut board = Board::new("kicad", Polygon::default(), stackup);
    let mut codes: BTreeMap<String, NetId> = BTreeMap::new();
    let mut net_codes: Codes = BTreeMap::new();
    let mut origin = BTreeMap::new();
    let mut edge_segments = Vec::new();
    let mut edge_rings = Vec::new();
    let mut sync_text: Option<String> = None;
    for (idx, node) in tree.items().iter().enumerate().skip(1) {
        match node.head() {
            Some("property") if node.arg(0) == Some(SYNC_PROPERTY) => {
                origin.insert(idx, Key::Sync);
                sync_text = node.arg(1).map(str::to_owned);
            }
            Some("net") => {
                let code = node.arg(0).unwrap_or("0").to_string();
                let name = node.arg(1).unwrap_or("").to_string();
                origin.insert(idx, Key::Net);
                if name.is_empty() {
                    continue;
                }
                let id = board.add_net(&name, None);
                net_codes.insert(id, code.clone());
                codes.insert(code, id);
                codes.insert(name, id);
            }
            Some("setup") => {
                origin.insert(idx, Key::Setup);
            }
            Some("footprint") if fiducial_of(node, ItemId(0)).is_some() => {
                let id = board.alloc();
                if let Some(f) = fiducial_of(node, id) {
                    board.fiducials.insert(id, f);
                    origin.insert(idx, Key::Fiducial(id));
                }
            }
            Some("footprint") => {
                let f = footprint::read(node, true);
                let id = board.alloc();
                let pad_nets = f
                    .pad_nets
                    .iter()
                    .filter_map(|(p, k)| codes.get(k).map(|n| (p.clone(), *n)))
                    .collect();
                board.footprints.insert(
                    id,
                    Footprint {
                        id,
                        reference: f.reference,
                        component: f.path,
                        value: f.value,
                        package: f.package,
                        at: f.at,
                        rotation: f.rotation,
                        side: f.side,
                        locked: f.locked,
                        pad_nets,
                        fitted: true,
                        properties: f.properties,
                    },
                );
                origin.insert(idx, Key::Footprint(id));
            }
            Some("segment") | Some("arc") => {
                let layer = node
                    .child_arg("layer")
                    .and_then(|l| copper_index(l, n_copper));
                let Some(layer) = layer else {
                    warnings.push(format!("track on a non-copper layer at item {idx}"));
                    continue;
                };
                let geom = if node.head() == Some("arc") {
                    TrackGeom::Arc {
                        start: at_of(node.child("start")).0,
                        mid: at_of(node.child("mid")).0,
                        end: at_of(node.child("end")).0,
                    }
                } else {
                    TrackGeom::Segment {
                        a: at_of(node.child("start")).0,
                        b: at_of(node.child("end")).0,
                    }
                };
                let id = board.alloc();
                board.tracks.insert(
                    id,
                    Track {
                        id,
                        net: net_of(node, &codes),
                        layer,
                        width: mm_of(node.child("width")).unwrap_or(mm(0.25)),
                        geom,
                        locked: node.has_flag("locked") || node.child("locked").is_some(),
                    },
                );
                origin.insert(idx, Key::Track(id));
            }
            Some("via") => {
                let ls: Vec<u8> = node
                    .child("layers")
                    .map(|l| {
                        l.items()
                            .iter()
                            .skip(1)
                            .filter_map(|x| x.atom())
                            .filter_map(|n| copper_index(n, n_copper))
                            .collect()
                    })
                    .unwrap_or_default();
                let (from, to) = match (ls.iter().min(), ls.iter().max()) {
                    (Some(a), Some(b)) => (*a, *b),
                    _ => (0, n_copper - 1),
                };
                let id = board.alloc();
                board.vias.insert(
                    id,
                    Via {
                        id,
                        net: net_of(node, &codes),
                        at: at_of(node.child("at")).0,
                        diameter: mm_of(node.child("size")).unwrap_or(mm(0.6)),
                        drill: mm_of(node.child("drill")).unwrap_or(mm(0.3)),
                        from,
                        to,
                        micro: node.has_flag("micro"),
                        tented: true,
                        locked: node.has_flag("locked") || node.child("locked").is_some(),
                    },
                );
                origin.insert(idx, Key::Via(id));
            }
            Some("zone") => {
                let mut layers: Vec<u8> = Vec::new();
                if let Some(l) = node.child_arg("layer") {
                    layers.extend(copper_index(l, n_copper));
                }
                if let Some(ls) = node.child("layers") {
                    for x in ls.items().iter().skip(1).filter_map(|x| x.atom()) {
                        if x == "*.Cu" {
                            layers.extend(0..n_copper);
                        } else if let Some(i) = copper_index(x, n_copper) {
                            layers.push(i);
                        }
                    }
                }
                layers.sort();
                layers.dedup();
                let outline = super::pts_of(node.child("polygon").and_then(|p| p.child("pts")));
                let poly = Polygon::from_ring(outline);
                let id = board.alloc();
                let name = node.child_arg("name").unwrap_or("").to_string();
                if let Some(k) = node.child("keepout") {
                    let na = |h: &str| k.child_arg(h) == Some("not_allowed");
                    board.keepouts.insert(
                        id,
                        Keepout {
                            id,
                            name,
                            outline: poly,
                            layers,
                            rules: KeepoutRules {
                                tracks: na("tracks"),
                                vias: na("vias"),
                                pads: na("pads"),
                                pours: na("copperpour"),
                                footprints: na("footprints"),
                                max_height: None,
                            },
                        },
                    );
                    origin.insert(idx, Key::Keepout(id));
                    continue;
                }
                let mut z = Zone::new(id, &name, net_of(node, &codes), layers, poly);
                z.priority = node.child_f("priority").unwrap_or(0.0) as i32;
                if let Some(cp) = node.child("connect_pads") {
                    z.clearance = mm_of(cp.child("clearance"));
                    z.connect = match cp.arg(0) {
                        Some("yes") => PadConnect::Solid,
                        Some("no") => PadConnect::None,
                        Some("thru_hole_only") => PadConnect::ThermalThtOnly,
                        _ => PadConnect::Thermal,
                    };
                }
                z.min_width = mm_of(node.child("min_thickness")).unwrap_or(mm(0.25));
                if let Some(f) = node.child("fill") {
                    z.thermal_gap = mm_of(f.child("thermal_gap")).unwrap_or(z.thermal_gap);
                    z.spoke_width = mm_of(f.child("thermal_bridge_width")).unwrap_or(z.spoke_width);
                    z.islands = match f.child_arg("island_removal_mode") {
                        Some("1") => IslandPolicy::Keep,
                        Some("2") => IslandPolicy::RemoveBelow(
                            f.child_f("island_area_min").unwrap_or(10.0) * 1e12,
                        ),
                        _ => IslandPolicy::Remove,
                    };
                }
                for fp in node.children("filled_polygon") {
                    let Some(l) = fp
                        .child_arg("layer")
                        .and_then(|l| copper_index(l, n_copper))
                    else {
                        continue;
                    };
                    let ring = super::pts_of(fp.child("pts"));
                    if ring.len() >= 3 {
                        let set = z.fill.entry(l).or_default();
                        let merged = set.union(&PolySet::from_rings(vec![ring]));
                        *set = merged;
                    }
                }
                board.zones.insert(id, z);
                origin.insert(idx, Key::Zone(id));
            }
            Some(h) if h.starts_with("gr_") && node.child_arg("layer") == Some("Edge.Cuts") => {
                origin.insert(idx, Key::Edge);
                let p = |k: &str| at_of(node.child(k)).0;
                match h {
                    "gr_line" => edge_segments.push((p("start"), p("end"))),
                    "gr_arc" => {
                        let (s, m, e) = (p("start"), p("mid"), p("end"));
                        if let Some((c, sw)) = crate::geom::shape::arc_from_three(s, m, e) {
                            let pts =
                                crate::geom::shape::arc_points(c, s, sw, crate::geom::ARC_TOL);
                            for w in pts.windows(2) {
                                edge_segments.push((w[0], w[1]));
                            }
                        }
                    }
                    "gr_rect" => edge_rings.push(crate::geom::poly::rect_ring(
                        crate::geom::Rect::new(p("start"), p("end")),
                    )),
                    "gr_circle" => {
                        let c = p("center");
                        edge_rings.push(crate::geom::poly::circle_ring(
                            c,
                            c.dist(p("end")) as Coord,
                            crate::geom::ARC_TOL,
                        ));
                    }
                    "gr_poly" => edge_rings.push(super::pts_of(node.child("pts"))),
                    _ => {}
                }
            }
            Some("gr_text") => {
                let id = ItemId(board.next_id);
                if let Some(t) = text_of(node, id, n_copper) {
                    board.alloc();
                    board.texts.insert(id, t);
                    origin.insert(idx, Key::Text(id));
                }
            }
            Some(h) if h.starts_with("gr_") => {
                let id = ItemId(board.next_id);
                if let Some(g) = graphic_of(node, id, n_copper) {
                    board.alloc();
                    board.graphics.insert(id, g);
                    origin.insert(idx, Key::Graphic(id));
                }
            }
            Some("dimension") => {
                let id = ItemId(board.next_id);
                if let Some(d) = dimension_of(node, id, n_copper) {
                    board.alloc();
                    board.dimensions.insert(id, d);
                    origin.insert(idx, Key::Dimension(id));
                }
            }
            _ => {}
        }
    }
    match outline_from(&edge_segments, edge_rings) {
        Some(o) => board.outline = o,
        None => warnings.push("no closed Edge.Cuts outline".into()),
    }
    if let Some(text) = sync_text {
        match read_sync_property(&text) {
            Ok(section) => {
                // The links replace the file's own symbol paths: they are the ECO's identity.
                let missing = section.install(&mut board);
                for r in missing {
                    warnings.push(format!(
                        "the sync record links {r}, which the board does not have"
                    ));
                }
            }
            Err(e) => warnings.push(format!("the board's sync record does not read: {e}")),
        }
    }
    let snapshot = board.clone();
    Ok(KicadPcb {
        tree,
        board,
        origin,
        snapshot,
        net_codes,
        warnings,
    })
}

type Codes = BTreeMap<NetId, String>;

/// A non-copper layer by its KiCad name; copper layers by [`copper_index`].
fn layer_of(name: &str, n: u8) -> Option<Layer> {
    if let Some(i) = copper_index(name, n) {
        return Some(Layer::Copper(i));
    }
    let side = |f: bool| if f { Side::Top } else { Side::Bottom };
    Some(match name {
        "F.Mask" | "B.Mask" => Layer::Mask(side(name.starts_with('F'))),
        "F.Paste" | "B.Paste" => Layer::Paste(side(name.starts_with('F'))),
        "F.SilkS" | "B.SilkS" => Layer::Silk(side(name.starts_with('F'))),
        "F.Fab" | "B.Fab" => Layer::Fab(side(name.starts_with('F'))),
        "F.CrtYd" | "B.CrtYd" => Layer::Courtyard(side(name.starts_with('F'))),
        "Edge.Cuts" => Layer::Edge,
        "Dwgs.User" => Layer::User(0),
        _ => Layer::User(
            name.strip_prefix("User.")?
                .parse()
                .ok()
                .filter(|k| *k > 0)?,
        ),
    })
}

/// The KiCad name of a layer.
fn layer_name(l: Layer, n: u8) -> String {
    let f =
        |s: Side, front: &str, back: &str| (if s == Side::Top { front } else { back }).to_owned();
    match l {
        Layer::Copper(i) => copper_layer(i, n),
        Layer::Mask(s) => f(s, "F.Mask", "B.Mask"),
        Layer::Paste(s) => f(s, "F.Paste", "B.Paste"),
        Layer::Silk(s) => f(s, "F.SilkS", "B.SilkS"),
        Layer::Fab(s) => f(s, "F.Fab", "B.Fab"),
        Layer::Courtyard(s) => f(s, "F.CrtYd", "B.CrtYd"),
        Layer::Edge => "Edge.Cuts".into(),
        Layer::User(0) => "Dwgs.User".into(),
        Layer::User(k) => format!("User.{k}"),
    }
}

fn font_size(node: &SExpr) -> (Option<Coord>, Option<Coord>) {
    let font = node.child("effects").and_then(|e| e.child("font"));
    (
        font.and_then(|f| mm_of(f.child("size"))),
        font.and_then(|f| mm_of(f.child("thickness"))),
    )
}

fn text_of(node: &SExpr, id: ItemId, n: u8) -> Option<BoardText> {
    let layer = layer_of(node.child_arg("layer")?, n)?;
    let (at, angle) = at_of(node.child("at"));
    let (size, thickness) = font_size(node);
    let size = size.unwrap_or(mm(1.0));
    let mirrored = node
        .child("effects")
        .and_then(|e| e.child("justify"))
        .is_some_and(|j| j.items().iter().skip(1).any(|x| x.atom() == Some("mirror")));
    Some(BoardText {
        id,
        layer,
        at,
        text: node.arg(0)?.to_owned(),
        size,
        thickness: thickness.unwrap_or(size / 7),
        angle,
        mirrored,
    })
}

fn text_node(t: &BoardText, n: u8) -> SExpr {
    let mut effects = vec![SExpr::list(
        "font",
        vec![
            SExpr::list("size", vec![len(t.size), len(t.size)]),
            SExpr::list("thickness", vec![len(t.thickness)]),
        ],
    )];
    if t.mirrored {
        effects.push(SExpr::list("justify", vec![SExpr::sym("mirror")]));
    }
    SExpr::list(
        "gr_text",
        vec![
            SExpr::str(&t.text),
            SExpr::list("at", vec![len(t.at.x), len(-t.at.y), SExpr::num(t.angle)]),
            SExpr::list("layer", vec![SExpr::str(&layer_name(t.layer, n))]),
            uuid_node(&format!("x{}", t.id.0)),
            SExpr::list("effects", effects),
        ],
    )
}

fn graphic_of(node: &SExpr, id: ItemId, n: u8) -> Option<BoardGraphic> {
    let (layer, graphic) = footprint::graphic(node, &|p| p)?;
    Some(BoardGraphic {
        id,
        layer: layer_of(&layer, n)?,
        graphic,
    })
}

fn board_graphic_node(g: &BoardGraphic, n: u8) -> SExpr {
    let mut node = footprint::graphic_node(
        &g.graphic,
        &layer_name(g.layer, n),
        &|p| p,
        &format!("g{}", g.id.0),
    );
    if let Some(head) = node.items_mut().and_then(|v| v.first_mut()) {
        if let Some(name) = head.atom() {
            *head = SExpr::sym(&name.replacen("fp_", "gr_", 1));
        }
    }
    node
}

/// An aligned dimension (`(dimension (type aligned) … (pts a b) (height h))`); other types stay
/// in the tree.
fn dimension_of(node: &SExpr, id: ItemId, n: u8) -> Option<Dimension> {
    if node.child_arg("type") != Some("aligned") {
        return None;
    }
    let pts = super::pts_of(node.child("pts"));
    let (a, b) = (*pts.first()?, *pts.get(1)?);
    Some(Dimension {
        id,
        layer: layer_of(node.child_arg("layer")?, n)?,
        a,
        b,
        // KiCad's height is measured to the left of a → b with y down: RunMat's offset sign.
        offset: -mm_of(node.child("height")).unwrap_or(0),
        precision: node
            .child("format")
            .and_then(|f| f.child_f("precision"))
            .map_or(2, |p| p as u8),
    })
}

fn dimension_node(d: &Dimension, n: u8) -> SExpr {
    SExpr::list(
        "dimension",
        vec![
            SExpr::list("type", vec![SExpr::sym("aligned")]),
            SExpr::list("layer", vec![SExpr::str(&layer_name(d.layer, n))]),
            uuid_node(&format!("d{}", d.id.0)),
            SExpr::list("pts", vec![xy("xy", d.a), xy("xy", d.b)]),
            SExpr::list("height", vec![len(-d.offset)]),
            SExpr::list(
                "format",
                vec![
                    SExpr::list("units", vec![SExpr::sym("2")]),
                    SExpr::list("units_format", vec![SExpr::sym("1")]),
                    SExpr::list("precision", vec![SExpr::sym(&d.precision.to_string())]),
                ],
            ),
            SExpr::list(
                "style",
                vec![
                    SExpr::list("thickness", vec![SExpr::num(0.15)]),
                    SExpr::list("arrow_length", vec![SExpr::num(1.27)]),
                    SExpr::list("text_position_mode", vec![SExpr::sym("0")]),
                    SExpr::list("extension_height", vec![SExpr::num(0.58642)]),
                    SExpr::list("keep_text_aligned", vec![SExpr::sym("yes")]),
                ],
            ),
        ],
    )
}

/// A fiducial: a footprint from a `Fiducial` library entry with one round pad and no net. The
/// mask opening is the pad plus twice its `solder_mask_margin` (the pad itself without one).
fn fiducial_of(node: &SExpr, id: ItemId) -> Option<Fiducial> {
    let lib = node.arg(0)?;
    if !lib.to_ascii_lowercase().contains("fiducial") {
        return None;
    }
    let pads: Vec<&SExpr> = node.children("pad").collect();
    let [pad] = pads.as_slice() else { return None };
    if pad.child("net").is_some() || pad.arg(2) != Some("circle") {
        return None;
    }
    let diameter = mm_of(pad.child("size"))?;
    let margin = mm_of(pad.child("solder_mask_margin")).unwrap_or(0);
    let side = if node.child_arg("layer") == Some("B.Cu") {
        Side::Bottom
    } else {
        Side::Top
    };
    let name = node
        .children("property")
        .find(|p| p.arg(0) == Some("Reference"))
        .and_then(|p| p.arg(1))
        .or_else(|| {
            node.children("fp_text")
                .find(|t| t.arg(0) == Some("reference"))
                .and_then(|t| t.arg(1))
        })
        .unwrap_or("FID")
        .to_owned();
    Some(Fiducial {
        id,
        name,
        at: at_of(node.child("at")).0,
        side,
        diameter,
        opening: diameter + 2 * margin,
        locked: node.has_flag("locked") || node.child("locked").is_some(),
    })
}

fn fiducial_node(f: &Fiducial) -> SExpr {
    let (cu, mask, silk) = if f.side == Side::Top {
        ("F.Cu", "F.Mask", "F.SilkS")
    } else {
        ("B.Cu", "B.Mask", "B.SilkS")
    };
    let mm_text = crate::syntax::values::mm_text;
    let mut items = vec![
        SExpr::str(&format!(
            "Fiducial:Fiducial_{}mm_Mask{}mm",
            mm_text(f.diameter),
            mm_text(f.opening)
        )),
        SExpr::list("layer", vec![SExpr::str(cu)]),
        uuid_node(&format!("f{}", f.id.0)),
        SExpr::list("at", vec![len(f.at.x), len(-f.at.y)]),
        SExpr::list(
            "property",
            vec![
                SExpr::str("Reference"),
                SExpr::str(&f.name),
                SExpr::list("at", vec![SExpr::num(0.0), SExpr::num(-1.5)]),
                SExpr::list("layer", vec![SExpr::str(silk)]),
                SExpr::list("hide", vec![SExpr::sym("yes")]),
            ],
        ),
        SExpr::list(
            "attr",
            vec![SExpr::sym("smd"), SExpr::sym("exclude_from_bom")],
        ),
        SExpr::list(
            "pad",
            vec![
                SExpr::str("1"),
                SExpr::sym("smd"),
                SExpr::sym("circle"),
                SExpr::list("at", vec![SExpr::num(0.0), SExpr::num(0.0)]),
                SExpr::list("size", vec![len(f.diameter), len(f.diameter)]),
                SExpr::list("layers", vec![SExpr::str(cu), SExpr::str(mask)]),
                SExpr::list(
                    "solder_mask_margin",
                    vec![len((f.opening - f.diameter) / 2)],
                ),
            ],
        ),
    ];
    if f.locked {
        items.insert(1, SExpr::sym("locked"));
    }
    SExpr::list("footprint", items)
}

fn net_ref(n: Option<NetId>, codes: &Codes) -> SExpr {
    match n.and_then(|id| codes.get(&id)) {
        Some(c) => SExpr::list("net", vec![SExpr::sym(c)]),
        None => SExpr::list("net", vec![SExpr::sym("0")]),
    }
}

fn merge(original: Option<&SExpr>, fresh: SExpr, modeled: &[&str]) -> SExpr {
    // Keep the original node's unknown children (uuid, tstamp, locked, future tokens), in
    // their places; take modelled children and atoms from the fresh node.
    let Some(orig) = original else { return fresh };
    let mut out = vec![orig.items()[0].clone()];
    // Atom arguments (flags such as `blind`, names) come from the fresh node.
    out.extend(
        fresh
            .items()
            .iter()
            .skip(1)
            .filter(|x| x.atom().is_some())
            .cloned(),
    );
    let mut used = vec![false; fresh.items().len()];
    for c in orig.items().iter().skip(1) {
        let Some(h) = c.head() else { continue };
        if modeled.contains(&h) {
            // Replace with the fresh child(ren) of the same head, first occurrence only.
            for (k, f) in fresh.items().iter().enumerate() {
                if !used[k] && f.head() == Some(h) {
                    used[k] = true;
                    out.push(f.clone());
                    break;
                }
            }
        } else {
            out.push(c.clone());
        }
    }
    for (k, f) in fresh.items().iter().enumerate().skip(1) {
        if !used[k] && f.atom().is_none() {
            out.push(f.clone());
        }
    }
    SExpr::List(out)
}

fn track_node(t: &Track, n: u8, b: &Codes, seed: &str) -> SExpr {
    let layer = SExpr::list("layer", vec![SExpr::str(&copper_layer(t.layer, n))]);
    let w = SExpr::list("width", vec![len(t.width)]);
    match t.geom {
        TrackGeom::Segment { a, b: e } => SExpr::list(
            "segment",
            vec![
                xy("start", a),
                xy("end", e),
                w,
                layer,
                net_ref(t.net, b),
                uuid_node(seed),
            ],
        ),
        TrackGeom::Arc { start, mid, end } => SExpr::list(
            "arc",
            vec![
                xy("start", start),
                xy("mid", mid),
                xy("end", end),
                w,
                layer,
                net_ref(t.net, b),
                uuid_node(seed),
            ],
        ),
    }
}

fn via_node(v: &Via, n: u8, b: &Codes, seed: &str) -> SExpr {
    let mut items = Vec::new();
    match v.kind(n) {
        ViaKind::Micro => items.push(SExpr::sym("micro")),
        ViaKind::Blind | ViaKind::Buried => items.push(SExpr::sym("blind")),
        ViaKind::Through => {}
    }
    items.push(xy("at", v.at));
    items.push(SExpr::list("size", vec![len(v.diameter)]));
    items.push(SExpr::list("drill", vec![len(v.drill)]));
    items.push(SExpr::list(
        "layers",
        vec![
            SExpr::str(&copper_layer(v.from, n)),
            SExpr::str(&copper_layer(v.to, n)),
        ],
    ));
    items.push(net_ref(v.net, b));
    items.push(uuid_node(seed));
    SExpr::list("via", items)
}

fn zone_node(z: &Zone, n: u8, b: &Board, codes: &Codes) -> SExpr {
    let mut items = vec![net_ref(z.net, codes)];
    items.push(SExpr::list(
        "net_name",
        vec![SExpr::str(z.net.map(|x| b.net_name(x)).unwrap_or(""))],
    ));
    items.push(SExpr::list(
        "layers",
        z.layers
            .iter()
            .map(|l| SExpr::str(&copper_layer(*l, n)))
            .collect(),
    ));
    items.push(uuid_node(&format!("zone{}", z.id.0)));
    items.push(SExpr::list("name", vec![SExpr::str(&z.name)]));
    items.push(SExpr::list(
        "hatch",
        vec![SExpr::sym("edge"), SExpr::num(0.5)],
    ));
    if z.priority != 0 {
        items.push(SExpr::list(
            "priority",
            vec![SExpr::sym(&z.priority.to_string())],
        ));
    }
    let mode = match z.connect {
        PadConnect::Solid => Some("yes"),
        PadConnect::None => Some("no"),
        PadConnect::ThermalThtOnly => Some("thru_hole_only"),
        PadConnect::Thermal => None,
    };
    let mut cp = Vec::new();
    if let Some(m) = mode {
        cp.push(SExpr::sym(m));
    }
    cp.push(SExpr::list(
        "clearance",
        vec![len(z.clearance.unwrap_or(mm(0.5)))],
    ));
    items.push(SExpr::list("connect_pads", cp));
    items.push(SExpr::list("min_thickness", vec![len(z.min_width)]));
    let mut fill = vec![
        SExpr::sym("yes"),
        SExpr::list("thermal_gap", vec![len(z.thermal_gap)]),
        SExpr::list("thermal_bridge_width", vec![len(z.spoke_width)]),
    ];
    match z.islands {
        IslandPolicy::Remove => {}
        IslandPolicy::Keep => fill.push(SExpr::list("island_removal_mode", vec![SExpr::sym("1")])),
        IslandPolicy::RemoveBelow(a) => {
            fill.push(SExpr::list("island_removal_mode", vec![SExpr::sym("2")]));
            fill.push(SExpr::list("island_area_min", vec![SExpr::num(a * 1e-12)]));
        }
    }
    items.push(SExpr::list("fill", fill));
    items.push(SExpr::list("polygon", vec![pts_node(&z.outline.outer)]));
    for (l, set) in &z.fill {
        for p in &set.polys {
            // KiCad stores fills as outlines with holes bridged; we write each ring.
            for r in p.rings() {
                items.push(SExpr::list(
                    "filled_polygon",
                    vec![
                        SExpr::list("layer", vec![SExpr::str(&copper_layer(*l, n))]),
                        pts_node(r),
                    ],
                ));
            }
        }
    }
    SExpr::list("zone", items)
}

fn keepout_node(k: &Keepout, n: u8) -> SExpr {
    let na = |b: bool| SExpr::sym(if b { "not_allowed" } else { "allowed" });
    let layers = if k.layers.is_empty() {
        vec![SExpr::str("*.Cu")]
    } else {
        k.layers
            .iter()
            .map(|l| SExpr::str(&copper_layer(*l, n)))
            .collect()
    };
    SExpr::list(
        "zone",
        vec![
            SExpr::list("net", vec![SExpr::sym("0")]),
            SExpr::list("net_name", vec![SExpr::str("")]),
            SExpr::list("layers", layers),
            uuid_node(&format!("keepout{}", k.id.0)),
            SExpr::list("name", vec![SExpr::str(&k.name)]),
            SExpr::list("hatch", vec![SExpr::sym("edge"), SExpr::num(0.5)]),
            SExpr::list(
                "keepout",
                vec![
                    SExpr::list("tracks", vec![na(k.rules.tracks)]),
                    SExpr::list("vias", vec![na(k.rules.vias)]),
                    SExpr::list("pads", vec![na(k.rules.pads)]),
                    SExpr::list("copperpour", vec![na(k.rules.pours)]),
                    SExpr::list("footprints", vec![na(k.rules.footprints)]),
                ],
            ),
            SExpr::list("polygon", vec![pts_node(&k.outline.outer)]),
        ],
    )
}

fn edge_nodes(o: &Polygon) -> Vec<SExpr> {
    o.rings()
        .enumerate()
        .map(|(i, r)| {
            SExpr::list(
                "gr_poly",
                vec![
                    pts_node(r),
                    SExpr::list(
                        "stroke",
                        vec![
                            SExpr::list("width", vec![SExpr::num(0.05)]),
                            SExpr::list("type", vec![SExpr::sym("solid")]),
                        ],
                    ),
                    SExpr::list("fill", vec![SExpr::sym("none")]),
                    SExpr::list("layer", vec![SExpr::str("Edge.Cuts")]),
                    uuid_node(&format!("edge{i}")),
                ],
            )
        })
        .collect()
}

fn net_table(b: &Board) -> Vec<SExpr> {
    let mut v = vec![SExpr::list("net", vec![SExpr::sym("0"), SExpr::str("")])];
    for (i, n) in b.nets.iter().enumerate() {
        v.push(SExpr::list(
            "net",
            vec![SExpr::sym(&(i + 1).to_string()), SExpr::str(&n.name)],
        ));
    }
    v
}

fn fp_node(f: &Footprint, b: &Board, codes: &Codes, orig: Option<&SExpr>) -> SExpr {
    let lookup = |pad: &str| {
        f.pad_nets.get(pad).and_then(|n| {
            codes
                .get(n)
                .map(|c| (c.parse::<i64>().unwrap_or(0), b.net_name(*n).to_string()))
        })
    };
    let fresh = footprint::write(&f.package, Some((f, &lookup)));
    let Some(o) = orig else { return fresh };
    // Keep the original's unknown children; regenerate placement and pads.
    let mut out = merge(Some(o), fresh.clone(), &["at", "layer", "pad"]);
    // Replace every original pad by the fresh pad of the same name, keeping unknown pad
    // children (pinfunction, pintype, uuid) from the original pad.
    if let Some(items) = out.items_mut() {
        let fresh_pads: BTreeMap<String, &SExpr> = fresh
            .children("pad")
            .map(|p| (p.arg(0).unwrap_or("").to_string(), p))
            .collect();
        let orig_pads: Vec<SExpr> = o.children("pad").cloned().collect();
        items.retain(|c| c.head() != Some("pad"));
        for op in orig_pads {
            let name = op.arg(0).unwrap_or("").to_string();
            if let Some(fp) = fresh_pads.get(&name) {
                items.push(merge(
                    Some(&op),
                    (*fp).clone(),
                    &[
                        "at",
                        "size",
                        "drill",
                        "layers",
                        "net",
                        "roundrect_rratio",
                        "rect_delta",
                        "chamfer_ratio",
                        "chamfer",
                    ],
                ));
            }
        }
        for (name, fp) in &fresh_pads {
            if !o.children("pad").any(|p| p.arg(0) == Some(name)) {
                items.push((*fp).clone());
            }
        }
        // Reference/Value properties follow the board.
        for c in items.iter_mut() {
            if c.head() == Some("property") {
                let key = c.arg(0).unwrap_or("").to_string();
                let val = match key.as_str() {
                    "Reference" => Some(f.reference.clone()),
                    "Value" => Some(f.value.clone()),
                    k => f.properties.get(k).cloned(),
                };
                if let (Some(v), Some(list)) = (val, c.items_mut()) {
                    if list.len() > 2 {
                        list[2] = SExpr::str(&v);
                    }
                }
            }
        }
    }
    out
}

impl KicadPcb {
    /// A new KiCad document for a RunMat board (no original file).
    pub fn from_board(board: &Board) -> KicadPcb {
        let n = board.copper_count();
        let mut items = vec![
            SExpr::sym("kicad_pcb"),
            SExpr::list("version", vec![SExpr::sym("20240108")]),
            SExpr::list("generator", vec![SExpr::str("runmat")]),
            SExpr::list("generator_version", vec![SExpr::str("0.7")]),
            SExpr::list(
                "general",
                vec![
                    SExpr::list("thickness", vec![len(board.stackup.thickness())]),
                    SExpr::list("legacy_teardrops", vec![SExpr::sym("no")]),
                ],
            ),
            SExpr::list("paper", vec![SExpr::str("A4")]),
        ];
        let mut layers = Vec::new();
        for (i, name) in layer_names(n) {
            let ord = if i + 1 == n && n > 1 { 31 } else { i as i64 };
            layers.push(SExpr::List(vec![
                SExpr::sym(&ord.to_string()),
                SExpr::str(&name),
                SExpr::sym("signal"),
            ]));
        }
        for (ord, name, user) in [
            (32, "B.Adhes", "B.Adhesive"),
            (33, "F.Adhes", "F.Adhesive"),
            (34, "B.Paste", ""),
            (35, "F.Paste", ""),
            (36, "B.SilkS", "B.Silkscreen"),
            (37, "F.SilkS", "F.Silkscreen"),
            (38, "B.Mask", ""),
            (39, "F.Mask", ""),
            (44, "Edge.Cuts", ""),
            (46, "B.CrtYd", "B.Courtyard"),
            (47, "F.CrtYd", "F.Courtyard"),
            (48, "B.Fab", ""),
            (49, "F.Fab", ""),
        ] {
            let mut v = vec![
                SExpr::sym(&ord.to_string()),
                SExpr::str(name),
                SExpr::sym("user"),
            ];
            if !user.is_empty() {
                v.push(SExpr::str(user));
            }
            layers.push(SExpr::List(v));
        }
        items.push(SExpr::list("layers", layers));
        items.push(SExpr::list(
            "setup",
            vec![
                write_stackup(&board.stackup),
                SExpr::list("pad_to_mask_clearance", vec![SExpr::num(0.0)]),
            ],
        ));
        let doc = KicadPcb {
            tree: SExpr::List(items),
            board: board.clone(),
            origin: BTreeMap::new(),
            snapshot: Board::new("", Polygon::default(), board.stackup.clone()),
            net_codes: BTreeMap::new(),
            warnings: vec![],
        };
        // Everything is "new": `write` appends nets, items and the outline.
        let mut d = doc;
        d.snapshot.nets.clear();
        d
    }

    /// Writes the document, preserving every unmodelled token.
    pub fn write(&self) -> String {
        let b = &self.board;
        let s = &self.snapshot;
        let n = b.copper_count();
        let mut out: Vec<SExpr> = vec![self.tree.items()[0].clone()];
        let mut nets_written = false;
        let mut edge_written = false;
        let nets_changed = b.nets != s.nets;
        let codes: Codes = if nets_changed {
            (0..b.nets.len() as u32)
                .map(|i| (NetId(i), (i + 1).to_string()))
                .collect()
        } else {
            self.net_codes.clone()
        };
        let outline_changed = b.outline != s.outline;
        for (idx, node) in self.tree.items().iter().enumerate().skip(1) {
            match self.origin.get(&idx) {
                None => {
                    if node.head() == Some("setup") || node.head() == Some("general") {
                        out.push(self.setup_or_general(node));
                    } else {
                        out.push(node.clone());
                    }
                }
                Some(Key::Setup) => out.push(self.setup_or_general(node)),
                Some(Key::Sync) => {
                    let of = crate::syntax::layout::SyncSection::of;
                    if of(b) == of(s) {
                        out.push(node.clone());
                    } else if let Some(p) = sync_property(b) {
                        out.push(p);
                    }
                }
                Some(Key::Net) => {
                    if !nets_changed {
                        out.push(node.clone());
                    } else if !nets_written {
                        out.extend(net_table(b));
                        nets_written = true;
                    }
                }
                Some(Key::Edge) => {
                    if !outline_changed {
                        out.push(node.clone());
                    } else if !edge_written {
                        out.extend(edge_nodes(&b.outline));
                        edge_written = true;
                    }
                }
                Some(Key::Footprint(id)) => match (b.footprints.get(id), s.footprints.get(id)) {
                    (Some(f), Some(o)) if f == o && !nets_changed => out.push(node.clone()),
                    (Some(f), _) => out.push(fp_node(f, b, &codes, Some(node))),
                    (None, _) => {}
                },
                Some(Key::Track(id)) => match (b.tracks.get(id), s.tracks.get(id)) {
                    (Some(t), Some(o)) if t == o && !nets_changed => out.push(node.clone()),
                    (Some(t), _) => out.push(merge(
                        Some(node),
                        track_node(t, n, &codes, &format!("t{}", id.0)),
                        &["start", "mid", "end", "width", "layer", "net"],
                    )),
                    (None, _) => {}
                },
                Some(Key::Via(id)) => match (b.vias.get(id), s.vias.get(id)) {
                    (Some(v), Some(o)) if v == o && !nets_changed => out.push(node.clone()),
                    (Some(v), _) => out.push(merge(
                        Some(node),
                        via_node(v, n, &codes, &format!("v{}", id.0)),
                        &["at", "size", "drill", "layers", "net"],
                    )),
                    (None, _) => {}
                },
                Some(Key::Zone(id)) => match (b.zones.get(id), s.zones.get(id)) {
                    (Some(z), Some(o)) if z == o && !nets_changed => out.push(node.clone()),
                    (Some(z), _) => {
                        let mut m = merge(
                            Some(node),
                            zone_node(z, n, b, &codes),
                            &[
                                "net",
                                "net_name",
                                "layer",
                                "layers",
                                "name",
                                "priority",
                                "connect_pads",
                                "min_thickness",
                                "fill",
                                "polygon",
                            ],
                        );
                        // Fills are derived: always the current ones.
                        if let Some(v) = m.items_mut() {
                            v.retain(|c| c.head() != Some("filled_polygon"));
                        }
                        let fresh = zone_node(z, n, b, &codes);
                        if let Some(v) = m.items_mut() {
                            v.extend(fresh.children("filled_polygon").cloned());
                        }
                        out.push(m);
                    }
                    (None, _) => {}
                },
                Some(Key::Text(id)) => match (b.texts.get(id), s.texts.get(id)) {
                    (Some(t), Some(o)) if t == o => out.push(node.clone()),
                    (Some(t), _) => out.push(merge(
                        Some(node),
                        text_node(t, n),
                        &["at", "layer", "effects"],
                    )),
                    (None, _) => {}
                },
                Some(Key::Graphic(id)) => match (b.graphics.get(id), s.graphics.get(id)) {
                    (Some(g), Some(o)) if g == o => out.push(node.clone()),
                    (Some(g), Some(o))
                        if std::mem::discriminant(&g.graphic)
                            == std::mem::discriminant(&o.graphic) =>
                    {
                        out.push(merge(
                            Some(node),
                            board_graphic_node(g, n),
                            &[
                                "start", "mid", "end", "center", "pts", "stroke", "width", "fill",
                                "layer",
                            ],
                        ))
                    }
                    (Some(g), _) => out.push(board_graphic_node(g, n)),
                    (None, _) => {}
                },
                Some(Key::Dimension(id)) => match (b.dimensions.get(id), s.dimensions.get(id)) {
                    (Some(d), Some(o)) if d == o => out.push(node.clone()),
                    (Some(d), _) => {
                        let mut m = merge(
                            Some(node),
                            dimension_node(d, n),
                            &["layer", "pts", "height"],
                        );
                        if let Some(f) = m
                            .items_mut()
                            .and_then(|v| v.iter_mut().find(|c| c.head() == Some("format")))
                        {
                            f.set_child(SExpr::list(
                                "precision",
                                vec![SExpr::sym(&d.precision.to_string())],
                            ));
                        }
                        out.push(m);
                    }
                    (None, _) => {}
                },
                Some(Key::Fiducial(id)) => match (b.fiducials.get(id), s.fiducials.get(id)) {
                    (Some(f), Some(o)) if f == o => out.push(node.clone()),
                    (Some(f), Some(o))
                        if f.diameter == o.diameter
                            && f.opening == o.opening
                            && f.side == o.side =>
                    {
                        let mut m = node.clone();
                        m.set_child(SExpr::list("at", vec![len(f.at.x), len(-f.at.y)]));
                        if let Some(v) = m.items_mut() {
                            v.retain(|x| x.atom() != Some("locked"));
                            if f.locked {
                                v.insert(2, SExpr::sym("locked"));
                            }
                            for p in v.iter_mut().filter(|p| {
                                p.head() == Some("property") && p.arg(0) == Some("Reference")
                            }) {
                                if let Some(items) = p.items_mut() {
                                    if items.len() > 2 {
                                        items[2] = SExpr::str(&f.name);
                                    }
                                }
                            }
                        }
                        out.push(m);
                    }
                    (Some(f), _) => out.push(fiducial_node(f)),
                    (None, _) => {}
                },
                Some(Key::Keepout(id)) => match (b.keepouts.get(id), s.keepouts.get(id)) {
                    (Some(k), Some(o)) if k == o => out.push(node.clone()),
                    (Some(k), _) => out.push(merge(
                        Some(node),
                        keepout_node(k, n),
                        &["layers", "layer", "name", "keepout", "polygon"],
                    )),
                    (None, _) => {}
                },
            }
        }
        // New items.
        if nets_changed && !nets_written {
            // Nets go right after `setup` (or at the end of the header).
            let pos = out
                .iter()
                .position(|x| x.head() == Some("setup"))
                .map(|p| p + 1)
                .unwrap_or(out.len());
            let nn = net_table(b);
            for (k, x) in nn.into_iter().enumerate() {
                out.insert(pos + k, x);
            }
        }
        let known_fp: Vec<ItemId> = self
            .origin
            .values()
            .filter_map(|k| {
                if let Key::Footprint(i) = k {
                    Some(*i)
                } else {
                    None
                }
            })
            .collect();
        for f in b.footprints.values().filter(|f| !known_fp.contains(&f.id)) {
            out.push(fp_node(f, b, &codes, None));
        }
        let has = |k: &dyn Fn(&Key) -> bool| -> Vec<ItemId> {
            self.origin
                .values()
                .filter(|x| k(x))
                .map(|x| match x {
                    Key::Track(i)
                    | Key::Via(i)
                    | Key::Zone(i)
                    | Key::Keepout(i)
                    | Key::Text(i)
                    | Key::Graphic(i)
                    | Key::Dimension(i)
                    | Key::Fiducial(i)
                    | Key::Footprint(i) => *i,
                    _ => ItemId(0),
                })
                .collect()
        };
        let kt = has(&|k| matches!(k, Key::Track(_)));
        let kv = has(&|k| matches!(k, Key::Via(_)));
        let kz = has(&|k| matches!(k, Key::Zone(_)));
        let kk = has(&|k| matches!(k, Key::Keepout(_)));
        for t in b.tracks.values().filter(|t| !kt.contains(&t.id)) {
            out.push(track_node(t, n, &codes, &format!("t{}", t.id.0)));
        }
        for v in b.vias.values().filter(|v| !kv.contains(&v.id)) {
            out.push(via_node(v, n, &codes, &format!("v{}", v.id.0)));
        }
        for z in b.zones.values().filter(|z| !kz.contains(&z.id)) {
            out.push(zone_node(z, n, b, &codes));
        }
        for k in b.keepouts.values().filter(|k| !kk.contains(&k.id)) {
            out.push(keepout_node(k, n));
        }
        let kx = has(&|k| matches!(k, Key::Text(_)));
        let kg = has(&|k| matches!(k, Key::Graphic(_)));
        let kd = has(&|k| matches!(k, Key::Dimension(_)));
        let kf = has(&|k| matches!(k, Key::Fiducial(_)));
        for f in b.fiducials.values().filter(|f| !kf.contains(&f.id)) {
            out.push(fiducial_node(f));
        }
        for t in b.texts.values().filter(|t| !kx.contains(&t.id)) {
            out.push(text_node(t, n));
        }
        for g in b.graphics.values().filter(|g| !kg.contains(&g.id)) {
            out.push(board_graphic_node(g, n));
        }
        for d in b.dimensions.values().filter(|d| !kd.contains(&d.id)) {
            out.push(dimension_node(d, n));
        }
        if outline_changed && !edge_written {
            out.extend(edge_nodes(&b.outline));
        }
        let of = crate::syntax::layout::SyncSection::of;
        if !self.origin.values().any(|k| *k == Key::Sync) && of(b) != of(s) {
            if let Some(p) = sync_property(b) {
                // After the header items, before the board's own items.
                let pos = out
                    .iter()
                    .position(|x| x.head() == Some("net") || x.head() == Some("footprint"))
                    .unwrap_or(out.len());
                out.insert(pos, p);
            }
        }
        sexpr::write(&SExpr::List(out))
    }

    fn setup_or_general(&self, node: &SExpr) -> SExpr {
        let mut n = node.clone();
        if node.head() == Some("setup") && self.board.stackup != self.snapshot.stackup {
            n.set_child(write_stackup(&self.board.stackup));
        }
        if node.head() == Some("general")
            && self.board.stackup.thickness() != self.snapshot.stackup.thickness()
        {
            n.set_child(SExpr::list(
                "thickness",
                vec![len(self.board.stackup.thickness())],
            ));
        }
        n
    }
}

/// Convenience: board → KiCad text.
pub fn write_board(board: &Board) -> String {
    KicadPcb::from_board(board).write()
}
