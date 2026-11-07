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

use crate::layer_set::CopperIndex;
use super::footprint;
use super::sexpr::{self, SExpr};
use super::{at_of, copper_index, copper_layer, len, mm_of};
use crate::geom::poly::{area2, PolySet, Polygon, Ring};
use crate::geom::{mm, Coord, Point};
use crate::layout::*;
use std::collections::BTreeMap;

mod items;
mod write;

use items::*;

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

fn layer_names(n: CopperIndex) -> Vec<(CopperIndex, String)> {
    (0..n).map(|i| (i, copper_layer(i, n))).collect()
}

fn read_stackup(setup: Option<&SExpr>, n_copper: CopperIndex, thickness: Coord) -> Stackup {
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
pub(crate) fn outline_from(segments: &[(Point, Point)], rings: Vec<Ring>) -> Option<Polygon> {
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
                .count() as CopperIndex
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
                let ls: Vec<CopperIndex> = node
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
                let mut layers: Vec<CopperIndex> = Vec::new();
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


/// The copper layers a KiCad board holds: `F.Cu`, `In1.Cu` … `In30.Cu` and `B.Cu`.
pub const KICAD_MAX_COPPER: CopperIndex = 32;

/// Whether KiCad can hold the board's copper stack; the error names both layer counts and the
/// outputs that write any number of layers.
pub fn check_copper_count(board: &Board) -> Result<(), String> {
    let n = board.copper_count();
    if n <= KICAD_MAX_COPPER {
        return Ok(());
    }
    Err(format!(
        "KiCad supports at most {KICAD_MAX_COPPER} copper layers; this board has {n}. Export to ODB++, IPC-2581 or Gerber instead."
    ))
}

/// Board → KiCad text. A board with more copper layers than KiCad holds is refused
/// ([`check_copper_count`]): no layer is dropped or merged.
pub fn write_board(board: &Board) -> Result<String, String> {
    check_copper_count(board)?;
    Ok(KicadPcb::from_board(board).write())
}
