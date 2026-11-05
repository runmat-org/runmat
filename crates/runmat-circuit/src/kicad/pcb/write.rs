//! Writing a board as KiCad text: tracks, vias, zones, keepouts, the edge, the net table and
//! footprints, merged into the original file's nodes so what RunMat doesn't model survives.

use super::*;
use super::super::footprint;
use super::super::sexpr::{self, SExpr};
use super::super::{copper_layer, len, pts_node, uuid_node, xy};
use crate::geom::poly::Polygon;
use crate::geom::mm;
use std::collections::BTreeMap;

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
                    (Some(f), _) => out.push(fp_node(f, b, &codes, Some(node), None)),
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
        // A footprint new to the file takes the board's designator placement (the display's).
        let fresh: Vec<&Footprint> = b
            .footprints
            .values()
            .filter(|f| !known_fp.contains(&f.id))
            .collect();
        let designators = if fresh.is_empty() {
            BTreeMap::new()
        } else {
            crate::designators::designators(b)
        };
        for f in fresh {
            out.push(fp_node(f, b, &codes, None, designators.get(&f.id)));
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

    pub(super) fn setup_or_general(&self, node: &SExpr) -> SExpr {
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

pub(super) fn net_ref(n: Option<NetId>, codes: &Codes) -> SExpr {
    match n.and_then(|id| codes.get(&id)) {
        Some(c) => SExpr::list("net", vec![SExpr::sym(c)]),
        None => SExpr::list("net", vec![SExpr::sym("0")]),
    }
}

pub(super) fn merge(original: Option<&SExpr>, fresh: SExpr, modeled: &[&str]) -> SExpr {
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

pub(super) fn track_node(t: &Track, n: u8, b: &Codes, seed: &str) -> SExpr {
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

pub(super) fn via_node(v: &Via, n: u8, b: &Codes, seed: &str) -> SExpr {
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

pub(super) fn zone_node(z: &Zone, n: u8, b: &Board, codes: &Codes) -> SExpr {
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

pub(super) fn keepout_node(k: &Keepout, n: u8) -> SExpr {
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

pub(super) fn edge_nodes(o: &Polygon) -> Vec<SExpr> {
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

pub(super) fn net_table(b: &Board) -> Vec<SExpr> {
    let mut v = vec![SExpr::list("net", vec![SExpr::sym("0"), SExpr::str("")])];
    for (i, n) in b.nets.iter().enumerate() {
        v.push(SExpr::list(
            "net",
            vec![SExpr::sym(&(i + 1).to_string()), SExpr::str(&n.name)],
        ));
    }
    v
}

/// A footprint node. A footprint read from the file keeps its own reference placement; a new
/// one is written with `designator`.
pub(super) fn fp_node(
    f: &Footprint,
    b: &Board,
    codes: &Codes,
    orig: Option<&SExpr>,
    designator: Option<&crate::designators::Designator>,
) -> SExpr {
    let lookup = |pad: &str| {
        f.pad_nets.get(pad).and_then(|n| {
            codes
                .get(n)
                .map(|c| (c.parse::<i64>().unwrap_or(0), b.net_name(*n).to_string()))
        })
    };
    let fresh = footprint::write_with_designator(
        &f.package,
        Some((f, &lookup)),
        designator.map(|d| (crate::designators::to_local(f, d.at), d.size)),
    );
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
