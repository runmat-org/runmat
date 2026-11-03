//! The incremental design-rules check.
//!
//! [`Drc::run`] checks a whole board; [`Drc::update`] re-checks only what an edit touched:
//! the changed items' own checks, their pairs with neighbours inside the largest clearance,
//! and the per-net checks (connectivity, length, skew, differential pairs) of the nets they
//! belong to. Findings are keyed by `(code, items, layer)` in an ordered map, so the result
//! after any sequence of updates equals a fresh run and is independent of edit order.
//!
//! | Code | Checks |
//! |---|---|
//! | `clearance` | copper of different nets closer than the pair rule `clearance` (zone fills included) |
//! | `track-width` | track narrower than `track_width_min` or wider than `track_width_max` |
//! | `annular-ring` | via or plated-hole ring below `annular_min` |
//! | `hole-size` | drill below `hole_min` |
//! | `via-type` | via kind not in `via_types`; microvia spanning more than one layer pair |
//! | `hole-to-hole` | drilled holes closer than `hole_to_hole` |
//! | `silk-to-pad` | silkscreen closer than `silk_clearance` to a pad on the same side |
//! | `courtyard-overlap` | courtyards on one side closer than `courtyard_clearance` (overlap by default) |
//! | `edge-clearance` | copper closer than `edge_clearance` to the board edge or a cutout, or outside the board |
//! | `keepout` | a track, via, pad or footprint inside a keepout that forbids it |
//! | `layer-not-in-region` | copper on a layer the local (rigid-flex) stackup does not have |
//! | `via-in-bend` | a via or pad inside a bend area |
//! | `unrouted` | a net whose copper forms more than one island |
//! | `length` | routed net length outside `length_min`..`length_max` |
//! | `skew` | length spread in a `match_group` (or a differential pair) above `skew_max` |
//! | `diff-pair-gap` | coupled differential-pair gap off `diff_gap` by more than 10 % |
//! | `diff-pair-uncoupled` | uncoupled length above `diff_uncoupled_max` |
//! | `min-spokes` | a thermal-relief pad with fewer connected spokes than `spoke_min` |
//!
//! Impedance targets (`impedance`) are recorded in [`Drc::impedance_targets`], not checked.

use crate::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use crate::connectivity::{islands, layer_mask, net_nodes};
use crate::elab::Severity;
use crate::geom::shape::{point_segment_distance, Prim};
use crate::geom::{Coord, Point, Rect};
use crate::layout::{Board, Layer, NetId, Side, ViaKind};
use crate::library::{Graphic, PadKind};
use crate::spatial::Grid;
use crate::zone::FillReport;
use runmat_rules::SourceKey;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
pub struct DrcFinding {
    pub code: String,
    pub severity: Severity,
    /// Items involved, sorted; per-net findings include `BItem::Net`.
    pub items: Vec<BItem>,
    pub layer: Option<u8>,
    pub at: Point,
    /// Measured value (nm for lengths) and the rule's required value.
    pub actual: Option<f64>,
    pub required: Option<f64>,
    pub rule: String,
    pub source: SourceKey,
    pub message: String,
}

type FKey = (String, Vec<BItem>, Option<u8>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CKey {
    owner: BItem,
    sub: u32,
}

#[derive(Clone, Debug)]
struct CPrim {
    net: Option<NetId>,
    layers: u64,
    prim: Prim,
}

/// DRC state kept between edits.
pub struct Drc {
    index: Grid<CKey>,
    prims: BTreeMap<CKey, CPrim>,
    owners: BTreeMap<BItem, Vec<CKey>>,
    findings: BTreeMap<FKey, DrcFinding>,
    pub impedance_targets: BTreeMap<String, f64>,
    diff_pairs: Vec<(NetId, NetId)>,
    spokes: Vec<FillReport>,
    zone_cache: Vec<ZoneFillCache>,
}

/// A zone fill on one layer with its bounding boxes (whole set, per polygon).
type ZoneFillCache = (
    BItem,
    u8,
    crate::geom::poly::PolySet,
    Option<NetId>,
    Rect,
    Vec<Rect>,
);

fn key_items(mut v: Vec<BItem>) -> Vec<BItem> {
    v.sort();
    v.dedup();
    v
}

fn owner_prims(board: &Board, it: BItem) -> Vec<CPrim> {
    let n = board.copper_count();
    match it {
        BItem::Track(t) => board
            .tracks
            .get(&t)
            .map(|t| {
                t.prims()
                    .into_iter()
                    .map(|p| CPrim {
                        net: t.net,
                        layers: 1 << t.layer,
                        prim: p,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        BItem::Via(v) => board
            .vias
            .get(&v)
            .map(|v| {
                vec![CPrim {
                    net: v.net,
                    layers: layer_mask(v.layers()),
                    prim: Prim::circle(v.at, v.diameter / 2),
                }]
            })
            .unwrap_or_default(),
        BItem::Pad(f, k) => board
            .footprints
            .get(&f)
            .and_then(|fp| fp.pads(n).into_iter().nth(k as usize))
            .filter(|p| p.kind != PadKind::Npth && !p.layers.is_empty())
            .map(|p| {
                vec![CPrim {
                    net: p.net,
                    layers: layer_mask(p.layers.iter().copied()),
                    prim: p.prim,
                }]
            })
            .unwrap_or_default(),
        BItem::Zone(z) => {
            let Some(z) = board.zones.get(&z) else {
                return vec![];
            };
            let mut out = Vec::new();
            for (layer, set) in &z.fill {
                for r in set.rings() {
                    let m = r.len();
                    for i in 0..m {
                        out.push(CPrim {
                            net: z.net,
                            layers: 1 << layer,
                            prim: Prim::segment(r[i], r[(i + 1) % m], 0),
                        });
                    }
                }
            }
            out
        }
        _ => vec![],
    }
}

fn copper_owners(board: &Board) -> Vec<BItem> {
    let mut v = Vec::new();
    for f in board.footprints.values() {
        for k in 0..f.package.pads.len() {
            v.push(BItem::Pad(f.id, k as u32));
        }
    }
    v.extend(board.tracks.keys().map(|k| BItem::Track(*k)));
    v.extend(board.vias.keys().map(|k| BItem::Via(*k)));
    v.extend(board.zones.keys().map(|k| BItem::Zone(*k)));
    v
}

fn lowest(mask: u64) -> u8 {
    mask.trailing_zeros() as u8
}

fn graphic_prims(g: &Graphic, t: &crate::geom::Transform) -> Vec<Prim> {
    let seg = |a: Point, b: Point, w: Coord| Prim::segment(t.apply(a), t.apply(b), w);
    match g {
        Graphic::Line { points, width } => {
            points.windows(2).map(|w| seg(w[0], w[1], *width)).collect()
        }
        Graphic::Rect { a, b, width, .. } => {
            let c = [*a, Point::new(b.x, a.y), *b, Point::new(a.x, b.y)];
            (0..4).map(|i| seg(c[i], c[(i + 1) % 4], *width)).collect()
        }
        Graphic::Circle {
            center,
            radius,
            width,
            fill,
        } => {
            if *fill {
                vec![Prim::circle(t.apply(*center), radius + width / 2)]
            } else {
                let ring = crate::geom::poly::circle_ring(*center, *radius, crate::geom::mm(0.02));
                let m = ring.len();
                (0..m)
                    .map(|i| seg(ring[i], ring[(i + 1) % m], *width))
                    .collect()
            }
        }
        Graphic::Arc {
            start,
            mid,
            end,
            width,
        } => match crate::geom::shape::arc_from_three(*start, *mid, *end) {
            Some((c, sw)) => crate::geom::shape::arc_points(c, *start, sw, crate::geom::mm(0.01))
                .windows(2)
                .map(|w| seg(w[0], w[1], *width))
                .collect(),
            None => vec![seg(*start, *end, *width)],
        },
        Graphic::Polygon { points, width, .. } => {
            let m = points.len();
            (0..m)
                .map(|i| seg(points[i], points[(i + 1) % m], *width))
                .collect()
        }
        Graphic::Text {
            at,
            text,
            size,
            angle,
        } => {
            if text.starts_with("${") {
                return vec![];
            }
            let w = *size * text.chars().count() as Coord * 3 / 5;
            let tt = t.compose(&crate::geom::Transform::new(*at, *angle, false));
            vec![Prim::rect(w.max(1), (*size).max(1), &tt)]
        }
    }
}

impl Drc {
    /// Full check.
    pub fn run(
        board: &Board,
        rules: &BoardRules,
        diff_pairs: &[(NetId, NetId)],
        spokes: &[FillReport],
    ) -> Drc {
        let mut d = Drc {
            index: Grid::new(crate::geom::mm(1.0)),
            prims: BTreeMap::new(),
            owners: BTreeMap::new(),
            findings: BTreeMap::new(),
            impedance_targets: BTreeMap::new(),
            diff_pairs: diff_pairs.to_vec(),
            spokes: spokes.to_vec(),
            zone_cache: Vec::new(),
        };
        d.rebuild_zone_cache(board);
        let owners = copper_owners(board);
        for o in &owners {
            d.index_owner(board, *o);
        }
        let g = BoardGraph::new(board, diff_pairs);
        for o in &owners {
            d.check_owner(board, rules, &g, *o);
        }
        d.check_holes(board, rules, None);
        d.check_footprints(board, rules, &g, None);
        let nets: BTreeSet<NetId> = (0..board.nets.len() as u32).map(NetId).collect();
        d.check_nets(board, rules, &g, &nets);
        d
    }

    fn index_owner(&mut self, board: &Board, o: BItem) {
        let ps = owner_prims(board, o);
        let mut keys = Vec::with_capacity(ps.len());
        for (k, p) in ps.into_iter().enumerate() {
            let key = CKey {
                owner: o,
                sub: k as u32,
            };
            self.index.insert(key, p.prim.bbox());
            self.prims.insert(key, p);
            keys.push(key);
        }
        self.owners.insert(o, keys);
    }

    fn unindex_owner(&mut self, o: BItem) {
        if let Some(keys) = self.owners.remove(&o) {
            for k in keys {
                self.index.remove(k);
                self.prims.remove(&k);
            }
        }
    }

    fn put(&mut self, f: DrcFinding) {
        let key = (f.code.clone(), f.items.clone(), f.layer);
        match self.findings.get(&key) {
            // Keep the worst instance (smallest actual value) per key.
            Some(old) if old.actual.unwrap_or(f64::MAX) <= f.actual.unwrap_or(f64::MAX) => {}
            _ => {
                self.findings.insert(key, f);
            }
        }
    }

    /// Findings, sorted by key.
    pub fn findings(&self) -> Vec<&DrcFinding> {
        self.findings.values().collect()
    }
    pub fn count(&self, code: &str) -> usize {
        self.findings.values().filter(|f| f.code == code).count()
    }

    fn clearance_pairs(&mut self, board: &Board, rules: &BoardRules, o: BItem) {
        let radius = rules.max_clearance().max(board.limits.min_clearance);
        let keys = self.owners.get(&o).cloned().unwrap_or_default();
        let zone_fills = std::mem::take(&mut self.zone_cache);
        let mut found: Vec<DrcFinding> = Vec::new();
        for k in keys {
            let p = self.prims[&k].clone();
            let q = p.prim.bbox().inflate(radius);
            for nk in self.index.query(&q) {
                if nk.owner == o {
                    continue;
                }
                let np = &self.prims[&nk];
                let shared = p.layers & np.layers;
                if shared == 0 {
                    continue;
                }
                if p.net.is_some() && p.net == np.net {
                    continue;
                }
                // Zone edges are only checked from the item side.
                let req = rules.pair_len(o, nk.owner, props::CLEARANCE);
                let gap = p.prim.gap(&np.prim);
                if gap < req as f64 {
                    let (rule, source) = rules.pair_origin(o, nk.owner, props::CLEARANCE);
                    found.push(DrcFinding {
                        code: "clearance".into(),
                        severity: Severity::Error,
                        items: key_items(vec![o, nk.owner]),
                        layer: Some(lowest(shared)),
                        at: p.prim.anchor(),
                        actual: Some(gap.max(0.0)),
                        required: Some(req as f64),
                        rule,
                        source,
                        message: format!(
                            "clearance {:.4} mm < {:.4} mm",
                            gap.max(0.0) * 1e-6,
                            req as f64 * 1e-6
                        ),
                    });
                }
            }
            // Containment in another net's fill. Only needed when no edge of that fill is
            // near (an overlap across an edge is already found above), so the expensive
            // point-in-polygon test runs only for primitives deep inside or far outside fills.
            if !matches!(o, BItem::Zone(_)) {
                let near_zone_edges: BTreeSet<BItem> = self
                    .index
                    .query(&p.prim.bbox().inflate(radius))
                    .into_iter()
                    .filter(|k| matches!(k.owner, BItem::Zone(_)))
                    .map(|k| k.owner)
                    .collect();
                let anchor = p.prim.anchor();
                for (zi, layer, set, znet, bbox, boxes) in &zone_fills {
                    if p.layers & (1 << layer) == 0 || (p.net.is_some() && p.net == *znet) {
                        continue;
                    }
                    if near_zone_edges.contains(zi) || !bbox.intersects(&p.prim.bbox()) {
                        continue;
                    }
                    let inside =
                        set.polys.iter().zip(boxes).any(|(poly, b)| {
                            b.contains(anchor) && poly.contains(anchor) == Some(true)
                        });
                    if inside {
                        let req = rules.pair_len(o, *zi, props::CLEARANCE);
                        let (rule, source) = rules.pair_origin(o, *zi, props::CLEARANCE);
                        found.push(DrcFinding {
                            code: "clearance".into(),
                            severity: Severity::Error,
                            items: key_items(vec![o, *zi]),
                            layer: Some(*layer),
                            at: p.prim.anchor(),
                            actual: Some(0.0),
                            required: Some(req as f64),
                            rule,
                            source,
                            message: "copper inside another net's zone fill".into(),
                        });
                    }
                }
            }
        }
        self.zone_cache = zone_fills;
        for f in found {
            self.put(f);
        }
    }

    fn rebuild_zone_cache(&mut self, board: &Board) {
        self.zone_cache = board
            .zones
            .values()
            .flat_map(|z| {
                z.fill.iter().map(move |(l, s)| {
                    let boxes: Vec<Rect> = s.polys.iter().map(|p| p.bbox()).collect();
                    let bb = boxes.iter().fold(Rect::EMPTY, |a, b| a.union(*b));
                    (BItem::Zone(z.id), *l, s.clone(), z.net, bb, boxes)
                })
            })
            .collect();
    }

    fn check_owner(&mut self, board: &Board, rules: &BoardRules, g: &BoardGraph, o: BItem) {
        self.clearance_pairs(board, rules, o);
        let n = board.copper_count();
        let anchor = self
            .owners
            .get(&o)
            .and_then(|ks| ks.first())
            .map(|k| self.prims[k].prim.anchor());
        // Edge clearance and outside-board.
        if let Some(keys) = self.owners.get(&o).cloned() {
            if !matches!(o, BItem::Zone(_)) {
                let req = rules.pair_len(o, BItem::Edge, props::EDGE_CLEARANCE);
                let mut worst: Option<(f64, Point)> = None;
                for k in &keys {
                    let p = &self.prims[k].prim;
                    let outside = board.outline.contains(p.anchor()) == Some(false);
                    let mut gap = edge_gap(board, p);
                    if outside {
                        gap = -1.0;
                    }
                    if gap < req as f64 && worst.is_none_or(|w| gap < w.0) {
                        worst = Some((gap, p.anchor()));
                    }
                }
                if let Some((gap, at)) = worst {
                    let (rule, source) = rules.pair_origin(o, BItem::Edge, props::EDGE_CLEARANCE);
                    self.put(DrcFinding {
                        code: "edge-clearance".into(),
                        severity: Severity::Error,
                        items: vec![o],
                        layer: None,
                        at,
                        actual: Some(gap.max(0.0)),
                        required: Some(req as f64),
                        rule,
                        source,
                        message: if gap < 0.0 {
                            "copper outside the board outline".into()
                        } else {
                            format!("{:.4} mm from the board edge", gap * 1e-6)
                        },
                    });
                }
            }
        }
        match o {
            BItem::Track(t) => {
                let tr = &board.tracks[&t];
                let min = rules.len(g, o, props::TRACK_WIDTH_MIN).unwrap_or(0);
                let max = rules.len(g, o, props::TRACK_WIDTH_MAX);
                if tr.width < min || max.is_some_and(|m| tr.width > m) {
                    let (rule, source) = rules.single_origin(
                        g,
                        o,
                        if tr.width < min {
                            props::TRACK_WIDTH_MIN
                        } else {
                            props::TRACK_WIDTH_MAX
                        },
                    );
                    self.put(DrcFinding {
                        code: "track-width".into(),
                        severity: Severity::Error,
                        items: vec![o],
                        layer: Some(tr.layer),
                        at: tr.endpoints().0,
                        actual: Some(tr.width as f64),
                        required: Some(if tr.width < min { min } else { max.unwrap() } as f64),
                        rule,
                        source,
                        message: format!("track width {:.4} mm", tr.width as f64 * 1e-6),
                    });
                }
                self.keepout_check(board, o, &[tr.layer], "tracks");
                self.region_check(board, o, &[tr.layer]);
            }
            BItem::Via(v) => {
                let via = &board.vias[&v];
                self.ring_and_hole(rules, g, o, via.diameter, via.drill, via.at, via.micro);
                let kind = via.kind(n);
                if let Some(RuleValue::ViaKinds(ks)) = rules.value(g, o, props::VIA_TYPES) {
                    if !ks.contains(&kind) {
                        let (rule, source) = rules.single_origin(g, o, props::VIA_TYPES);
                        self.put(DrcFinding {
                            code: "via-type".into(),
                            severity: Severity::Error,
                            items: vec![o],
                            layer: Some(via.from),
                            at: via.at,
                            actual: None,
                            required: None,
                            rule,
                            source,
                            message: format!(
                                "{kind:?} via ({}–{}) is not allowed here",
                                via.from, via.to
                            ),
                        });
                    }
                }
                if kind == ViaKind::Micro && via.to - via.from != 1 {
                    self.put(DrcFinding {
                        code: "via-type".into(),
                        severity: Severity::Error,
                        items: vec![o],
                        layer: Some(via.from),
                        at: via.at,
                        actual: None,
                        required: None,
                        rule: "fab".into(),
                        source: SourceKey::default(),
                        message: "a microvia must span exactly one layer pair".into(),
                    });
                }
                let layers: Vec<u8> = via.layers().collect();
                self.keepout_check(board, o, &layers, "vias");
                self.region_check(board, o, &layers);
                self.bend_check(board, o, via.at, via.diameter / 2);
            }
            BItem::Pad(f, k) => {
                let fp = &board.footprints[&f];
                let pads = fp.pads(n);
                let p = &pads[k as usize];
                if let (Some(d), true) = (p.drill, p.plated) {
                    let b = p.prim.bbox();
                    let size = b.width().min(b.height());
                    self.ring_and_hole(rules, g, o, size, d, p.center, false);
                } else if let Some(d) = p.drill {
                    let min = rules.len(g, o, props::HOLE_MIN).unwrap_or(0);
                    if d < min {
                        self.hole_finding(rules, g, o, d, min, p.center);
                    }
                }
                let layers: Vec<u8> = p.layers.iter().copied().collect();
                self.keepout_check(board, o, &layers, "pads");
                self.region_check(board, o, &layers);
                if p.drill.is_some() {
                    let b = p.prim.bbox();
                    self.bend_check(board, o, p.center, b.width().max(b.height()) / 2);
                }
            }
            BItem::Zone(_) => {}
            _ => {}
        }
        let _ = anchor;
    }

    #[allow(clippy::too_many_arguments)]
    fn ring_and_hole(
        &mut self,
        rules: &BoardRules,
        g: &BoardGraph,
        o: BItem,
        dia: Coord,
        drill: Coord,
        at: Point,
        micro: bool,
    ) {
        let ring = (dia - drill) / 2;
        let amin = rules.len(g, o, props::ANNULAR_MIN).unwrap_or(0);
        if ring < amin {
            let (rule, source) = rules.single_origin(g, o, props::ANNULAR_MIN);
            self.put(DrcFinding {
                code: "annular-ring".into(),
                severity: Severity::Error,
                items: vec![o],
                layer: None,
                at,
                actual: Some(ring as f64),
                required: Some(amin as f64),
                rule,
                source,
                message: format!("annular ring {:.4} mm", ring as f64 * 1e-6),
            });
        }
        let hmin = if micro {
            rules
                .len(g, o, props::HOLE_MIN)
                .unwrap_or(0)
                .min(crate::geom::mm(0.1))
        } else {
            rules.len(g, o, props::HOLE_MIN).unwrap_or(0)
        };
        if drill < hmin {
            self.hole_finding(rules, g, o, drill, hmin, at);
        }
    }

    fn hole_finding(
        &mut self,
        rules: &BoardRules,
        g: &BoardGraph,
        o: BItem,
        d: Coord,
        min: Coord,
        at: Point,
    ) {
        let (rule, source) = rules.single_origin(g, o, props::HOLE_MIN);
        self.put(DrcFinding {
            code: "hole-size".into(),
            severity: Severity::Error,
            items: vec![o],
            layer: None,
            at,
            actual: Some(d as f64),
            required: Some(min as f64),
            rule,
            source,
            message: format!("drill {:.4} mm", d as f64 * 1e-6),
        });
    }

    fn keepout_check(&mut self, board: &Board, o: BItem, layers: &[u8], what: &str) {
        let keys = self.owners.get(&o).cloned().unwrap_or_default();
        for k in board.keepouts.values() {
            let forbidden = match what {
                "tracks" => k.rules.tracks,
                "vias" => k.rules.vias,
                "pads" => k.rules.pads,
                _ => false,
            };
            if !forbidden || !(k.layers.is_empty() || layers.iter().any(|l| k.layers.contains(l))) {
                continue;
            }
            let ko = Prim::polygon(k.outline.rings().cloned().collect());
            if keys.iter().any(|ck| self.prims[ck].prim.gap(&ko) < 0.0) {
                self.put(DrcFinding {
                    code: "keepout".into(),
                    severity: Severity::Error,
                    items: vec![o],
                    layer: layers.first().copied(),
                    at: self.prims[&keys[0]].prim.anchor(),
                    actual: None,
                    required: None,
                    rule: k.name.clone(),
                    source: SourceKey::default(),
                    message: format!("{what} are not allowed in keepout `{}`", k.name),
                });
            }
        }
    }

    fn region_check(&mut self, board: &Board, o: BItem, layers: &[u8]) {
        if board.regions.is_empty() {
            return;
        }
        let Some(k) = self.owners.get(&o).and_then(|v| v.first()) else {
            return;
        };
        let at = self.prims[k].prim.anchor();
        let present = board.copper_at(at);
        let missing: Vec<u8> = layers
            .iter()
            .copied()
            .filter(|l| !present.contains(l))
            .collect();
        if !missing.is_empty() {
            self.put(DrcFinding {
                code: "layer-not-in-region".into(),
                severity: Severity::Error,
                items: vec![o],
                layer: Some(missing[0]),
                at,
                actual: None,
                required: None,
                rule: board.stackup_at(at).name.clone(),
                source: SourceKey::default(),
                message: format!(
                    "copper on {} where the `{}` stackup has no such layer",
                    crate::layout::copper_name(missing[0], board.copper_count()),
                    board.stackup_at(at).name
                ),
            });
        }
    }

    fn bend_check(&mut self, board: &Board, o: BItem, at: Point, r: Coord) {
        for reg in &board.regions {
            for b in &reg.bends {
                let d = point_segment_distance(at, b.a, b.b);
                if d < (b.keepout + r) as f64 {
                    self.put(DrcFinding {
                        code: "via-in-bend".into(),
                        severity: Severity::Error,
                        items: vec![o],
                        layer: None,
                        at,
                        actual: Some(d),
                        required: Some((b.keepout + r) as f64),
                        rule: reg.name.clone(),
                        source: SourceKey::default(),
                        message: format!(
                            "plated hole inside the bend area of region `{}`",
                            reg.name
                        ),
                    });
                }
            }
        }
    }

    fn check_holes(&mut self, board: &Board, rules: &BoardRules, only: Option<&BTreeSet<BItem>>) {
        let mut holes: Vec<(BItem, Point, Coord)> = Vec::new();
        for v in board.vias.values() {
            holes.push((BItem::Via(v.id), v.at, v.drill / 2));
        }
        for p in board.all_pads() {
            if let Some(d) = p.drill {
                holes.push((BItem::Pad(p.footprint, p.index as u32), p.center, d / 2));
            }
        }
        let mut grid: Grid<usize> = Grid::new(crate::geom::mm(2.0));
        for (i, h) in holes.iter().enumerate() {
            grid.insert(i, Rect::new(h.1, h.1).inflate(h.2));
        }
        let radius = rules
            .pair
            .max_value_by(props::HOLE_TO_HOLE, |v| v.as_len().unwrap_or(0))
            .unwrap_or(0);
        for (i, h) in holes.iter().enumerate() {
            if only.is_some_and(|s| !s.contains(&h.0)) {
                continue;
            }
            for j in grid.query(&Rect::new(h.1, h.1).inflate(h.2 + radius)) {
                if j == i {
                    continue;
                }
                let o = &holes[j];
                // Pads of one footprint (connectors) still need hole spacing.
                let gap = h.1.dist(o.1) - (h.2 + o.2) as f64;
                let req = rules.pair_len(h.0, o.0, props::HOLE_TO_HOLE);
                if gap < req as f64 {
                    let (rule, source) = rules.pair_origin(h.0, o.0, props::HOLE_TO_HOLE);
                    self.put(DrcFinding {
                        code: "hole-to-hole".into(),
                        severity: Severity::Error,
                        items: key_items(vec![h.0, o.0]),
                        layer: None,
                        at: h.1,
                        actual: Some(gap.max(0.0)),
                        required: Some(req as f64),
                        rule,
                        source,
                        message: format!("holes {:.4} mm apart", gap.max(0.0) * 1e-6),
                    });
                }
            }
        }
    }

    fn check_footprints(
        &mut self,
        board: &Board,
        rules: &BoardRules,
        g: &BoardGraph,
        only: Option<&BTreeSet<BItem>>,
    ) {
        let n = board.copper_count();
        // Courtyards.
        let mut grid: Grid<crate::layout::ItemId> = Grid::new(crate::geom::mm(5.0));
        let mut cy: BTreeMap<crate::layout::ItemId, Prim> = BTreeMap::new();
        for f in board.footprints.values() {
            let rings = f.courtyard();
            if rings.is_empty() {
                continue;
            }
            let p = Prim::polygon(rings);
            grid.insert(f.id, p.bbox());
            cy.insert(f.id, p);
        }
        for (id, p) in &cy {
            let fa = &board.footprints[id];
            if only.is_some_and(|s| !s.contains(&BItem::Footprint(*id))) {
                continue;
            }
            for other in grid.query(&p.bbox().inflate(crate::geom::mm(1.0))) {
                if other == *id {
                    continue;
                }
                let fb = &board.footprints[&other];
                if fa.side != fb.side {
                    continue;
                }
                let (a, b) = (BItem::Footprint(*id), BItem::Footprint(other));
                let req = rules.pair_len(a, b, props::COURTYARD_CLEARANCE);
                let gap = p.gap(&cy[&other]);
                // With no required clearance, touching is fine; only a real overlap counts.
                let violates = if req > 0 {
                    gap < req as f64
                } else {
                    gap <= 0.0 && courtyard_overlap(p, &cy[&other])
                };
                if violates {
                    let (rule, source) = rules.pair_origin(a, b, props::COURTYARD_CLEARANCE);
                    self.put(DrcFinding {
                        code: "courtyard-overlap".into(),
                        severity: Severity::Warning,
                        items: key_items(vec![a, b]),
                        layer: None,
                        at: fa.at,
                        actual: Some(gap),
                        required: Some(req as f64),
                        rule,
                        source,
                        message: format!(
                            "courtyards of {} and {} overlap",
                            fa.reference, fb.reference
                        ),
                    });
                }
            }
            // Footprint keepouts.
            for k in board.keepouts.values() {
                if k.rules.footprints || k.rules.max_height.is_some() {
                    let ko = Prim::polygon(k.outline.rings().cloned().collect());
                    if p.gap(&ko) < 0.0 {
                        let too_tall = match (k.rules.max_height, fa.package.body) {
                            (Some(h), Some(b)) => b.height > h,
                            (Some(_), None) => false,
                            (None, _) => true,
                        };
                        if k.rules.footprints || too_tall {
                            self.put(DrcFinding {
                                code: "keepout".into(),
                                severity: Severity::Error,
                                items: vec![BItem::Footprint(*id)],
                                layer: None,
                                at: fa.at,
                                actual: None,
                                required: None,
                                rule: k.name.clone(),
                                source: SourceKey::default(),
                                message: format!("{} is inside keepout `{}`", fa.reference, k.name),
                            });
                        }
                    }
                }
            }
        }
        // Silk to pad (footprint silk against every pad on that side's outer copper).
        let outer = |side: Side| if side == Side::Top { 0u8 } else { n - 1 };
        let mut silk: Vec<(BItem, Side, Prim)> = Vec::new();
        for f in board.footprints.values() {
            let t = f.transform();
            for gph in &f.package.silk {
                for p in graphic_prims(gph, &t) {
                    silk.push((BItem::Footprint(f.id), f.side, p));
                }
            }
        }
        for gr in board.graphics.values() {
            if let Layer::Silk(side) = gr.layer {
                for p in graphic_prims(&gr.graphic, &crate::geom::Transform::IDENTITY) {
                    silk.push((BItem::Edge, side, p));
                }
            }
        }
        let mut found = Vec::new();
        for (owner, side, sp) in &silk {
            if only.is_some_and(|s| !s.contains(owner)) {
                continue;
            }
            let layer = outer(*side);
            let q = sp
                .bbox()
                .inflate(board.limits.min_silk_to_pad.max(crate::geom::mm(0.5)));
            for k in self.index.query(&q) {
                let BItem::Pad(..) = k.owner else { continue };
                let cp = &self.prims[&k];
                if cp.layers & (1 << layer) == 0 {
                    continue;
                }
                let req = rules.pair_len(*owner, k.owner, props::SILK_CLEARANCE);
                let gap = sp.gap(&cp.prim);
                if gap < req as f64 {
                    let (rule, source) = rules.pair_origin(*owner, k.owner, props::SILK_CLEARANCE);
                    found.push(DrcFinding {
                        code: "silk-to-pad".into(),
                        severity: Severity::Warning,
                        items: key_items(vec![*owner, k.owner]),
                        layer: Some(layer),
                        at: sp.anchor(),
                        actual: Some(gap.max(0.0)),
                        required: Some(req as f64),
                        rule,
                        source,
                        message: format!("silkscreen {:.4} mm from a pad", gap.max(0.0) * 1e-6),
                    });
                }
            }
        }
        for f in found {
            self.put(f);
        }
        let _ = g;
    }

    fn check_nets(
        &mut self,
        board: &Board,
        rules: &BoardRules,
        g: &BoardGraph,
        nets: &BTreeSet<NetId>,
    ) {
        let nodes = net_nodes(board);
        let mut lengths: BTreeMap<NetId, f64> = BTreeMap::new();
        for t in board.tracks.values() {
            if let Some(n) = t.net {
                *lengths.entry(n).or_default() += t.length();
            }
        }
        for v in board.vias.values() {
            if let Some(n) = v.net {
                *lengths.entry(n).or_default() += board.stackup.span_length(v.from, v.to) as f64;
            }
        }
        for &n in nets {
            let net_item = BItem::Net(n.0);
            if let Some(ns) = nodes.get(&n) {
                let pads = ns
                    .iter()
                    .filter(|x| matches!(x.item, BItem::Pad(..)))
                    .count();
                if pads >= 2 {
                    let c = islands(n, ns.clone());
                    if c.islands > 1 {
                        let mut items = vec![net_item];
                        // Name one pad per island so the finding cross-probes.
                        let mut seen = BTreeSet::new();
                        for (i, nd) in c.nodes.iter().enumerate() {
                            if matches!(nd.item, BItem::Pad(..)) && seen.insert(c.island[i]) {
                                items.push(nd.item);
                            }
                        }
                        self.put(DrcFinding {
                            code: "unrouted".into(),
                            severity: Severity::Error,
                            items: key_items(items),
                            layer: None,
                            at: c.nodes[0].anchors[0],
                            actual: Some(c.islands as f64),
                            required: Some(1.0),
                            rule: "connectivity".into(),
                            source: SourceKey::default(),
                            message: format!(
                                "net `{}` has {} unconnected islands",
                                board.net_name(n),
                                c.islands
                            ),
                        });
                    }
                }
            }
            let len = lengths.get(&n).copied().unwrap_or(0.0);
            let lmin = rules.len(g, net_item, props::LENGTH_MIN);
            let lmax = rules.len(g, net_item, props::LENGTH_MAX);
            if lmin.is_some_and(|m| len < m as f64) || lmax.is_some_and(|m| len > m as f64) {
                let p = if lmin.is_some_and(|m| len < m as f64) {
                    props::LENGTH_MIN
                } else {
                    props::LENGTH_MAX
                };
                let (rule, source) = rules.single_origin(g, net_item, p);
                self.put(DrcFinding {
                    code: "length".into(),
                    severity: Severity::Error,
                    items: vec![net_item],
                    layer: None,
                    at: Point::ORIGIN,
                    actual: Some(len),
                    required: Some(lmin.or(lmax).unwrap_or(0) as f64),
                    rule,
                    source,
                    message: format!(
                        "net `{}` routed length {:.3} mm",
                        board.net_name(n),
                        len * 1e-6
                    ),
                });
            }
            if let Some(RuleValue::Num(z)) = rules.value(g, net_item, props::IMPEDANCE) {
                self.impedance_targets
                    .insert(board.net_name(n).to_string(), z);
            }
        }
        // Match groups.
        let mut groups: BTreeMap<String, Vec<NetId>> = BTreeMap::new();
        for i in 0..board.nets.len() as u32 {
            if let Some(RuleValue::Text(gname)) = rules.value(g, BItem::Net(i), props::MATCH_GROUP)
            {
                groups.entry(gname).or_default().push(NetId(i));
            }
        }
        for (gname, members) in &groups {
            if !members.iter().any(|m| nets.contains(m)) || members.len() < 2 {
                continue;
            }
            let ls: Vec<f64> = members
                .iter()
                .map(|m| lengths.get(m).copied().unwrap_or(0.0))
                .collect();
            let spread = ls.iter().cloned().fold(f64::MIN, f64::max)
                - ls.iter().cloned().fold(f64::MAX, f64::min);
            let skew = members
                .iter()
                .filter_map(|m| rules.len(g, BItem::Net(m.0), props::SKEW_MAX))
                .min();
            if let Some(s) = skew {
                if spread > s as f64 {
                    self.put(DrcFinding {
                        code: "skew".into(),
                        severity: Severity::Error,
                        items: key_items(members.iter().map(|m| BItem::Net(m.0)).collect()),
                        layer: None,
                        at: Point::ORIGIN,
                        actual: Some(spread),
                        required: Some(s as f64),
                        rule: gname.clone(),
                        source: SourceKey::default(),
                        message: format!(
                            "match group `{gname}` length spread {:.3} mm",
                            spread * 1e-6
                        ),
                    });
                }
            }
        }
        // Differential pairs.
        for &(p, nn) in &self.diff_pairs.clone() {
            if !nets.contains(&p) && !nets.contains(&nn) {
                continue;
            }
            self.check_diff_pair(board, rules, g, p, nn, &lengths);
        }
        // Thermal spokes.
        for rep in &self.spokes.clone() {
            for (&(f, k), &(reached, total)) in &rep.spokes {
                let it = BItem::Pad(f, k);
                let Some(fp) = board.footprints.get(&f) else {
                    continue;
                };
                let Some(net) = fp.pad_nets.get(&fp.package.pads[k as usize].name) else {
                    continue;
                };
                if !nets.contains(net) || total == 0 {
                    continue;
                }
                let min = rules
                    .value(g, it, props::SPOKE_MIN)
                    .and_then(|v| v.int())
                    .unwrap_or(2) as u32;
                if reached < min {
                    let (rule, source) = rules.single_origin(g, it, props::SPOKE_MIN);
                    self.put(DrcFinding {
                        code: "min-spokes".into(),
                        severity: Severity::Warning,
                        items: key_items(vec![it, BItem::Zone(rep.zone), BItem::Net(net.0)]),
                        layer: Some(rep.layer),
                        at: fp.at,
                        actual: Some(reached as f64),
                        required: Some(min as f64),
                        rule,
                        source,
                        message: format!(
                            "{}.{} has {reached} of {min} thermal spokes connected",
                            fp.reference, fp.package.pads[k as usize].name
                        ),
                    });
                }
            }
        }
    }

    fn check_diff_pair(
        &mut self,
        board: &Board,
        rules: &BoardRules,
        g: &BoardGraph,
        p: NetId,
        n: NetId,
        lengths: &BTreeMap<NetId, f64>,
    ) {
        let pi = BItem::Net(p.0);
        let gap_rule = rules.len(g, pi, props::DIFF_GAP);
        let unc_max = rules.len(g, pi, props::DIFF_UNCOUPLED_MAX);
        let tp: Vec<_> = board.tracks.values().filter(|t| t.net == Some(p)).collect();
        let tn: Vec<_> = board.tracks.values().filter(|t| t.net == Some(n)).collect();
        let mut coupled = 0.0;
        let total: f64 = tp.iter().map(|t| t.length()).sum();
        let mut worst_gap: Option<(f64, Point, crate::layout::ItemId, crate::layout::ItemId, u8)> =
            None;
        for a in &tp {
            let (a0, a1) = a.endpoints();
            let (ax, ay) = ((a1.x - a0.x) as f64, (a1.y - a0.y) as f64);
            let la = ax.hypot(ay);
            if la == 0.0 {
                continue;
            }
            let (ux, uy) = (ax / la, ay / la);
            let mut best = 0.0f64;
            for b in &tn {
                if b.layer != a.layer {
                    continue;
                }
                let (b0, b1) = b.endpoints();
                let (bx, by) = ((b1.x - b0.x) as f64, (b1.y - b0.y) as f64);
                let lb = bx.hypot(by);
                if lb == 0.0 || (ux * by - uy * bx).abs() / lb > 0.02 {
                    continue;
                }
                // Perpendicular distance between centrelines and the overlap of projections.
                let d = ((b0.x - a0.x) as f64 * -uy + (b0.y - a0.y) as f64 * ux).abs();
                let gap = d - (a.width + b.width) as f64 / 2.0;
                let reference = gap_rule.map(|x| x as f64).unwrap_or(a.width as f64);
                if gap > reference * 3.0 || gap < 0.0 {
                    continue;
                }
                let s0 = (b0.x - a0.x) as f64 * ux + (b0.y - a0.y) as f64 * uy;
                let s1 = (b1.x - a0.x) as f64 * ux + (b1.y - a0.y) as f64 * uy;
                let (lo, hi) = (s0.min(s1).max(0.0), s0.max(s1).min(la));
                if hi <= lo {
                    continue;
                }
                best = best.max(hi - lo);
                if let Some(r) = gap_rule {
                    let dev = (gap - r as f64).abs();
                    if dev > 0.1 * r as f64 + 1.0
                        && worst_gap.is_none_or(|w| dev > (w.0 - r as f64).abs())
                    {
                        worst_gap = Some((gap, a0, a.id, b.id, a.layer));
                    }
                }
            }
            coupled += best.min(la);
        }
        if let (Some((gap, at, ta, tb, layer)), Some(r)) = (worst_gap, gap_rule) {
            let (rule, source) = rules.single_origin(g, pi, props::DIFF_GAP);
            self.put(DrcFinding {
                code: "diff-pair-gap".into(),
                severity: Severity::Error,
                items: key_items(vec![
                    pi,
                    BItem::Net(n.0),
                    BItem::Track(ta),
                    BItem::Track(tb),
                ]),
                layer: Some(layer),
                at,
                actual: Some(gap),
                required: Some(r as f64),
                rule,
                source,
                message: format!(
                    "differential gap {:.4} mm, rule {:.4} mm",
                    gap * 1e-6,
                    r as f64 * 1e-6
                ),
            });
        }
        let uncoupled = (total - coupled).max(0.0);
        if let Some(m) = unc_max {
            if uncoupled > m as f64 {
                let (rule, source) = rules.single_origin(g, pi, props::DIFF_UNCOUPLED_MAX);
                self.put(DrcFinding {
                    code: "diff-pair-uncoupled".into(),
                    severity: Severity::Error,
                    items: key_items(vec![pi, BItem::Net(n.0)]),
                    layer: None,
                    at: Point::ORIGIN,
                    actual: Some(uncoupled),
                    required: Some(m as f64),
                    rule,
                    source,
                    message: format!("uncoupled length {:.3} mm", uncoupled * 1e-6),
                });
            }
        }
        if let Some(s) = rules.len(g, pi, props::SKEW_MAX) {
            let lp = lengths.get(&p).copied().unwrap_or(0.0);
            let ln = lengths.get(&n).copied().unwrap_or(0.0);
            if (lp - ln).abs() > s as f64 {
                let (rule, source) = rules.single_origin(g, pi, props::SKEW_MAX);
                self.put(DrcFinding {
                    code: "skew".into(),
                    severity: Severity::Error,
                    items: key_items(vec![pi, BItem::Net(n.0)]),
                    layer: None,
                    at: Point::ORIGIN,
                    actual: Some((lp - ln).abs()),
                    required: Some(s as f64),
                    rule,
                    source,
                    message: format!("pair skew {:.3} mm", (lp - ln).abs() * 1e-6),
                });
            }
        }
    }

    /// Re-checks after an edit. `changed` are copper items (and footprints) that were added
    /// or modified, `removed` those deleted; a moved footprint should be passed with all its
    /// pads. Rules must already reflect the new board (`rules.single.items_changed` etc.).
    pub fn update(
        &mut self,
        board: &Board,
        rules: &BoardRules,
        changed: &[BItem],
        removed: &[BItem],
        spokes: Option<&[FillReport]>,
    ) {
        if let Some(s) = spokes {
            self.spokes = s.to_vec();
        }
        let mut touched: BTreeSet<BItem> = BTreeSet::new();
        let mut nets: BTreeSet<NetId> = BTreeSet::new();
        let expand = |it: BItem, touched: &mut BTreeSet<BItem>| {
            touched.insert(it);
            if let BItem::Footprint(f) = it {
                if let Some(fp) = board.footprints.get(&f) {
                    for k in 0..fp.package.pads.len() {
                        touched.insert(BItem::Pad(f, k as u32));
                    }
                }
            }
        };
        for &c in changed.iter().chain(removed) {
            expand(c, &mut touched);
        }
        // Nets before and after.
        for t in &touched {
            if let Some(ps) = self.owners.get(t) {
                for k in ps {
                    if let Some(n) = self.prims[k].net {
                        nets.insert(n);
                    }
                }
            }
        }
        // Footprints of touched pads (courtyard/silk); removed footprints are dropped below.
        let mut fps: BTreeSet<BItem> = BTreeSet::new();
        for t in &touched {
            match t {
                BItem::Pad(f, _) | BItem::Footprint(f) => {
                    fps.insert(BItem::Footprint(*f));
                }
                _ => {}
            }
        }
        // Drop findings involving touched items, their nets, or their footprints.
        let net_items: BTreeSet<BItem> = nets.iter().map(|n| BItem::Net(n.0)).collect();
        for t in &touched {
            self.unindex_owner(*t);
        }
        for &c in changed {
            let mut items = BTreeSet::from([c]);
            expand(c, &mut items);
            for i in items {
                if matches!(
                    i,
                    BItem::Pad(..) | BItem::Track(_) | BItem::Via(_) | BItem::Zone(_)
                ) {
                    self.index_owner(board, i);
                }
            }
        }
        for t in &touched {
            if let Some(ps) = self.owners.get(t) {
                for k in ps {
                    if let Some(n) = self.prims[k].net {
                        nets.insert(n);
                    }
                }
            }
        }
        let net_items: BTreeSet<BItem> = net_items
            .into_iter()
            .chain(nets.iter().map(|n| BItem::Net(n.0)))
            .collect();
        self.findings.retain(|(_, items, _), _| {
            !items
                .iter()
                .any(|i| touched.contains(i) || net_items.contains(i) || fps.contains(i))
        });
        self.findings
            .retain(|(code, items, _), _| !(code == "silk-to-pad" && items.contains(&BItem::Edge)));
        if touched.iter().any(|t| matches!(t, BItem::Zone(_))) {
            self.rebuild_zone_cache(board);
        }
        let g = BoardGraph::new(board, &self.diff_pairs);
        let present: Vec<BItem> = touched
            .iter()
            .copied()
            .filter(|t| self.owners.contains_key(t))
            .collect();
        for o in present {
            self.check_owner(board, rules, &g, o);
        }
        let fp_live: BTreeSet<BItem> = fps
            .iter()
            .copied()
            .filter(|f| matches!(f, BItem::Footprint(id) if board.footprints.contains_key(id)))
            .collect();
        self.check_holes(board, rules, Some(&touched));
        self.check_footprints(board, rules, &g, Some(&fp_live));
        // Board-level silk is cheap; re-run its part.
        self.check_footprints(board, rules, &g, Some(&BTreeSet::from([BItem::Edge])));
        // Courtyard pairs where the other side changed: re-check neighbours of touched
        // footprints too.
        let mut neighbours = BTreeSet::new();
        for f in &fp_live {
            if let BItem::Footprint(id) = f {
                let bb = board.footprints[id].bbox().inflate(crate::geom::mm(1.0));
                for o in board.footprints.values() {
                    if o.bbox().intersects(&bb) {
                        neighbours.insert(BItem::Footprint(o.id));
                    }
                }
            }
        }
        self.check_footprints(board, rules, &g, Some(&neighbours));
        self.check_nets(board, rules, &g, &nets);
    }

    /// A stable text digest of all findings, for determinism tests.
    pub fn digest(&self) -> String {
        let mut s = String::new();
        for f in self.findings.values() {
            s.push_str(&format!(
                "{} {:?} {:?} {:?} {:?}\n",
                f.code,
                f.items,
                f.layer,
                f.actual.map(|a| a.round() as i64),
                f.rule
            ));
        }
        s
    }
}

fn courtyard_overlap(a: &Prim, b: &Prim) -> bool {
    match (&a.core, &b.core) {
        (crate::geom::shape::Core::Poly(ra), crate::geom::shape::Core::Poly(rb)) => {
            crate::geom::poly::boolean(ra, rb, crate::geom::poly::BoolOp::Intersection).area() > 0.0
        }
        _ => true,
    }
}

fn edge_gap(board: &Board, p: &Prim) -> f64 {
    let mut best = f64::INFINITY;
    let bb = p.bbox();
    for r in board.outline.rings() {
        let m = r.len();
        for i in 0..m {
            let (a, b) = (r[i], r[(i + 1) % m]);
            let eb = Rect::new(a, b);
            if eb.inflate(crate::geom::mm(5.0)).intersects(&bb) {
                best = best.min(p.gap(&Prim::segment(a, b, 0)));
            }
        }
    }
    best
}
