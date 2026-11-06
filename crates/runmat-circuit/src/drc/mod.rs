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
//! Impedance targets (`impedance`) are recorded in [`Drc::impedance_targets`] and checked track
//! by track against the stackup's closed forms ([`crate::impedance`], `code = impedance`).

use crate::board_rules::{BItem, BoardGraph, BoardRules};
use crate::connectivity::layer_mask;
use crate::elab::Severity;
use crate::geom::shape::Prim;
use crate::geom::{Coord, Point};
use crate::layout::{Board, NetId};
use crate::library::{Graphic, PadKind};
use crate::spatial::{EdgeIndex, Grid};
use crate::zone::FillReport;
use runmat_rules::SourceKey;
use std::collections::{BTreeMap, BTreeSet};

/// Whether a measured gap (nm, measured in floating point) is short of the rule `req`. The gap is
/// taken at the board's 1 nm resolution first, so a gap equal to its rule passes and one 1 nm
/// under it fails. Every DRC distance check compares through this.
pub(crate) fn short_of(gap: f64, req: Coord) -> bool {
    (gap.round() as Coord) < req
}

/// A measured length and its limit in mm for a message: four decimals, or nanometre precision
/// when four would read the same, so a failing value never reads equal to its limit.
pub(crate) fn mm_pair(value: f64, limit: Coord) -> (String, String) {
    let v = value.round().max(0.0) * 1e-6;
    let l = limit as f64 * 1e-6;
    let (a, b) = (format!("{v:.4}"), format!("{l:.4}"));
    if a == b {
        (format!("{v:.6}"), format!("{l:.6}"))
    } else {
        (a, b)
    }
}

mod copper;
mod nets;
mod parts;

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

/// A copper primitive in the DRC store, keyed in the spatial grid by its slot.
#[derive(Clone, Debug)]
struct CPrim {
    owner: BItem,
    sub: u32,
    net: Option<NetId>,
    layers: u64,
    prim: Prim,
}

/// One zone's fill on one layer, with its edges indexed (fills are large: they are never put
/// in the item grid, and are checked from the item side).
#[derive(Clone, Debug)]
struct ZoneFill {
    zone: BItem,
    layer: u8,
    net: Option<NetId>,
    edges: EdgeIndex,
}

/// DRC state kept between edits.
pub struct Drc {
    index: Grid<u32>,
    slots: Vec<Option<CPrim>>,
    free: Vec<u32>,
    owners: BTreeMap<BItem, Vec<u32>>,
    findings: BTreeMap<FKey, DrcFinding>,
    pub impedance_targets: BTreeMap<String, f64>,
    diff_pairs: Vec<(NetId, NetId)>,
    spokes: Vec<FillReport>,
    zones: Vec<ZoneFill>,
    outline: EdgeIndex,
}

/// Grid cell for copper items: near the typical item size plus clearance.
const ITEM_CELL: Coord = 1_000_000;
/// Cell for zone-fill and outline edges.
const EDGE_CELL: Coord = 1_000_000;

fn key_items(mut v: Vec<BItem>) -> Vec<BItem> {
    v.sort();
    v.dedup();
    v
}

/// Copper primitives of a non-zone owner as `(net, layer mask, primitive)`.
fn owner_prims(board: &Board, it: BItem) -> Vec<(Option<NetId>, u64, Prim)> {
    match it {
        BItem::Track(t) => board
            .tracks
            .get(&t)
            .map(|t| {
                t.prims()
                    .into_iter()
                    .map(|p| (t.net, 1u64 << t.layer, p))
                    .collect()
            })
            .unwrap_or_default(),
        BItem::Via(v) => board
            .vias
            .get(&v)
            .map(|v| {
                vec![(
                    v.net,
                    layer_mask(v.layers()),
                    Prim::circle(v.at, v.diameter / 2),
                )]
            })
            .unwrap_or_default(),
        BItem::Pad(f, k) => board
            .footprints
            .get(&f)
            .and_then(|fp| fp.pad(k as usize, board.copper_count()))
            .filter(|p| p.kind != PadKind::Npth && !p.layers.is_empty())
            .map(|p| vec![(p.net, layer_mask(p.layers.iter().copied()), p.prim)])
            .unwrap_or_default(),
        _ => vec![],
    }
}

/// The two ends of a clearance finding's gap between copper items (not pours), on its
/// layer: the view dimensions the gap between them.
pub fn gap_ends(board: &Board, f: &DrcFinding) -> Option<(Point, Point)> {
    let [a, b] = f.items.as_slice() else { return None };
    if f.code != "clearance" || matches!(a, BItem::Zone(_)) || matches!(b, BItem::Zone(_)) {
        return None;
    }
    let on = |m: u64| f.layer.is_none_or(|l| m & (1u64 << l) != 0);
    let (pa, pb) = (owner_prims(board, *a), owner_prims(board, *b));
    let mut best: Option<(f64, (Point, Point))> = None;
    for (_, ma, x) in pa.iter().filter(|p| on(p.1)) {
        for (_, mb, y) in pb.iter().filter(|p| on(p.1)) {
            if ma & mb == 0 {
                continue;
            }
            let g = x.gap(y);
            if best.is_none_or(|(bg, _)| g < bg) {
                best = Some((g, x.closest(y)));
            }
        }
    }
    best.map(|b| b.1)
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

fn outline_index(board: &Board) -> EdgeIndex {
    EdgeIndex::new(board.outline.rings(), EDGE_CELL)
}

fn zone_fills(board: &Board) -> Vec<ZoneFill> {
    board
        .zones
        .values()
        .flat_map(|z| {
            z.fill.iter().map(move |(l, s)| ZoneFill {
                zone: BItem::Zone(z.id),
                layer: *l,
                net: z.net,
                edges: EdgeIndex::new(s.polys.iter().flat_map(|p| p.rings()), EDGE_CELL),
            })
        })
        .collect()
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
            index: Grid::new(ITEM_CELL),
            slots: Vec::new(),
            free: Vec::new(),
            owners: BTreeMap::new(),
            findings: BTreeMap::new(),
            impedance_targets: BTreeMap::new(),
            diff_pairs: diff_pairs.to_vec(),
            spokes: spokes.to_vec(),
            zones: zone_fills(board),
            outline: outline_index(board),
        };
        let owners = copper_owners(board);
        for o in &owners {
            d.index_owner(board, *o);
        }
        let g = BoardGraph::new(board, diff_pairs);
        for o in &owners {
            d.check_owner(board, rules, &g, *o, true);
        }
        d.check_zone_pairs(rules, None);
        d.check_holes(board, rules, None);
        d.check_footprints(board, rules, &g, None);
        let nets: BTreeSet<NetId> = (0..board.nets.len() as u32).map(NetId).collect();
        d.check_nets(board, rules, &g, &nets);
        d
    }

    fn slot(&self, s: u32) -> &CPrim {
        self.slots[s as usize].as_ref().expect("live DRC slot")
    }

    fn index_owner(&mut self, board: &Board, o: BItem) {
        let ps = owner_prims(board, o);
        let mut keys = Vec::with_capacity(ps.len());
        for (k, (net, layers, prim)) in ps.into_iter().enumerate() {
            let cp = CPrim {
                owner: o,
                sub: k as u32,
                net,
                layers,
                prim,
            };
            let bb = cp.prim.bbox();
            let s = match self.free.pop() {
                Some(s) => {
                    self.slots[s as usize] = Some(cp);
                    s
                }
                None => {
                    self.slots.push(Some(cp));
                    (self.slots.len() - 1) as u32
                }
            };
            self.index.insert(s, bb);
            keys.push(s);
        }
        self.owners.insert(o, keys);
    }

    fn unindex_owner(&mut self, o: BItem) {
        if let Some(keys) = self.owners.remove(&o) {
            for s in keys {
                self.index.remove(s);
                self.slots[s as usize] = None;
                self.free.push(s);
            }
        }
    }

    fn put(&mut self, f: DrcFinding) {
        let key = (f.code.clone(), f.items.clone(), f.layer);
        // Keep the worst instance (smallest actual value) per key; ties go to the smaller
        // location so the result does not depend on check order.
        let worse = |new: &DrcFinding, old: &DrcFinding| {
            let (a, b) = (
                new.actual.unwrap_or(f64::MAX),
                old.actual.unwrap_or(f64::MAX),
            );
            a < b || (a == b && (new.at.x, new.at.y) < (old.at.x, old.at.y))
        };
        match self.findings.get(&key) {
            Some(old) if !worse(&f, old) => {}
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
            // A removed footprint's pads are only known to the store.
            if let BItem::Footprint(f) = c {
                touched.extend(
                    self.owners
                        .range(BItem::Pad(f, 0)..=BItem::Pad(f, u32::MAX))
                        .map(|(k, _)| *k),
                );
            }
        }
        // Nets before and after.
        let nets_of = |d: &Drc, nets: &mut BTreeSet<NetId>, after: bool| {
            for t in &touched {
                if let Some(ps) = d.owners.get(t) {
                    for k in ps {
                        if let Some(n) = d.slot(*k).net {
                            nets.insert(n);
                        }
                    }
                }
                if let BItem::Zone(z) = t {
                    let net = if after {
                        board.zones.get(z).and_then(|z| z.net)
                    } else {
                        d.zones.iter().find(|f| f.zone == *t).and_then(|f| f.net)
                    };
                    nets.extend(net);
                }
            }
        };
        nets_of(self, &mut nets, false);
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
        for t in &touched {
            self.unindex_owner(*t);
        }
        for &c in changed {
            let mut items = BTreeSet::from([c]);
            expand(c, &mut items);
            for i in items {
                match i {
                    BItem::Pad(..) | BItem::Track(_) | BItem::Via(_) => self.index_owner(board, i),
                    BItem::Zone(z) if board.zones.contains_key(&z) => {
                        self.owners.insert(i, Vec::new());
                    }
                    _ => {}
                }
            }
        }
        nets_of(self, &mut nets, true);
        let net_items: BTreeSet<BItem> = nets.iter().map(|n| BItem::Net(n.0)).collect();
        self.findings.retain(|(_, items, _), _| {
            !items
                .iter()
                .any(|i| touched.contains(i) || net_items.contains(i) || fps.contains(i))
        });
        self.findings
            .retain(|(code, items, _), _| !(code == "silk-to-pad" && items.contains(&BItem::Edge)));
        if touched.iter().any(|t| matches!(t, BItem::Zone(_))) {
            self.zones = zone_fills(board);
        }
        self.outline = outline_index(board);
        let g = BoardGraph::new(board, &self.diff_pairs);
        let present: Vec<BItem> = touched
            .iter()
            .copied()
            .filter(|t| self.owners.contains_key(t))
            .collect();
        for o in present {
            self.check_owner(board, rules, &g, o, false);
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
