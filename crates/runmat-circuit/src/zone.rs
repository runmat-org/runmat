//! Copper pour fill.
//!
//! For each zone and each of its copper layers, highest priority first:
//!
//! 1. **Area:** the zone outline intersected with the board outline shrunk by the zone's
//!    edge clearance (cutouts included).
//! 2. **Cutters:** every copper object on the layer of another net, grown by the clearance
//!    the rules give for the pair (or the zone's own clearance when larger); every hole grown
//!    by the same clearance; same-net pads with thermal reliefs grown by the thermal gap;
//!    pour keepouts; filled areas of higher-priority zones of other nets, grown by clearance.
//! 3. **Spokes:** for each thermally connected pad, four spokes along the pad's axes, of the
//!    rule's spoke width, reaching past the gap; spokes are clipped by the other-net cutters.
//! 4. **Minimum width:** a morphological opening by half the zone's minimum width removes
//!    slivers and necks.
//! 5. **Islands:** each remaining polygon that touches no same-net copper on the layer is
//!    removed (or kept, or removed below an area, per the zone's policy).
//!
//! The report gives, per pad, how many spokes reached the fill: the DRC's minimum-spoke
//! check reads it.

use crate::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use crate::geom::poly::{boolean, deflate, open, rect_ring, BoolOp, PolySet, Ring};
use crate::geom::shape::Prim;
use crate::geom::{Coord, Point, Rect, Transform};
use crate::layer_set::CopperIndex;
use crate::layout::{Board, IslandPolicy, ItemId, NetId, PadConnect, PlacedPad};
use crate::library::PadKind;
use crate::spatial::EdgeIndex;
use std::collections::{BTreeMap, BTreeSet};

mod items;
mod rekey;
use items::{layer_items, pad_extent, LayerItem, LayerItems};

/// Tolerance for arcs in fills: 5 µm.
pub const FILL_TOL: Coord = 5_000;

/// Extra growth of every cutter so integer rounding never puts fill inside a clearance.
pub const FILL_MARGIN: Coord = 10;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FillReport {
    pub zone: ItemId,
    pub layer: CopperIndex,
    /// `(footprint, pad index) → (spokes reaching the fill, spokes generated)`.
    pub spokes: BTreeMap<(ItemId, u32), (u32, u32)>,
    pub islands_removed: u32,
    pub polygons: u32,
    pub area: f64,
}

fn spokes_for(p: &PlacedPad, fp_rotation: f64, gap: Coord, width: Coord) -> Vec<Ring> {
    let b = p.prim.bbox();
    let half = (b.width().max(b.height())) / 2;
    let reach = half + gap + width;
    let t = Transform::new(p.center, fp_rotation, false);
    let hw = width / 2;
    let mut out = Vec::new();
    for k in 0..4 {
        let r = Transform::rotation(90.0 * k as f64);
        let pts = [
            Point::new(0, -hw),
            Point::new(reach, -hw),
            Point::new(reach, hw),
            Point::new(0, hw),
        ];
        let mut ring: Ring = pts.iter().map(|q| t.apply(r.apply(*q))).collect();
        if crate::geom::poly::area2(&ring) < 0 {
            ring.reverse();
        }
        out.push(ring);
    }
    out
}

/// Fills are computed per tile of this absolute grid, then merged: an edit refills only the
/// tiles whose inputs it changed.
pub const FILL_TILE: Coord = 10_000_000;

/// The pieces of every zone layer's fill, per tile, kept between fills.
#[derive(Clone, Debug, Default)]
pub struct FillCache {
    layers: BTreeMap<(ItemId, CopperIndex), LayerCache>,
}

#[derive(Clone, Debug)]
struct LayerCache {
    /// What the layer's fill depends on besides the copper around it: the zone (without its
    /// fill) and the area it may fill.
    zone: crate::layout::Zone,
    area: PolySet,
    tiles: BTreeMap<(i64, i64), PolySet>,
    merged: PolySet,
    report: FillReport,
    /// The report names old ids (after [`FillCache::rekey`]): merge again on the next fill.
    remerge: bool,
}

impl FillCache {
    /// The pieces of a zone layer's fill by tile (their union is the fill, islands included).
    pub fn pieces(
        &self,
        zone: ItemId,
        layer: CopperIndex,
    ) -> Option<&BTreeMap<(i64, i64), PolySet>> {
        self.layers.get(&(zone, layer)).map(|c| &c.tiles)
    }
}

/// Where fill inputs changed since a [`FillCache`] was filled: rectangles per copper layer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FillDirty {
    pub everywhere: bool,
    pub rects: BTreeMap<CopperIndex, Vec<Rect>>,
}

impl FillDirty {
    pub fn everywhere() -> Self {
        FillDirty {
            everywhere: true,
            ..Default::default()
        }
    }
    pub fn add(&mut self, layer: CopperIndex, r: Rect) {
        if !r.is_empty() {
            self.rects.entry(layer).or_default().push(r);
        }
    }
    fn hits(&self, layer: CopperIndex, r: &Rect) -> bool {
        self.everywhere
            || self
                .rects
                .get(&layer)
                .is_some_and(|rs| rs.iter().any(|d| d.intersects(r)))
    }
}

/// What a fill changed: per zone layer, the tiles refilled.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FillChange {
    pub tiles: BTreeMap<(ItemId, CopperIndex), Vec<(i64, i64)>>,
    /// Zone layers whose merged fill changed.
    pub merged: BTreeSet<(ItemId, CopperIndex)>,
}

/// The rectangle of tile `(i, j)`.
pub fn tile_rect(t: (i64, i64)) -> Rect {
    Rect::new(
        Point::new(t.0 * FILL_TILE, t.1 * FILL_TILE),
        Point::new((t.0 + 1) * FILL_TILE, (t.1 + 1) * FILL_TILE),
    )
}

fn tiles_of(r: &Rect) -> impl Iterator<Item = (i64, i64)> {
    let f = |v: Coord| v.div_euclid(FILL_TILE);
    let (x0, y0, x1, y1) = (f(r.min.x), f(r.min.y), f(r.max.x), f(r.max.y));
    (y0..=y1).flat_map(move |j| (x0..=x1).map(move |i| (i, j)))
}

/// The relief a same-net pad gets from zone `z`: `None` when it joins solid.
fn relief(
    rules: &BoardRules,
    g: &BoardGraph,
    z: &crate::layout::Zone,
    it: &LayerItem,
    p: &PlacedPad,
) -> Option<PadConnect> {
    let connect = match rules.value(g, it.key, props::ZONE_CONNECT) {
        Some(RuleValue::Text(t)) => match t.as_str() {
            "solid" => PadConnect::Solid,
            "none" => PadConnect::None,
            "thermal" => PadConnect::Thermal,
            _ => z.connect,
        },
        _ => z.connect,
    };
    let connect = match connect {
        PadConnect::ThermalThtOnly if p.kind == PadKind::Tht => PadConnect::Thermal,
        PadConnect::ThermalThtOnly => PadConnect::Solid,
        c => c,
    };
    let connect = if p.thermal_pad {
        PadConnect::Solid
    } else {
        connect
    };
    (connect != PadConnect::Solid).then_some(connect)
}

/// Fills every zone of the board (derived data is replaced). Returns one report per zone
/// layer, in fill order.
pub fn fill_zones(
    board: &mut Board,
    rules: &BoardRules,
    graph_diff: &[(NetId, NetId)],
) -> Vec<FillReport> {
    let mut cache = FillCache::default();
    fill_zones_cached(
        board,
        rules,
        graph_diff,
        &mut cache,
        &FillDirty::everywhere(),
    )
    .0
}

/// [`fill_zones`] that refills only the tiles `dirty` reaches, reusing `cache` for the rest.
/// The result equals a fresh fill when `dirty` covers every input that changed since the
/// cache was filled.
pub fn fill_zones_cached(
    board: &mut Board,
    rules: &BoardRules,
    graph_diff: &[(NetId, NetId)],
    cache: &mut FillCache,
    dirty: &FillDirty,
) -> (Vec<FillReport>, FillChange) {
    let mut order: Vec<(i32, ItemId)> = board.zones.values().map(|z| (-z.priority, z.id)).collect();
    order.sort();
    let mut reports = Vec::new();
    let mut change = FillChange::default();
    let mut fills: BTreeMap<(ItemId, CopperIndex), PolySet> = BTreeMap::new();
    // Zone layers gone from the board leave the cache.
    let live: BTreeSet<(ItemId, CopperIndex)> = board
        .zones
        .values()
        .flat_map(|z| z.layers.iter().map(move |l| (z.id, *l)))
        .collect();
    cache.layers.retain(|k, _| live.contains(k));
    {
        let g = BoardGraph::new(board, graph_diff);
        let board_area = {
            let mut rings = vec![board.outline.outer.clone()];
            rings.extend(board.outline.holes.iter().cloned());
            boolean(&rings, &[], BoolOp::Union)
        };
        let pads = board.all_pads();
        let extent = pad_extent(&pads);
        // Each layer's items, made when a merge needs the whole layer; a tile refill takes the
        // items of the region it refills.
        let mut per_layer: BTreeMap<CopperIndex, LayerItems> = BTreeMap::new();
        // Tiles refilled so far per layer: lower-priority zones there refill too.
        let mut refilled: BTreeMap<CopperIndex, Vec<Rect>> = BTreeMap::new();
        let max_clr = rules.max_clearance();
        for &(_, zid) in &order {
            let z = &board.zones[&zid];
            let zkey = BItem::Zone(zid);
            let edge = rules.pair_len(zkey, BItem::Edge, props::EDGE_CLEARANCE);
            let mut bare = z.clone();
            bare.fill.clear();
            for &layer in &z.layers {
                let spoke = spoke_reach(z, extent.get(&layer).copied().unwrap_or(0));
                let ck = (zid, layer);
                let stale = cache.layers.get(&ck).is_none_or(|c| c.zone != bare);
                if stale {
                    let area = deflate(&board_area, edge, FILL_TOL).intersection(&PolySet {
                        polys: vec![z.outline.clone()],
                    });
                    cache.layers.insert(
                        ck,
                        LayerCache {
                            zone: bare.clone(),
                            area,
                            tiles: BTreeMap::new(),
                            merged: PolySet::default(),
                            report: FillReport::default(),
                            remerge: false,
                        },
                    );
                }
                let zclr = z.clearance.unwrap_or(0);
                let margin = z.min_width + 8 * FILL_TOL + 4 * FILL_MARGIN;
                let reach = max_clr.max(zclr) + FILL_MARGIN + margin + spoke;
                let area_box = cache.layers[&ck].area.bbox();
                let wanted: Vec<(i64, i64)> = if area_box.is_empty() {
                    Vec::new()
                } else {
                    tiles_of(&area_box).collect()
                };
                let mut redo: Vec<(i64, i64)> = Vec::new();
                for t in &wanted {
                    let probe = tile_rect(*t).inflate(reach);
                    let again = stale
                        || dirty.hits(layer, &probe)
                        || refilled
                            .get(&layer)
                            .is_some_and(|rs| rs.iter().any(|r| r.intersects(&probe)));
                    if again {
                        redo.push(*t);
                    }
                }
                let wanted_set: BTreeSet<(i64, i64)> = wanted.iter().copied().collect();
                let before = cache.layers[&ck].tiles.len();
                cache
                    .layers
                    .get_mut(&ck)
                    .unwrap()
                    .tiles
                    .retain(|t, _| wanted_set.contains(t));
                let dropped = cache.layers[&ck].tiles.len() != before;
                let mut done: Vec<(i64, i64)> = Vec::new();
                let higher = if redo.is_empty() {
                    HigherFills { pieces: Vec::new() }
                } else {
                    fills_cut(cache, rules, board, z, layer, &fills)
                };
                // What the refilled tiles' cutters can come from (fill_tile's `near`).
                let near = |t: (i64, i64)| {
                    tile_rect(t).inflate(margin + max_clr.max(zclr) + FILL_MARGIN + spoke)
                };
                let region = redo.iter().map(|t| near(*t)).reduce(|a, b| a.union(b));
                let local = region.map(|r| layer_items(board, &pads, layer, Some(&r)));
                for t in redo {
                    let piece = fill_tile(
                        board,
                        rules,
                        &g,
                        z,
                        layer,
                        local.as_ref().expect("a region for the tiles"),
                        spoke,
                        &cache.layers[&ck].area,
                        &higher,
                        t,
                        margin,
                    );
                    let tiles = &mut cache.layers.get_mut(&ck).unwrap().tiles;
                    let old = tiles.get(&t);
                    if old != Some(&piece) && !(old.is_none() && piece.is_empty()) {
                        done.push(t);
                    }
                    if piece.is_empty() {
                        tiles.remove(&t);
                    } else {
                        tiles.insert(t, piece);
                    }
                }
                if !done.is_empty() {
                    let r = refilled.entry(layer).or_default();
                    r.extend(done.iter().map(|t| tile_rect(*t)));
                    change.tiles.insert(ck, done.clone());
                }
                let remerge = cache.layers[&ck].remerge;
                if stale || dropped || remerge || !done.is_empty() {
                    let li = per_layer
                        .entry(layer)
                        .or_insert_with(|| layer_items(board, &pads, layer, None));
                    let (merged, report) =
                        merge_layer(rules, &g, board, z, layer, li, &cache.layers[&ck].tiles);
                    let c = cache.layers.get_mut(&ck).unwrap();
                    if c.merged != merged || stale {
                        change.merged.insert(ck);
                    }
                    c.merged = merged;
                    c.report = report;
                    c.remerge = false;
                }
                let c = &cache.layers[&ck];
                reports.push(c.report.clone());
                fills.insert(ck, c.merged.clone());
            }
        }
    }
    for z in board.zones.values_mut() {
        z.fill.clear();
    }
    for ((zid, layer), f) in fills {
        board.zones.get_mut(&zid).unwrap().fill.insert(layer, f);
    }
    (reports, change)
}

/// How far a thermal spoke of a same-net pad on the layer reaches past the pad's box.
/// `pad` is the largest side of a plated pad's box on the layer ([`pad_extent`]).
fn spoke_reach(z: &crate::layout::Zone, pad: Coord) -> Coord {
    pad + z.thermal_gap.max(0) + z.spoke_width.max(0) + crate::geom::mm(1.0)
}

/// The fills of higher-priority zones of other nets on `layer`, each grown by its clearance
/// to `z`: the cutters a tile of `z` takes from them, as pieces with their tiles.
struct HigherFills {
    pieces: Vec<(Rect, PolySet, Coord)>,
}

fn fills_cut(
    cache: &FillCache,
    rules: &BoardRules,
    board: &Board,
    z: &crate::layout::Zone,
    layer: CopperIndex,
    done: &BTreeMap<(ItemId, CopperIndex), PolySet>,
) -> HigherFills {
    let mut pieces = Vec::new();
    for &(oz, ol) in done.keys() {
        if ol != layer {
            continue;
        }
        let other = &board.zones[&oz];
        if other.net == z.net && z.net.is_some() {
            continue;
        }
        let clr = rules
            .pair_len(BItem::Zone(z.id), BItem::Zone(oz), props::CLEARANCE)
            .max(z.clearance.unwrap_or(0))
            + FILL_MARGIN;
        if let Some(c) = cache.layers.get(&(oz, ol)) {
            for (t, p) in &c.tiles {
                pieces.push((tile_rect(*t), p.clone(), clr));
            }
        }
    }
    HigherFills { pieces }
}

/// One tile of zone `z` on `layer`: the fill computed on the tile grown by `margin` (so the
/// minimum-width opening sees past the tile edge), clipped to the tile.
#[allow(clippy::too_many_arguments)]
fn fill_tile(
    board: &Board,
    rules: &BoardRules,
    g: &BoardGraph,
    z: &crate::layout::Zone,
    layer: CopperIndex,
    li: &LayerItems,
    spoke: Coord,
    area: &PolySet,
    higher: &HigherFills,
    t: (i64, i64),
    margin: Coord,
) -> PolySet {
    let tile = tile_rect(t);
    let window = tile.inflate(margin);
    if !area.bbox().intersects(&window) {
        return PolySet::default();
    }
    let window_set = PolySet {
        polys: vec![crate::geom::poly::Polygon::from_ring(rect_ring(window))],
    };
    let area_w = area.intersection(&window_set);
    if area_w.is_empty() {
        return PolySet::default();
    }
    let zkey = BItem::Zone(z.id);
    let zclr = z.clearance.unwrap_or(0);
    let near = window.inflate(rules.max_clearance().max(zclr) + FILL_MARGIN + spoke);
    let mut cutters: Vec<Ring> = Vec::new();
    let mut hard: Vec<Ring> = Vec::new();
    let mut thermal: Vec<(&PlacedPad, Coord, Coord)> = Vec::new();
    for it in li.near(&near) {
        let same = z.net.is_some() && it.net == z.net;
        let clr = rules.pair_len(zkey, it.key, props::CLEARANCE).max(zclr) + FILL_MARGIN;
        if !same {
            let rings = it.prim.outline(clr, FILL_TOL);
            hard.extend(rings.iter().cloned());
            cutters.extend(rings);
            continue;
        }
        if let Some(p) = &it.pad {
            match relief(rules, g, z, it, p) {
                None => {}
                Some(PadConnect::None) => cutters.extend(it.prim.outline(clr, FILL_TOL)),
                Some(_) => {
                    let gap = rules
                        .len(g, it.key, props::THERMAL_GAP)
                        .unwrap_or(z.thermal_gap);
                    let sw = rules
                        .len(g, it.key, props::SPOKE_WIDTH)
                        .unwrap_or(z.spoke_width);
                    cutters.extend(it.prim.outline(gap + FILL_MARGIN, FILL_TOL));
                    thermal.push((p, gap, sw));
                }
            }
        }
    }
    for k in board.keepouts.values() {
        if k.rules.pours
            && (k.layers.is_empty() || k.layers.contains(&layer))
            && k.outline.bbox().intersects(&window)
        {
            cutters.push(k.outline.outer.clone());
        }
    }
    for (rect, piece, clr) in &higher.pieces {
        if !rect.inflate(*clr).intersects(&window) {
            continue;
        }
        cutters.extend(
            crate::geom::offset::offset_set(
                piece,
                *clr,
                crate::geom::offset::Join::Round,
                FILL_TOL,
            )
            .rings(),
        );
    }
    let mut fill = boolean(&area_w.rings(), &cutters, BoolOp::Difference);
    if !thermal.is_empty() {
        let mut all_spokes = Vec::new();
        for (p, gap, sw) in &thermal {
            let rot = board.footprints[&p.footprint].rotation;
            all_spokes.extend(spokes_for(p, rot, *gap, *sw));
        }
        let spokes = boolean(&all_spokes, &hard, BoolOp::Difference).intersection(&area_w);
        fill = fill.union(&spokes);
    }
    if z.min_width > 0 {
        fill = open(&fill, z.min_width / 2, FILL_TOL);
    }
    let tile_set = PolySet {
        polys: vec![crate::geom::poly::Polygon::from_ring(rect_ring(tile))],
    };
    fill.intersection(&tile_set)
}

/// The union of tile pieces. Pieces are clipped to their tiles, so they meet only along tile
/// lines: edges on a tile line are split at every vertex on that line, opposite runs from the
/// two sides cancel, and the remaining edges link into rings.
pub fn stitch(tiles: &BTreeMap<(i64, i64), PolySet>) -> PolySet {
    let mut edges: Vec<(Point, Point)> = Vec::new();
    // (vertical, line coordinate) → (from, to) along the line.
    let mut seams: BTreeMap<(bool, Coord), Vec<(Coord, Coord)>> = BTreeMap::new();
    for piece in tiles.values() {
        for poly in &piece.polys {
            for ring in poly.rings() {
                let n = ring.len();
                for i in 0..n {
                    let (a, b) = (ring[i], ring[(i + 1) % n]);
                    if a == b {
                        continue;
                    }
                    if a.x == b.x && a.x.rem_euclid(FILL_TILE) == 0 {
                        seams.entry((true, a.x)).or_default().push((a.y, b.y));
                    } else if a.y == b.y && a.y.rem_euclid(FILL_TILE) == 0 {
                        seams.entry((false, a.y)).or_default().push((a.x, b.x));
                    } else {
                        edges.push((a, b));
                    }
                }
            }
        }
    }
    for ((vertical, c), runs) in seams {
        let mut cuts: Vec<Coord> = runs.iter().flat_map(|(a, b)| [*a, *b]).collect();
        cuts.sort_unstable();
        cuts.dedup();
        // Net direction per elementary interval.
        let mut net = vec![0i32; cuts.len().saturating_sub(1)];
        for (a, b) in runs {
            let (lo, hi, d) = if a < b { (a, b, 1) } else { (b, a, -1) };
            let i0 = cuts.partition_point(|x| *x < lo);
            let i1 = cuts.partition_point(|x| *x < hi);
            for v in &mut net[i0..i1] {
                *v += d;
            }
        }
        let at = |s: Coord| {
            if vertical {
                Point::new(c, s)
            } else {
                Point::new(s, c)
            }
        };
        for (i, d) in net.iter().enumerate() {
            match d.signum() {
                1 => edges.push((at(cuts[i]), at(cuts[i + 1]))),
                -1 => edges.push((at(cuts[i + 1]), at(cuts[i]))),
                _ => {}
            }
        }
    }
    crate::geom::poly::build_rings(edges)
}

/// A zone layer's fill from its tiles: the pieces merged, islands removed per the zone's
/// policy, and the spoke report.
fn merge_layer(
    rules: &BoardRules,
    g: &BoardGraph,
    board: &Board,
    z: &crate::layout::Zone,
    layer: CopperIndex,
    li: &LayerItems,
    tiles: &BTreeMap<(i64, i64), PolySet>,
) -> (PolySet, FillReport) {
    let fill = stitch(tiles);
    let mut kept = Vec::new();
    let mut removed = 0;
    for poly in fill.polys {
        let pb = poly.bbox();
        let idx = EdgeIndex::new(poly.rings(), crate::geom::mm(1.0));
        let connected = z.net.is_some()
            && li.near(&pb).into_iter().any(|it| {
                it.net == z.net && it.prim.bbox().intersects(&pb) && idx.touches(&it.prim)
            });
        let keep = match z.islands {
            IslandPolicy::Keep => true,
            IslandPolicy::Remove => connected,
            IslandPolicy::RemoveBelow(a) => connected || poly.area() >= a,
        };
        if keep {
            kept.push(poly);
        } else {
            removed += 1;
        }
    }
    let fill = PolySet { polys: kept };
    let mut rep = FillReport {
        zone: z.id,
        layer,
        islands_removed: removed,
        polygons: fill.polys.len() as u32,
        area: fill.area(),
        ..Default::default()
    };
    let fill_idx = EdgeIndex::new(
        fill.polys.iter().flat_map(|p| p.rings()),
        crate::geom::mm(1.0),
    );
    let fb = fill.bbox();
    if !fb.is_empty() && z.net.is_some() {
        for it in li.near(&fb.inflate(crate::geom::mm(1.0))) {
            if it.net != z.net {
                continue;
            }
            let Some(p) = &it.pad else { continue };
            if !matches!(relief(rules, g, z, it, p), Some(c) if c != PadConnect::None) {
                continue;
            }
            let gap = rules
                .len(g, it.key, props::THERMAL_GAP)
                .unwrap_or(z.thermal_gap);
            let sw = rules
                .len(g, it.key, props::SPOKE_WIDTH)
                .unwrap_or(z.spoke_width);
            let rot = board.footprints[&p.footprint].rotation;
            let rings = spokes_for(p, rot, gap, sw);
            let mut reached = 0;
            for r in &rings {
                let tip = spoke_probe(r, p, gap);
                if fill_idx.contains(tip) == Some(true) {
                    reached += 1;
                }
            }
            rep.spokes
                .insert((p.footprint, p.index as u32), (reached, rings.len() as u32));
        }
    }
    (fill, rep)
}

/// A point on the spoke centreline a little past the thermal gap.
fn spoke_probe(ring: &Ring, p: &PlacedPad, gap: Coord) -> Point {
    // The spoke ring is [base−, tip−, tip+, base+]; its tip midpoint gives the direction.
    let tip = Point::new((ring[1].x + ring[2].x) / 2, (ring[1].y + ring[2].y) / 2);
    let d = tip - p.center;
    let len = d.len();
    let b = p.prim.bbox();
    let half = (b.width().max(b.height())) as f64 / 2.0;
    let along = (half + gap as f64 + 1000.0).min(len - 1.0);
    let s = along / len.max(1.0);
    Point::from_f(
        p.center.x as f64 + d.x as f64 * s,
        p.center.y as f64 + d.y as f64 * s,
    )
}

/// Whether a fill polygon set touches (overlaps) a primitive. For many queries against one
/// fill, build an [`EdgeIndex`] once instead.
pub fn touches(fill: &PolySet, prim: &Prim) -> bool {
    EdgeIndex::new(
        fill.polys.iter().flat_map(|p| p.rings()),
        crate::geom::mm(1.0),
    )
    .touches(prim)
}

/// A rectangle as a zone outline, for tests and generators.
pub fn rect_outline(r: Rect) -> crate::geom::poly::Polygon {
    crate::geom::poly::Polygon::from_ring(rect_ring(r))
}
