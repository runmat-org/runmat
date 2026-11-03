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
use crate::layout::{Board, IslandPolicy, ItemId, NetId, PadConnect, PlacedPad};
use crate::library::PadKind;
use std::collections::BTreeMap;

/// Tolerance for arcs in fills: 5 µm.
pub const FILL_TOL: Coord = 5_000;

/// Extra growth of every cutter so integer rounding never puts fill inside a clearance.
pub const FILL_MARGIN: Coord = 10;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FillReport {
    pub zone: ItemId,
    pub layer: u8,
    /// `(footprint, pad index) → (spokes reaching the fill, spokes generated)`.
    pub spokes: BTreeMap<(ItemId, u32), (u32, u32)>,
    pub islands_removed: u32,
    pub polygons: u32,
    pub area: f64,
}

struct LayerItem {
    key: BItem,
    net: Option<NetId>,
    prim: Prim,
    pad: Option<PlacedPad>,
    _hole: Option<(Point, Coord)>,
}

fn layer_items(board: &Board, layer: u8) -> Vec<LayerItem> {
    let mut out = Vec::new();
    for p in board.all_pads() {
        if p.layers.contains(&layer) || p.kind == PadKind::Npth {
            let hole = p.drill.map(|d| (p.center, d / 2));
            if p.kind == PadKind::Npth {
                // NPTH: only the hole matters.
                if let Some((c, r)) = hole {
                    out.push(LayerItem {
                        key: BItem::Pad(p.footprint, p.index as u32),
                        net: None,
                        prim: Prim::circle(c, r),
                        pad: None,
                        _hole: hole,
                    });
                }
                continue;
            }
            out.push(LayerItem {
                key: BItem::Pad(p.footprint, p.index as u32),
                net: p.net,
                prim: p.prim.clone(),
                _hole: hole,
                pad: Some(p),
            });
        }
    }
    for t in board.tracks.values() {
        if t.layer == layer {
            for prim in t.prims() {
                out.push(LayerItem {
                    key: BItem::Track(t.id),
                    net: t.net,
                    prim,
                    pad: None,
                    _hole: None,
                });
            }
        }
    }
    for v in board.vias.values() {
        if v.layers().any(|l| l == layer) {
            out.push(LayerItem {
                key: BItem::Via(v.id),
                net: v.net,
                prim: Prim::circle(v.at, v.diameter / 2),
                pad: None,
                _hole: Some((v.at, v.drill / 2)),
            });
        } else {
            // A via passing outside its span still drills through? Blind/buried vias do not.
        }
    }
    out
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

/// Fills every zone of the board (derived data is replaced). Returns one report per zone
/// layer, in fill order.
pub fn fill_zones(
    board: &mut Board,
    rules: &BoardRules,
    graph_diff: &[(NetId, NetId)],
) -> Vec<FillReport> {
    let mut order: Vec<(i32, ItemId)> = board.zones.values().map(|z| (-z.priority, z.id)).collect();
    order.sort();
    let mut fills: BTreeMap<(ItemId, u8), PolySet> = BTreeMap::new();
    let mut reports = Vec::new();
    {
        let g = BoardGraph::new(board, graph_diff);
        let board_area = {
            let mut rings = vec![board.outline.outer.clone()];
            rings.extend(board.outline.holes.iter().cloned());
            boolean(&rings, &[], BoolOp::Union)
        };
        let mut per_layer: BTreeMap<u8, Vec<LayerItem>> = BTreeMap::new();
        for &(_, zid) in &order {
            let z = &board.zones[&zid];
            let zkey = BItem::Zone(zid);
            let edge = rules.pair_len(zkey, BItem::Edge, props::EDGE_CLEARANCE);
            let area = deflate(&board_area, edge, FILL_TOL).intersection(&PolySet {
                polys: vec![z.outline.clone()],
            });
            for &layer in &z.layers {
                let items = per_layer
                    .entry(layer)
                    .or_insert_with(|| layer_items(board, layer));
                let mut cutters: Vec<Ring> = Vec::new();
                let mut thermal: Vec<(&PlacedPad, Coord, Coord)> = Vec::new();
                let zbox = area
                    .bbox()
                    .inflate(rules.max_clearance() + z.clearance.unwrap_or(0));
                for it in items.iter() {
                    if !it.prim.bbox().intersects(&zbox) {
                        continue;
                    }
                    let same = z.net.is_some() && it.net == z.net;
                    let clr = rules
                        .pair_len(zkey, it.key, props::CLEARANCE)
                        .max(z.clearance.unwrap_or(0))
                        + FILL_MARGIN;
                    if !same {
                        cutters.extend(it.prim.outline(clr, FILL_TOL));
                        continue;
                    }
                    // Same net: holes still need clearance where the hole is not plated.
                    if let Some(p) = &it.pad {
                        let connect = match rules.value(&g, it.key, props::ZONE_CONNECT) {
                            Some(RuleValue::Text(t)) => match t.as_str() {
                                "solid" => PadConnect::Solid,
                                "none" => PadConnect::None,
                                "thermal" => PadConnect::Thermal,
                                _ => z.connect,
                            },
                            _ => z.connect,
                        };
                        let connect = match connect {
                            PadConnect::ThermalThtOnly if p.kind == PadKind::Tht => {
                                PadConnect::Thermal
                            }
                            PadConnect::ThermalThtOnly => PadConnect::Solid,
                            c => c,
                        };
                        let connect = if p.thermal_pad {
                            PadConnect::Solid
                        } else {
                            connect
                        };
                        match connect {
                            PadConnect::Solid => {}
                            PadConnect::None => cutters.extend(it.prim.outline(clr, FILL_TOL)),
                            _ => {
                                let gap = rules
                                    .len(&g, it.key, props::THERMAL_GAP)
                                    .unwrap_or(z.thermal_gap);
                                let sw = rules
                                    .len(&g, it.key, props::SPOKE_WIDTH)
                                    .unwrap_or(z.spoke_width);
                                cutters.extend(it.prim.outline(gap + FILL_MARGIN, FILL_TOL));
                                thermal.push((p, gap, sw));
                            }
                        }
                    }
                }
                for k in board.keepouts.values() {
                    if k.rules.pours && (k.layers.is_empty() || k.layers.contains(&layer)) {
                        cutters.push(k.outline.outer.clone());
                    }
                }
                for (&(oz, ol), f) in &fills {
                    if ol != layer {
                        continue;
                    }
                    let other = &board.zones[&oz];
                    if other.net == z.net && z.net.is_some() {
                        continue;
                    }
                    let clr = rules
                        .pair_len(zkey, BItem::Zone(oz), props::CLEARANCE)
                        .max(z.clearance.unwrap_or(0))
                        + FILL_MARGIN;
                    cutters.extend(crate::geom::poly::inflate(&f.rings(), clr, FILL_TOL).rings());
                }
                let mut fill = boolean(&area.rings(), &cutters, BoolOp::Difference);
                // Spokes, clipped by everything except the thermal gaps themselves.
                let mut spoke_rings: Vec<(usize, Vec<Ring>)> = Vec::new();
                if !thermal.is_empty() {
                    let mut hard: Vec<Ring> = Vec::new();
                    for it in items.iter() {
                        let same = z.net.is_some() && it.net == z.net;
                        if !same && it.prim.bbox().intersects(&zbox) {
                            let clr = rules
                                .pair_len(zkey, it.key, props::CLEARANCE)
                                .max(z.clearance.unwrap_or(0))
                                + FILL_MARGIN;
                            hard.extend(it.prim.outline(clr, FILL_TOL));
                        }
                    }
                    let mut all_spokes = Vec::new();
                    for (k, (p, gap, sw)) in thermal.iter().enumerate() {
                        let rot = board.footprints[&p.footprint].rotation;
                        let s = spokes_for(p, rot, *gap, *sw);
                        all_spokes.extend(s.iter().cloned());
                        spoke_rings.push((k, s));
                    }
                    let spokes = boolean(&all_spokes, &hard, BoolOp::Difference);
                    let spokes = spokes.intersection(&area);
                    fill = fill.union(&spokes);
                }
                if z.min_width > 0 {
                    // Opening keeps the spokes' own width when spoke width ≥ min width.
                    fill = open(&fill, z.min_width / 2, FILL_TOL);
                }
                // Islands.
                let mut kept = Vec::new();
                let mut removed = 0;
                for poly in fill.polys {
                    let single = PolySet {
                        polys: vec![poly.clone()],
                    };
                    let connected = z.net.is_some()
                        && items.iter().any(|it| {
                            it.net == z.net
                                && it.prim.bbox().intersects(&poly.bbox())
                                && touches(&single, &it.prim)
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
                    zone: zid,
                    layer,
                    islands_removed: removed,
                    polygons: fill.polys.len() as u32,
                    area: fill.area(),
                    ..Default::default()
                };
                for (k, rings) in spoke_rings {
                    let (p, gap, _) = thermal[k];
                    let mut reached = 0;
                    for r in &rings {
                        // Probe just beyond the gap along the spoke centreline.
                        let tip = spoke_probe(r, p, gap);
                        if fill.contains(tip) == Some(true) {
                            reached += 1;
                        }
                    }
                    rep.spokes
                        .insert((p.footprint, p.index as u32), (reached, rings.len() as u32));
                }
                reports.push(rep);
                fills.insert((zid, layer), fill);
            }
        }
    }
    for z in board.zones.values_mut() {
        z.fill.clear();
    }
    for ((zid, layer), f) in fills {
        board.zones.get_mut(&zid).unwrap().fill.insert(layer, f);
    }
    reports
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

/// Whether a fill polygon set touches (overlaps) a primitive.
pub fn touches(fill: &PolySet, prim: &Prim) -> bool {
    if fill.contains(prim.anchor()) != Some(false) {
        return true;
    }
    for poly in &fill.polys {
        for r in poly.rings() {
            let p = Prim::polygon(vec![r.clone()]);
            if p.gap(prim) <= 0.0 {
                // Ring boundary within the primitive: they overlap (the primitive straddles an
                // edge of the fill).
                return poly.contains(prim.anchor()) != Some(false) || ring_crosses(r, prim);
            }
        }
    }
    false
}

fn ring_crosses(r: &Ring, prim: &Prim) -> bool {
    let n = r.len();
    (0..n).any(|i| {
        let e = Prim::segment(r[i], r[(i + 1) % n], 0);
        e.gap(prim) <= 0.0
    })
}

/// A rectangle as a zone outline, for tests and generators.
pub fn rect_outline(r: Rect) -> crate::geom::poly::Polygon {
    crate::geom::poly::Polygon::from_ring(rect_ring(r))
}
