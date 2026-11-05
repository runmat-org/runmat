//! The copper on a layer that a fill cuts around or connects to: pads, tracks and vias, with a
//! grid over their boxes.
//!
//! A tile refill reads only the items near its tile, so after a small edit the fill gathers the
//! items in the refilled region ([`layer_items`] with a region) instead of the whole layer. The
//! items are the whole layer's that touch the region, in the same order, so a query inside the
//! region returns what the whole layer's query would.

use crate::board_rules::BItem;
use crate::geom::shape::Prim;
use crate::geom::{Coord, Point, Rect};
use crate::layout::{Board, NetId, PlacedPad, TrackGeom};
use crate::library::PadKind;
use std::collections::BTreeMap;

pub(super) struct LayerItem {
    pub(super) key: BItem,
    pub(super) net: Option<NetId>,
    pub(super) prim: Prim,
    pub(super) pad: Option<PlacedPad>,
    pub(super) _hole: Option<(Point, Coord)>,
}

/// A layer's items with a grid over their boxes.
pub(super) struct LayerItems {
    pub(super) items: Vec<LayerItem>,
    grid: crate::spatial::Grid<u32>,
}

impl LayerItems {
    pub(super) fn near(&self, r: &Rect) -> Vec<&LayerItem> {
        let mut keys = self.grid.query(r);
        keys.sort_unstable();
        keys.dedup();
        keys.into_iter().map(|k| &self.items[k as usize]).collect()
    }
}

/// The items on `layer`, or only those whose boxes touch `region`.
pub(super) fn layer_items(board: &Board, pads: &[PlacedPad], layer: u8, region: Option<&Rect>) -> LayerItems {
    let inside = |b: Rect| region.is_none_or(|r| r.intersects(&b));
    let mut out = Vec::new();
    for p in pads {
        if p.layers.contains(&layer) || p.kind == PadKind::Npth {
            let hole = p.drill.map(|d| (p.center, d / 2));
            if p.kind == PadKind::Npth {
                // NPTH: only the hole matters.
                if let Some((c, r)) = hole {
                    let prim = Prim::circle(c, r);
                    if inside(prim.bbox()) {
                        out.push(LayerItem { key: BItem::Pad(p.footprint, p.index as u32), net: None, prim, pad: None, _hole: hole });
                    }
                }
                continue;
            }
            if inside(p.prim.bbox()) {
                out.push(LayerItem { key: BItem::Pad(p.footprint, p.index as u32), net: p.net, prim: p.prim.clone(), _hole: hole, pad: Some(p.clone()) });
            }
        }
    }
    for t in board.tracks.values() {
        if t.layer != layer {
            continue;
        }
        // A segment's box without making its shape; arcs make theirs.
        if let (Some(r), TrackGeom::Segment { a, b }) = (region, &t.geom) {
            if !r.intersects(&Rect::new(*a, *a).union(Rect::new(*b, *b)).inflate(t.width / 2 + 1)) {
                continue;
            }
        }
        for prim in t.prims() {
            if inside(prim.bbox()) {
                out.push(LayerItem { key: BItem::Track(t.id), net: t.net, prim, pad: None, _hole: None });
            }
        }
    }
    for v in board.vias.values() {
        if v.layers().any(|l| l == layer) {
            let prim = Prim::circle(v.at, v.diameter / 2);
            if inside(prim.bbox()) {
                out.push(LayerItem { key: BItem::Via(v.id), net: v.net, prim, pad: None, _hole: Some((v.at, v.drill / 2)) });
            }
        }
    }
    let mut grid = crate::spatial::Grid::new(crate::geom::mm(2.0));
    for (k, it) in out.iter().enumerate() {
        grid.insert(k as u32, it.prim.bbox());
    }
    LayerItems { items: out, grid }
}

/// Per copper layer, the largest side of a plated pad's box there: how far a thermal spoke reaches.
pub(super) fn pad_extent(pads: &[PlacedPad]) -> BTreeMap<u8, Coord> {
    let mut out: BTreeMap<u8, Coord> = BTreeMap::new();
    for p in pads.iter().filter(|p| p.kind != PadKind::Npth) {
        let b = p.prim.bbox();
        let side = b.width().max(b.height());
        for l in &p.layers {
            let e = out.entry(*l).or_insert(0);
            *e = (*e).max(side);
        }
    }
    out
}
