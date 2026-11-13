//! The copper a router must keep clear of: every pad, track and via on the board by layer, the
//! drilled holes, keepouts and the board edge, answering "may net `n` put this primitive
//! here" exactly (primitive gaps, not a raster).

use crate::connectivity::layer_mask;
use crate::geom::shape::Prim;
use crate::geom::{Coord, Point, Rect};
use crate::layer_set::{CopperIndex, LayerSet};
use crate::layout::{Board, NetId};
use crate::spatial::{EdgeIndex, Grid};

/// One obstacle: its net (none for unconnected copper), the copper layers it is on, its shape.
struct Obstacle {
    net: Option<NetId>,
    mask: LayerSet,
    prim: Prim,
}

/// The board's copper and holes as clearance obstacles.
pub struct Space {
    grid: Grid<u32>,
    items: Vec<Obstacle>,
    holes: Grid<u32>,
    hole_list: Vec<(Point, Coord)>,
    edge: EdgeIndex,
    /// Copper to copper of other nets: the largest any pair needs, and the search radius.
    pub clearance: Coord,
    /// Each net's class clearance (by net id; 0 for a net with none).
    net_clearance: Vec<Coord>,
    /// What every pair keeps: the board's minimum, or a written rule's larger value.
    floor: Coord,
    /// Hole edge to hole edge.
    pub hole_to_hole: Coord,
    /// Copper to the board edge.
    pub edge_clearance: Coord,
    /// Keepout areas closed to routing: (layer mask, outline primitive).
    keepouts: Vec<(LayerSet, Prim)>,
    /// Poured copper: a route may end on it (through a via) but it is no obstacle, since pours
    /// refill around new copper.
    fills: Vec<(NetId, LayerSet, Prim)>,
}

impl Space {
    /// The board's copper, with at most `clearance` between nets (the largest clearance the
    /// board's rules ask for). Two nets keep the larger of their net classes' clearances, or
    /// the board's minimum, so every route the router places passes DRC without spacing a
    /// signal from a signal at a power class's value. A written clearance rule holds for every
    /// pair.
    pub fn new(board: &Board, clearance: Coord) -> Space {
        let (net_clearance, floor) = pair_clearances(board);
        let mut rings = vec![board.outline.outer.clone()];
        rings.extend(board.outline.holes.iter().cloned());
        let mut s = Space {
            grid: Grid::new(crate::geom::mm(1.0)),
            items: Vec::new(),
            holes: Grid::new(crate::geom::mm(2.0)),
            hole_list: Vec::new(),
            edge: EdgeIndex::new(rings.iter(), crate::geom::mm(2.0)),
            clearance,
            net_clearance,
            floor,
            hole_to_hole: board.limits.min_hole_to_hole,
            edge_clearance: board.limits.min_copper_to_edge,
            keepouts: Vec::new(),
            fills: Vec::new(),
        };
        for z in board.zones.values() {
            let Some(net) = z.net else { continue };
            for (layer, set) in &z.fill {
                for poly in &set.polys {
                    let mut rings = vec![poly.outer.clone()];
                    rings.extend(poly.holes.iter().cloned());
                    s.fills
                        .push((net, LayerSet::single(*layer), Prim::polygon(rings)));
                }
            }
        }
        let n = board.copper_count();
        for p in board.all_pads() {
            let mask = layer_mask(p.layers.iter().copied());
            if !mask.is_empty() {
                s.add(p.net, mask, p.prim.clone());
            }
            if let Some(d) = p.drill {
                s.add_hole(p.center, d / 2);
            }
        }
        for t in board.tracks.values() {
            for prim in t.prims() {
                s.add(t.net, LayerSet::single(t.layer), prim);
            }
        }
        for v in board.vias.values() {
            s.add(
                v.net,
                layer_mask(v.layers()),
                Prim::circle(v.at, v.diameter / 2),
            );
            s.add_hole(v.at, v.drill / 2);
        }
        for k in board.keepouts.values() {
            if !(k.rules.tracks || k.rules.vias) {
                continue;
            }
            let layers: Vec<CopperIndex> = if k.layers.is_empty() {
                (0..n).collect()
            } else {
                k.layers.clone()
            };
            let mut rings = vec![k.outline.outer.clone()];
            rings.extend(k.outline.holes.iter().cloned());
            s.keepouts.push((layer_mask(layers), Prim::polygon(rings)));
        }
        s
    }

    /// Records copper placed by the router.
    pub fn add(&mut self, net: Option<NetId>, mask: LayerSet, prim: Prim) {
        let k = self.items.len() as u32;
        self.grid.insert(k, prim.bbox());
        self.items.push(Obstacle { net, mask, prim });
    }

    /// Records a drilled hole of radius `r`.
    pub fn add_hole(&mut self, at: Point, r: Coord) {
        let k = self.hole_list.len() as u32;
        self.holes.insert(k, Rect::new(at, at).inflate(r));
        self.hole_list.push((at, r));
    }

    /// Whether `prim` of net `net` on the layers of `mask` keeps `clearance + extra` from all
    /// other copper and the edge clearance from the board edge, inside the outline.
    pub fn free(&self, net: Option<NetId>, mask: &LayerSet, prim: &Prim, extra: Coord) -> bool {
        let class = |n: Option<NetId>| {
            n.and_then(|n| self.net_clearance.get(n.0 as usize).copied())
                .unwrap_or(0)
        };
        let own = class(net).max(self.floor);
        let bb = prim.bbox();
        let anchor = prim.anchor();
        if self.edge.contains(anchor) != Some(true) {
            return false;
        }
        if self.edge.boundary_gap(prim, self.edge_clearance + extra)
            < (self.edge_clearance + extra) as f64
        {
            return false;
        }
        let mut ok = true;
        self.grid
            .visit(&bb.inflate(self.clearance + extra), |k, _| {
                if !ok {
                    return;
                }
                let o = &self.items[k as usize];
                if !o.mask.intersects(mask) || (net.is_some() && o.net == net) {
                    return;
                }
                let need = (own.max(class(o.net)).min(self.clearance) + extra) as f64;
                if prim.gap(&o.prim) < need {
                    ok = false;
                }
            });
        ok && !self
            .keepouts
            .iter()
            .any(|(m, k)| m.intersects(mask) && prim.gap(k) <= 0.0)
    }

    /// Whether a hole of radius `r` at `at` keeps the hole-to-hole distance.
    pub fn hole_free(&self, at: Point, r: Coord) -> bool {
        let mut ok = true;
        let reach = r + self.hole_to_hole + crate::geom::mm(3.0);
        self.holes.visit(&Rect::new(at, at).inflate(reach), |k, _| {
            let (c, cr) = self.hole_list[k as usize];
            if at.dist(c) - ((r + cr) as f64) < self.hole_to_hole as f64 {
                ok = false;
            }
        });
        ok
    }

    /// Whether poured copper of `net` on any layer covers `p` (a via there joins the pour).
    pub fn own_fill(&self, net: NetId, p: Point) -> bool {
        self.fills
            .iter()
            .any(|(n, _, prim)| *n == net && prim.bbox().contains(p) && prim.contains(p))
    }

    /// The copper items of `net` on layer mask `mask` that cover `p`: an anchor's own pad, track
    /// or via there. Other copper of the net nearby can belong to another island.
    pub fn covering(&self, net: NetId, mask: &LayerSet, p: Point) -> Vec<u32> {
        let mut out = Vec::new();
        self.grid.visit(&Rect::new(p, p), |k, _| {
            let o = &self.items[k as usize];
            if o.net == Some(net)
                && o.mask.intersects(mask)
                && o.prim.contains(p)
                && !out.contains(&k)
            {
                out.push(k);
            }
        });
        out
    }

    /// Whether one of `items` (from [`Space::covering`]) covers `p`.
    pub fn covers(&self, items: &[u32], p: Point) -> bool {
        items
            .iter()
            .any(|k| self.items[*k as usize].prim.contains(p))
    }

    /// Whether copper of `net` on layer mask `mask` covers `p` (the route may start or end there).
    pub fn own(&self, net: NetId, mask: &LayerSet, p: Point) -> bool {
        let mut hit = false;
        self.grid.visit(&Rect::new(p, p), |k, _| {
            if hit {
                return;
            }
            let o = &self.items[k as usize];
            if o.net == Some(net) && o.mask.intersects(mask) && o.prim.contains(p) {
                hit = true;
            }
        });
        hit
    }
}

/// Each net's class clearance, and the clearance every pair keeps (the board's minimum, or the
/// largest written clearance rule), each with 5 µm of rounding margin.
fn pair_clearances(board: &Board) -> (Vec<Coord>, Coord) {
    use crate::board_rules::{props, BoardRules};
    let margin = 5_000;
    let class_value = |class: &str| {
        board.net_class_rules.get(class).and_then(|c| {
            c.decls
                .iter()
                .find(|(k, _)| k == props::CLEARANCE)
                .and_then(|(_, v)| v.as_len())
        })
    };
    let by_net = board
        .nets
        .iter()
        .map(|n| {
            n.class
                .as_deref()
                .and_then(class_value)
                .map_or(0, |c| c + margin)
        })
        .collect();
    let written = if board.net_class_rules.is_empty() {
        BoardRules::for_board(board).0.max_clearance()
    } else {
        let mut only_written = board.clone();
        only_written.net_class_rules.clear();
        BoardRules::for_board(&only_written).0.max_clearance()
    };
    (by_net, board.limits.min_clearance.max(written) + margin)
}
