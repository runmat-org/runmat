//! Physical connectivity on the board: which copper of a net actually touches.
//!
//! Nodes are tracks, vias, pads and individual zone-fill polygons. Two nodes of the same net
//! connect when they share a copper layer and touch (gap ≤ 0); vias and through-hole pads
//! span layers, so they join layers. Each net's nodes split into *islands*; a net with more
//! than one island is unrouted, and the ratsnest connects its islands.
//!
//! Logical connectivity (which pins belong to which net across hierarchy and buses) is
//! [`crate::elab`]; this module is its physical counterpart.

use crate::board_rules::BItem;
use crate::geom::shape::Prim;
use crate::geom::{Point, Rect};
use crate::layout::{Board, NetId};
use crate::spatial::Grid;
use std::collections::BTreeMap;

/// One copper node of a net.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub item: BItem,
    /// For zone fills: (layer, polygon index).
    pub fill: Option<(u8, u32)>,
    pub layers: u64,
    pub prims: Vec<Prim>,
    /// Points the ratsnest may connect to (pad centres, track ends, via centres).
    pub anchors: Vec<Point>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetConnectivity {
    pub net: NetId,
    pub nodes: Vec<Node>,
    /// Island index of each node.
    pub island: Vec<usize>,
    pub islands: usize,
}

pub fn layer_mask(layers: impl IntoIterator<Item = u8>) -> u64 {
    layers
        .into_iter()
        .fold(0u64, |m, l| m | (1u64 << l.min(63)))
}

/// Copper nodes of every net, keyed by net.
pub fn net_nodes(board: &Board) -> BTreeMap<NetId, Vec<Node>> {
    let mut out: BTreeMap<NetId, Vec<Node>> = BTreeMap::new();
    for p in board.all_pads() {
        let Some(n) = p.net else { continue };
        if p.layers.is_empty() {
            continue;
        }
        out.entry(n).or_default().push(Node {
            item: BItem::Pad(p.footprint, p.index as u32),
            fill: None,
            layers: layer_mask(p.layers.iter().copied()),
            prims: vec![p.prim.clone()],
            anchors: vec![p.center],
        });
    }
    for t in board.tracks.values() {
        let Some(n) = t.net else { continue };
        let (a, b) = t.endpoints();
        out.entry(n).or_default().push(Node {
            item: BItem::Track(t.id),
            fill: None,
            layers: 1u64 << t.layer,
            prims: t.prims(),
            anchors: vec![a, b],
        });
    }
    for v in board.vias.values() {
        let Some(n) = v.net else { continue };
        out.entry(n).or_default().push(Node {
            item: BItem::Via(v.id),
            fill: None,
            layers: layer_mask(v.layers()),
            prims: vec![Prim::circle(v.at, v.diameter / 2)],
            anchors: vec![v.at],
        });
    }
    for z in board.zones.values() {
        let Some(n) = z.net else { continue };
        for (layer, set) in &z.fill {
            for (k, poly) in set.polys.iter().enumerate() {
                let anchor = poly.outer[0];
                out.entry(n).or_default().push(Node {
                    item: BItem::Zone(z.id),
                    fill: Some((*layer, k as u32)),
                    layers: 1u64 << layer,
                    prims: vec![Prim::polygon(poly.rings().cloned().collect())],
                    anchors: vec![anchor],
                });
            }
        }
    }
    out
}

fn find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

/// Islands of one net's nodes.
pub fn islands(net: NetId, nodes: Vec<Node>) -> NetConnectivity {
    let n = nodes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut grid: Grid<(usize, usize)> = Grid::new(crate::geom::mm(2.0));
    let mut boxes: Vec<Vec<Rect>> = Vec::with_capacity(n);
    for (i, nd) in nodes.iter().enumerate() {
        let bs: Vec<Rect> = nd.prims.iter().map(|p| p.bbox()).collect();
        // Large fill polygons are not put in the grid; they are tested against candidates.
        if nd.fill.is_none() {
            for (k, b) in bs.iter().enumerate() {
                grid.insert((i, k), *b);
            }
        }
        boxes.push(bs);
    }
    for i in 0..n {
        for (k, b) in boxes[i].iter().enumerate() {
            let cands: Vec<(usize, usize)> = if nodes[i].fill.is_some() {
                grid.query(b)
            } else {
                grid.query(&b.inflate(1))
            };
            for (j, kj) in cands {
                if j == i || nodes[i].layers & nodes[j].layers == 0 {
                    continue;
                }
                if find(&mut parent, i) == find(&mut parent, j) {
                    continue;
                }
                if nodes[i].prims[k].gap(&nodes[j].prims[kj]) <= 0.0 {
                    let (a, c) = (find(&mut parent, i), find(&mut parent, j));
                    parent[a.max(c)] = a.min(c);
                }
            }
        }
    }
    // Fill polygons of the same zone and layer that touch each other are already merged by
    // the boolean; different zones on a layer may touch.
    for i in 0..n {
        if nodes[i].fill.is_none() {
            continue;
        }
        for j in i + 1..n {
            if nodes[j].fill.is_none() || nodes[i].layers & nodes[j].layers == 0 {
                continue;
            }
            if boxes[i][0].intersects(&boxes[j][0])
                && nodes[i].prims[0].gap(&nodes[j].prims[0]) <= 0.0
            {
                let (a, c) = (find(&mut parent, i), find(&mut parent, j));
                parent[a.max(c)] = a.min(c);
            }
        }
    }
    let mut ids: BTreeMap<usize, usize> = BTreeMap::new();
    let mut island = vec![0; n];
    for (i, slot) in island.iter_mut().enumerate() {
        let r = find(&mut parent, i);
        let next = ids.len();
        *slot = *ids.entry(r).or_insert(next);
    }
    NetConnectivity {
        net,
        islands: ids.len(),
        island,
        nodes,
    }
}

/// Connectivity of every net.
pub fn board_connectivity(board: &Board) -> BTreeMap<NetId, NetConnectivity> {
    net_nodes(board)
        .into_iter()
        .map(|(n, nodes)| (n, islands(n, nodes)))
        .collect()
}
