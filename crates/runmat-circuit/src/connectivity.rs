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
use crate::geom::shape::{Core, Prim};
use crate::geom::{Point, Rect};
use crate::layout::{Board, NetId};
use crate::spatial::{EdgeIndex, Grid};
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
///
/// Small nodes go in a grid and are tested pairwise. A fill polygon is never tested as a
/// polygon against each candidate (that would cost its whole vertex count per test): its
/// edges are indexed once, and each candidate is probed against the index.
pub fn islands(net: NetId, nodes: Vec<Node>) -> NetConnectivity {
    let n = nodes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut grid: Grid<(usize, usize)> = Grid::new(crate::geom::mm(2.0));
    let mut boxes: Vec<Vec<Rect>> = Vec::with_capacity(n);
    for (i, nd) in nodes.iter().enumerate() {
        let bs: Vec<Rect> = nd.prims.iter().map(|p| p.bbox()).collect();
        if nd.fill.is_none() {
            for (k, b) in bs.iter().enumerate() {
                grid.insert((i, k), *b);
            }
        }
        boxes.push(bs);
    }
    let union = |parent: &mut Vec<usize>, i: usize, j: usize| {
        let (a, c) = (find(parent, i), find(parent, j));
        parent[a.max(c)] = a.min(c);
    };
    let mut fills: Vec<(usize, EdgeIndex)> = Vec::new();
    for i in 0..n {
        if nodes[i].fill.is_some() {
            if let Core::Poly(rs) = &nodes[i].prims[0].core {
                fills.push((i, EdgeIndex::new(rs.iter(), crate::geom::mm(1.0))));
            }
            continue;
        }
        for (k, b) in boxes[i].iter().enumerate() {
            for (j, kj) in grid.query(&b.inflate(1)) {
                if j <= i || nodes[i].layers & nodes[j].layers == 0 {
                    continue;
                }
                if find(&mut parent, i) == find(&mut parent, j) {
                    continue;
                }
                if nodes[i].prims[k].gap(&nodes[j].prims[kj]) <= 0.0 {
                    union(&mut parent, i, j);
                }
            }
        }
    }
    for (i, idx) in &fills {
        for (j, kj) in grid.query(&boxes[*i][0]) {
            if nodes[*i].layers & nodes[j].layers == 0
                || find(&mut parent, *i) == find(&mut parent, j)
            {
                continue;
            }
            if idx.touches(&nodes[j].prims[kj]) {
                union(&mut parent, *i, j);
            }
        }
    }
    // Fill polygons of the same zone and layer that touch each other are already merged by
    // the boolean; different zones on a layer may touch.
    for (a, (i, ia)) in fills.iter().enumerate() {
        for (j, _) in &fills[a + 1..] {
            if nodes[*i].layers & nodes[*j].layers == 0
                || !boxes[*i][0].intersects(&boxes[*j][0])
                || find(&mut parent, *i) == find(&mut parent, *j)
            {
                continue;
            }
            let other = &nodes[*j].prims[0];
            let Core::Poly(rs) = &other.core else {
                continue;
            };
            let touch = rs.iter().any(|r| {
                (0..r.len()).any(|e| {
                    ia.probe(&Prim::segment(r[e], r[(e + 1) % r.len()], 0), 1)
                        .gap
                        <= 0.0
                })
            }) || ia.touches(other);
            if touch {
                union(&mut parent, *i, *j);
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
