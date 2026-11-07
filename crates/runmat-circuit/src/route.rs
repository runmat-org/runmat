//! Routing primitives (push-and-shove is P5 and builds on these).
//!
//! - [`ratsnest`]: per net, the exact minimum spanning tree over the net's copper *islands*
//!   (Borůvka over a component-labelled k-d tree of anchors), giving the airwires still to
//!   route.
//! - [`walkaround`]: the geometry the interactive router's walkaround mode needs: obstacles
//!   of other nets on the layer grown by clearance plus half the track width and merged with
//!   the polygon booleans, then the shortest path around them on the visibility graph.
//! - [`insert_track`] / [`insert_via`]: add an object, run the incremental DRC, and roll back
//!   if the object introduced an error.
//! - [`trace_length`]: the tuning length between two pads (shortest path over tracks and
//!   vias, vias counted by their span through the stackup), plus [`net_length`].

use crate::layer_set::CopperIndex;
use crate::board_rules::{props, BItem, BoardGraph, BoardRules, TrackProbe};
use crate::connectivity::{board_connectivity, NetConnectivity};
use crate::drc::{Drc, DrcFinding};
use crate::elab::Severity;
use crate::geom::poly::{boolean, BoolOp, PolySet};
use crate::geom::shape::{segments_intersect, Prim};
use crate::geom::{Coord, Point, Rect};
use crate::layout::{Board, ItemId, NetId, Track, TrackGeom, Via};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

#[derive(Clone, Debug, PartialEq)]
pub struct Airwire {
    pub net: NetId,
    pub a: Point,
    pub b: Point,
    pub from: BItem,
    pub to: BItem,
    pub length: f64,
}

/// Airwires for one net's connectivity: the minimum spanning tree over the net's islands,
/// where joining two islands costs the shortest distance between their anchors.
///
/// Exact and sub-quadratic: Borůvka rounds over a k-d tree of the anchors. Each node of the
/// tree records the component all its anchors belong to (if they agree), so a query for the
/// nearest anchor outside a component skips whole subtrees inside it; a large island
/// surrounded by its own anchors costs little. Distances are exact squared integers and edges
/// are ordered by `(distance², lower anchor, higher anchor)`, a strict total order, so the
/// tree is the unique MST under that order (what Kruskal over every pair gives). Airwires are
/// listed in that order.
pub fn net_ratsnest(c: &NetConnectivity) -> Vec<Airwire> {
    if c.islands <= 1 {
        return vec![];
    }
    let mut anchors: Vec<(Point, usize, BItem)> = Vec::new();
    for (i, n) in c.nodes.iter().enumerate() {
        for a in &n.anchors {
            anchors.push((*a, c.island[i], n.item));
        }
    }
    let pts: Vec<Point> = anchors.iter().map(|a| a.0).collect();
    let mut comp: Vec<usize> = anchors.iter().map(|a| a.1).collect();
    let tree = KdTree::new(&pts);
    let mut uf: Vec<usize> = (0..c.islands).collect();
    fn root(uf: &mut [usize], mut x: usize) -> usize {
        while uf[x] != x {
            uf[x] = uf[uf[x]];
            x = uf[x];
        }
        x
    }
    let mut edges: Vec<MstEdge> = Vec::new();
    let mut components = c.islands;
    while components > 1 {
        let labels = tree.labels(&comp);
        // Cheapest outgoing edge per component.
        let mut best: BTreeMap<usize, MstEdge> = BTreeMap::new();
        for i in 0..pts.len() {
            if let Some(e) = tree.nearest_foreign(&pts, &comp, &labels, i) {
                let k = comp[i];
                if best.get(&k).is_none_or(|b| e < *b) {
                    best.insert(k, e);
                }
            }
        }
        if best.is_empty() {
            break;
        }
        let mut merged = false;
        for e in best.into_values() {
            let (ra, rb) = (root(&mut uf, comp[e.a]), root(&mut uf, comp[e.b]));
            if ra != rb {
                uf[ra.max(rb)] = ra.min(rb);
                components -= 1;
                edges.push(e);
                merged = true;
            }
        }
        if !merged {
            break;
        }
        for k in comp.iter_mut() {
            *k = root(&mut uf, *k);
        }
    }
    edges.sort();
    edges
        .into_iter()
        .map(|e| Airwire {
            net: c.net,
            a: pts[e.a],
            b: pts[e.b],
            from: anchors[e.a].2,
            to: anchors[e.b].2,
            length: (e.d2 as f64).sqrt(),
        })
        .collect()
}

/// An MST candidate edge between anchors `a < b`, ordered by `(d², a, b)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MstEdge {
    d2: i128,
    a: usize,
    b: usize,
}

fn d2(p: Point, q: Point) -> i128 {
    let (dx, dy) = ((p.x - q.x) as i128, (p.y - q.y) as i128);
    dx * dx + dy * dy
}

/// A static 2-d tree over points (median splits, alternating axes).
struct KdTree {
    /// Point indices in tree order; node `n` covers `order[lo..hi]`.
    order: Vec<usize>,
    nodes: Vec<KdNode>,
}

struct KdNode {
    lo: usize,
    hi: usize,
    bbox: Rect,
    children: Option<(usize, usize)>,
}

const KD_LEAF: usize = 8;
const MIXED: usize = usize::MAX;

impl KdTree {
    fn new(pts: &[Point]) -> KdTree {
        let mut t = KdTree {
            order: (0..pts.len()).collect(),
            nodes: Vec::new(),
        };
        t.build(pts, 0, pts.len(), 0);
        t
    }
    fn build(&mut self, pts: &[Point], lo: usize, hi: usize, depth: usize) -> usize {
        let bbox = Rect::from_points(self.order[lo..hi].iter().map(|&i| pts[i]));
        let id = self.nodes.len();
        self.nodes.push(KdNode {
            lo,
            hi,
            bbox,
            children: None,
        });
        if hi - lo > KD_LEAF {
            let mid = (lo + hi) / 2;
            let x_axis = bbox.width() >= bbox.height() || depth > 64;
            self.order[lo..hi].select_nth_unstable_by_key(mid - lo, |&i| {
                if x_axis {
                    (pts[i].x, pts[i].y, i)
                } else {
                    (pts[i].y, pts[i].x, i)
                }
            });
            let l = self.build(pts, lo, mid, depth + 1);
            let r = self.build(pts, mid, hi, depth + 1);
            self.nodes[id].children = Some((l, r));
        }
        id
    }
    /// Per node, the component shared by all its points, or `MIXED`.
    fn labels(&self, comp: &[usize]) -> Vec<usize> {
        let mut out = vec![MIXED; self.nodes.len()];
        for n in (0..self.nodes.len()).rev() {
            let node = &self.nodes[n];
            out[n] = match node.children {
                Some((l, r)) if out[l] == out[r] => out[l],
                Some(_) => MIXED,
                None => {
                    let first = comp[self.order[node.lo]];
                    if self.order[node.lo..node.hi]
                        .iter()
                        .all(|&i| comp[i] == first)
                    {
                        first
                    } else {
                        MIXED
                    }
                }
            };
        }
        out
    }
    /// The least edge (under the MST order) from point `q` to a point of another component.
    fn nearest_foreign(
        &self,
        pts: &[Point],
        comp: &[usize],
        labels: &[usize],
        q: usize,
    ) -> Option<MstEdge> {
        let mut best: Option<MstEdge> = None;
        let mut stack = vec![0usize];
        let p = pts[q];
        let box_d2 = |r: &Rect| {
            let dx = (r.min.x - p.x).max(0).max(p.x - r.max.x) as i128;
            let dy = (r.min.y - p.y).max(0).max(p.y - r.max.y) as i128;
            dx * dx + dy * dy
        };
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            if labels[n] == comp[q] {
                continue;
            }
            // `>` keeps equal-distance subtrees: their indices may win the tie.
            if best.is_some_and(|b| box_d2(&node.bbox) > b.d2) {
                continue;
            }
            match node.children {
                Some((l, r)) => {
                    let (dl, dr) = (box_d2(&self.nodes[l].bbox), box_d2(&self.nodes[r].bbox));
                    // Visit the nearer child first (pushed last).
                    if dl <= dr {
                        stack.push(r);
                        stack.push(l);
                    } else {
                        stack.push(l);
                        stack.push(r);
                    }
                }
                None => {
                    for &i in &self.order[node.lo..node.hi] {
                        if comp[i] == comp[q] {
                            continue;
                        }
                        let e = MstEdge {
                            d2: d2(p, pts[i]),
                            a: q.min(i),
                            b: q.max(i),
                        };
                        if best.is_none_or(|b| e < b) {
                            best = Some(e);
                        }
                    }
                }
            }
        }
        best
    }
}

/// Airwires for every net, ordered by net.
pub fn ratsnest(board: &Board) -> Vec<Airwire> {
    board_connectivity(board)
        .values()
        .flat_map(net_ratsnest)
        .collect()
}

/// Obstacles for routing `net` on `layer` with `width`: other-net copper grown by the rule
/// clearance plus half the width, merged.
pub fn obstacles(
    board: &Board,
    rules: &BoardRules,
    net: Option<NetId>,
    layer: CopperIndex,
    width: Coord,
    region: Rect,
) -> PolySet {
    let tol = crate::geom::mm(0.01);
    let mut rings = Vec::new();
    let n = board.copper_count();
    let probe = rules.probe(board, TrackProbe { net, layer, width });
    let mut add = |item: BItem, item_net: Option<NetId>, p: &Prim| {
        if net.is_some() && item_net == net {
            return;
        }
        let clr = rules
            .probe_len(&probe, item, props::CLEARANCE)
            .max(board.limits.min_clearance);
        let grow = clr + width / 2 + 1000;
        if !p.bbox().inflate(grow).intersects(&region) {
            return;
        }
        rings.extend(p.outline(grow, tol));
    };
    for p in board.all_pads() {
        if p.layers.contains(&layer) {
            add(BItem::Pad(p.footprint, p.index as u32), p.net, &p.prim);
        }
    }
    for t in board.tracks.values() {
        if t.layer == layer {
            for p in t.prims() {
                add(BItem::Track(t.id), t.net, &p);
            }
        }
    }
    for v in board.vias.values() {
        if v.layers().any(|l| l == layer) {
            add(BItem::Via(v.id), v.net, &Prim::circle(v.at, v.diameter / 2));
        }
    }
    for z in board.zones.values() {
        if let Some(f) = z.fill.get(&layer) {
            if net.is_none() || z.net != net {
                let grow = rules
                    .probe_len(&probe, BItem::Zone(z.id), props::CLEARANCE)
                    .max(board.limits.min_clearance)
                    + width / 2
                    + 1000;
                rings.extend(
                    crate::geom::offset::offset_set(f, grow, crate::geom::offset::Join::Round, tol)
                        .rings(),
                );
            }
        }
    }
    for k in board.keepouts.values() {
        if k.rules.tracks && (k.layers.is_empty() || k.layers.contains(&layer)) {
            rings.push(k.outline.outer.clone());
        }
    }
    let _ = n;
    boolean(&rings, &[], BoolOp::Union)
}

fn blocked(set: &PolySet, a: Point, b: Point) -> bool {
    for poly in &set.polys {
        let bb = Rect::new(a, b);
        if !poly.bbox().intersects(&bb) {
            continue;
        }
        for r in poly.rings() {
            let m = r.len();
            for i in 0..m {
                let (c, d) = (r[i], r[(i + 1) % m]);
                if c == a || c == b || d == a || d == b {
                    continue;
                }
                if segments_intersect(a, b, c, d) {
                    // Touching a vertex tangentially is allowed only at endpoints; a crossing
                    // of an edge interior blocks.
                    let o1 = crate::geom::orient(a, b, c);
                    let o2 = crate::geom::orient(a, b, d);
                    if o1 != 0 && o2 != 0 {
                        return true;
                    }
                }
            }
        }
        let mid = Point::new(a.x + (b.x - a.x) / 2, a.y + (b.y - a.y) / 2);
        if poly.contains(mid) == Some(true) {
            return true;
        }
    }
    false
}

/// Shortest path from `a` to `b` around `obstacles` (any-angle), or `None` if `a` or `b` is
/// inside an obstacle or no path exists. The path includes both ends.
pub fn shortest_path(obstacles: &PolySet, a: Point, b: Point) -> Option<Vec<Point>> {
    if obstacles.contains(a) == Some(true) || obstacles.contains(b) == Some(true) {
        return None;
    }
    if !blocked(obstacles, a, b) {
        return Some(vec![a, b]);
    }
    let mut pts = vec![a, b];
    for p in &obstacles.polys {
        for r in p.rings() {
            let m = r.len();
            for i in 0..m {
                // Only convex corners of obstacles (reflex corners of free space) can lie on a
                // shortest path.
                let (prev, cur, next) = (r[(i + m - 1) % m], r[i], r[(i + 1) % m]);
                if crate::geom::orient(prev, cur, next) > 0 {
                    pts.push(cur);
                }
            }
        }
    }
    let n = pts.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    dist[0] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((0u64, 0usize)));
    let mut done = vec![false; n];
    while let Some(Reverse((_, u))) = heap.pop() {
        if done[u] {
            continue;
        }
        done[u] = true;
        if u == 1 {
            break;
        }
        for v in 0..n {
            if done[v] || v == u {
                continue;
            }
            let d = dist[u] + pts[u].dist(pts[v]);
            if d < dist[v] && !blocked(obstacles, pts[u], pts[v]) {
                dist[v] = d;
                prev[v] = u;
                heap.push(Reverse((d as u64, v)));
            }
        }
    }
    if !dist[1].is_finite() {
        return None;
    }
    let mut path = vec![1];
    while *path.last().unwrap() != 0 {
        path.push(prev[*path.last().unwrap()]);
    }
    path.reverse();
    Some(path.into_iter().map(|i| pts[i]).collect())
}

/// Walkaround route for `net` from `a` to `b` on `layer` with `width`.
pub fn walkaround(
    board: &Board,
    rules: &BoardRules,
    net: Option<NetId>,
    layer: CopperIndex,
    width: Coord,
    a: Point,
    b: Point,
) -> Option<Vec<Point>> {
    let margin = crate::geom::mm(10.0) + a.dist(b) as Coord / 2;
    let region = Rect::new(a, b).inflate(margin);
    let obs = obstacles(board, rules, net, layer, width, region);
    shortest_path(&obs, a, b)
}

#[allow(clippy::too_many_arguments)]
/// Adds a track, re-runs the DRC incrementally and keeps it only if it introduced no error.
/// Returns the new id, or the errors it would have caused.
pub fn insert_track(
    board: &mut Board,
    rules: &mut BoardRules,
    drc: &mut Drc,
    net: Option<NetId>,
    layer: CopperIndex,
    width: Coord,
    a: Point,
    b: Point,
) -> Result<ItemId, Vec<DrcFinding>> {
    let id = board.alloc();
    board.tracks.insert(
        id,
        Track {
            id,
            net,
            layer,
            width,
            geom: TrackGeom::Segment { a, b },
            locked: false,
        },
    );
    commit_or_rollback(board, rules, drc, BItem::Track(id))
}

/// Adds a via spanning `from..=to`, checked like [`insert_track`].
#[allow(clippy::too_many_arguments)]
pub fn insert_via(
    board: &mut Board,
    rules: &mut BoardRules,
    drc: &mut Drc,
    net: Option<NetId>,
    at: Point,
    diameter: Coord,
    drill: Coord,
    span: (CopperIndex, CopperIndex),
) -> Result<ItemId, Vec<DrcFinding>> {
    let id = board.alloc();
    board.vias.insert(
        id,
        Via {
            id,
            net,
            at,
            diameter,
            drill,
            from: span.0.min(span.1),
            to: span.0.max(span.1),
            micro: false,
            tented: true,
            locked: false,
        },
    );
    commit_or_rollback(board, rules, drc, BItem::Via(id))
}

fn refresh_rules(board: &Board, rules: &mut BoardRules, it: BItem, removed: bool) {
    let g = BoardGraph::new(board, &[]);
    if removed {
        rules.single.items_changed(&g, &[], &[it]);
        rules.pair.items_changed(&g, &[], &[it]);
    } else {
        rules.single.items_changed(&g, &[it], &[]);
        rules.pair.items_changed(&g, &[it], &[]);
    }
}

fn commit_or_rollback(
    board: &mut Board,
    rules: &mut BoardRules,
    drc: &mut Drc,
    it: BItem,
) -> Result<ItemId, Vec<DrcFinding>> {
    refresh_rules(board, rules, it, false);
    let before: BTreeSet<String> = drc.findings().iter().map(|f| finding_key(f)).collect();
    drc.update(board, rules, &[it], &[], None);
    let new_errors: Vec<DrcFinding> = drc
        .findings()
        .into_iter()
        .filter(|f| {
            f.severity == Severity::Error
                && f.items.contains(&it)
                && f.code != "unrouted"
                && !before.contains(&finding_key(f))
        })
        .cloned()
        .collect();
    let id = match it {
        BItem::Track(i) | BItem::Via(i) => i,
        _ => unreachable!(),
    };
    if new_errors.is_empty() {
        return Ok(id);
    }
    match it {
        BItem::Track(i) => {
            board.tracks.remove(&i);
        }
        BItem::Via(i) => {
            board.vias.remove(&i);
        }
        _ => {}
    }
    refresh_rules(board, rules, it, true);
    drc.update(board, rules, &[], &[it], None);
    Err(new_errors)
}

fn finding_key(f: &DrcFinding) -> String {
    format!("{}|{:?}|{:?}", f.code, f.items, f.layer)
}

/// Total routed length of a net: tracks plus via spans through the stackup (nm).
pub fn net_length(board: &Board, net: NetId) -> f64 {
    let t: f64 = board
        .tracks
        .values()
        .filter(|t| t.net == Some(net))
        .map(|t| t.length())
        .sum();
    let v: f64 = board
        .vias
        .values()
        .filter(|v| v.net == Some(net))
        .map(|v| board.stackup.span_length(v.from, v.to) as f64)
        .sum();
    t + v
}

/// Tuning length between two pads (by footprint reference and pad name): the shortest path
/// over the net's tracks and vias, each via counted by the depth between the layers it joins.
/// `None` when the pads are not connected by routed copper.
pub fn trace_length(board: &Board, from: (&str, &str), to: (&str, &str)) -> Option<f64> {
    let n = board.copper_count();
    let pad = |r: &str, p: &str| {
        let fp = board.footprint_by_ref(r)?;
        fp.pads(n).into_iter().find(|x| x.name == p)
    };
    let pa = pad(from.0, from.1)?;
    let pb = pad(to.0, to.1)?;
    let net = pa.net?;
    if pb.net != Some(net) {
        return None;
    }
    // Nodes: (point, layer); 0 = start pad, 1 = end pad.
    let mut ids: BTreeMap<(Point, CopperIndex), usize> = BTreeMap::new();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![], vec![]];
    let node = |p: Point,
                l: CopperIndex,
                ids: &mut BTreeMap<(Point, CopperIndex), usize>,
                adj: &mut Vec<Vec<(usize, f64)>>| {
        *ids.entry((p, l)).or_insert_with(|| {
            adj.push(vec![]);
            adj.len() - 1
        })
    };
    let edge = |a: usize, b: usize, w: f64, adj: &mut Vec<Vec<(usize, f64)>>| {
        adj[a].push((b, w));
        adj[b].push((a, w));
    };
    let tracks: Vec<&Track> = board
        .tracks
        .values()
        .filter(|t| t.net == Some(net))
        .collect();
    let mut ends: Vec<(Point, CopperIndex, usize)> = Vec::new();
    for t in &tracks {
        let (a, b) = t.endpoints();
        let ia = node(a, t.layer, &mut ids, &mut adj);
        let ib = node(b, t.layer, &mut ids, &mut adj);
        edge(ia, ib, t.length(), &mut adj);
        ends.push((a, t.layer, ia));
        ends.push((b, t.layer, ib));
    }
    for v in board.vias.values().filter(|v| v.net == Some(net)) {
        let mut prev: Option<(CopperIndex, usize)> = None;
        for l in v.layers() {
            let i = node(v.at, l, &mut ids, &mut adj);
            if let Some((pl, pi)) = prev {
                edge(pi, i, board.stackup.span_length(pl, l) as f64, &mut adj);
            }
            prev = Some((l, i));
            // Track ends inside the via land join it.
            for &(p, tl, ti) in &ends {
                if tl == l && p != v.at && p.dist(v.at) <= v.diameter as f64 / 2.0 {
                    edge(i, ti, p.dist(v.at), &mut adj);
                }
            }
        }
    }
    for (pi, pp) in [(0usize, &pa), (1usize, &pb)] {
        for &(p, l, ti) in &ends {
            if pp.layers.contains(&l) && pp.prim.contains(p) {
                edge(pi, ti, p.dist(pp.center), &mut adj);
            }
        }
    }
    // Dijkstra.
    let mut dist = vec![f64::INFINITY; adj.len()];
    dist[0] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((0u64, 0usize)));
    while let Some(Reverse((_, u))) = heap.pop() {
        if u == 1 {
            break;
        }
        for &(v, w) in &adj[u].clone() {
            let d = dist[u] + w;
            if d < dist[v] {
                dist[v] = d;
                heap.push(Reverse(((d * 1000.0) as u64, v)));
            }
        }
    }
    dist[1].is_finite().then_some(dist[1])
}

/// Length difference of a differential pair between matching pad pairs (P minus N).
pub fn pair_skew(
    board: &Board,
    p: ((&str, &str), (&str, &str)),
    n: ((&str, &str), (&str, &str)),
) -> Option<f64> {
    Some(trace_length(board, p.0, p.1)? - trace_length(board, n.0, n.1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::Node;

    fn net(points: &[(i64, i64, usize)]) -> NetConnectivity {
        let islands = points.iter().map(|p| p.2).max().unwrap() + 1;
        NetConnectivity {
            net: NetId(0),
            nodes: points
                .iter()
                .enumerate()
                .map(|(i, p)| Node {
                    item: BItem::Via(ItemId(i as u64)),
                    fill: None,
                    layers: crate::layer_set::LayerSet::single(0),
                    prims: vec![],
                    anchors: vec![Point::new(p.0, p.1)],
                })
                .collect(),
            island: points.iter().map(|p| p.2).collect(),
            islands,
        }
    }

    /// Kruskal over every anchor pair, under the same edge order.
    fn reference(c: &NetConnectivity) -> Vec<(Point, Point, i128)> {
        let pts: Vec<(Point, usize)> = c
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.anchors.iter().map(move |a| (*a, c.island[i])))
            .collect();
        let mut all = Vec::new();
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                if pts[i].1 != pts[j].1 {
                    all.push(MstEdge {
                        d2: d2(pts[i].0, pts[j].0),
                        a: i,
                        b: j,
                    });
                }
            }
        }
        all.sort();
        let mut uf: Vec<usize> = (0..c.islands).collect();
        fn root(uf: &mut [usize], mut x: usize) -> usize {
            while uf[x] != x {
                x = uf[x];
            }
            x
        }
        let mut out = Vec::new();
        for e in all {
            let (ra, rb) = (root(&mut uf, pts[e.a].1), root(&mut uf, pts[e.b].1));
            if ra != rb {
                uf[ra.max(rb)] = ra.min(rb);
                out.push((pts[e.a].0, pts[e.b].0, e.d2));
            }
        }
        out
    }

    #[test]
    fn ratsnest_matches_exact_kruskal() {
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for case in 0..40 {
            let n = 2 + (next() % 400) as usize;
            let islands = 1 + (next() % n.min(60) as u64) as usize;
            // Some cases on a coarse lattice to force distance ties.
            let lattice = case % 3 == 0;
            let pts: Vec<(i64, i64, usize)> = (0..n)
                .map(|i| {
                    let (x, y) = if lattice {
                        (
                            (next() % 20) as i64 * 500_000,
                            (next() % 20) as i64 * 500_000,
                        )
                    } else {
                        ((next() % 100_000_000) as i64, (next() % 60_000_000) as i64)
                    };
                    // Every island gets at least one anchor; the rest cluster loosely.
                    (
                        x,
                        y,
                        if i < islands {
                            i
                        } else {
                            (next() as usize) % islands
                        },
                    )
                })
                .collect();
            let c = net(&pts);
            let got: Vec<(Point, Point, i128)> = net_ratsnest(&c)
                .iter()
                .map(|w| (w.a, w.b, d2(w.a, w.b)))
                .collect();
            let want = reference(&c);
            assert_eq!(got.len(), islands - 1, "case {case}");
            assert_eq!(got, want, "case {case}");
        }
    }

    #[test]
    fn ratsnest_scales() {
        // 20 000 anchors in 2 000 islands: the old per-island Prim step was quadratic.
        let mut pts = Vec::new();
        for i in 0..20_000i64 {
            pts.push(((i % 200) * 700_000, (i / 200) * 650_000, (i / 10) as usize));
        }
        let c = net(&pts);
        let t = std::time::Instant::now();
        let w = net_ratsnest(&c);
        assert_eq!(w.len(), 1_999);
        assert!(t.elapsed().as_secs_f64() < 5.0, "{:?}", t.elapsed());
    }
}
