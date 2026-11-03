//! Routing primitives (push-and-shove is P5 and builds on these).
//!
//! - [`ratsnest`]: per net, a minimum spanning tree over the net's copper *islands* (Prim's
//!   algorithm on anchors, with zero cost inside an island), giving the airwires still to
//!   route.
//! - [`walkaround`]: the geometry the interactive router's walkaround mode needs: obstacles
//!   of other nets on the layer grown by clearance plus half the track width and merged with
//!   the polygon booleans, then the shortest path around them on the visibility graph.
//! - [`insert_track`] / [`insert_via`]: add an object, run the incremental DRC, and roll back
//!   if the object introduced an error.
//! - [`trace_length`]: the tuning length between two pads (shortest path over tracks and
//!   vias, vias counted by their span through the stackup), plus [`net_length`].

use crate::board_rules::{props, BItem, BoardGraph, BoardRules};
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

/// Airwires for one net's connectivity.
pub fn net_ratsnest(c: &NetConnectivity) -> Vec<Airwire> {
    if c.islands <= 1 {
        return vec![];
    }
    // Anchors with their island.
    let mut anchors: Vec<(Point, usize, BItem)> = Vec::new();
    for (i, n) in c.nodes.iter().enumerate() {
        for a in &n.anchors {
            anchors.push((*a, c.island[i], n.item));
        }
    }
    let m = anchors.len();
    let mut in_tree = vec![false; m];
    let mut island_in = vec![false; c.islands];
    let mut best = vec![f64::INFINITY; m];
    let mut from = vec![usize::MAX; m];
    let mut out = Vec::new();
    let add_island = |isl: usize,
                      in_tree: &mut Vec<bool>,
                      island_in: &mut Vec<bool>,
                      best: &mut Vec<f64>,
                      from: &mut Vec<usize>| {
        island_in[isl] = true;
        let members: Vec<usize> = (0..m).filter(|&k| anchors[k].1 == isl).collect();
        for &k in &members {
            in_tree[k] = true;
        }
        for j in 0..m {
            if in_tree[j] {
                continue;
            }
            for &k in &members {
                let d = anchors[j].0.dist(anchors[k].0);
                if d < best[j] || (d == best[j] && k < from[j]) {
                    best[j] = d;
                    from[j] = k;
                }
            }
        }
    };
    add_island(
        anchors[0].1,
        &mut in_tree,
        &mut island_in,
        &mut best,
        &mut from,
    );
    while island_in.iter().any(|x| !x) {
        let mut pick = None;
        for j in 0..m {
            if !in_tree[j]
                && !island_in[anchors[j].1]
                && pick.is_none_or(|p: usize| best[j] < best[p])
            {
                pick = Some(j);
            }
        }
        let Some(j) = pick else { break };
        let k = from[j];
        out.push(Airwire {
            net: c.net,
            a: anchors[k].0,
            b: anchors[j].0,
            from: anchors[k].2,
            to: anchors[j].2,
            length: best[j],
        });
        add_island(
            anchors[j].1,
            &mut in_tree,
            &mut island_in,
            &mut best,
            &mut from,
        );
    }
    out
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
    layer: u8,
    width: Coord,
    region: Rect,
) -> PolySet {
    let tol = crate::geom::mm(0.01);
    let mut rings = Vec::new();
    let n = board.copper_count();
    let probe = BItem::Track(ItemId(u64::MAX));
    let mut add = |item: BItem, item_net: Option<NetId>, p: &Prim| {
        if net.is_some() && item_net == net {
            return;
        }
        let clr = rules
            .pair_len(probe, item, props::CLEARANCE)
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
                let grow = board.limits.min_clearance + width / 2 + 1000;
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
    layer: u8,
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
    layer: u8,
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
    span: (u8, u8),
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
    let mut ids: BTreeMap<(Point, u8), usize> = BTreeMap::new();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![], vec![]];
    let node = |p: Point,
                l: u8,
                ids: &mut BTreeMap<(Point, u8), usize>,
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
    let mut ends: Vec<(Point, u8, usize)> = Vec::new();
    for t in &tracks {
        let (a, b) = t.endpoints();
        let ia = node(a, t.layer, &mut ids, &mut adj);
        let ib = node(b, t.layer, &mut ids, &mut adj);
        edge(ia, ib, t.length(), &mut adj);
        ends.push((a, t.layer, ia));
        ends.push((b, t.layer, ib));
    }
    for v in board.vias.values().filter(|v| v.net == Some(net)) {
        let mut prev: Option<(u8, usize)> = None;
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
