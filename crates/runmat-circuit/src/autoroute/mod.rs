//! Automatic routing: fanout of pads to vias, and a gridded maze router.
//!
//! - [`fanout`]: dog-bone fanout. BGA balls get a short track to a via on the diagonal between
//!   balls, away from the package centre; other SMD pads get a via just past the pad's outer
//!   end (or beside it). Pads already joined to a via are left alone, so a run can repeat.
//! - [`autoroute`]: routes airwires one at a time with an A* search over a square grid in the
//!   eight 0°/45°/90° directions on the routing layers, changing layer through vias. Every
//!   placed object keeps the given clearance from other nets, the hole-to-hole minimum and the
//!   edge clearance, so routes pass DRC by construction. The path is then straightened: runs
//!   merge into single segments and corners cut where a 0°/45°/90° shortcut stays clear.
//!   Connections it cannot route stay airwires and are listed in the report.
//!
//! Both work on a [`Board`] model; the runtime's `board.fanout` and `board.autoroute` commands
//! run them and write the new copper into the layout as statements.

mod fanout;
mod search;
mod space;
mod star;

pub use fanout::{fanout, FanoutOptions, FanoutReport};
pub use space::Space;

use crate::layer_set::CopperIndex;
use crate::connectivity::layer_mask;
use crate::layer_set::LayerSet;
use crate::geom::shape::Prim;
use crate::geom::{mm, Coord, Point};
use crate::layout::{Board, NetId};
use search::{Node, Query, Search};
use std::collections::BTreeMap;

/// What [`autoroute`] routes and how.
#[derive(Clone, Debug)]
pub struct RouteOptions {
    /// Nets to route; airwires of other nets are left.
    pub nets: Vec<NetId>,
    /// Copper layers to route on, preferred first.
    pub layers: Vec<CopperIndex>,
    /// Track width per net; nets not listed use `default_width`.
    pub widths: BTreeMap<NetId, Coord>,
    pub default_width: Coord,
    /// Via diameter and drill.
    pub via: (Coord, Coord),
    /// Clearance kept to other nets (at least the largest the board's rules require).
    pub clearance: Coord,
    /// Grid pitch.
    pub grid: Coord,
}

impl RouteOptions {
    pub fn new(nets: Vec<NetId>, layers: Vec<CopperIndex>, clearance: Coord) -> Self {
        RouteOptions {
            nets,
            layers,
            widths: BTreeMap::new(),
            default_width: mm(0.2),
            via: (mm(0.45), mm(0.2)),
            clearance,
            grid: mm(0.05),
        }
    }
}

/// What [`autoroute`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouteReport {
    /// Airwires routed.
    pub routed: u32,
    /// Airwires left unrouted.
    pub failed: Vec<Unrouted>,
    pub tracks: u32,
    pub vias: u32,
}

/// An airwire the router left, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Unrouted {
    pub net: String,
    pub a: Point,
    pub b: Point,
    pub why: Why,
}

/// Why a connection stayed unrouted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// No track of the net's width can leave the first end with the clearance.
    StartBlocked,
    /// No track can reach the second end.
    EndBlocked,
    /// The search ran out of room or budget between the ends.
    NoPath,
    /// The only path found would put copper inside another net's clearance at an end.
    Crowded,
}

impl Why {
    pub fn text(self) -> &'static str {
        match self {
            Why::StartBlocked => "no room to leave the first pad",
            Why::EndBlocked => "no room to reach the second pad",
            Why::NoPath => "no path found",
            Why::Crowded => "no path keeps the clearance at its ends",
        }
    }
}

/// Routes the airwires of `opts.nets`, shortest first.
pub fn autoroute(board: &mut Board, opts: &RouteOptions) -> RouteReport {
    let mut space = Space::new(board, opts.clearance);
    let wires = crate::route::ratsnest(board);
    autoroute_in(board, &mut space, opts, wires)
}

/// [`autoroute`] over given airwires (a ratsnest that counts poured copper) with a [`Space`]
/// the caller built (and keeps using).
pub fn autoroute_in(
    board: &mut Board,
    space: &mut Space,
    opts: &RouteOptions,
    wires: Vec<crate::route::Airwire>,
) -> RouteReport {
    let mut report = RouteReport::default();
    let wanted: std::collections::BTreeSet<NetId> = opts.nets.iter().copied().collect();
    let wires: Vec<crate::route::Airwire> = wires
        .into_iter()
        .filter(|a| wanted.contains(&a.net))
        .collect();
    let n = board.copper_count();
    let pads = board.all_pads();
    let layer_cost: Vec<f64> = opts
        .layers
        .iter()
        .enumerate()
        .map(|(i, _)| 1.0 + 0.25 * i as f64)
        .collect();
    let rooms: Vec<crate::geom::Rect> = board.rooms.iter().map(|r| r.outline.bbox()).collect();
    let mut wires = star::star_between_rooms(wires, &rooms);
    wires.sort_by(|a, b| {
        a.length
            .partial_cmp(&b.length)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| (a.net, a.a, a.b).cmp(&(b.net, b.a, b.b)))
    });
    for w in wires {
        let avoid: Vec<(crate::geom::Rect, Option<Point>)> = rooms
            .iter()
            .filter_map(|r| match (r.contains(w.a), r.contains(w.b)) {
                (true, true) => None,
                (true, false) => Some((*r, Some(w.a))),
                (false, true) => Some((*r, Some(w.b))),
                (false, false) => Some((*r, None)),
            })
            .collect();
        let width = opts
            .widths
            .get(&w.net)
            .copied()
            .unwrap_or(opts.default_width);
        let neck = [w.a, w.b]
            .iter()
            .filter_map(|p| pad_width(&pads, w.net, *p))
            .min()
            .map_or(width, |pad| pad.min(width).max(board.limits.min_track));
        let q = Query {
            net: w.net,
            width,
            via: opts.via,
            via_mask: layer_mask(0..n),
            layers: &opts.layers,
            grid: opts.grid,
            origin: w.a,
            via_cost: 25.0,
            layer_cost: &layer_cost,
            budget: 3_000_000,
            neck,
            anchors: [w.a, w.b],
            neck_reach: mm(0.8),
            avoid: &avoid,
        };
        match route_one(space, q, w.a, w.b) {
            Ok(route) => {
                report.routed += 1;
                for (layer, width, a, b) in route.tracks {
                    board.add_track(Some(w.net), layer, width, a, b);
                    space.add(Some(w.net), LayerSet::single(layer), Prim::segment(a, b, width));
                    report.tracks += 1;
                }
                for p in route.vias {
                    let id = board.add_via(Some(w.net), p, opts.via.0, opts.via.1);
                    let v = &board.vias[&id];
                    space.add(
                        Some(w.net),
                        layer_mask(v.layers()),
                        Prim::circle(p, v.diameter / 2),
                    );
                    space.add_hole(p, opts.via.1 / 2);
                    report.vias += 1;
                }
            }
            Err(why) => report.failed.push(Unrouted {
                net: board.net_name(w.net).to_owned(),
                a: w.a,
                b: w.b,
                why,
            }),
        }
    }
    report
}

/// The narrow side of the pad of `net` that covers `p`, if one does.
fn pad_width(pads: &[crate::layout::PlacedPad], net: NetId, p: Point) -> Option<Coord> {
    pads.iter()
        .filter(|pad| pad.net == Some(net) && pad.prim.contains(p))
        .map(|pad| {
            let bb = pad.prim.bbox();
            bb.width().min(bb.height())
        })
        .min()
}

struct Route {
    /// Layer, width, ends.
    tracks: Vec<(CopperIndex, Coord, Point, Point)>,
    vias: Vec<Point>,
}

/// Grid nodes near `p` on the routing layers that the net's copper covers and that are open;
/// else, where `p` is on the net's pour on another layer, open nodes over the pour where a via
/// fits (the route ends with a via into the pour; the flag says so).
fn ends(s: &mut Search, space: &Space, p: Point) -> (Vec<Node>, bool) {
    let own = copper_ends(s, p);
    if !own.is_empty() {
        return (own, false);
    }
    if !space.own_fill(s.q.net, p) {
        return (Vec::new(), false);
    }
    let g = s.q.grid;
    let o = s.q.origin;
    let (cx, cy) = ((p.x - o.x) / g, (p.y - o.y) / g);
    let r = (mm(0.6) / g).max(1);
    let mut out = Vec::new();
    for x in cx - r..=cx + r {
        for y in cy - r..=cy + r {
            let at = search::point(&s.q, (x, y, 0));
            if !space.own_fill(s.q.net, at) || !s.via_site(x, y) {
                continue;
            }
            for l in 0..s.q.layers.len() as CopperIndex {
                if s.open_node((x, y, l)) {
                    out.push((x, y, l));
                }
            }
        }
    }
    (out, true)
}

fn copper_ends(s: &mut Search, p: Point) -> Vec<Node> {
    let g = s.q.grid;
    let o = s.q.origin;
    let (cx, cy) = ((p.x - o.x) / g, (p.y - o.y) / g);
    let r = (mm(1.2) / g).max(1);
    // Only the copper that covers `p` on the node's layer (the anchor's pad, track or via): other
    // copper of the net nearby can be another island, or the anchor's copper on another layer,
    // and a stub from there to `p` would join nothing.
    // Nodes whose straight stub to `p` keeps the clearance first: the stub is drawn without the
    // search's checks, so a node on the net's copper beside another net's pad would run it past
    // that pad.
    let mut clear = Vec::new();
    let mut out = Vec::new();
    let mut any_own = Vec::new();
    for l in 0..s.q.layers.len() as CopperIndex {
        let items = s.anchor_items(l, p);
        if items.is_empty() {
            continue;
        }
        for x in cx - r..=cx + r {
            for y in cy - r..=cy + r {
                let n = (x, y, l);
                if s.on_items(n, &items) {
                    any_own.push(n);
                    if s.open_node(n) {
                        if s.stub_free(n, p) {
                            clear.push(n);
                        }
                        out.push(n);
                    }
                }
            }
        }
    }
    if !clear.is_empty() {
        clear
    } else if out.is_empty() {
        any_own
    } else {
        out
    }
}

fn route_one(space: &Space, q: Query, a: Point, b: Point) -> Result<Route, Why> {
    let mut s = Search::new(q, space);
    let (sources, via_a) = ends(&mut s, space, a);
    let (targets, via_b) = ends(&mut s, space, b);
    if sources.is_empty() {
        return Err(Why::StartBlocked);
    }
    if targets.is_empty() {
        return Err(Why::EndBlocked);
    }
    let nodes = s.run(&sources, &targets, b).ok_or(Why::NoPath)?;
    let first = search::point(&s.q, nodes[0]);
    let last = search::point(&s.q, *nodes.last().expect("a path"));
    // A route that starts or ends on a pour does so with a via at its end node.
    let (a, b) = (if via_a { first } else { a }, if via_b { last } else { b });
    let mut route = geometry(&s, space, &nodes, a, b);
    if via_a {
        route.vias.push(first);
    }
    if via_b {
        route.vias.push(last);
    }
    // The stubs and an end that is not open are drawn without the search's checks: a route that
    // would put copper inside another net's clearance is left unrouted.
    if !clear_of_others(space, &s.q, &route) {
        return Err(Why::Crowded);
    }
    Ok(route)
}

/// Whether every track and via of `route` keeps the clearance to other nets' copper, the
/// board edge and the hole-to-hole distance.
fn clear_of_others(space: &Space, q: &Query, route: &Route) -> bool {
    let net = Some(q.net);
    let (d, drill) = q.via;
    route
        .tracks
        .iter()
        .all(|(layer, width, a, b)| space.free(net, &LayerSet::single(*layer), &Prim::segment(*a, *b, *width), 0))
        && route
            .vias
            .iter()
            .all(|p| space.free(net, &q.via_mask, &Prim::circle(*p, d / 2), 0) && space.hole_free(*p, drill / 2))
}

/// The node path as tracks and vias: split into runs by layer and by track width (the
/// neck-down near pads), straight runs merged, corners cut where clear, and short stubs to the
/// two anchors (inside the net's own copper).
fn geometry(s: &Search, space: &Space, nodes: &[Node], a: Point, b: Point) -> Route {
    let q = &s.q;
    let mut runs: Vec<(CopperIndex, Coord, Vec<Point>)> = Vec::new();
    let mut vias = Vec::new();
    for n in nodes {
        let p = search::point(q, *n);
        let w = q.width_at(p);
        match runs.last_mut() {
            Some((l, rw, pts)) if *l == n.2 && *rw == w => pts.push(p),
            Some((l, rw, pts)) if *l == n.2 => {
                // The step between a neck node and a full-width node is drawn narrow.
                let last = *pts.last().expect("a run has a point");
                if *rw < w {
                    pts.push(p);
                    runs.push((n.2, w, vec![p]));
                } else {
                    runs.push((n.2, w, vec![last, p]));
                }
            }
            Some(_) => {
                vias.push(p);
                runs.push((n.2, w, vec![p]));
            }
            None => runs.push((n.2, w, vec![p])),
        }
    }
    // The stubs to the anchors are drawn at the anchor's width (the neck on a pad), as the
    // ends were checked.
    let (wa, wb) = (q.width_at(a), q.width_at(b));
    if wa < runs[0].1 {
        let (l, start) = (runs[0].0, runs[0].2[0]);
        runs.insert(0, (l, wa, vec![a, start]));
    } else {
        runs[0].2.insert(0, a);
    }
    let last = runs.len() - 1;
    if wb < runs[last].1 {
        let (l, end) = (runs[last].0, *runs[last].2.last().expect("a run has a point"));
        runs.push((l, wb, vec![end, b]));
    } else {
        runs[last].2.push(b);
    }
    let last = runs.len() - 1;
    let mut tracks = Vec::new();
    for (i, (l, width, pts)) in runs.into_iter().enumerate() {
        let layer = q.layers[l as usize];
        let mut pts = merge(&pts);
        pts = shortcut(space, q.net, layer, width, &pts, i == 0, i == last);
        for w in pts.windows(2) {
            if w[0] != w[1] {
                tracks.push((layer, width, w[0], w[1]));
            }
        }
    }
    Route { tracks, vias }
}

fn direction(a: Point, b: Point) -> Option<(i64, i64)> {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    if dx == 0 && dy == 0 {
        return None;
    }
    if dx == 0 || dy == 0 || dx.abs() == dy.abs() {
        Some((dx.signum(), dy.signum()))
    } else {
        None
    }
}

/// Drops the middle point of every straight run.
fn merge(pts: &[Point]) -> Vec<Point> {
    let mut out: Vec<Point> = Vec::new();
    for p in pts {
        if out.last() == Some(p) {
            continue;
        }
        if out.len() >= 2 {
            let (x, y) = (out[out.len() - 2], out[out.len() - 1]);
            if direction(x, y).is_some() && direction(x, y) == direction(y, *p) {
                out.pop();
            }
        }
        out.push(*p);
    }
    out
}

/// Replaces sub-paths by one 0°/45°/90° segment where that segment keeps the clearance. The
/// anchor stubs (first and last segment when `keep_first`/`keep_last`) stay as they are.
fn shortcut(
    space: &Space,
    net: NetId,
    layer: CopperIndex,
    width: Coord,
    pts: &[Point],
    keep_first: bool,
    keep_last: bool,
) -> Vec<Point> {
    let lo = usize::from(keep_first);
    let hi = pts.len().saturating_sub(usize::from(keep_last) + 1);
    if hi <= lo + 1 {
        return pts.to_vec();
    }
    let mut out: Vec<Point> = pts[..=lo].to_vec();
    let mut i = lo;
    while i < hi {
        let mut j = hi;
        while j > i + 1 {
            let (p, r) = (pts[i], pts[j]);
            if direction(p, r).is_some()
                && space.free(Some(net), &LayerSet::single(layer), &Prim::segment(p, r, width), 0)
            {
                break;
            }
            j -= 1;
        }
        out.push(pts[j]);
        i = j;
    }
    out.extend_from_slice(&pts[hi + 1..]);
    merge(&out)
}
