//! The maze search: A* over a square grid with the eight 0°/45°/90° directions on each routing
//! layer, plus layer changes through a via. A grid node is open to a net when a track of the
//! net's width centred there keeps the clearance plus the grid's sampling slack, so the
//! straight run between two open neighbours keeps the clearance exactly.

use super::space::Space;
use crate::geom::shape::Prim;
use crate::geom::{Coord, Point};
use crate::layout::NetId;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// A grid node: column, row, index into the routing layers.
pub(super) type Node = (i64, i64, u8);

/// What one search needs: the net, its track width and via, the routing layers (board copper
/// indices) and the costs.
pub(super) struct Query<'a> {
    pub net: NetId,
    pub width: Coord,
    pub via: (Coord, Coord),
    /// Copper layers the via spans (through: every layer).
    pub via_mask: u64,
    pub layers: &'a [u8],
    pub grid: Coord,
    /// Grid origin: the route's start anchor, so a track leaves a pad along its centre line.
    pub origin: Point,
    /// Cost of a layer change, in grid steps.
    pub via_cost: f64,
    /// Cost multiplier per routing layer (keeps routes on the preferred layer).
    pub layer_cost: &'a [f64],
    /// Search bound: nodes expanded before giving up.
    pub budget: usize,
    /// Neck-down: within `neck_reach` of an end that is a pad narrower than the track, the
    /// track narrows to `neck` (the pad's width, at least the board's minimum track), as
    /// Altium's neck-down rule does, so a wide power track can enter a fine-pitch pad.
    pub neck: Coord,
    pub anchors: [Point; 2],
    pub neck_reach: Coord,
    /// Rooms that don't hold both ends of the connection, each with the end it holds (if
    /// any): a step inside one, farther than [`ROOM_EXIT`] from that end, costs [`ROOM_COST`]
    /// times as much, so a route between channels leaves its room near its pin and runs
    /// around the other channels' rooms.
    pub avoid: &'a [(crate::geom::Rect, Option<Point>)],
}

/// How far from its pin a route between rooms may run inside the pin's room at no extra cost.
pub(super) const ROOM_EXIT: Coord = 5_000_000;

/// Cost multiplier for a step through a room the connection doesn't belong to.
pub(super) const ROOM_COST: f64 = 2.5;

impl Query<'_> {
    /// The track width at `p`.
    pub fn width_at(&self, p: Point) -> Coord {
        if self.neck < self.width
            && self
                .anchors
                .iter()
                .any(|a| a.dist(p) <= self.neck_reach as f64)
        {
            self.neck
        } else {
            self.width
        }
    }
}

#[derive(PartialEq)]
struct Open {
    f: f64,
    g: f64,
    node: Node,
    dir: u8,
}

impl Eq for Open {}

impl Ord for Open {
    fn cmp(&self, o: &Self) -> Ordering {
        o.f.partial_cmp(&self.f)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&o.node))
    }
}

impl PartialOrd for Open {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

const DIRS: [(i64, i64); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// No direction yet (a source node).
const START: u8 = 8;
/// Arrived by a via.
const VIA: u8 = 9;

pub(super) fn point(q: &Query, n: Node) -> Point {
    Point::new(q.origin.x + n.0 * q.grid, q.origin.y + n.1 * q.grid)
}

/// The search with its per-node openness memo.
pub(super) struct Search<'a> {
    pub q: Query<'a>,
    space: &'a Space,
    open: HashMap<Node, bool>,
    via_open: HashMap<(i64, i64), bool>,
    slack: Coord,
}

impl<'a> Search<'a> {
    pub fn new(q: Query<'a>, space: &'a Space) -> Self {
        // A segment between two neighbours lies within half a diagonal step of one of them.
        let slack = (q.grid as f64 * std::f64::consts::FRAC_1_SQRT_2).ceil() as Coord;
        Search {
            q,
            space,
            open: HashMap::new(),
            via_open: HashMap::new(),
            slack,
        }
    }

    fn mask(&self, l: u8) -> u64 {
        1u64 << self.q.layers[l as usize]
    }

    /// Whether net copper covers the node (sources and targets).
    pub fn own(&self, n: Node) -> bool {
        self.space
            .own(self.q.net, self.mask(n.2), point(&self.q, n))
    }

    /// Whether a track of the net may pass through the node.
    pub fn open_node(&mut self, n: Node) -> bool {
        self.is_open(n)
    }

    fn is_open(&mut self, n: Node) -> bool {
        if let Some(v) = self.open.get(&n) {
            return *v;
        }
        let p = point(&self.q, n);
        let prim = Prim::circle(p, self.q.width_at(p) / 2);
        let v = self
            .space
            .free(Some(self.q.net), self.mask(n.2), &prim, self.slack);
        self.open.insert(n, v);
        v
    }

    /// Whether a via of the net fits at the grid point.
    pub fn via_site(&mut self, x: i64, y: i64) -> bool {
        self.via_ok(x, y)
    }

    fn via_ok(&mut self, x: i64, y: i64) -> bool {
        if let Some(v) = self.via_open.get(&(x, y)) {
            return *v;
        }
        let p = point(&self.q, (x, y, 0));
        let (d, drill) = self.q.via;
        let v = self.space.free(
            Some(self.q.net),
            self.q.via_mask,
            &Prim::circle(p, d / 2),
            0,
        ) && self.space.hole_free(p, drill / 2);
        self.via_open.insert((x, y), v);
        v
    }

    /// The cheapest path from any of `sources` to any of `targets`, as nodes.
    pub fn run(&mut self, sources: &[Node], targets: &[Node], goal: Point) -> Option<Vec<Node>> {
        let target: std::collections::HashSet<Node> = targets.iter().copied().collect();
        let h = |n: Node, q: &Query| {
            let p = point(q, n);
            let dx = ((p.x - goal.x).abs() as f64) / q.grid as f64;
            let dy = ((p.y - goal.y).abs() as f64) / q.grid as f64;
            let (a, b) = if dx > dy { (dx, dy) } else { (dy, dx) };
            a - b + b * std::f64::consts::SQRT_2
        };
        let mut heap = BinaryHeap::new();
        let mut best: HashMap<(Node, u8), f64> = HashMap::new();
        let mut parent: HashMap<(Node, u8), (Node, u8)> = HashMap::new();
        for s in sources {
            best.insert((*s, START), 0.0);
            heap.push(Open {
                f: h(*s, &self.q),
                g: 0.0,
                node: *s,
                dir: START,
            });
        }
        let mut expanded = 0usize;
        while let Some(Open { g, node, dir, .. }) = heap.pop() {
            if best.get(&(node, dir)).is_some_and(|b| *b < g) {
                continue;
            }
            if target.contains(&node) {
                let mut path = vec![node];
                let mut k = (node, dir);
                while let Some(p) = parent.get(&k) {
                    path.push(p.0);
                    k = *p;
                }
                path.reverse();
                return Some(path);
            }
            expanded += 1;
            if expanded > self.q.budget {
                return None;
            }
            let cost = self.q.layer_cost[node.2 as usize];
            for (d, (dx, dy)) in DIRS.iter().enumerate() {
                let next = (node.0 + dx, node.1 + dy, node.2);
                if !(self.is_open(next) || target.contains(&next) && self.own(next)) {
                    continue;
                }
                let step = if dx * dy == 0 {
                    1.0
                } else {
                    std::f64::consts::SQRT_2
                };
                // Bends cost a little (fewer corners), sharp bends more.
                let turn = match dir {
                    START | VIA => 0.0,
                    prev => {
                        let delta = (d as i32 - prev as i32)
                            .rem_euclid(8)
                            .min((prev as i32 - d as i32).rem_euclid(8));
                        match delta {
                            0 => 0.0,
                            1 => 0.6,
                            2 => 3.0,
                            _ => 40.0,
                        }
                    }
                };
                let foreign = !self.q.avoid.is_empty() && {
                    let at = point(&self.q, next);
                    self.q.avoid.iter().any(|(r, end)| {
                        r.contains(at) && end.is_none_or(|e| e.dist(at) > ROOM_EXIT as f64)
                    })
                };
                let ng = g + step * cost * if foreign { ROOM_COST } else { 1.0 } + turn;
                let key = (next, d as u8);
                if best.get(&key).is_none_or(|b| ng < *b) {
                    best.insert(key, ng);
                    parent.insert(key, (node, dir));
                    heap.push(Open {
                        f: ng + h(next, &self.q),
                        g: ng,
                        node: next,
                        dir: d as u8,
                    });
                }
            }
            // A via to every other routing layer.
            if self.q.layers.len() > 1 && dir != VIA && self.via_ok(node.0, node.1) {
                for l in 0..self.q.layers.len() as u8 {
                    if l == node.2 {
                        continue;
                    }
                    let next = (node.0, node.1, l);
                    if !self.is_open(next) && !target.contains(&next) {
                        continue;
                    }
                    let ng = g + self.q.via_cost;
                    let key = (next, VIA);
                    if best.get(&key).is_none_or(|b| ng < *b) {
                        best.insert(key, ng);
                        parent.insert(key, (node, dir));
                        heap.push(Open {
                            f: ng + h(next, &self.q),
                            g: ng,
                            node: next,
                            dir: VIA,
                        });
                    }
                }
            }
        }
        None
    }
}
