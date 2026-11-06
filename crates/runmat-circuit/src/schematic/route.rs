//! Connections on one page: orthogonal wires on the 1.27 mm grid, junction dots, power ports,
//! net labels and no-connect markers.
//!
//! - **Rails** get power ports, never long wires: a pin on the top edge gets its port pointing
//!   up, one on the bottom edge pointing down. Neighbouring side pins on one rail share a comb
//!   (a stub each, a joining wire and one port), pointing up or down when nothing on that side
//!   is beyond them, else outward.
//! - **Signal nets** on one page are wired: a maze search on the grid joins each pin to the
//!   net's tree, leaving every pin straight out, turning as little as it can and crossing other
//!   nets only at right angles (never at a bend, a junction or a pin). Symbol bodies, pin lines
//!   and texts are walls, so no wire crosses a symbol. A net whose wire would be long (more
//!   than [`LONG`] grid steps for one branch) gets a label at each pin instead.
//! - **Labels:** a hierarchical label where a net leaves the sheet through a port, a global
//!   label for a global net, a local label otherwise, each on a 2.54 mm stub.
//!
//! Every pin's tip and its two stub points are kept for its own net, so a label always fits.

use super::symbol::PinSide;
use super::{text_width, NetInfo, GRID, LABEL_TEXT};
use crate::geom::{Coord, Point};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// The longest wire branch, in grid steps (about 90 mm).
pub(super) const LONG: usize = 72;
/// Most pins a net may have and still be wired.
const MOST_PINS: usize = 16;
const STUB_STEPS: Coord = 2;
const NONE: u32 = u32::MAX;

/// One pin to connect.
#[derive(Clone, Debug)]
pub(super) struct Term {
    pub net: Option<usize>,
    pub tip: Point,
    pub side: PinSide,
    /// The pin's index on its side, counted from the top (left and right pins) and whether it
    /// is the last on that side.
    pub first_on_side: bool,
    pub last_on_side: bool,
    /// Row neighbours on the same side: the terms just above and below, by index.
    pub above: Option<usize>,
    pub below: Option<usize>,
}

/// What [`connect`] draws.
#[derive(Default)]
pub(super) struct Drawn {
    pub wires: Vec<(usize, Vec<Point>)>,
    pub labels: Vec<Label>,
    pub junctions: Vec<Point>,
    pub no_connects: Vec<Point>,
}

#[derive(Clone, Debug)]
pub(super) struct Label {
    pub net: usize,
    pub at: Point,
    pub kind: &'static str,
    pub side: PinSide,
}

#[derive(Clone, Copy, Default)]
struct G {
    block: bool,
    net: u32,
    h: bool,
    v: bool,
    /// A bend, junction, wire end or pin: nothing else may pass.
    hard: bool,
    cross: bool,
}

struct Grid {
    x0: Coord,
    y0: Coord,
    nx: Coord,
    ny: Coord,
    c: Vec<G>,
}

const DIRS: [(Coord, Coord); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

impl Grid {
    fn new(bbox: [Coord; 4]) -> Grid {
        let pad = 16 * GRID;
        let x0 = (bbox[0] - pad).div_euclid(GRID) * GRID;
        let y0 = (bbox[1] - pad).div_euclid(GRID) * GRID;
        let nx = (bbox[2] + pad - x0) / GRID + 1;
        let ny = (bbox[3] + pad - y0) / GRID + 1;
        let empty = G {
            net: NONE,
            ..Default::default()
        };
        Grid {
            x0,
            y0,
            nx,
            ny,
            c: vec![empty; (nx * ny) as usize],
        }
    }
    fn at(&self, p: Point) -> Option<usize> {
        let (gx, gy) = ((p.x - self.x0).div_euclid(GRID), (p.y - self.y0).div_euclid(GRID));
        self.cell(gx, gy)
    }
    fn cell(&self, gx: Coord, gy: Coord) -> Option<usize> {
        (gx >= 0 && gy >= 0 && gx < self.nx && gy < self.ny).then(|| (gy * self.nx + gx) as usize)
    }
    fn xy(&self, i: usize) -> (Coord, Coord) {
        (i as Coord % self.nx, i as Coord / self.nx)
    }
    fn point(&self, i: usize) -> Point {
        let (gx, gy) = self.xy(i);
        Point::new(self.x0 + gx * GRID, self.y0 + gy * GRID)
    }
    fn step(&self, i: usize, d: usize) -> Option<usize> {
        let (gx, gy) = self.xy(i);
        self.cell(gx + DIRS[d].0, gy + DIRS[d].1)
    }
    /// Blocks every grid point inside `r` grown by `grow`.
    fn block(&mut self, r: [Coord; 4], grow: Coord) {
        let gx0 = (r[0] - grow - self.x0).div_euclid(GRID).max(0);
        let gy0 = (r[1] - grow - self.y0).div_euclid(GRID).max(0);
        let gx1 = (r[2] + grow - self.x0).div_euclid(GRID).min(self.nx - 1);
        let gy1 = (r[3] + grow - self.y0).div_euclid(GRID).min(self.ny - 1);
        for gy in gy0..=gy1 {
            for gx in gx0..=gx1 {
                let p = Point::new(self.x0 + gx * GRID, self.y0 + gy * GRID);
                if p.x >= r[0] - grow && p.x <= r[2] + grow && p.y >= r[1] - grow && p.y <= r[3] + grow
                {
                    let i = (gy * self.nx + gx) as usize;
                    self.c[i].block = true;
                }
            }
        }
    }
}

fn dir_of(side: PinSide) -> usize {
    match side {
        PinSide::Right => 0,
        PinSide::Left => 1,
        PinSide::Top => 2,
        PinSide::Bottom => 3,
    }
}

fn offset(p: Point, side: PinSide, steps: Coord) -> Point {
    let (dx, dy) = side.out();
    Point::new(p.x + dx * steps * GRID, p.y + dy * steps * GRID)
}

/// Walls (bodies), pin lines (from the body to just before the tip) and text boxes.
pub(super) struct Walls {
    pub bodies: Vec<[Coord; 4]>,
    pub pin_lines: Vec<(Point, Point)>,
    pub texts: Vec<[Coord; 4]>,
}

/// Connects a page's pins. `bbox` bounds everything placed.
pub(super) fn connect(terms: &[Term], nets: &[NetInfo], walls: &Walls, bbox: [Coord; 4]) -> Drawn {
    let mut g = Grid::new(bbox);
    for b in &walls.bodies {
        g.block(*b, 0);
    }
    for t in &walls.texts {
        g.block(*t, GRID / 2);
    }
    for (a, b) in &walls.pin_lines {
        g.block([a.x.min(b.x), a.y.min(b.y), a.x.max(b.x), a.y.max(b.y)], 0);
    }
    let mut out = Drawn::default();
    // Tips and their stubs are kept for their own nets; a tip with no net gets a cross.
    for t in terms {
        let Some(i) = g.at(t.tip) else { continue };
        match t.net {
            None => {
                g.c[i].block = true;
                out.no_connects.push(t.tip);
            }
            Some(n) => {
                g.c[i] = G {
                    net: n as u32,
                    hard: true,
                    ..Default::default()
                };
                for k in 1..=STUB_STEPS {
                    if let Some(j) = g.at(offset(t.tip, t.side, k)) {
                        if !g.c[j].block && g.c[j].net == NONE {
                            g.c[j].net = n as u32;
                        }
                    }
                }
            }
        }
    }
    let mut by_net: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (k, t) in terms.iter().enumerate() {
        if let Some(n) = t.net {
            by_net.entry(n).or_default().push(k);
        }
    }
    // Rails first, then labelled nets, then wired nets from the shortest span up.
    let mut wired: Vec<(Coord, usize)> = Vec::new();
    for (&n, ks) in &by_net {
        if nets[n].rail {
            rail(&mut g, &mut out, terms, ks, n, nets);
        } else if nets[n].global || ks.len() == 1 || ks.len() > MOST_PINS {
            for &k in ks {
                label(&mut g, &mut out, &terms[k], n, nets, walls);
            }
        } else {
            let xs = ks.iter().map(|k| terms[*k].tip.x);
            let ys = ks.iter().map(|k| terms[*k].tip.y);
            let span = xs.clone().max().unwrap() - xs.min().unwrap() + ys.clone().max().unwrap()
                - ys.min().unwrap();
            wired.push((span, n));
        }
    }
    wired.sort();
    for (_, n) in wired {
        let ks = &by_net[&n];
        match route_net(&g, terms, ks, n) {
            Some(paths) => {
                commit(&mut g, &paths, n);
                let mut deg: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
                for path in &paths {
                    for w in path.windows(2) {
                        deg.entry(w[0]).or_default().insert(w[1]);
                        deg.entry(w[1]).or_default().insert(w[0]);
                    }
                    out.wires.push((n, corners(&g, path)));
                }
                for (cell, near) in deg {
                    if near.len() >= 3 {
                        out.junctions.push(g.point(cell));
                    }
                }
                // A net that also leaves the sheet gets its port's label at its first pin; a node
                // whose pins all leave up or down (a divider) gets it on a stub off the node's
                // right end instead, so the label never sits on a part's lead.
                let upright = ks.iter().all(|k| !terms[*k].side.horizontal());
                if nets[n].external && !(upright && tail_label(&mut g, &mut out, &paths, n, nets)) {
                    label(&mut g, &mut out, &terms[ks[0]], n, nets, walls);
                }
            }
            None => {
                for &k in ks {
                    label(&mut g, &mut out, &terms[k], n, nets, walls);
                }
            }
        }
    }
    out.junctions.sort();
    out.junctions.dedup();
    out
}

fn mark_wire(g: &mut Grid, cells: &[usize], n: usize) {
    for (k, &c) in cells.iter().enumerate() {
        let horiz = |a: usize, b: usize| g.xy(a).1 == g.xy(b).1;
        let mut h = false;
        let mut v = false;
        if k > 0 {
            if horiz(cells[k - 1], c) {
                h = true
            } else {
                v = true
            }
        }
        if k + 1 < cells.len() {
            if horiz(c, cells[k + 1]) {
                h = true
            } else {
                v = true
            }
        }
        let cell = &mut g.c[c];
        if cell.net != NONE && cell.net != n as u32 && (cell.h || cell.v) {
            cell.cross = true;
        } else {
            cell.net = n as u32;
        }
        cell.h |= h;
        cell.v |= v;
        if (h && v) || k == 0 || k + 1 == cells.len() {
            cell.hard = true;
        }
    }
}

fn commit(g: &mut Grid, paths: &[Vec<usize>], n: usize) {
    for p in paths {
        mark_wire(g, p, n);
    }
}

/// A path's corner points.
fn corners(g: &Grid, path: &[usize]) -> Vec<Point> {
    let mut pts: Vec<Point> = Vec::new();
    for (k, &c) in path.iter().enumerate() {
        let p = g.point(c);
        if k > 0 && k + 1 < path.len() {
            let (a, b) = (g.point(path[k - 1]), g.point(path[k + 1]));
            if (a.x == p.x && p.x == b.x) || (a.y == p.y && p.y == b.y) {
                continue;
            }
        }
        pts.push(p);
    }
    pts
}

/// Wires one net: each pin joins the tree, nearest first. `None` when a branch can't be found
/// or would be long.
fn route_net(g: &Grid, terms: &[Term], ks: &[usize], n: usize) -> Option<Vec<Vec<usize>>> {
    let mut order: Vec<usize> = ks.to_vec();
    order.sort_by_key(|k| (terms[*k].tip.x, Reverse(terms[*k].tip.y)));
    let first = order.remove(0);
    let mut tree: BTreeMap<usize, usize> = BTreeMap::new();
    let start = g.at(terms[first].tip)?;
    tree.insert(start, 0);
    // A tip in the tree with no wire yet is entered only straight in.
    let mut bare: BTreeMap<usize, usize> = BTreeMap::from([(start, dir_of(terms[first].side))]);
    let mut paths = Vec::new();
    while !order.is_empty() {
        let bb = tree.keys().fold([Coord::MAX, Coord::MAX, Coord::MIN, Coord::MIN], |b, &c| {
            let (x, y) = g.xy(c);
            [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)]
        });
        let dist = |k: &usize| {
            let (x, y) = g.xy(g.at(terms[*k].tip).unwrap_or(0));
            (bb[0] - x).max(0) + (x - bb[2]).max(0) + (bb[1] - y).max(0) + (y - bb[3]).max(0)
        };
        let pick = (0..order.len()).min_by_key(|&i| (dist(&order[i]), i)).unwrap();
        let k = order.remove(pick);
        let t = &terms[k];
        let from = g.at(t.tip)?;
        let path = search(g, from, dir_of(t.side), &tree, &bare, n, bb)?;
        for w in path.windows(2) {
            *tree.entry(w[0]).or_default() += 1;
            *tree.entry(w[1]).or_default() += 1;
        }
        bare.remove(path.last().unwrap());
        paths.push(path);
    }
    Some(paths)
}

/// A* from a pin tip (first step straight out) to any cell of the tree.
fn search(
    g: &Grid,
    from: usize,
    out_dir: usize,
    tree: &BTreeMap<usize, usize>,
    bare: &BTreeMap<usize, usize>,
    n: usize,
    bb: [Coord; 4],
) -> Option<Vec<usize>> {
    let me = n as u32;
    let (fx, fy) = g.xy(from);
    let window = [
        bb[0].min(fx) - 24,
        bb[1].min(fy) - 24,
        bb[2].max(fx) + 24,
        bb[3].max(fy) + 24,
    ];
    let h = |c: usize| {
        let (x, y) = g.xy(c);
        (bb[0] - x).max(0) + (x - bb[2]).max(0) + (bb[1] - y).max(0) + (y - bb[3]).max(0)
    };
    let opposite = |d: usize| d ^ 1;
    let mut best: BTreeMap<(usize, usize), (Coord, usize, (usize, usize))> = BTreeMap::new();
    let mut heap = BinaryHeap::new();
    let start = (from, out_dir);
    best.insert(start, (0, 0, start));
    heap.push(Reverse((h(from), 0 as Coord, from, out_dir, 0usize)));
    let mut expanded = 0usize;
    while let Some(Reverse((_, cost, c, d, len))) = heap.pop() {
        if best.get(&(c, d)).is_some_and(|b| b.0 < cost) {
            continue;
        }
        expanded += 1;
        if expanded > 60_000 || len > LONG {
            continue;
        }
        let crossing = g.c[c].net != NONE && g.c[c].net != me && c != from;
        for nd in 0..4 {
            if (c == from && nd != out_dir) || nd == opposite(d) || (crossing && nd != d) {
                continue;
            }
            let Some(q) = g.step(c, nd) else { continue };
            let (qx, qy) = g.xy(q);
            if qx < window[0] || qy < window[1] || qx > window[2] || qy > window[3] {
                continue;
            }
            let cell = &g.c[q];
            if cell.block {
                continue;
            }
            let bend = if nd != d { 3 } else { 0 };
            if let Some(&deg) = tree.get(&q) {
                // Join the tree: straight into a bare tip, else as a T (never a four-way).
                let ok = match bare.get(&q) {
                    Some(&tip_out) => nd == opposite(tip_out),
                    None => deg < 3 && !(cell.hard && deg == 1 && g.c[q].net == me && is_tip(g, q)),
                };
                if ok {
                    let mut path = vec![q];
                    let mut s = (c, d);
                    loop {
                        path.push(s.0);
                        if s == start {
                            break;
                        }
                        s = best[&s].2;
                    }
                    path.reverse();
                    return Some(path);
                }
                continue;
            }
            let step = if cell.net == NONE || cell.net == me {
                if cell.net == me && cell.hard {
                    continue;
                }
                1
            } else {
                let horizontal = nd < 2;
                let perpendicular = if horizontal {
                    cell.v && !cell.h
                } else {
                    cell.h && !cell.v
                };
                if cell.hard || cell.cross || !perpendicular {
                    continue;
                }
                8
            };
            let nc = cost + step + bend;
            let key = (q, nd);
            if best.get(&key).is_some_and(|b| b.0 <= nc) {
                continue;
            }
            best.insert(key, (nc, len + 1, (c, d)));
            heap.push(Reverse((nc + h(q), nc, q, nd, len + 1)));
        }
    }
    None
}

/// Whether a cell holds a pin tip (a hard cell no wire passes straight through).
fn is_tip(g: &Grid, q: usize) -> bool {
    let c = &g.c[q];
    c.hard && !(c.h && c.v)
}

/// A label on a stub at a pin.
fn label(g: &mut Grid, out: &mut Drawn, t: &Term, n: usize, nets: &[NetInfo], _w: &Walls) {
    let end = offset(t.tip, t.side, STUB_STEPS);
    let cells: Vec<usize> = (0..=STUB_STEPS)
        .filter_map(|k| g.at(offset(t.tip, t.side, k)))
        .collect();
    mark_wire(g, &cells, n);
    out.wires.push((n, vec![t.tip, end]));
    let kind = if nets[n].global {
        "global"
    } else if nets[n].external {
        "hier"
    } else {
        "local"
    };
    let side = if t.side.horizontal() {
        t.side
    } else {
        PinSide::Right
    };
    let l = Label {
        net: n,
        at: end,
        kind,
        side,
    };
    g.block(label_extent(&l, &nets[n].short), GRID / 2);
    out.labels.push(l);
}

/// A port's label on a stub run right from the rightmost point of a wired net (the highest of
/// equals), when the stub and the label's box are clear. Returns whether it was placed.
fn tail_label(g: &mut Grid, out: &mut Drawn, paths: &[Vec<usize>], n: usize, nets: &[NetInfo]) -> bool {
    let Some(&from) = paths
        .iter()
        .flatten()
        .max_by_key(|&&c| (g.point(c).x, g.point(c).y))
    else {
        return false;
    };
    let p = g.point(from);
    let steps = 2 * STUB_STEPS;
    let end = offset(p, PinSide::Right, steps);
    let l = Label {
        net: n,
        at: end,
        kind: "hier",
        side: PinSide::Right,
    };
    let stub: Vec<usize> = (1..=steps).filter_map(|k| g.at(offset(p, PinSide::Right, k))).collect();
    let free = |g: &Grid, c: usize| !g.c[c].block && g.c[c].net == NONE;
    if stub.len() != steps as usize || !stub.iter().all(|&c| free(g, c)) {
        return false;
    }
    let r = label_extent(&l, &nets[n].short);
    let grow = GRID / 2;
    let mut y = r[1] - grow;
    while y <= r[3] + grow {
        let mut x = r[0] + GRID;
        while x <= r[2] + grow {
            if g.at(Point::new(x, y)).is_none_or(|c| !free(g, c)) {
                return false;
            }
            x += GRID;
        }
        y += GRID;
    }
    // A point the wire passes or turns at becomes a T: it gets a junction dot.
    let through = paths
        .iter()
        .flat_map(|path| path.windows(2))
        .filter(|w| w[0] == from || w[1] == from)
        .count();
    let mut cells = vec![from];
    cells.extend(&stub);
    mark_wire(g, &cells, n);
    out.wires.push((n, vec![p, end]));
    if through >= 2 {
        out.junctions.push(p);
    }
    g.block(r, grow);
    out.labels.push(l);
    true
}

/// Power ports for one rail: top and bottom pins point their ports up and down; side pins
/// share combs.
fn rail(g: &mut Grid, out: &mut Drawn, terms: &[Term], ks: &[usize], n: usize, nets: &[NetInfo]) {
    let ground = nets[n].ground;
    let mut done: BTreeSet<usize> = BTreeSet::new();
    let port = |g: &mut Grid, out: &mut Drawn, at: Point, side: PinSide| {
        let l = Label {
            net: n,
            at,
            kind: "power",
            side,
        };
        g.block(label_extent(&l, &nets[n].short), GRID / 2);
        out.labels.push(l);
    };
    for &k in ks {
        if done.contains(&k) {
            continue;
        }
        let t = &terms[k];
        if !t.side.horizontal() {
            done.insert(k);
            port(g, out, t.tip, t.side);
            continue;
        }
        // The run of neighbouring pins on this side and rail.
        let mut run = vec![k];
        let mut at = k;
        while let Some(b) = terms[at].below {
            if terms[b].net == Some(n) && ks.contains(&b) {
                run.push(b);
                at = b;
            } else {
                break;
            }
        }
        done.extend(run.iter().copied());
        let ends: Vec<Point> = run
            .iter()
            .map(|&r| offset(terms[r].tip, terms[r].side, STUB_STEPS))
            .collect();
        for (&r, e) in run.iter().zip(&ends) {
            let cells: Vec<usize> = (0..=STUB_STEPS)
                .filter_map(|s| g.at(offset(terms[r].tip, terms[r].side, s)))
                .collect();
            mark_wire(g, &cells, n);
            out.wires.push((n, vec![terms[r].tip, *e]));
        }
        let (top, bottom) = (ends[0], ends[ends.len() - 1]);
        if run.len() > 1 {
            out.wires.push((n, vec![top, bottom]));
            for e in &ends[1..ends.len() - 1] {
                out.junctions.push(*e);
            }
        }
        let first = terms[run[0]].first_on_side;
        let last = terms[*run.last().unwrap()].last_on_side;
        if !ground && first {
            port(g, out, top, PinSide::Top);
        } else if ground && last {
            port(g, out, bottom, PinSide::Bottom);
        } else if run.len() > 1 {
            let far = offset(top, t.side, STUB_STEPS);
            out.wires.push((n, vec![top, far]));
            out.junctions.push(top);
            port(g, out, far, t.side);
        } else {
            port(g, out, top, t.side);
        }
    }
}

/// Port drawing sizes (mm): the stem, the bar's half width and the ground bars' spacing.
pub const PORT_STEM: f64 = 1.27;
pub const PORT_BAR: f64 = 1.27;
pub const GROUND_STEP: f64 = 0.5;
pub const PORT_TEXT_GAP: f64 = 0.6;
/// A global or hierarchical label's padding around its text.
pub const FLAG_PAD: f64 = 1.6;
pub const FLAG_HALF: f64 = 0.9;

/// The box a label covers on the page, `[x0, y0, x1, y1]` in nm, as the canvas draws it.
pub(super) fn label_extent(l: &Label, text: &str) -> [Coord; 4] {
    let mm = |v: f64| (v * 1.0e6).round() as Coord;
    let tw = text_width(text, LABEL_TEXT);
    let size = LABEL_TEXT;
    let (x, y) = (l.at.x, l.at.y);
    let (dx, dy) = l.side.out();
    match l.kind {
        "power" => {
            let ground = text_is_ground(text);
            let depth = mm(PORT_STEM + if ground { 2.0 * GROUND_STEP } else { 0.0 });
            let gap = mm(PORT_TEXT_GAP);
            let bar = mm(PORT_BAR);
            if dx == 0 {
                let half = bar.max(tw / 2);
                let far = depth + gap + size;
                if dy > 0 {
                    [x - half, y, x + half, y + far]
                } else {
                    [x - half, y - far, x + half, y]
                }
            } else {
                let far = depth + gap + tw;
                let half = bar.max(size * 3 / 5);
                if dx > 0 {
                    [x, y - half, x + far, y + half]
                } else {
                    [x - far, y - half, x, y + half]
                }
            }
        }
        "global" | "hier" => {
            let len = tw + mm(FLAG_PAD + FLAG_HALF);
            let half = mm(FLAG_HALF);
            if dx < 0 {
                [x - len, y - half, x, y + half]
            } else {
                [x, y - half, x + len, y + half]
            }
        }
        _ => {
            let pad = mm(0.4);
            if dx < 0 {
                [x - pad - tw, y, x, y + size]
            } else {
                [x, y, x + pad + tw, y + size]
            }
        }
    }
}

/// Whether a rail name draws as ground (`GND`, `AGND`, `VSS`, `-5V`).
pub fn text_is_ground(text: &str) -> bool {
    let u = text.to_ascii_uppercase();
    u.contains("GND") || u.starts_with("VSS") || u.starts_with("VEE") || u.starts_with('-')
}
