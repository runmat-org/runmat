//! Schematic connectivity of one `.kicad_sch` sheet, by KiCad's rules.
//!
//! KiCad's documented rules ("Electrical Connections" in the schematic editor manual):
//!
//! - Wires connect to other wires or to pins only where their ends coincide exactly. A wire
//!   end, pin or sheet pin lying on the middle of another wire does not connect to it, and
//!   wires that cross are not connected.
//! - A junction joins every wire passing through or ending at its point: the editor splits the
//!   wires there, so the junction becomes an end of each.
//! - A label connects to a wire anywhere along it, and to every wire under its anchor (two
//!   crossing wires under one label are joined, which ERC reports as a warning).
//! - A pin cannot connect directly to a bus.
//!
//! When it opens a schematic, KiCad's editor also tidies the wires, which decides how
//! collinear wires connect:
//!
//! - Collinear wires (or buses) with the same stroke that overlap become one wire spanning
//!   both. Only the two outer ends remain ends, so an item ending inside the overlap no longer
//!   touches a wire end.
//! - Collinear wires that only touch end to end become one wire too, unless the touch point is
//!   a junction (three or more wire directions or pins meet there).
//!
//! [`groups`] applies these rules and returns a group number for each attachment point.

use crate::geom::{orient, Point};
use std::collections::{BTreeMap, BTreeSet};

/// A wire or bus segment, with its stroke text (only identical strokes merge).
#[derive(Clone, Debug)]
pub(super) struct Seg {
    pub a: Point,
    pub b: Point,
    pub bus: bool,
    pub stroke: String,
}

/// How an attached item connects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// A symbol pin (including a power symbol's pin): connects at its point, never to a bus.
    Pin,
    /// A sheet pin: connects at its point, to a wire or a bus.
    SheetPin,
    /// A local, global or hierarchical label: connects anywhere along a wire or bus.
    Label,
    /// A no-connect marker: connects at its point.
    Marker,
}

fn on_inside(s: &Seg, p: Point) -> bool {
    p != s.a
        && p != s.b
        && orient(s.a, s.b, p) == 0
        && p.x >= s.a.x.min(s.b.x)
        && p.x <= s.a.x.max(s.b.x)
        && p.y >= s.a.y.min(s.b.y)
        && p.y <= s.a.y.max(s.b.y)
}

/// Splits every segment passing through a junction, so the junction is an end of each part.
fn break_at_junctions(segs: &mut Vec<Seg>, junctions: &[Point]) {
    for &j in junctions {
        let mut extra = Vec::new();
        for s in segs.iter_mut() {
            if on_inside(s, j) {
                extra.push(Seg { a: j, b: s.b, ..s.clone() });
                s.b = j;
            }
        }
        segs.extend(extra);
    }
}

/// The direction from `p` along a segment, reduced so equal directions compare equal.
fn direction(p: Point, q: Point) -> (i64, i64) {
    let (dx, dy) = (q.x - p.x, q.y - p.y);
    let g = gcd(dx.unsigned_abs(), dy.unsigned_abs()).max(1) as i64;
    (dx / g, dy / g)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// True where three or more wire (or bus) directions, pins and sheet pins meet: the point
/// is a junction, so collinear wires touching there stay separate.
fn is_junction(segs: &[Seg], pins: &BTreeSet<Point>, pin_count: &BTreeMap<Point, usize>, p: Point) -> bool {
    for bus in [false, true] {
        let mut exits: BTreeSet<(i64, i64)> = BTreeSet::new();
        let mut ended = false;
        let mut middles = Vec::new();
        for s in segs.iter().filter(|s| s.bus == bus) {
            if s.a == p {
                ended = true;
                exits.insert(direction(p, s.b));
            } else if s.b == p {
                ended = true;
                exits.insert(direction(p, s.a));
            } else if on_inside(s, p) {
                middles.push(s);
            }
        }
        let mut count = 0;
        if !bus && pins.contains(&p) {
            ended = true;
            count = pin_count.get(&p).copied().unwrap_or(0);
        }
        if ended {
            for s in middles {
                exits.insert(direction(p, s.a));
                exits.insert(direction(p, s.b));
            }
        }
        if exits.len() + count >= 3 {
            return true;
        }
    }
    false
}

/// Joins collinear segments of the same kind and stroke that overlap, or that touch end to
/// end where the touch point is not a junction.
fn merge_collinear(segs: &mut Vec<Seg>, pins: &BTreeSet<Point>, pin_count: &BTreeMap<Point, usize>) {
    loop {
        let mut merged = None;
        'pairs: for i in 0..segs.len() {
            for j in i + 1..segs.len() {
                let (s, t) = (&segs[i], &segs[j]);
                if s.bus != t.bus || s.stroke != t.stroke {
                    continue;
                }
                if orient(s.a, s.b, t.a) != 0 || orient(s.a, s.b, t.b) != 0 {
                    continue;
                }
                let (mut s0, mut s1) = (s.a.min(s.b), s.a.max(s.b));
                let (mut t0, mut t1) = (t.a.min(t.b), t.a.max(t.b));
                if t0 < s0 {
                    std::mem::swap(&mut s0, &mut t0);
                    std::mem::swap(&mut s1, &mut t1);
                }
                if s1 < t0 {
                    continue;
                }
                if s1 == t0 && is_junction(segs, pins, pin_count, s1) {
                    continue;
                }
                merged = Some((i, j, s0, s1.max(t1)));
                break 'pairs;
            }
        }
        let Some((i, j, a, b)) = merged else { return };
        segs[i].a = a;
        segs[i].b = b;
        segs.remove(j);
    }
}

struct Union(Vec<usize>);

impl Union {
    fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] != x {
            self.0[x] = self.0[self.0[x]];
            x = self.0[x];
        }
        x
    }
    fn join(&mut self, a: usize, b: usize) {
        let (x, y) = (self.find(a), self.find(b));
        if x != y {
            self.0[x.max(y)] = x.min(y);
        }
    }
}

/// A connectable item at a point.
#[derive(Clone, Copy)]
enum Item {
    Seg(usize, bool),
    Attach(usize, Kind),
    Junction(usize),
}

/// Groups `attach` (point and kind of each attached item) into connections over the sheet's
/// `segments` and `junctions`; returns one group number per attachment.
pub(super) fn groups(mut segments: Vec<Seg>, junctions: &[Point], attach: &[(Point, Kind)]) -> Vec<usize> {
    segments.retain(|s| s.a != s.b);
    break_at_junctions(&mut segments, junctions);
    let mut pin_count: BTreeMap<Point, usize> = BTreeMap::new();
    for (p, k) in attach {
        if matches!(k, Kind::Pin | Kind::SheetPin) {
            *pin_count.entry(*p).or_default() += 1;
        }
    }
    let pins: BTreeSet<Point> = pin_count.keys().copied().collect();
    merge_collinear(&mut segments, &pins, &pin_count);

    let (ns, na) = (segments.len(), attach.len());
    let node = |it: Item| match it {
        Item::Seg(i, _) => i,
        Item::Attach(k, _) => ns + k,
        Item::Junction(j) => ns + na + j,
    };
    let mut uf = Union((0..ns + na + junctions.len()).collect());
    let mut at: BTreeMap<Point, Vec<Item>> = BTreeMap::new();
    for (i, s) in segments.iter().enumerate() {
        at.entry(s.a).or_default().push(Item::Seg(i, s.bus));
        at.entry(s.b).or_default().push(Item::Seg(i, s.bus));
    }
    for (k, (p, kind)) in attach.iter().enumerate() {
        at.entry(*p).or_default().push(Item::Attach(k, *kind));
    }
    for (j, p) in junctions.iter().enumerate() {
        at.entry(*p).or_default().push(Item::Junction(j));
    }
    for items in at.values() {
        // A junction on a bus belongs to the bus; elsewhere to the wires.
        let bus_here = items.iter().any(|i| matches!(i, Item::Seg(_, true)));
        let layer = |i: &Item| match i {
            Item::Seg(_, bus) => Some(*bus),
            Item::Attach(_, Kind::Pin) => Some(false),
            Item::Junction(_) => Some(bus_here),
            _ => None,
        };
        for (x, a) in items.iter().enumerate() {
            for b in &items[x + 1..] {
                match (layer(a), layer(b)) {
                    (Some(la), Some(lb)) if la != lb => continue,
                    _ => uf.join(node(*a), node(*b)),
                }
            }
        }
    }
    // Labels along a wire or bus.
    for (k, (p, kind)) in attach.iter().enumerate() {
        if *kind != Kind::Label {
            continue;
        }
        for (i, s) in segments.iter().enumerate() {
            if on_inside(s, *p) {
                uf.join(ns + k, i);
            }
        }
    }
    (0..na).map(|k| uf.find(ns + k)).collect()
}
