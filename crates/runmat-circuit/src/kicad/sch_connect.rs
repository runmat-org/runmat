//! Schematic connectivity of one `.kicad_sch` sheet.
//!
//! Written independently from KiCad's documented behavior: the schematic editor manual
//! ("Electrical Connections", "Wire Junctions", "Labels", "Buses", "Power Symbols") and the
//! netlists KiCad itself writes for sample files. No KiCad source code is included.
//!
//! The rules, each checked by `tests/kicad_connectivity.rs` against KiCad's own netlist of
//! `tests/kicad/connectivity.kicad_sch` (written by `kicad-cli sch export netlist`, stored as
//! `tests/kicad/connectivity.net`):
//!
//! - Wires connect to other wires or to pins only where their ends coincide exactly (manual).
//!   A wire end, pin or sheet pin lying on the middle of another wire does not connect to it,
//!   and wires that cross are not connected. Collinear wires that overlap are no exception:
//!   each connects only at its own ends (observed in KiCad's netlist).
//! - A junction joins every wire passing through or ending at its point (manual).
//! - Pins stacked on the same point connect (manual).
//! - A label connects to a wire anywhere along it (observed), and to every wire under its
//!   anchor: two crossing wires under one label are joined, which KiCad's ERC reports as a
//!   label connecting more than one wire (observed).
//! - A pin cannot connect directly to a bus (manual). KiCad 10's netlist export does join two
//!   pins drawn on the ends of one bus with no bus label; we follow the manual there, and the
//!   test checks that case on its own.
//!
//! [`groups`] applies these rules and returns a group number for each attachment point.

use crate::geom::{orient, Point};
use std::collections::BTreeMap;

/// A wire or bus segment.
#[derive(Clone, Debug)]
pub(super) struct Seg {
    pub a: Point,
    pub b: Point,
    pub bus: bool,
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
