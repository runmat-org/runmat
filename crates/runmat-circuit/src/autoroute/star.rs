//! Connections between rooms leave from the hub outside them.
//!
//! The ratsnest is a minimum spanning tree, so a net that reaches the same pin in every copied
//! room (a converter's CONVST or SCLK in each channel) chains room to room: each airwire runs
//! from one channel's pin straight into the next channel, and the route runs through the rooms
//! however their crossing is priced. A layout engineer routes such a net from its source outside
//! the rooms to each room. This rewires a net that has anchors outside every room: within a room,
//! and among the anchors outside the rooms, the tree's own airwires stay; the outside groups join
//! each other by their nearest anchors, and each room's group joins the nearest outside anchor.
//! A net with no anchor outside the rooms keeps its tree.

use crate::board_rules::BItem;
use crate::geom::{Point, Rect};
use crate::route::Airwire;
use std::collections::BTreeMap;

/// `wires` with every room-to-room airwire replaced as the module describes.
pub(super) fn star_between_rooms(wires: Vec<Airwire>, rooms: &[Rect]) -> Vec<Airwire> {
    let mut by_net: BTreeMap<crate::layout::NetId, Vec<Airwire>> = BTreeMap::new();
    for w in wires {
        by_net.entry(w.net).or_default().push(w);
    }
    let room_of = |p: Point| rooms.iter().position(|r| r.contains(p));
    let mut out = Vec::new();
    for (_, net) in by_net {
        let between = net
            .iter()
            .any(|w| matches!((room_of(w.a), room_of(w.b)), (Some(x), Some(y)) if x != y));
        let outside = net
            .iter()
            .any(|w| room_of(w.a).is_none() || room_of(w.b).is_none());
        if !between || !outside {
            out.extend(net);
            continue;
        }
        out.extend(rewire(net, &room_of));
    }
    out
}

fn rewire(net: Vec<Airwire>, room_of: &dyn Fn(Point) -> Option<usize>) -> Vec<Airwire> {
    // The anchors, each with its item, in a fixed order.
    let mut anchors: Vec<(Point, BItem)> = Vec::new();
    let index = |anchors: &mut Vec<(Point, BItem)>, p: Point, item: BItem| {
        anchors
            .iter()
            .position(|(q, _)| *q == p)
            .unwrap_or_else(|| {
                anchors.push((p, item));
                anchors.len() - 1
            })
    };
    let mut parent: Vec<usize> = Vec::new();
    let mut kept = Vec::new();
    let mut edges = Vec::new();
    for w in &net {
        let (a, b) = (
            index(&mut anchors, w.a, w.from),
            index(&mut anchors, w.b, w.to),
        );
        edges.push((a, b, room_of(w.a) == room_of(w.b)));
    }
    parent.extend(0..anchors.len());
    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut i = i;
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for (w, (a, b, same)) in net.iter().zip(&edges) {
        if *same {
            let (ra, rb) = (find(&mut parent, *a), find(&mut parent, *b));
            parent[ra.max(rb)] = ra.min(rb);
            kept.push(w.clone());
        }
    }
    // Groups by root: whether they sit outside the rooms, and their anchors.
    let mut groups: BTreeMap<usize, (bool, Vec<usize>)> = BTreeMap::new();
    for i in 0..anchors.len() {
        let root = find(&mut parent, i);
        let g = groups
            .entry(root)
            .or_insert((room_of(anchors[i].0).is_none(), Vec::new()));
        g.1.push(i);
    }
    let net_id = net[0].net;
    let wire = |a: usize, b: usize| Airwire {
        net: net_id,
        a: anchors[a].0,
        b: anchors[b].0,
        from: anchors[a].1,
        to: anchors[b].1,
        length: anchors[a].0.dist(anchors[b].0),
    };
    let nearest = |xs: &[usize], ys: &[usize]| -> Option<(usize, usize, f64)> {
        let mut best: Option<(usize, usize, f64)> = None;
        for &x in xs {
            for &y in ys {
                let d = anchors[x].0.dist(anchors[y].0);
                if best.is_none_or(|(_, _, b)| d < b) {
                    best = Some((x, y, d));
                }
            }
        }
        best
    };
    // The outside groups join into one tree (Prim's, by nearest anchors).
    let outside: Vec<&Vec<usize>> = groups.values().filter(|g| g.0).map(|g| &g.1).collect();
    let mut joined: Vec<usize> = outside[0].clone();
    let mut left: Vec<&Vec<usize>> = outside[1..].to_vec();
    while !left.is_empty() {
        let (k, (x, y, _)) = left
            .iter()
            .enumerate()
            .filter_map(|(k, g)| nearest(&joined, g).map(|n| (k, n)))
            .min_by(|a, b| a.1 .2.total_cmp(&b.1 .2))
            .expect("groups have anchors");
        kept.push(wire(x, y));
        joined.extend(left.remove(k).iter().copied());
    }
    // Each room's group joins the nearest outside anchor.
    for g in groups.values().filter(|g| !g.0) {
        if let Some((x, y, _)) = nearest(&joined, &g.1) {
            kept.push(wire(x, y));
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::NetId;

    fn wire(a: Point, b: Point) -> Airwire {
        Airwire {
            net: NetId(1),
            a,
            b,
            from: BItem::Edge,
            to: BItem::Edge,
            length: a.dist(b),
        }
    }

    /// Three rooms stacked in a column, a pin in each at the same x, and the source to the right:
    /// the tree chains room to room; the rewired net runs from the source to each room.
    #[test]
    fn a_chain_through_rooms_becomes_a_star_from_the_source() {
        let rooms: Vec<Rect> = (0..3)
            .map(|k| {
                Rect::new(
                    Point::mm(0.0, 30.0 * k as f64),
                    Point::mm(40.0, 30.0 * k as f64 + 25.0),
                )
            })
            .collect();
        let pins: Vec<Point> = (0..3)
            .map(|k| Point::mm(20.0, 30.0 * k as f64 + 12.0))
            .collect();
        let source = Point::mm(80.0, 40.0);
        let tree = vec![
            wire(pins[0], pins[1]),
            wire(pins[1], pins[2]),
            wire(pins[1], source),
        ];
        let star = star_between_rooms(tree, &rooms);
        assert_eq!(star.len(), 3);
        for w in &star {
            assert!(
                w.a == source || w.b == source,
                "{w:?} doesn't start at the source"
            );
        }
        // A net without an anchor outside the rooms keeps its tree.
        let inside = vec![wire(pins[0], pins[1]), wire(pins[1], pins[2])];
        assert_eq!(star_between_rooms(inside.clone(), &rooms), inside);
    }
}
