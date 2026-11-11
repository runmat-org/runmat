//! Return paths over plane splits.
//!
//! A controlled line's return current flows in its reference plane, right under the track.
//! Where the plane has a gap, or the track crosses from one plane net to another (a split
//! between ground and a supply pour), the return current detours around the gap, so the line
//! sees a step in impedance and radiates.
//!
//! A track's reference planes are the copper layers next to it in the stack that carry a pour.
//! Each controlled track (a net with an `impedance` or `diff_impedance` target, or
//! `return_path = true`; `return_path = false` turns the check off) is sampled along its
//! centreline every tenth of a millimetre on each reference plane:
//! - **split**: the pour under the track changes net;
//! - **gap**: the track runs over no pour for more than `return_gap_max` (0.5 mm when unset,
//!   longer than a via antipad, so a track's own via doesn't count).

use crate::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use crate::geom::{mm, Coord, Point};
use crate::layer_set::CopperIndex;
use crate::layout::{Board, ItemId, NetId};

/// The longest stretch over no plane copper that isn't a finding, when `return_gap_max` is unset.
pub const DEFAULT_GAP_MAX: Coord = 500_000;
const STEP: f64 = 100_000.0;

#[derive(Clone, Debug, PartialEq)]
pub enum CrossingKind {
    /// The pour under the track changes from net `from` to net `to`.
    Split { from: NetId, to: NetId },
    /// The track runs `length` nm over no pour.
    Gap { length: Coord },
}

/// One place a controlled track leaves its reference plane.
#[derive(Clone, Debug, PartialEq)]
pub struct Crossing {
    pub track: ItemId,
    pub net: NetId,
    pub layer: CopperIndex,
    pub plane: CopperIndex,
    /// Where the plane copper ends under the track.
    pub at: Point,
    pub kind: CrossingKind,
    /// The pours on either side (a split) or the one the track leaves (a gap).
    pub zones: Vec<ItemId>,
}

/// Whether `net` gets the return-path check.
fn checked(rules: &BoardRules, g: &BoardGraph, net: NetId) -> bool {
    let it = BItem::Net(net.0);
    match rules.value(g, it, props::RETURN_PATH) {
        Some(RuleValue::Bool(b)) => b,
        _ => {
            rules.value(g, it, props::IMPEDANCE).is_some()
                || rules.value(g, it, props::DIFF_IMPEDANCE).is_some()
        }
    }
}

/// Every place a controlled track crosses a split or a gap in a reference plane.
pub fn crossings(board: &Board, rules: &BoardRules, g: &BoardGraph) -> Vec<Crossing> {
    let n = board.stackup.copper_count();
    // Each layer's pour fills, with their nets.
    let planes = |l: CopperIndex| {
        board
            .zones
            .values()
            .filter_map(move |z| Some((z.id, z.net?, z.fill.get(&l).filter(|f| !f.is_empty())?)))
            .collect::<Vec<_>>()
    };
    let by_layer: Vec<_> = (0..n).map(planes).collect();
    let mut wanted = std::collections::BTreeMap::new();
    let mut out = vec![];
    for t in board.tracks.values() {
        let Some(net) = t.net else { continue };
        if !*wanted.entry(net).or_insert_with(|| checked(rules, g, net)) {
            continue;
        }
        let gap_max = rules
            .len(g, BItem::Track(t.id), props::RETURN_GAP_MAX)
            .unwrap_or(DEFAULT_GAP_MAX) as f64;
        let neighbours = [
            t.layer.checked_sub(1),
            (t.layer + 1 < n).then_some(t.layer + 1),
        ];
        for plane in neighbours.into_iter().flatten() {
            let fills = &by_layer[plane as usize];
            if fills.is_empty() {
                continue;
            }
            let under = |p: Point| {
                fills
                    .iter()
                    .find(|(_, _, f)| f.contains(p).unwrap_or(true))
                    .map(|(z, net, _)| (*z, *net))
            };
            // Samples along the centreline, in order.
            let mut pts = vec![];
            for prim in t.prims() {
                if let crate::geom::shape::Core::Segment(a, b) = prim.core {
                    let len = a.dist(b);
                    let k = (len / STEP).ceil().max(1.0) as usize;
                    for i in 0..=k {
                        let f = i as f64 / k as f64;
                        pts.push(Point {
                            x: a.x + ((b.x - a.x) as f64 * f) as Coord,
                            y: a.y + ((b.y - a.y) as f64 * f) as Coord,
                        });
                    }
                }
            }
            let mut last: Option<(ItemId, NetId, Point)> = None;
            let mut void: Option<(Point, f64)> = None;
            let mut prev: Option<Point> = None;
            for p in pts {
                let step = prev.map_or(0.0, |q| q.dist(p));
                prev = Some(p);
                match under(p) {
                    Some((z, pn)) => {
                        if let Some((lz, ln, at)) = last {
                            if ln != pn {
                                out.push(Crossing {
                                    track: t.id,
                                    net,
                                    layer: t.layer,
                                    plane,
                                    at: void.map_or(at, |v| v.0),
                                    kind: CrossingKind::Split { from: ln, to: pn },
                                    zones: vec![lz, z],
                                });
                            } else if let Some((at, len)) = void {
                                if len > gap_max {
                                    out.push(Crossing {
                                        track: t.id,
                                        net,
                                        layer: t.layer,
                                        plane,
                                        at,
                                        kind: CrossingKind::Gap {
                                            length: len as Coord,
                                        },
                                        zones: vec![lz],
                                    });
                                }
                            }
                        }
                        last = Some((z, pn, p));
                        void = None;
                    }
                    None => {
                        void = Some(match void {
                            Some((at, len)) => (at, len + step),
                            None => (p, step),
                        });
                    }
                }
            }
            // A track that ends over no plane after leaving one.
            if let (Some((lz, _, _)), Some((at, len))) = (last, void) {
                if len > gap_max {
                    out.push(Crossing {
                        track: t.id,
                        net,
                        layer: t.layer,
                        plane,
                        at,
                        kind: CrossingKind::Gap {
                            length: len as Coord,
                        },
                        zones: vec![lz],
                    });
                }
            }
        }
    }
    out
}

/// `0.5 mm`-style text for a gap length.
pub fn gap_text(len: Coord) -> String {
    format!("{:.2} mm", len as f64 / mm(1.0) as f64)
}
