//! Signals through series parts: one tuned signal across a series resistor, a ferrite or any
//! other two-pin part on its path.
//!
//! A net whose rules set `through_parts = true` is followed through each two-pad part it
//! touches when the part's other net goes on to exactly one more pad (a point-to-point
//! continuation, such as the resistor's far side to the receiver). The signal's timing is the
//! copper of every net on the path plus each part's `through_delay` (a delay such as `8 ps`,
//! set by a rule on the footprint; zero when none is written). A part's own body adds no
//! length.
//!
//! ```text
//! rule dq = { select = ":net(DQ*)", match_group = byte0, through_parts = true }
//! rule series = { select = "footprint#R21, footprint#R22", through_delay = 6 ps }
//! ```

use super::delay::Timing;
use crate::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use crate::layout::{Board, ItemId, NetId};
use std::collections::{BTreeMap, BTreeSet};

/// A signal: its nets in path order from the net it starts at, the parts it runs through, and
/// its total timing.
#[derive(Clone, Debug, PartialEq)]
pub struct Signal {
    pub nets: Vec<NetId>,
    /// Designators of the series parts.
    pub parts: Vec<String>,
    pub timing: Timing,
}

/// How many pads each net reaches.
fn pad_counts(board: &Board) -> BTreeMap<NetId, usize> {
    let mut out = BTreeMap::new();
    for f in board.footprints.values() {
        for n in f.pad_nets.values() {
            *out.entry(*n).or_default() += 1;
        }
    }
    out
}

/// The two-pad parts each net touches: (footprint, the other pad's net).
fn series_parts(board: &Board) -> BTreeMap<NetId, Vec<(ItemId, NetId)>> {
    let mut out: BTreeMap<NetId, Vec<(ItemId, NetId)>> = BTreeMap::new();
    for f in board.footprints.values() {
        if f.package.pads.len() != 2 {
            continue;
        }
        let nets: Vec<NetId> = f
            .package
            .pads
            .iter()
            .filter_map(|p| f.pad_nets.get(&p.name).copied())
            .collect();
        if let [a, b] = nets[..] {
            if a != b {
                out.entry(a).or_default().push((f.id, b));
                out.entry(b).or_default().push((f.id, a));
            }
        }
    }
    out
}

/// The signal starting at `net`: `net` alone unless its rules set `through_parts`.
pub fn signal_of(
    board: &Board,
    rules: &BoardRules,
    g: &BoardGraph,
    net: NetId,
    timing: &BTreeMap<NetId, Timing>,
) -> Signal {
    let follow = matches!(
        rules.value(g, BItem::Net(net.0), props::THROUGH_PARTS),
        Some(RuleValue::Bool(true))
    );
    let mut sig = Signal {
        nets: vec![net],
        parts: Vec::new(),
        timing: timing.get(&net).copied().unwrap_or_default(),
    };
    if !follow {
        return sig;
    }
    let counts = pad_counts(board);
    let parts = series_parts(board);
    let mut seen: BTreeSet<NetId> = BTreeSet::from([net]);
    let mut frontier = vec![net];
    while let Some(at) = frontier.pop() {
        for &(f, other) in parts.get(&at).map(Vec::as_slice).unwrap_or(&[]) {
            // The far net continues the signal only to one more pad (point to point).
            if seen.contains(&other) || counts.get(&other).copied().unwrap_or(0) != 2 {
                continue;
            }
            seen.insert(other);
            frontier.push(other);
            sig.nets.push(other);
            let fp = &board.footprints[&f];
            sig.parts.push(fp.reference.clone());
            sig.timing += timing.get(&other).copied().unwrap_or_default();
            if let Some(RuleValue::Time(ps)) =
                rules.value(g, BItem::Footprint(f), props::THROUGH_DELAY)
            {
                sig.timing.delay += ps;
            }
        }
    }
    sig
}
