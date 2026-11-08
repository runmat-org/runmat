//! Via stubs and back-drill.
//!
//! A via is plated through every copper layer it is drilled through (`from..=to`), but its net
//! only connects on some of them: the layers where its tracks or pads touch the barrel. The
//! plated barrel past the outermost connected layer is a stub. An open stub much shorter than a
//! quarter wave loads the line like a shunt capacitance `C = t_stub / Z`, which delays an edge on
//! a `Z` line by `Z · C / 2`, so a stub adds half its own propagation delay to its net
//! ([`ViaStub::delay`], counted by [`super::delay::net_timing`]).
//!
//! Rules on the via's net or class:
//! - `stub_max`: the longest stub allowed (DRC `via-stub`).
//! - `backdrill = true`: drill out each stub that reaches the board surface, from that surface,
//!   down to the connected layer (its *must-not-cut* layer), leaving `backdrill_stub` of barrel
//!   (the fabricator's depth tolerance, 0.2 mm when unset).
//! - `backdrill_oversize`: how much wider the back-drill is than the via's drill (0.2 mm when
//!   unset).
//!
//! A stub inside the board (a buried via's, or a blind via's inner end) can't be back-drilled;
//! only a shorter span fixes it.
//!
//! Lengths are in nanometres; depths are measured from the outer copper's surface.

use super::delay::DelayModel;
use crate::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use crate::geom::shape::Prim;
use crate::geom::{mm, Coord};
use crate::layer_set::CopperIndex;
use crate::layout::{Board, ItemId, NetId, Via};

/// The back-drill's residual stub when `backdrill_stub` is unset.
pub const DEFAULT_BACKDRILL_STUB: Coord = 200_000;
/// The back-drill's oversize over the via drill when `backdrill_oversize` is unset.
pub const DEFAULT_BACKDRILL_OVERSIZE: Coord = 200_000;

/// The board face a back-drill enters from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Face {
    Top,
    Bottom,
}

impl Face {
    pub fn word(self) -> &'static str {
        match self {
            Face::Top => "top",
            Face::Bottom => "bottom",
        }
    }
}

/// A back-drill: a larger, controlled-depth hole that removes a via's stub.
#[derive(Clone, Debug, PartialEq)]
pub struct Backdrill {
    pub face: Face,
    pub diameter: Coord,
    /// From the outer copper's surface.
    pub depth: Coord,
    /// The connected layer the drill stops short of.
    pub must_not_cut: CopperIndex,
    /// The barrel left past the must-not-cut layer's centre.
    pub residual: Coord,
}

/// One end of a via's stub.
#[derive(Clone, Debug, PartialEq)]
pub struct Stub {
    pub face: Face,
    /// The plated barrel past the connected layer's centre, before any back-drill.
    pub drilled: Coord,
    /// What is left after the back-drill (equal to `drilled` without one).
    pub length: Coord,
    pub backdrill: Option<Backdrill>,
    /// Whether the stub reaches the board surface (a back-drill can remove it).
    pub open_to_surface: bool,
}

/// A via whose barrel runs past the layers its net connects on.
#[derive(Clone, Debug, PartialEq)]
pub struct ViaStub {
    pub via: ItemId,
    pub net: NetId,
    /// The outermost connected copper layers, top-most first.
    pub connected: (CopperIndex, CopperIndex),
    pub stubs: Vec<Stub>,
    /// The longest stub left.
    pub length: Coord,
    /// The net's `stub_max`, if any.
    pub max: Option<Coord>,
    /// The delay the stubs add, ps.
    pub delay: f64,
}

/// The copper layers in `v`'s span where its net's tracks or pads touch the barrel.
pub fn connected_layers(board: &Board, v: &Via) -> Vec<CopperIndex> {
    let Some(net) = v.net else { return vec![] };
    let barrel = Prim::circle(v.at, v.diameter / 2);
    let mut on = std::collections::BTreeSet::new();
    for t in board.tracks.values() {
        if t.net == Some(net) && (v.from..=v.to).contains(&t.layer) && t.prims().iter().any(|p| barrel.gap(p) <= 0.0) {
            on.insert(t.layer);
        }
    }
    let n = board.stackup.copper_count();
    for fp in board.footprints.values() {
        if !fp.pad_nets.values().any(|&x| x == net) {
            continue;
        }
        for pad in fp.pads(n) {
            if pad.net == Some(net) && barrel.gap(&pad.prim) <= 0.0 {
                on.extend(pad.layers.iter().copied().filter(|l| (v.from..=v.to).contains(l)));
            }
        }
    }
    on.into_iter().collect()
}

fn len_rule(rules: &BoardRules, g: &BoardGraph, it: BItem, prop: &str, default: Coord) -> Coord {
    rules.len(g, it, prop).unwrap_or(default)
}

/// Every via on the board with a stub, with its back-drill where the rules ask for one.
pub fn via_stubs(board: &Board, rules: &BoardRules, g: &BoardGraph) -> Vec<ViaStub> {
    let model = DelayModel::new(&board.stackup);
    let s = &board.stackup;
    let n = s.copper_count();
    let face_z = |l: CopperIndex, top: bool| -> Option<Coord> {
        let k = s.copper_index(l)?;
        let above: Coord = s.layers[..k].iter().map(|x| x.thickness).sum();
        Some(if top { above } else { above + s.layers[k].thickness })
    };
    // The board's outer copper surfaces.
    let (Some(top_surface), Some(bottom_surface)) = (face_z(0, true), face_z(n.saturating_sub(1), false)) else {
        return vec![];
    };
    let mut out = vec![];
    for v in board.vias.values() {
        let Some(net) = v.net else { continue };
        let on = connected_layers(board, v);
        let (Some(&lo), Some(&hi)) = (on.first(), on.last()) else { continue };
        let it = BItem::Via(v.id);
        let backdrill = matches!(rules.value(g, it, props::BACKDRILL), Some(RuleValue::Bool(true)));
        let residual = len_rule(rules, g, it, props::BACKDRILL_STUB, DEFAULT_BACKDRILL_STUB);
        let oversize = len_rule(rules, g, it, props::BACKDRILL_OVERSIZE, DEFAULT_BACKDRILL_OVERSIZE);
        let mut stubs = vec![];
        let mut delay = 0.0;
        for (face, used, end) in [(Face::Top, lo, v.from), (Face::Bottom, hi, v.to)] {
            if used == end {
                continue;
            }
            let (Some(c), Some(e)) = (s.copper_z(used), face_z(end, face == Face::Top)) else { continue };
            let drilled = (c - e).abs();
            let open = match face {
                Face::Top => e == top_surface,
                Face::Bottom => e == bottom_surface,
            };
            let bd = (backdrill && open && drilled > residual).then(|| Backdrill {
                face,
                diameter: v.drill + oversize,
                depth: drilled - residual,
                must_not_cut: used,
                residual,
            });
            let length = bd.as_ref().map_or(drilled, |b| b.residual);
            // The stub's own delay at the dielectric it crosses, half of it seen by the edge.
            let stub_delay = model.via(used, end).delay * (length as f64 / s.span_length(used, end).max(1) as f64);
            delay += stub_delay / 2.0;
            stubs.push(Stub { face, drilled, length, backdrill: bd, open_to_surface: open });
        }
        if stubs.is_empty() {
            continue;
        }
        let length = stubs.iter().map(|x| x.length).max().unwrap_or(0);
        out.push(ViaStub { via: v.id, net, connected: (lo, hi), stubs, length, max: rules.len(g, it, props::STUB_MAX), delay });
    }
    out
}

/// A back-drill depth as the fab drawing states it, mm.
pub fn depth_mm(depth: Coord) -> f64 {
    (depth as f64 / mm(1.0) as f64 * 1000.0).round() / 1000.0
}
