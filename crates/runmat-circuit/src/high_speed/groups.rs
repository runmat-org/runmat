//! Matched-length groups: nets tuned to one target, in length or in delay.
//!
//! A group is the nets whose rules give the same `match_group` name. Its rules may also set:
//!
//! - `match_target`: the target, a length (`42 mm`) or a delay (`300 ps`). With no target the
//!   group matches its longest member (in delay, its slowest).
//! - `match_tolerance`: how far a member may sit from the target, a length or a delay. Without
//!   it, 0.1 mm or 1 ps.
//!
//! A delay target or tolerance makes the group a delay group: its members are compared in
//! picoseconds, from each layer's propagation velocity ([`super::delay`]), so a member routed
//! on a stripline layer reaches the target with less copper than one on a microstrip layer.
//!
//! ```text
//! rule byte0 = { select = ":net(DQ0, DQ1, DQ2, DQ3, DQ4, DQ5, DQ6, DQ7, DM0)", match_group = byte0, match_target = 420 ps, match_tolerance = 5 ps }
//! rule addr = { select = ":net(A*)", match_group = addr, match_tolerance = 0.5 mm }
//! ```
//!
//! A member's timing is its signal's ([`super::signals`]): a net that runs through a series
//! resistor to another net counts the copper on both sides and the part's own delay.

use super::delay::Timing;
use crate::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use crate::layout::{Board, NetId};
use std::collections::BTreeMap;

/// What a group compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// mm.
    Length,
    /// ps.
    Delay,
}

impl Basis {
    pub fn unit(self) -> &'static str {
        match self {
            Basis::Length => "mm",
            Basis::Delay => "ps",
        }
    }
    /// A member's value in this basis (mm or ps).
    pub fn of(self, t: &Timing) -> f64 {
        match self {
            Basis::Length => t.length * 1e-6,
            Basis::Delay => t.delay,
        }
    }
}

/// A member against its group's target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Within the tolerance.
    Matched,
    /// Shorter (or faster) than the target by more than the tolerance: it needs tuning.
    Short,
    /// Longer (or slower): its route must get shorter.
    Long,
    /// Not routed yet.
    Unrouted,
}

impl Status {
    pub fn word(self) -> &'static str {
        match self {
            Status::Matched => "matched",
            Status::Short => "short",
            Status::Long => "long",
            Status::Unrouted => "unrouted",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    pub net: NetId,
    pub name: String,
    pub timing: Timing,
    /// In the group's basis (mm or ps).
    pub value: f64,
    /// `value − target`.
    pub deviation: f64,
    pub status: Status,
    /// Series parts the member's signal runs through.
    pub through: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub name: String,
    pub basis: Basis,
    /// mm or ps.
    pub target: f64,
    /// Whether the target was written (else it is the longest member's).
    pub written: bool,
    pub tolerance: f64,
    pub members: Vec<Member>,
    /// The rule that names the group, for findings.
    pub rule: String,
    pub source: runmat_rules::SourceKey,
}

/// A rule value in a basis: a length in mm or a delay in ps; `None` for the other kind.
fn in_basis(v: &RuleValue, basis: Basis) -> Option<f64> {
    match (v, basis) {
        (RuleValue::Len(l), Basis::Length) => Some(*l as f64 * 1e-6),
        (RuleValue::Time(ps), Basis::Delay) => Some(*ps),
        _ => None,
    }
}

/// The board's matched-length groups, each member with its timing (`timing`, by net) against
/// the target.
pub fn groups(
    board: &Board,
    rules: &BoardRules,
    g: &BoardGraph,
    timing: &BTreeMap<NetId, Timing>,
) -> Vec<Group> {
    let mut by_name: BTreeMap<String, Vec<NetId>> = BTreeMap::new();
    for i in 0..board.nets.len() as u32 {
        if let Some(RuleValue::Text(name)) = rules.value(g, BItem::Net(i), props::MATCH_GROUP) {
            by_name.entry(name).or_default().push(NetId(i));
        }
    }
    by_name
        .into_iter()
        .map(|(name, nets)| {
            let first = BItem::Net(nets[0].0);
            let value = |p| nets.iter().find_map(|n| rules.value(g, BItem::Net(n.0), p));
            let (target_v, tol_v) = (value(props::MATCH_TARGET), value(props::MATCH_TOLERANCE));
            let basis = if matches!(target_v, Some(RuleValue::Time(_)))
                || matches!(tol_v, Some(RuleValue::Time(_)))
            {
                Basis::Delay
            } else {
                Basis::Length
            };
            let signals: BTreeMap<NetId, super::signals::Signal> = nets
                .iter()
                .map(|n| (*n, super::signals::signal_of(board, rules, g, *n, timing)))
                .collect();
            let routed = |n: &NetId| Some(signals[n].timing).filter(|t| t.length > 0.0);
            let written = target_v.as_ref().and_then(|v| in_basis(v, basis));
            let longest = nets
                .iter()
                .filter_map(routed)
                .map(|t| basis.of(&t))
                .fold(0.0, f64::max);
            let target = written.unwrap_or(longest);
            let tolerance =
                tol_v
                    .as_ref()
                    .and_then(|v| in_basis(v, basis))
                    .unwrap_or(match basis {
                        Basis::Length => 0.1,
                        Basis::Delay => 1.0,
                    });
            let members = nets
                .iter()
                .map(|n| {
                    let t = routed(n);
                    let value = t.map(|t| basis.of(&t)).unwrap_or(0.0);
                    let deviation = value - target;
                    let status = match t {
                        None => Status::Unrouted,
                        Some(_) if deviation.abs() <= tolerance + 1e-9 => Status::Matched,
                        Some(_) if deviation < 0.0 => Status::Short,
                        Some(_) => Status::Long,
                    };
                    Member {
                        net: *n,
                        name: board.net_name(*n).to_owned(),
                        timing: t.unwrap_or_default(),
                        value,
                        deviation,
                        status,
                        through: signals[n].parts.clone(),
                    }
                })
                .collect();
            let (rule, source) = rules.single_origin(g, first, props::MATCH_GROUP);
            Group {
                name,
                basis,
                target,
                written: written.is_some(),
                tolerance,
                members,
                rule,
                source,
            }
        })
        .collect()
}
