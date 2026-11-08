//! Tuning a net, or every short member of a group, to a length or delay target.
//!
//! The meanders go on the net's longest straight track ([`crate::pair::tune`]); two short
//! members of a group that form a declared pair are meandered together along their coupled
//! run ([`super::pair_tune`]), so they keep their gap. A delay target
//! becomes a length on that track's layer: the missing picoseconds divided by the layer's
//! ps/mm at the track's width. A net whose rules set `through_parts` is tuned as its whole
//! signal: the copper and part delays beyond it count toward the target.

use super::delay::{net_timing, DelayModel};
use super::groups::{groups, Basis, Status};
use super::signals::signal_of;
use crate::board_rules::{BoardGraph, BoardRules};
use crate::layout::{Board, NetId};
use crate::pair::{longest_run, tune, Pattern, Tuning};

/// A tuning target: a length (mm) or a delay (ps).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    Length(f64),
    Delay(f64),
}

/// The result of tuning one net, in the target's unit.
#[derive(Clone, Debug, PartialEq)]
pub struct Tuned {
    pub net: String,
    pub basis: Basis,
    pub before: f64,
    pub after: f64,
    pub tuning: Tuning,
}

/// Tunes `net` (as its signal) to `target` within `tolerance` (mm or ps, as the target).
pub fn tune_net(board: &mut Board, net: NetId, target: Target, tolerance: f64, pattern: Pattern, pairs: &[(NetId, NetId)]) -> Result<Tuned, String> {
    let name = board.net_name(net).to_owned();
    let signal = |board: &Board| {
        let (mut rules, _) = BoardRules::for_board(board);
        let g = BoardGraph::new(board, pairs);
        rules.rebuild(&g);
        let t = net_timing(board, &rules, &g);
        (signal_of(board, &rules, &g, net, &t), t.get(&net).copied().unwrap_or_default())
    };
    let (sig, own) = signal(board);
    match target {
        Target::Length(mm) => {
            let outside = (sig.timing.length - own.length) * 1e-6;
            let tuning = tune(board, net, mm, tolerance, pairs, pattern, outside)?;
            Ok(Tuned { net: name, basis: Basis::Length, before: tuning.before, after: tuning.after, tuning })
        }
        Target::Delay(ps) => {
            let run = longest_run(board, net).ok_or_else(|| format!("{name} has no straight track to meander"))?;
            let per_mm = DelayModel::new(&board.stackup).ps_per_mm(run.layer, run.width);
            let before = sig.timing.delay;
            if (ps - before).abs() <= tolerance {
                return Err(format!("{name} is {before:.2} ps, already within {tolerance} ps of {ps} ps"));
            }
            if ps < before {
                return Err(format!("{name} is {before:.2} ps, slower than {ps} ps; meanders only add delay, so shorten its route"));
            }
            let own_mm = own.length * 1e-6;
            let length_target = own_mm + (ps - before) / per_mm;
            let tuning = tune(board, net, length_target, tolerance / per_mm, pairs, pattern, 0.0)?;
            let after = signal(board).0.timing.delay;
            Ok(Tuned { net: name, basis: Basis::Delay, before, after, tuning })
        }
    }
}

/// Tunes every short member of group `group` to the group's target. Members already matched
/// or too long are left; each short member's result is reported, success or not.
pub fn tune_group(board: &mut Board, group: &str, pattern: Pattern, pairs: &[(NetId, NetId)]) -> Result<Vec<Result<Tuned, String>>, String> {
    let (mut rules, _) = BoardRules::for_board(board);
    let found = {
        let g = BoardGraph::new(board, pairs);
        rules.rebuild(&g);
        groups(board, &rules, &g, &net_timing(board, &rules, &g))
    };
    let names: Vec<String> = found.iter().map(|g| g.name.clone()).collect();
    let grp = found
        .into_iter()
        .find(|g| g.name == group)
        .ok_or_else(|| format!("`{group}` is no match group on this board (groups: {})", if names.is_empty() { "none".into() } else { names.join(", ") }))?;
    let target = match grp.basis {
        Basis::Length => Target::Length(grp.target),
        Basis::Delay => Target::Delay(grp.target),
    };
    // Two short members that form a declared pair are tuned together, so they stay coupled.
    let short: Vec<_> = grp.members.iter().filter(|m| m.status == Status::Short).collect();
    let mut out = vec![];
    let mut done = std::collections::BTreeSet::new();
    for m in &short {
        if done.contains(&m.net) {
            continue;
        }
        let partner = pairs.iter().find_map(|&(a, b)| (a == m.net).then_some(b).or((b == m.net).then_some(a)));
        if let Some(q) = partner.and_then(|q| short.iter().find(|x| x.net == q)) {
            done.extend([m.net, q.net]);
            let (p, n) = pairs.iter().copied().find(|&(a, b)| (a, b) == (m.net, q.net) || (b, a) == (m.net, q.net)).unwrap_or((m.net, q.net));
            out.extend(tune_pair_to(board, p, n, &grp, pattern, pairs));
            continue;
        }
        done.insert(m.net);
        out.push(tune_net(board, m.net, target, grp.tolerance, pattern, pairs));
    }
    Ok(out)
}

/// Tunes a pair of short members together by their mean shortfall, reported per net.
fn tune_pair_to(board: &mut Board, p: NetId, n: NetId, grp: &super::groups::Group, pattern: Pattern, pairs: &[(NetId, NetId)]) -> Vec<Result<Tuned, String>> {
    let dev = |net: NetId| grp.members.iter().find(|m| m.net == net).map_or(0.0, |m| m.deviation);
    let short = -(dev(p) + dev(n)) / 2.0;
    let missing_mm = match grp.basis {
        Basis::Length => short,
        Basis::Delay => {
            let Some(run) = super::pair_tune::coupled_run(board, p, n) else {
                return vec![Err(format!("{}/{} has no coupled run to meander", board.net_name(p), board.net_name(n)))];
            };
            short / DelayModel::new(&board.stackup).ps_per_mm(run.p.layer, run.p.width)
        }
    };
    let value = |board: &Board, net: NetId| {
        let (mut rules, _) = BoardRules::for_board(board);
        let g = BoardGraph::new(board, pairs);
        rules.rebuild(&g);
        let t = net_timing(board, &rules, &g);
        let sig = signal_of(board, &rules, &g, net, &t);
        match grp.basis {
            Basis::Length => sig.timing.length * 1e-6,
            Basis::Delay => sig.timing.delay,
        }
    };
    let before = (value(board, p), value(board, n));
    match super::pair_tune::tune_pair(board, p, n, missing_mm, pairs, pattern) {
        Ok(t) => {
            let after = (value(board, p), value(board, n));
            let one = |net: NetId, b: f64, a: f64, lb: f64, la: f64| Tuned {
                net: board.net_name(net).to_owned(),
                basis: grp.basis,
                before: b,
                after: a,
                tuning: Tuning { before: lb, after: la, bumps: t.bumps, amplitude: t.amplitude },
            };
            vec![Ok(one(p, before.0, after.0, t.before.0, t.after.0)), Ok(one(n, before.1, after.1, t.before.1, t.after.1))]
        }
        Err(e) => vec![Err(e)],
    }
}
