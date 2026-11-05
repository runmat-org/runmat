//! Net checks: connectivity (unrouted islands), thermal spokes, lengths and differential pairs.

use super::*;
use crate::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use crate::connectivity::{islands, net_nodes};
use crate::elab::Severity;
use crate::geom::Point;
use crate::layout::{Board, NetId};
use std::collections::{BTreeMap, BTreeSet};

impl Drc {
    pub(super) fn check_nets(
        &mut self,
        board: &Board,
        rules: &BoardRules,
        g: &BoardGraph,
        nets: &BTreeSet<NetId>,
    ) {
        let nodes = net_nodes(board);
        let mut lengths: BTreeMap<NetId, f64> = BTreeMap::new();
        for t in board.tracks.values() {
            if let Some(n) = t.net {
                *lengths.entry(n).or_default() += t.length();
            }
        }
        for v in board.vias.values() {
            if let Some(n) = v.net {
                *lengths.entry(n).or_default() += board.stackup.span_length(v.from, v.to) as f64;
            }
        }
        for &n in nets {
            let net_item = BItem::Net(n.0);
            if let Some(ns) = nodes.get(&n) {
                let pads = ns
                    .iter()
                    .filter(|x| matches!(x.item, BItem::Pad(..)))
                    .count();
                if pads >= 2 {
                    let c = islands(n, ns.clone());
                    if c.islands > 1 {
                        let mut items = vec![net_item];
                        // Name one pad per island so the finding cross-probes.
                        let mut seen = BTreeSet::new();
                        for (i, nd) in c.nodes.iter().enumerate() {
                            if matches!(nd.item, BItem::Pad(..)) && seen.insert(c.island[i]) {
                                items.push(nd.item);
                            }
                        }
                        self.put(DrcFinding {
                            code: "unrouted".into(),
                            severity: Severity::Error,
                            items: key_items(items),
                            layer: None,
                            at: c.nodes[0].anchors[0],
                            actual: Some(c.islands as f64),
                            required: Some(1.0),
                            rule: "connectivity".into(),
                            source: SourceKey::default(),
                            message: format!(
                                "net `{}` has {} unconnected islands",
                                board.net_name(n),
                                c.islands
                            ),
                        });
                    }
                }
            }
            let len = lengths.get(&n).copied().unwrap_or(0.0);
            let lmin = rules.len(g, net_item, props::LENGTH_MIN);
            let lmax = rules.len(g, net_item, props::LENGTH_MAX);
            if lmin.is_some_and(|m| len < m as f64) || lmax.is_some_and(|m| len > m as f64) {
                let p = if lmin.is_some_and(|m| len < m as f64) {
                    props::LENGTH_MIN
                } else {
                    props::LENGTH_MAX
                };
                let (rule, source) = rules.single_origin(g, net_item, p);
                self.put(DrcFinding {
                    code: "length".into(),
                    severity: Severity::Error,
                    items: vec![net_item],
                    layer: None,
                    at: Point::ORIGIN,
                    actual: Some(len),
                    required: Some(lmin.or(lmax).unwrap_or(0) as f64),
                    rule,
                    source,
                    message: format!(
                        "net `{}` routed length {:.3} mm",
                        board.net_name(n),
                        len * 1e-6
                    ),
                });
            }
            if let Some(RuleValue::Num(z)) = rules.value(g, net_item, props::IMPEDANCE) {
                self.impedance_targets
                    .insert(board.net_name(n).to_string(), z);
            }
        }
        // Match groups.
        let mut groups: BTreeMap<String, Vec<NetId>> = BTreeMap::new();
        for i in 0..board.nets.len() as u32 {
            if let Some(RuleValue::Text(gname)) = rules.value(g, BItem::Net(i), props::MATCH_GROUP)
            {
                groups.entry(gname).or_default().push(NetId(i));
            }
        }
        for (gname, members) in &groups {
            if !members.iter().any(|m| nets.contains(m)) || members.len() < 2 {
                continue;
            }
            let ls: Vec<f64> = members
                .iter()
                .map(|m| lengths.get(m).copied().unwrap_or(0.0))
                .collect();
            let spread = ls.iter().cloned().fold(f64::MIN, f64::max)
                - ls.iter().cloned().fold(f64::MAX, f64::min);
            let skew = members
                .iter()
                .filter_map(|m| rules.len(g, BItem::Net(m.0), props::SKEW_MAX))
                .min();
            if let Some(s) = skew {
                if spread > s as f64 {
                    self.put(DrcFinding {
                        code: "skew".into(),
                        severity: Severity::Error,
                        items: key_items(members.iter().map(|m| BItem::Net(m.0)).collect()),
                        layer: None,
                        at: Point::ORIGIN,
                        actual: Some(spread),
                        required: Some(s as f64),
                        rule: gname.clone(),
                        source: SourceKey::default(),
                        message: format!(
                            "match group `{gname}` length spread {:.3} mm",
                            spread * 1e-6
                        ),
                    });
                }
            }
        }
        // Differential pairs.
        for &(p, nn) in &self.diff_pairs.clone() {
            if !nets.contains(&p) && !nets.contains(&nn) {
                continue;
            }
            self.check_diff_pair(board, rules, g, p, nn, &lengths);
        }
        // Thermal spokes.
        for rep in &self.spokes.clone() {
            for (&(f, k), &(reached, total)) in &rep.spokes {
                let it = BItem::Pad(f, k);
                let Some(fp) = board.footprints.get(&f) else {
                    continue;
                };
                let Some(net) = fp.pad_nets.get(&fp.package.pads[k as usize].name) else {
                    continue;
                };
                if !nets.contains(net) || total == 0 {
                    continue;
                }
                let min = rules
                    .value(g, it, props::SPOKE_MIN)
                    .and_then(|v| v.int())
                    .unwrap_or(2) as u32;
                if reached < min {
                    let (rule, source) = rules.single_origin(g, it, props::SPOKE_MIN);
                    self.put(DrcFinding {
                        code: "min-spokes".into(),
                        severity: Severity::Warning,
                        items: key_items(vec![it, BItem::Zone(rep.zone), BItem::Net(net.0)]),
                        layer: Some(rep.layer),
                        at: fp.at,
                        actual: Some(reached as f64),
                        required: Some(min as f64),
                        rule,
                        source,
                        message: format!(
                            "{}.{} has {reached} of {min} thermal spokes connected",
                            fp.reference, fp.package.pads[k as usize].name
                        ),
                    });
                }
            }
        }
    }

    pub(super) fn check_diff_pair(
        &mut self,
        board: &Board,
        rules: &BoardRules,
        g: &BoardGraph,
        p: NetId,
        n: NetId,
        lengths: &BTreeMap<NetId, f64>,
    ) {
        let pi = BItem::Net(p.0);
        let gap_rule = rules.len(g, pi, props::DIFF_GAP);
        let unc_max = rules.len(g, pi, props::DIFF_UNCOUPLED_MAX);
        let tp: Vec<_> = board.tracks.values().filter(|t| t.net == Some(p)).collect();
        let tn: Vec<_> = board.tracks.values().filter(|t| t.net == Some(n)).collect();
        let mut coupled = 0.0;
        let total: f64 = tp.iter().map(|t| t.length()).sum();
        let mut worst_gap: Option<(f64, Point, crate::layout::ItemId, crate::layout::ItemId, u8)> =
            None;
        for a in &tp {
            let (a0, a1) = a.endpoints();
            let (ax, ay) = ((a1.x - a0.x) as f64, (a1.y - a0.y) as f64);
            let la = ax.hypot(ay);
            if la == 0.0 {
                continue;
            }
            let (ux, uy) = (ax / la, ay / la);
            let mut best = 0.0f64;
            for b in &tn {
                if b.layer != a.layer {
                    continue;
                }
                let (b0, b1) = b.endpoints();
                let (bx, by) = ((b1.x - b0.x) as f64, (b1.y - b0.y) as f64);
                let lb = bx.hypot(by);
                if lb == 0.0 || (ux * by - uy * bx).abs() / lb > 0.02 {
                    continue;
                }
                // Perpendicular distance between centrelines and the overlap of projections.
                let d = ((b0.x - a0.x) as f64 * -uy + (b0.y - a0.y) as f64 * ux).abs();
                let gap = d - (a.width + b.width) as f64 / 2.0;
                let reference = gap_rule.map(|x| x as f64).unwrap_or(a.width as f64);
                if gap > reference * 3.0 || gap < 0.0 {
                    continue;
                }
                let s0 = (b0.x - a0.x) as f64 * ux + (b0.y - a0.y) as f64 * uy;
                let s1 = (b1.x - a0.x) as f64 * ux + (b1.y - a0.y) as f64 * uy;
                let (lo, hi) = (s0.min(s1).max(0.0), s0.max(s1).min(la));
                if hi <= lo {
                    continue;
                }
                best = best.max(hi - lo);
                if let Some(r) = gap_rule {
                    let dev = (gap - r as f64).abs();
                    if dev > 0.1 * r as f64 + 1.0
                        && worst_gap.is_none_or(|w| dev > (w.0 - r as f64).abs())
                    {
                        worst_gap = Some((gap, a0, a.id, b.id, a.layer));
                    }
                }
            }
            coupled += best.min(la);
        }
        if let (Some((gap, at, ta, tb, layer)), Some(r)) = (worst_gap, gap_rule) {
            let (rule, source) = rules.single_origin(g, pi, props::DIFF_GAP);
            self.put(DrcFinding {
                code: "diff-pair-gap".into(),
                severity: Severity::Error,
                items: key_items(vec![
                    pi,
                    BItem::Net(n.0),
                    BItem::Track(ta),
                    BItem::Track(tb),
                ]),
                layer: Some(layer),
                at,
                actual: Some(gap),
                required: Some(r as f64),
                rule,
                source,
                message: format!(
                    "differential gap {:.4} mm, rule {:.4} mm",
                    gap * 1e-6,
                    r as f64 * 1e-6
                ),
            });
        }
        let uncoupled = (total - coupled).max(0.0);
        if let Some(m) = unc_max {
            if uncoupled > m as f64 {
                let (rule, source) = rules.single_origin(g, pi, props::DIFF_UNCOUPLED_MAX);
                self.put(DrcFinding {
                    code: "diff-pair-uncoupled".into(),
                    severity: Severity::Error,
                    items: key_items(vec![pi, BItem::Net(n.0)]),
                    layer: None,
                    at: Point::ORIGIN,
                    actual: Some(uncoupled),
                    required: Some(m as f64),
                    rule,
                    source,
                    message: format!("uncoupled length {:.3} mm", uncoupled * 1e-6),
                });
            }
        }
        if let Some(s) = rules.len(g, pi, props::SKEW_MAX) {
            let lp = lengths.get(&p).copied().unwrap_or(0.0);
            let ln = lengths.get(&n).copied().unwrap_or(0.0);
            if (lp - ln).abs() > s as f64 {
                let (rule, source) = rules.single_origin(g, pi, props::SKEW_MAX);
                self.put(DrcFinding {
                    code: "skew".into(),
                    severity: Severity::Error,
                    items: key_items(vec![pi, BItem::Net(n.0)]),
                    layer: None,
                    at: Point::ORIGIN,
                    actual: Some((lp - ln).abs()),
                    required: Some(s as f64),
                    rule,
                    source,
                    message: format!("pair skew {:.3} mm", (lp - ln).abs() * 1e-6),
                });
            }
        }
    }
}
