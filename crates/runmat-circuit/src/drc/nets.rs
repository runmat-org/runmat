//! Net checks: connectivity (unrouted islands), thermal spokes, lengths and differential pairs.

use crate::layer_set::CopperIndex;
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
                    at: net_anchor(board, n),
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
            if let Some(z) = rules.value(g, net_item, props::IMPEDANCE).as_ref().and_then(crate::high_speed::controlled::ohms) {
                self.impedance_targets
                    .insert(board.net_name(n).to_string(), z);
                self.check_impedance(board, rules, g, n, z);
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
                        at: net_anchor(board, members[0]),
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
        // Groups with a target or tolerance: each member against the target, in length or
        // delay, through series parts where its rules say so.
        let timing = crate::high_speed::delay::net_timing(board);
        for group in crate::high_speed::groups::groups(board, rules, g, &timing) {
            let checked = group.members.iter().any(|m| nets.contains(&m.net))
                && group.members.iter().any(|m| {
                    let it = BItem::Net(m.net.0);
                    rules.value(g, it, props::MATCH_TARGET).is_some() || rules.value(g, it, props::MATCH_TOLERANCE).is_some()
                });
            if !checked {
                continue;
            }
            let unit = group.basis.unit();
            for m in &group.members {
                use crate::high_speed::groups::Status;
                if matches!(m.status, Status::Matched | Status::Unrouted) {
                    continue;
                }
                self.put(DrcFinding {
                    code: "length-match".into(),
                    severity: Severity::Error,
                    items: vec![BItem::Net(m.net.0)],
                    layer: None,
                    at: net_anchor(board, m.net),
                    actual: Some(m.value),
                    required: Some(group.target),
                    rule: group.rule.clone(),
                    source: group.source.clone(),
                    message: format!(
                        "net `{}` in group `{}` is {:.3} {unit}, {} the {}target {:.3} {unit} by {:.3} {unit} (tolerance {:.3} {unit})",
                        m.name,
                        group.name,
                        m.value,
                        if m.deviation < 0.0 { "short of" } else { "over" },
                        if group.written { "" } else { "longest member's " },
                        group.target,
                        m.deviation.abs(),
                        group.tolerance
                    ),
                });
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
        let mut worst_gap: Option<(f64, f64, Point, crate::layout::ItemId, crate::layout::ItemId, CopperIndex)> =
            None;
        // Coupled runs by layer, width and gap (1 µm), for the differential impedance check.
        let mut runs: BTreeMap<(CopperIndex, crate::geom::Coord, i64), Vec<(crate::layout::ItemId, crate::layout::ItemId, Point)>> =
            BTreeMap::new();
        for a in &tp {
            // A class's per-layer impedance gap applies on the track's own layer.
            let gap_rule = rules.len(g, BItem::Track(a.id), props::DIFF_GAP).or(gap_rule);
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
                // Coupled within half again the pair's gap; farther apart (a breakout to the
                // pads) the run counts as uncoupled length, not a gap or impedance error.
                if gap > reference * 1.5 || gap < 0.0 {
                    continue;
                }
                let s0 = (b0.x - a0.x) as f64 * ux + (b0.y - a0.y) as f64 * uy;
                let s1 = (b1.x - a0.x) as f64 * ux + (b1.y - a0.y) as f64 * uy;
                let (lo, hi) = (s0.min(s1).max(0.0), s0.max(s1).min(la));
                if hi <= lo {
                    continue;
                }
                best = best.max(hi - lo);
                runs.entry((a.layer, a.width, (gap / 1000.0).round() as i64)).or_default().push((a.id, b.id, a0));
                if let Some(r) = gap_rule {
                    let dev = (gap - r as f64).abs();
                    if dev > 0.1 * r as f64 + 1.0
                        && worst_gap.is_none_or(|w| dev > (w.0 - w.1).abs())
                    {
                        worst_gap = Some((gap, r as f64, a0, a.id, b.id, a.layer));
                    }
                }
            }
            coupled += best.min(la);
        }
        if let Some((gap, r, at, ta, tb, layer)) = worst_gap {
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
                required: Some(r),
                rule,
                source,
                message: format!(
                    "differential gap {:.4} mm, rule {:.4} mm",
                    gap * 1e-6,
                    r * 1e-6
                ),
            });
        }
        if let Some(target) = rules.value(g, pi, props::DIFF_IMPEDANCE).as_ref().and_then(crate::high_speed::controlled::ohms) {
            self.impedance_targets.insert(format!("{}/{}", board.net_name(p), board.net_name(n)), target);
            self.check_diff_impedance(board, rules, g, (p, n), target, &runs);
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
                    at: net_anchor(board, p),
                    actual: Some(uncoupled),
                    required: Some(m as f64),
                    rule,
                    source,
                    message: format!("uncoupled length {:.3} mm", uncoupled * 1e-6),
                });
            }
        }
        if let Some(max_ps) = rules.value(g, pi, props::SKEW_MAX).and_then(|v| v.as_time()) {
            // In-pair skew as a delay: each net's delay from its layers' propagation velocity.
            let t = crate::high_speed::delay::net_timing(board);
            let d = |x: NetId| t.get(&x).map_or(0.0, |t| t.delay);
            let skew = (d(p) - d(n)).abs();
            if skew > max_ps {
                let (rule, source) = rules.single_origin(g, pi, props::SKEW_MAX);
                self.put(DrcFinding {
                    code: "skew".into(),
                    severity: Severity::Error,
                    items: key_items(vec![pi, BItem::Net(n.0)]),
                    layer: None,
                    at: net_anchor(board, p),
                    actual: Some(skew),
                    required: Some(max_ps),
                    rule,
                    source,
                    message: format!("pair skew {skew:.2} ps, limit {max_ps:.2} ps"),
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
                    at: net_anchor(board, p),
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

/// How far a track's impedance may sit from its net's target: the ±10 % a board house holds
/// controlled impedance to (IPC-6012).
pub const IMPEDANCE_TOLERANCE: f64 = 0.10;

impl Drc {
    /// The net's tracks against its impedance target, by the closed forms of
    /// [`crate::impedance`] on each track's layer: one finding per layer and width off by more
    /// than [`IMPEDANCE_TOLERANCE`].
    fn check_impedance(&mut self, board: &Board, rules: &BoardRules, g: &BoardGraph, n: NetId, target: f64) {
        let mut runs: BTreeMap<(CopperIndex, crate::geom::Coord), Vec<&crate::layout::Track>> = BTreeMap::new();
        for t in board.tracks.values().filter(|t| t.net == Some(n)) {
            runs.entry((t.layer, t.width)).or_default().push(t);
        }
        let (rule, source) = rules.single_origin(g, BItem::Net(n.0), props::IMPEDANCE);
        for ((layer, width), tracks) in runs {
            let Some(geometry) = crate::impedance::TraceGeometry::of(&board.stackup, layer) else { continue };
            let z = geometry.single_ended(width as f64 * 1e-9);
            if (z - target).abs() <= IMPEDANCE_TOLERANCE * target {
                continue;
            }
            let fit = geometry
                .width_for(target)
                .map(|w| format!("; {:.3} mm gives {target:.0} Ω", w as f64 * 1e-6))
                .unwrap_or_default();
            let mut items: Vec<BItem> = tracks.iter().map(|t| BItem::Track(t.id)).collect();
            items.push(BItem::Net(n.0));
            items.sort();
            let (a, _) = tracks[0].endpoints();
            self.put(DrcFinding {
                code: "impedance".into(),
                severity: Severity::Error,
                items,
                layer: Some(layer),
                at: a,
                actual: Some(z),
                required: Some(target),
                rule: rule.clone(),
                source: source.clone(),
                message: format!(
                    "net `{}` impedance {z:.1} Ω on {} at {:.3} mm, target {target:.0} Ω ±{:.0} %{fit}",
                    board.net_name(n),
                    crate::layout::copper_name(layer, board.copper_count()),
                    width as f64 * 1e-6,
                    IMPEDANCE_TOLERANCE * 100.0
                ),
            });
        }
    }

    /// A pair's coupled runs against its differential target: one finding per layer, width
    /// and gap whose closed-form differential impedance is off by more than
    /// [`IMPEDANCE_TOLERANCE`].
    #[allow(clippy::type_complexity)]
    fn check_diff_impedance(
        &mut self,
        board: &Board,
        rules: &BoardRules,
        g: &BoardGraph,
        (p, n): (NetId, NetId),
        target: f64,
        runs: &BTreeMap<(CopperIndex, crate::geom::Coord, i64), Vec<(crate::layout::ItemId, crate::layout::ItemId, Point)>>,
    ) {
        let (rule, source) = rules.single_origin(g, BItem::Net(p.0), props::DIFF_IMPEDANCE);
        for (&(layer, width, gap_um), segs) in runs {
            let Some(geometry) = crate::impedance::TraceGeometry::of(&board.stackup, layer) else { continue };
            let (w, s) = (width as f64 * 1e-9, gap_um as f64 * 1e-6);
            let z = geometry.differential(w, s);
            if (z - target).abs() <= IMPEDANCE_TOLERANCE * target {
                continue;
            }
            let fit = geometry
                .width_for_diff(target, s)
                .map(|w| format!("; {:.3} mm at this gap gives {target:.0} Ω", w as f64 * 1e-6))
                .unwrap_or_default();
            let mut items: Vec<BItem> = segs.iter().flat_map(|(a, b, _)| [BItem::Track(*a), BItem::Track(*b)]).collect();
            items.extend([BItem::Net(p.0), BItem::Net(n.0)]);
            items.sort();
            items.dedup();
            self.put(DrcFinding {
                code: "diff-impedance".into(),
                severity: Severity::Error,
                items,
                layer: Some(layer),
                at: segs[0].2,
                actual: Some(z),
                required: Some(target),
                rule: rule.clone(),
                source: source.clone(),
                message: format!(
                    "pair `{}`/`{}` differential impedance {z:.1} Ω on {} at {:.3} mm width, {:.3} mm gap, target {target:.0} Ω ±{:.0} %{fit}",
                    board.net_name(p),
                    board.net_name(n),
                    crate::layout::copper_name(layer, board.copper_count()),
                    w * 1e3,
                    s * 1e3,
                    IMPEDANCE_TOLERANCE * 100.0
                ),
            });
        }
    }
}

/// Where a finding about a whole net is drawn: the middle of the net's longest track, or the
/// first part with a pad on the net while it is unrouted. A finding at the board origin drew
/// its marker over whatever part sits there.
pub(crate) fn net_anchor(board: &Board, net: NetId) -> Point {
    let longest = board
        .tracks
        .values()
        .filter(|t| t.net == Some(net))
        .max_by(|a, b| a.length().total_cmp(&b.length()).then(b.id.cmp(&a.id)));
    if let Some(t) = longest {
        return match t.geom {
            crate::layout::TrackGeom::Arc { mid, .. } => mid,
            crate::layout::TrackGeom::Segment { a, b } => Point { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
        };
    }
    board
        .footprints
        .values()
        .find(|fp| fp.pad_nets.values().any(|&x| x == net))
        .map_or(Point::ORIGIN, |fp| fp.at)
}
