//! Copper checks: clearance between items and to pours, annular rings and holes, keepouts,
//! board regions and bend radii, and the hole checks.

use super::*;
use crate::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use crate::elab::Severity;
use crate::geom::shape::{point_segment_distance, Prim};
use crate::geom::{Coord, Point, Rect};
use crate::layout::{Board, ViaKind};
use std::collections::BTreeSet;

impl Drc {
    /// Item-to-item clearance from owner `o`. With `one_sided`, only pairs with a larger
    /// owner are checked (a full run visits both sides, so each pair is measured once).
    pub(super) fn clearance_pairs(&mut self, rules: &BoardRules, o: BItem, one_sided: bool) {
        let radius = rules.max_clearance();
        let pid = rules.pair.prop_id(props::CLEARANCE);
        let keys = self.owners.get(&o).cloned().unwrap_or_default();
        let mut found: Vec<DrcFinding> = Vec::new();
        for s in keys {
            let p = self.slot(s);
            let pb = p.prim.bbox();
            self.index.visit(&pb.inflate(radius), |ns, nb| {
                let np = self.slots[ns as usize].as_ref().unwrap();
                if np.owner == o || (one_sided && np.owner < o) {
                    return;
                }
                let shared = p.layers & np.layers;
                if shared == 0 || (p.net.is_some() && p.net == np.net) {
                    return;
                }
                let mut req = rules.pair_len_id(o, np.owner, pid);
                // Pads of one footprint are spaced by its land pattern: the board's clearance
                // applies to them only through a rule written for them; otherwise only pads
                // that touch are a finding (a land pattern that shorts its pins).
                let same_footprint = matches!((o, np.owner), (BItem::Pad(a, _), BItem::Pad(b, _)) if a == b);
                if same_footprint
                    && BoardRules::is_fallback(&rules.pair_origin(o, np.owner, props::CLEARANCE).0)
                {
                    req = 1;
                }
                if !pb.inflate(req).intersects(nb) {
                    return;
                }
                let gap = p.prim.gap(&np.prim);
                if gap < req as f64 {
                    let (rule, source) = rules.pair_origin(o, np.owner, props::CLEARANCE);
                    if same_footprint && req == 1 {
                        let mut f = clearance_finding(
                            vec![o, np.owner],
                            lowest(shared),
                            p.prim.anchor(),
                            gap,
                            0,
                            rule,
                            source,
                        );
                        f.message = "pads of one footprint on different nets touch".into();
                        found.push(f);
                        return;
                    }
                    let at = if (p.owner, p.sub) < (np.owner, np.sub) {
                        p.prim.anchor()
                    } else {
                        np.prim.anchor()
                    };
                    found.push(clearance_finding(
                        vec![o, np.owner],
                        lowest(shared),
                        at,
                        gap,
                        req,
                        rule,
                        source,
                    ));
                }
            });
            if matches!(o, BItem::Zone(_)) {
                continue;
            }
            for z in &self.zones {
                self.zone_probe(rules, p, z, &mut found);
            }
        }
        for f in found {
            self.put(f);
        }
    }

    /// One item primitive against one zone fill: clearance across the fill's edges, or
    /// containment when no edge crosses it.
    pub(super) fn zone_probe(&self, rules: &BoardRules, p: &CPrim, z: &ZoneFill, found: &mut Vec<DrcFinding>) {
        if p.layers & (1 << z.layer) == 0 || (p.net.is_some() && p.net == z.net) {
            return;
        }
        let pb = p.prim.bbox();
        let req = rules.pair_len_id(p.owner, z.zone, rules.pair.prop_id(props::CLEARANCE));
        if !z.edges.bbox().inflate(req).intersects(&pb) {
            return;
        }
        let pr = z.edges.probe(&p.prim, req);
        let origin = || rules.pair_origin(p.owner, z.zone, props::CLEARANCE);
        if pr.gap < req as f64 {
            let (rule, source) = origin();
            found.push(clearance_finding(
                vec![p.owner, z.zone],
                z.layer,
                p.prim.anchor(),
                pr.gap,
                req,
                rule,
                source,
            ));
        } else if pr.inside == Some(true) {
            let (rule, source) = origin();
            let mut f = clearance_finding(
                vec![p.owner, z.zone],
                z.layer,
                p.prim.anchor(),
                0.0,
                req,
                rule,
                source,
            );
            f.message = "copper inside another net's zone fill".into();
            found.push(f);
        }
    }

    /// Every item near the fills of `zones` against those fills, plus their zone-to-zone
    /// clearance (after an edit to a zone).
    pub(super) fn check_zone_items(&mut self, rules: &BoardRules, zones: &BTreeSet<BItem>) {
        let mut found = Vec::new();
        for z in self.zones.iter().filter(|z| zones.contains(&z.zone)) {
            let mut seen = BTreeSet::new();
            self.index
                .visit(&z.edges.bbox().inflate(rules.max_clearance()), |s, _| {
                    seen.insert(s);
                });
            for s in seen {
                self.zone_probe(rules, self.slot(s), z, &mut found);
            }
        }
        for f in found {
            self.put(f);
        }
        self.check_zone_pairs(rules, Some(zones));
    }

    /// Clearance between fills of different nets on one layer. With `only`, pairs involving
    /// those zones; otherwise every pair once.
    pub(super) fn check_zone_pairs(&mut self, rules: &BoardRules, only: Option<&BTreeSet<BItem>>) {
        let pid = rules.pair.prop_id(props::CLEARANCE);
        let mut found = Vec::new();
        for (i, a) in self.zones.iter().enumerate() {
            for b in &self.zones[i + 1..] {
                if a.layer != b.layer || a.zone == b.zone || (a.net.is_some() && a.net == b.net) {
                    continue;
                }
                if only.is_some_and(|o| !o.contains(&a.zone) && !o.contains(&b.zone)) {
                    continue;
                }
                let req = rules.pair_len_id(a.zone, b.zone, pid);
                if !a.edges.bbox().inflate(req).intersects(&b.edges.bbox()) {
                    continue;
                }
                let mut worst: Option<(f64, Point)> = None;
                a.edges.visit_near(&b.edges.bbox().inflate(req), |p, q| {
                    let e = Prim::segment(p, q, 0);
                    let g = b.edges.probe(&e, req).gap;
                    if g < req as f64 && worst.is_none_or(|w| g < w.0) {
                        worst = Some((g, p));
                    }
                });
                if let Some((g, at)) = worst {
                    let (rule, source) = rules.pair_origin(a.zone, b.zone, props::CLEARANCE);
                    found.push(clearance_finding(
                        vec![a.zone, b.zone],
                        a.layer,
                        at,
                        g,
                        req,
                        rule,
                        source,
                    ));
                }
            }
        }
        for f in found {
            self.put(f);
        }
    }

    pub(super) fn check_owner(
        &mut self,
        board: &Board,
        rules: &BoardRules,
        g: &BoardGraph,
        o: BItem,
        full: bool,
    ) {
        let n = board.copper_count();
        if let BItem::Zone(_) = o {
            // Zones are checked from the item side in a full run; after an edit to a zone,
            // every item near its fill is re-probed against it.
            if !full {
                self.check_zone_items(rules, &BTreeSet::from([o]));
            }
            return;
        }
        self.clearance_pairs(rules, o, full);
        // Edge clearance and outside-board.
        if let Some(keys) = self.owners.get(&o).cloned() {
            let req = rules.pair_len(o, BItem::Edge, props::EDGE_CLEARANCE);
            let mut worst: Option<(f64, Point)> = None;
            for k in &keys {
                let p = &self.slot(*k).prim;
                let pr = self.outline.probe(p, req);
                let gap = if pr.inside == Some(false) {
                    -1.0
                } else {
                    pr.gap
                };
                if gap < req as f64 && worst.is_none_or(|w| gap < w.0) {
                    worst = Some((gap, p.anchor()));
                }
            }
            if let Some((gap, at)) = worst {
                let (rule, source) = rules.pair_origin(o, BItem::Edge, props::EDGE_CLEARANCE);
                self.put(DrcFinding {
                    code: "edge-clearance".into(),
                    severity: Severity::Error,
                    items: vec![o],
                    layer: None,
                    at,
                    actual: Some(gap.max(0.0)),
                    required: Some(req as f64),
                    rule,
                    source,
                    message: if gap < 0.0 {
                        "copper outside the board outline".into()
                    } else {
                        format!("{:.4} mm from the board edge", gap * 1e-6)
                    },
                });
            }
        }
        match o {
            BItem::Track(t) => {
                let tr = &board.tracks[&t];
                let min = rules.len(g, o, props::TRACK_WIDTH_MIN).unwrap_or(0);
                let max = rules.len(g, o, props::TRACK_WIDTH_MAX);
                if tr.width < min || max.is_some_and(|m| tr.width > m) {
                    let (rule, source) = rules.single_origin(
                        g,
                        o,
                        if tr.width < min {
                            props::TRACK_WIDTH_MIN
                        } else {
                            props::TRACK_WIDTH_MAX
                        },
                    );
                    self.put(DrcFinding {
                        code: "track-width".into(),
                        severity: Severity::Error,
                        items: vec![o],
                        layer: Some(tr.layer),
                        at: tr.endpoints().0,
                        actual: Some(tr.width as f64),
                        required: Some(if tr.width < min { min } else { max.unwrap() } as f64),
                        rule,
                        source,
                        message: format!("track width {:.4} mm", tr.width as f64 * 1e-6),
                    });
                }
                self.keepout_check(board, o, &[tr.layer], "tracks");
                self.region_check(board, o, &[tr.layer]);
            }
            BItem::Via(v) => {
                let via = &board.vias[&v];
                self.ring_and_hole(rules, g, o, via.diameter, via.drill, via.at, via.micro);
                let kind = via.kind(n);
                if let Some(RuleValue::ViaKinds(ks)) = rules.value(g, o, props::VIA_TYPES) {
                    if !ks.contains(&kind) {
                        let (rule, source) = rules.single_origin(g, o, props::VIA_TYPES);
                        self.put(DrcFinding {
                            code: "via-type".into(),
                            severity: Severity::Error,
                            items: vec![o],
                            layer: Some(via.from),
                            at: via.at,
                            actual: None,
                            required: None,
                            rule,
                            source,
                            message: format!(
                                "{kind:?} via ({}–{}) is not allowed here",
                                via.from, via.to
                            ),
                        });
                    }
                }
                if kind == ViaKind::Micro && via.to - via.from != 1 {
                    self.put(DrcFinding {
                        code: "via-type".into(),
                        severity: Severity::Error,
                        items: vec![o],
                        layer: Some(via.from),
                        at: via.at,
                        actual: None,
                        required: None,
                        rule: "fab".into(),
                        source: SourceKey::default(),
                        message: "a microvia must span exactly one layer pair".into(),
                    });
                }
                let layers: Vec<u8> = via.layers().collect();
                self.keepout_check(board, o, &layers, "vias");
                self.region_check(board, o, &layers);
                self.bend_check(board, o, via.at, via.diameter / 2);
            }
            BItem::Pad(f, k) => {
                let fp = &board.footprints[&f];
                let Some(p) = fp.pad(k as usize, n) else {
                    return;
                };
                if let (Some(d), true) = (p.drill, p.plated) {
                    let b = p.prim.bbox();
                    let size = b.width().min(b.height());
                    self.ring_and_hole(rules, g, o, size, d, p.center, false);
                } else if let Some(d) = p.drill {
                    let min = rules.len(g, o, props::HOLE_MIN).unwrap_or(0);
                    if d < min {
                        self.hole_finding(rules, g, o, d, min, p.center);
                    }
                }
                let layers: Vec<u8> = p.layers.iter().copied().collect();
                self.keepout_check(board, o, &layers, "pads");
                self.region_check(board, o, &layers);
                if p.drill.is_some() {
                    let b = p.prim.bbox();
                    self.bend_check(board, o, p.center, b.width().max(b.height()) / 2);
                }
            }
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ring_and_hole(
        &mut self,
        rules: &BoardRules,
        g: &BoardGraph,
        o: BItem,
        dia: Coord,
        drill: Coord,
        at: Point,
        micro: bool,
    ) {
        let ring = (dia - drill) / 2;
        let amin = rules.len(g, o, props::ANNULAR_MIN).unwrap_or(0);
        if ring < amin {
            let (rule, source) = rules.single_origin(g, o, props::ANNULAR_MIN);
            self.put(DrcFinding {
                code: "annular-ring".into(),
                severity: Severity::Error,
                items: vec![o],
                layer: None,
                at,
                actual: Some(ring as f64),
                required: Some(amin as f64),
                rule,
                source,
                message: format!("annular ring {:.4} mm", ring as f64 * 1e-6),
            });
        }
        let hmin = if micro {
            rules
                .len(g, o, props::HOLE_MIN)
                .unwrap_or(0)
                .min(crate::geom::mm(0.1))
        } else {
            rules.len(g, o, props::HOLE_MIN).unwrap_or(0)
        };
        if drill < hmin {
            self.hole_finding(rules, g, o, drill, hmin, at);
        }
    }

    pub(super) fn hole_finding(
        &mut self,
        rules: &BoardRules,
        g: &BoardGraph,
        o: BItem,
        d: Coord,
        min: Coord,
        at: Point,
    ) {
        let (rule, source) = rules.single_origin(g, o, props::HOLE_MIN);
        self.put(DrcFinding {
            code: "hole-size".into(),
            severity: Severity::Error,
            items: vec![o],
            layer: None,
            at,
            actual: Some(d as f64),
            required: Some(min as f64),
            rule,
            source,
            message: format!("drill {:.4} mm", d as f64 * 1e-6),
        });
    }

    pub(super) fn keepout_check(&mut self, board: &Board, o: BItem, layers: &[u8], what: &str) {
        let keys = self.owners.get(&o).cloned().unwrap_or_default();
        for k in board.keepouts.values() {
            let forbidden = match what {
                "tracks" => k.rules.tracks,
                "vias" => k.rules.vias,
                "pads" => k.rules.pads,
                _ => false,
            };
            if !forbidden || !(k.layers.is_empty() || layers.iter().any(|l| k.layers.contains(l))) {
                continue;
            }
            let ko = Prim::polygon(k.outline.rings().cloned().collect());
            if keys.iter().any(|ck| self.slot(*ck).prim.gap(&ko) < 0.0) {
                self.put(DrcFinding {
                    code: "keepout".into(),
                    severity: Severity::Error,
                    items: vec![o],
                    layer: layers.first().copied(),
                    at: self.slot(keys[0]).prim.anchor(),
                    actual: None,
                    required: None,
                    rule: k.name.clone(),
                    source: SourceKey::default(),
                    message: format!("{what} are not allowed in keepout `{}`", k.name),
                });
            }
        }
    }

    pub(super) fn region_check(&mut self, board: &Board, o: BItem, layers: &[u8]) {
        if board.regions.is_empty() {
            return;
        }
        let Some(k) = self.owners.get(&o).and_then(|v| v.first()) else {
            return;
        };
        let at = self.slot(*k).prim.anchor();
        let present = board.copper_at(at);
        let missing: Vec<u8> = layers
            .iter()
            .copied()
            .filter(|l| !present.contains(l))
            .collect();
        if !missing.is_empty() {
            self.put(DrcFinding {
                code: "layer-not-in-region".into(),
                severity: Severity::Error,
                items: vec![o],
                layer: Some(missing[0]),
                at,
                actual: None,
                required: None,
                rule: board.stackup_at(at).name.clone(),
                source: SourceKey::default(),
                message: format!(
                    "copper on {} where the `{}` stackup has no such layer",
                    crate::layout::copper_name(missing[0], board.copper_count()),
                    board.stackup_at(at).name
                ),
            });
        }
    }

    pub(super) fn bend_check(&mut self, board: &Board, o: BItem, at: Point, r: Coord) {
        for reg in &board.regions {
            for b in &reg.bends {
                let d = point_segment_distance(at, b.a, b.b);
                if d < (b.keepout + r) as f64 {
                    self.put(DrcFinding {
                        code: "via-in-bend".into(),
                        severity: Severity::Error,
                        items: vec![o],
                        layer: None,
                        at,
                        actual: Some(d),
                        required: Some((b.keepout + r) as f64),
                        rule: reg.name.clone(),
                        source: SourceKey::default(),
                        message: format!(
                            "plated hole inside the bend area of region `{}`",
                            reg.name
                        ),
                    });
                }
            }
        }
    }

    pub(super) fn check_holes(&mut self, board: &Board, rules: &BoardRules, only: Option<&BTreeSet<BItem>>) {
        let mut holes: Vec<(BItem, Point, Coord)> = Vec::new();
        for v in board.vias.values() {
            holes.push((BItem::Via(v.id), v.at, v.drill / 2));
        }
        for p in board.all_pads() {
            if let Some(d) = p.drill {
                holes.push((BItem::Pad(p.footprint, p.index as u32), p.center, d / 2));
            }
        }
        let mut grid: Grid<usize> = Grid::new(crate::geom::mm(2.0));
        for (i, h) in holes.iter().enumerate() {
            grid.insert(i, Rect::new(h.1, h.1).inflate(h.2));
        }
        let radius = rules
            .pair
            .max_value_by(props::HOLE_TO_HOLE, |v| v.as_len().unwrap_or(0))
            .unwrap_or(0);
        for (i, h) in holes.iter().enumerate() {
            if only.is_some_and(|s| !s.contains(&h.0)) {
                continue;
            }
            for j in grid.query(&Rect::new(h.1, h.1).inflate(h.2 + radius)) {
                if j == i {
                    continue;
                }
                let o = &holes[j];
                // Pads of one footprint (connectors) still need hole spacing.
                let gap = h.1.dist(o.1) - (h.2 + o.2) as f64;
                let req = rules.pair_len(h.0, o.0, props::HOLE_TO_HOLE);
                if gap < req as f64 {
                    let (rule, source) = rules.pair_origin(h.0, o.0, props::HOLE_TO_HOLE);
                    self.put(DrcFinding {
                        code: "hole-to-hole".into(),
                        severity: Severity::Error,
                        items: key_items(vec![h.0, o.0]),
                        layer: None,
                        at: h.1,
                        actual: Some(gap.max(0.0)),
                        required: Some(req as f64),
                        rule,
                        source,
                        message: format!("holes {:.4} mm apart", gap.max(0.0) * 1e-6),
                    });
                }
            }
        }
    }
}

pub(super) fn clearance_finding(
    items: Vec<BItem>,
    layer: u8,
    at: Point,
    gap: f64,
    req: Coord,
    rule: String,
    source: SourceKey,
) -> DrcFinding {
    DrcFinding {
        code: "clearance".into(),
        severity: Severity::Error,
        items: key_items(items),
        layer: Some(layer),
        at,
        actual: Some(gap.max(0.0)),
        required: Some(req as f64),
        rule,
        source,
        message: format!(
            "clearance {:.4} mm < {:.4} mm",
            gap.max(0.0) * 1e-6,
            req as f64 * 1e-6
        ),
    }
}
