//! Footprint checks: courtyards, silkscreen to pads and the placement rules.

use super::*;
use crate::board_rules::{props, BItem, BoardGraph, BoardRules};
use crate::elab::Severity;
use crate::geom::shape::Prim;
use crate::layout::{Board, Layer, Side};
use std::collections::{BTreeMap, BTreeSet};

impl Drc {
    pub(super) fn check_footprints(
        &mut self,
        board: &Board,
        rules: &BoardRules,
        g: &BoardGraph,
        only: Option<&BTreeSet<BItem>>,
    ) {
        let n = board.copper_count();
        // Courtyards.
        let mut grid: Grid<crate::layout::ItemId> = Grid::new(crate::geom::mm(5.0));
        let mut cy: BTreeMap<crate::layout::ItemId, Prim> = BTreeMap::new();
        for f in board.footprints.values() {
            let rings = f.courtyard();
            if rings.is_empty() {
                continue;
            }
            let p = Prim::polygon(rings);
            grid.insert(f.id, p.bbox());
            cy.insert(f.id, p);
        }
        for (id, p) in &cy {
            let fa = &board.footprints[id];
            if only.is_some_and(|s| !s.contains(&BItem::Footprint(*id))) {
                continue;
            }
            for other in grid.query(&p.bbox().inflate(crate::geom::mm(1.0))) {
                if other == *id {
                    continue;
                }
                let fb = &board.footprints[&other];
                if fa.side != fb.side {
                    continue;
                }
                let (a, b) = (BItem::Footprint(*id), BItem::Footprint(other));
                let req = rules.pair_len(a, b, props::COURTYARD_CLEARANCE);
                let gap = p.gap(&cy[&other]);
                // With no required clearance, touching is fine; only a real overlap counts.
                let violates = if req > 0 {
                    super::short_of(gap, req)
                } else {
                    gap <= 0.0 && courtyard_overlap(p, &cy[&other])
                };
                if violates {
                    let (rule, source) = rules.pair_origin(a, b, props::COURTYARD_CLEARANCE);
                    self.put(DrcFinding {
                        code: "courtyard-overlap".into(),
                        severity: Severity::Warning,
                        items: key_items(vec![a, b]),
                        layer: None,
                        at: fa.at,
                        actual: Some(gap),
                        required: Some(req as f64),
                        rule,
                        source,
                        message: format!(
                            "courtyards of {} and {} overlap",
                            fa.reference, fb.reference
                        ),
                    });
                }
            }
            // Footprint keepouts.
            for k in board.keepouts.values() {
                if k.rules.footprints || k.rules.max_height.is_some() {
                    let ko = Prim::polygon(k.outline.rings().cloned().collect());
                    if p.gap(&ko) < 0.0 {
                        let too_tall = match (k.rules.max_height, &fa.package.body) {
                            (Some(h), Some(b)) => b.height > h,
                            (Some(_), None) => false,
                            (None, _) => true,
                        };
                        if k.rules.footprints || too_tall {
                            self.put(DrcFinding {
                                code: "keepout".into(),
                                severity: Severity::Error,
                                items: vec![BItem::Footprint(*id)],
                                layer: None,
                                at: fa.at,
                                actual: None,
                                required: None,
                                rule: k.name.clone(),
                                source: SourceKey::default(),
                                message: format!("{} is inside keepout `{}`", fa.reference, k.name),
                            });
                        }
                    }
                }
            }
        }
        // Silk to pad (footprint silk against every pad on that side's outer copper).
        let outer = |side: Side| if side == Side::Top { 0 } else { n - 1 };
        let mut silk: Vec<(BItem, Side, Prim)> = Vec::new();
        for f in board.footprints.values() {
            let t = f.transform();
            for gph in &f.package.silk {
                for p in graphic_prims(gph, &t) {
                    silk.push((BItem::Footprint(f.id), f.side, p));
                }
            }
        }
        for gr in board.graphics.values() {
            if let Layer::Silk(side) = gr.layer {
                for p in graphic_prims(&gr.graphic, &crate::geom::Transform::IDENTITY) {
                    silk.push((BItem::Edge, side, p));
                }
            }
        }
        let mut found = Vec::new();
        for (owner, side, sp) in &silk {
            if only.is_some_and(|s| !s.contains(owner)) {
                continue;
            }
            let layer = outer(*side);
            let q = sp
                .bbox()
                .inflate(board.limits.min_silk_to_pad.max(crate::geom::mm(0.5)));
            for k in self.index.query(&q) {
                let cp = self.slot(k);
                let BItem::Pad(..) = cp.owner else { continue };
                if !cp.layers.contains(layer) {
                    continue;
                }
                let req = rules.pair_len(*owner, cp.owner, props::SILK_CLEARANCE);
                let gap = sp.gap(&cp.prim);
                if super::short_of(gap, req) {
                    let (rule, source) = rules.pair_origin(*owner, cp.owner, props::SILK_CLEARANCE);
                    found.push(DrcFinding {
                        code: "silk-to-pad".into(),
                        severity: Severity::Warning,
                        items: key_items(vec![*owner, cp.owner]),
                        layer: Some(layer),
                        at: sp.anchor(),
                        actual: Some(gap.max(0.0)),
                        required: Some(req as f64),
                        rule,
                        source,
                        message: format!("silkscreen {:.4} mm from a pad", gap.max(0.0) * 1e-6),
                    });
                }
            }
        }
        for f in found {
            self.put(f);
        }
        let _ = g;
    }
}

pub(super) fn courtyard_overlap(a: &Prim, b: &Prim) -> bool {
    match (&a.core, &b.core) {
        (crate::geom::shape::Core::Poly(ra), crate::geom::shape::Core::Poly(rb)) => {
            crate::geom::poly::boolean(ra, rb, crate::geom::poly::BoolOp::Intersection).area() > 0.0
        }
        _ => true,
    }
}
