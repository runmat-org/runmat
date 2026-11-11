//! Edge flanges and hems on flat plates, developed for the plate's flat pattern.
//!
//! An edge flange on a plate edge adds a bend and a flat outside the plate's outline, so the
//! developed plate is the base flange's outline with a strip spliced in along that edge: the
//! plate stops at the flange's `shift` past the edge line (inside the edge for a flange
//! placed `material_inside` or `material_outside`), then the bend zone (the bend allowance)
//! and the flange's flat follow, as wide as the edge. The bend's centre line runs across the
//! middle of the bend zone.
//!
//! Each flange flat records the rigid motion from the developed strip to where the flange
//! put it. A flange grown from a flat a sketched bend moved, or from another flange's flat,
//! composes that flat's motion, so later flanges, cuts and the flat pattern find every flat
//! by its region of the developed plate.

use super::super::families::{apply_rotation, compose, inverse, line_seg};
use super::*;
use crate::regen::records::PlateFlangeEdge;

/// One flange edge as the flange lowering sees it, in world coordinates.
pub(in crate::regen::engine) struct FlangeEdgeIn {
    /// The edge's ends.
    pub start: [f64; 3],
    pub end: [f64; 3],
    /// The section frame: `x` out of the thickness face, `y` the outward normal of the sheet
    /// face the flange turns towards, origin on the edge line moved by `shift` along `x`.
    pub frame: Frame,
    pub shift: f64,
    pub theta: f64,
    pub radius: f64,
    pub flat: f64,
}

/// `(p − o)·n ≥ d` as `(o, n, d)`, in the developed plate's frame.
pub(super) type Keep = ([f64; 2], [f64; 2], f64);

/// A flat region of the developed plate: `keep` half-planes `(p − o)·n ≥ d` (all must hold),
/// and the motion from the developed plate to where the flat lies.
pub(super) struct Region {
    pub key: String,
    pub keep: Vec<Keep>,
    pub motion: Option<Transform>,
}

/// How far short of a bend's tangent line a region stops.
pub(super) const SHY: f64 = 1e-6;

impl Region {
    pub fn holds(&self, p: [f64; 2], tol: f64) -> bool {
        self.keep
            .iter()
            .all(|(o, n, d)| (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1] >= d - tol)
    }
}

fn plus(p: [f64; 2], v: [f64; 2], s: f64) -> [f64; 2] {
    [p[0] + s * v[0], p[1] + s * v[1]]
}

impl<'a, 'b> Run<'a, 'b> {
    /// The flanges of plate `maker` recorded on the faces of `body`, in feature order.
    pub(in crate::regen::engine) fn plate_flanges_on(
        &self,
        body: usize,
        maker: &FeatureId,
    ) -> Vec<FeatureId> {
        let makers: Vec<&FeatureId> = self.st.bodies[body]
            .names
            .faces
            .iter()
            .flatten()
            .map(|n| n.maker())
            .collect();
        let own = matches!(
            self.st.records.get(maker),
            Some(FeatureRecord::Sheet { plate: Some(_), .. })
        )
        .then(|| maker.clone());
        own.into_iter()
            .chain(
                self.doc
                    .features
                    .iter()
                    .filter(|f| {
                        matches!(
                            self.st.records.get(&f.id),
                            Some(FeatureRecord::PlateFlange { base, .. }) if base == maker
                        ) && makers.contains(&&f.id)
                    })
                    .map(|f| f.id.clone()),
            )
            .collect()
    }

    /// The outline of plate `maker`: its base flange's profile, or the outline a conversion
    /// recorded; `None` for another sheet.
    pub(in crate::regen::engine) fn plate_profile(
        &mut self,
        i: usize,
        id: &FeatureId,
        param: &str,
        maker: &FeatureId,
    ) -> Result<Option<CanonicalProfile>, Halt> {
        if let Some(FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { profile, .. })) =
            self.doc.feature(maker).map(|f| &f.kind)
        {
            let profile = profile.clone();
            return self.profile(i, id, param, &profile).map(Some);
        }
        Ok(match self.st.records.get(maker) {
            Some(FeatureRecord::Sheet { plate: Some(p), .. }) => Some(p.outline.clone()),
            _ => None,
        })
    }

    /// The developed strips of `flanges`: each flange's edges, and a converted plate's walls.
    pub(super) fn flange_edges(&self, flanges: &[FeatureId]) -> Vec<PlateFlangeEdge> {
        flanges
            .iter()
            .filter_map(|f| match self.st.records.get(f) {
                Some(FeatureRecord::PlateFlange { edges, .. }) => Some(edges.clone()),
                Some(FeatureRecord::Sheet { plate: Some(p), .. }) => Some(p.edges.clone()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    /// The flat regions of a plate bent by `bends` and flanged by `flanges`.
    pub(super) fn plate_regions(
        &self,
        bends: &[(FeatureId, Vec<FlatBendLine>)],
        flanges: &[FeatureId],
    ) -> Vec<Region> {
        let mut out = vec![Region {
            key: "plate".into(),
            keep: Vec::new(),
            motion: None,
        }];
        let record = |f: &FeatureId| match self.st.records.get(f) {
            Some(FeatureRecord::PlateBend {
                lines,
                moved,
                motion,
                parent,
                ..
            }) => Some((lines.clone(), *moved, *motion, parent.clone())),
            _ => None,
        };
        // (owner region, excluded half-plane) pairs.
        let mut leaves: Vec<(String, Keep)> = Vec::new();
        for (b, _) in bends {
            let Some((lines, (o, n), motion, parent)) = record(b) else {
                continue;
            };
            // The moved flat starts past the last bend line's allowance.
            let start = lines
                .iter()
                .map(|l| {
                    let mid = [(l.start[0] + l.end[0]) / 2.0, (l.start[1] + l.end[1]) / 2.0];
                    (mid[0] - o[0]) * n[0] + (mid[1] - o[1]) * n[1] + l.allowance / 2.0
                })
                .fold(0.0, f64::max);
            out.push(Region {
                key: b.0.clone(),
                keep: vec![(o, n, start + SHY)],
                motion: Some(motion),
            });
            // Each bend's moved half-plane leaves its parent's region (the plate's, without one).
            leaves.push((
                parent.map(|p| p.0).unwrap_or_else(|| "plate".into()),
                (o, [-n[0], -n[1]], SHY),
            ));
        }
        for e in self.flange_edges(flanges) {
            out.push(Region {
                key: e.key.clone(),
                keep: vec![(e.start, e.out, e.shift + e.allowance + SHY)],
                motion: Some(e.motion),
            });
            leaves.push((
                e.parent.clone().unwrap_or_else(|| "plate".into()),
                (e.start, [-e.out[0], -e.out[1]], -e.shift + SHY),
            ));
        }
        for (owner, h) in leaves {
            if let Some(r) = out.iter_mut().find(|r| r.key == owner) {
                r.keep.push(h);
            }
        }
        out
    }

    /// Records the flanges an edge flange or hem `id` grows on plate `maker` (on `body`), one
    /// per edge: where each develops on the plate, its bend line and its flat's motion.
    pub(in crate::regen::engine) fn record_plate_flange(
        &mut self,
        i: usize,
        id: &FeatureId,
        body: usize,
        maker: &FeatureId,
        rules: &SheetRules,
        edges: Vec<FlangeEdgeIn>,
    ) -> Result<(), Halt> {
        let Some(plate) = self.plate_profile(i, id, "edges", maker)? else {
            return Ok(());
        };
        let out = self.developed_flange_edges(i, id, body, maker, &plate, rules, edges)?;
        self.st.records.insert(
            id.clone(),
            FeatureRecord::PlateFlange {
                base: maker.clone(),
                edges: out,
            },
        );
        Ok(())
    }

    /// Where each of `edges` (on plate `maker`, outline `plate`, on `body`) develops on the
    /// plate: its strip, its bend line and its flat's motion.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::regen::engine) fn developed_flange_edges(
        &mut self,
        _i: usize,
        id: &FeatureId,
        body: usize,
        maker: &FeatureId,
        plate: &CanonicalProfile,
        rules: &SheetRules,
        edges: Vec<FlangeEdgeIn>,
    ) -> Result<Vec<PlateFlangeEdge>, Halt> {
        let pf = plate.frame;
        let t = rules.thickness;
        let (bends, flanges, _) = self.plate_bends_and_others(body, maker);
        let regions = self.plate_regions(&bends, &flanges);
        let local = |p: [f64; 3]| {
            let q = sub(p, pf.origin);
            [dot(q, pf.x), dot(q, pf.y), dot(q, pf.z)]
        };
        let tol = 1e-7;
        let mut out = Vec::new();
        for (k, e) in edges.into_iter().enumerate() {
            // The flat the edge lies on: its ends, moved back onto the developed plate, lie on
            // that region within the sheet.
            let mid = scale(add(e.start, e.end), 0.5);
            let region = regions.iter().find(|r| {
                let back = r.motion.as_ref().map(inverse);
                [e.start, e.end, mid].iter().all(|p| {
                    let p = back.as_ref().map_or(*p, |m| m.apply(*p));
                    let q = local(p);
                    q[2] >= -tol && q[2] <= t + tol && r.holds([q[0], q[1]], tol + SHY)
                })
            });
            let Some(region) = region else {
                return Err(self.not_lowered(
                    id,
                    "flanges on an edge of a bend zone or of a jog's middle flat (flange an edge of the plate or of a flat a bend or flange moved)",
                ));
            };
            let back = region.motion.as_ref().map(inverse);
            let to2d = |p: [f64; 3]| {
                let q = local(back.as_ref().map_or(p, |m| m.apply(p)));
                [q[0], q[1]]
            };
            let dir2d = |v: [f64; 3]| {
                let v = back.as_ref().map_or(v, |m| apply_rotation(m, v));
                [dot(v, pf.x), dot(v, pf.y)]
            };
            let fr = e.frame;
            let (start, end) = (to2d(e.start), to2d(e.end));
            let out_dir = dir2d(fr.x);
            // The flange turns towards the sheet face `y` points out of.
            let up = dot(
                back.as_ref().map_or(fr.y, |m| apply_rotation(m, fr.y)),
                pf.z,
            ) > 0.0;
            let allowance =
                crate::model::sheet_metal::bend_allowance(&rules.rule, t, e.radius, e.theta)
                    .map_err(|m| {
                        self.error(id, Some("angle"), DiagCode::RegenInvalid, m, Vec::new())
                    })?;
            let deduction =
                crate::model::sheet_metal::bend_deduction(&rules.rule, t, e.radius, e.theta)
                    .map_err(|m| {
                        self.error(id, Some("angle"), DiagCode::RegenInvalid, m, Vec::new())
                    })?;
            let at = e.shift + allowance / 2.0;
            let key = format!("{}.{k}", id.0);
            let line = FlatBendLine {
                key: key.clone(),
                feature: id.clone(),
                start: plus(start, out_dir, at),
                end: plus(end, out_dir, at),
                angle: e.theta,
                inner_radius: e.radius,
                allowance,
                deduction,
                direction: if up {
                    crate::model::sheet_metal::BendDirection::Up
                } else {
                    crate::model::sheet_metal::BendDirection::Down
                },
            };
            // Developed to bent, in the section frame: the flat's start (x = allowance on the
            // developed sheet) turns by θ about the bend's axis to the bend's far tangent line.
            let (s, c) = e.theta.sin_cos();
            let r = e.radius;
            let cols = [fr.x, fr.y, fr.z];
            let turn = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
            let mut rotation = [[0.0; 3]; 3];
            for (a, row) in rotation.iter_mut().enumerate() {
                for (b, v) in row.iter_mut().enumerate() {
                    // F · turn · Fᵀ
                    *v = (0..3)
                        .map(|p| {
                            (0..3)
                                .map(|q| cols[p][a] * turn[p][q] * cols[q][b])
                                .sum::<f64>()
                        })
                        .sum();
                }
            }
            let p0 = add(fr.origin, scale(fr.x, allowance));
            let p1 = add(
                fr.origin,
                add(scale(fr.x, r * s), scale(fr.y, r * (1.0 - c))),
            );
            let rot = Transform {
                rotation,
                translation: [0.0; 3],
            };
            let local_motion = Transform {
                rotation,
                translation: sub(p1, rot.apply(p0)),
            };
            let motion = match &region.motion {
                Some(m) => compose(&local_motion, m),
                None => local_motion,
            };
            out.push(PlateFlangeEdge {
                key,
                start,
                end,
                out: out_dir,
                shift: e.shift,
                allowance,
                flat: e.flat.max(0.0),
                line,
                motion,
                parent: (region.key != "plate").then(|| region.key.clone()),
                frame: fr,
                mitre: None,
            });
        }
        Ok(out)
    }

    /// The developed outline of plate `plate` with the strips of `flanges` spliced in along
    /// their edges, in feature order (a flange on another flange's tip extends that tip), then the
    /// corner features' edits.
    pub(super) fn developed_plate(
        &mut self,
        id: &FeatureId,
        maker: &FeatureId,
        plate: &CanonicalProfile,
        flanges: &[FeatureId],
    ) -> Result<CanonicalProfile, Halt> {
        let mut out = plate.clone();
        let tol = 1e-7;
        for e in self.flange_edges(flanges) {
            let reach = e.shift + e.allowance + e.flat;
            let lp = out.loops.first_mut().expect("an outer loop");
            // The outline line holding the edge.
            let found = lp.iter().position(|s| {
                if !matches!(s.kind, SegmentKind::Line) {
                    return false;
                }
                let d = [s.end[0] - s.start[0], s.end[1] - s.start[1]];
                let len2 = d[0] * d[0] + d[1] * d[1];
                [e.start, e.end].iter().all(|p| {
                    let w = [p[0] - s.start[0], p[1] - s.start[1]];
                    let along = (w[0] * d[0] + w[1] * d[1]) / len2;
                    let across = (w[0] * d[1] - w[1] * d[0]).abs() / len2.sqrt();
                    across <= tol && along >= -tol && along <= 1.0 + tol
                })
            });
            let Some(k) = found else {
                return Err(self.not_lowered(
                    id,
                    "flat patterns of a flange on an edge that is not on the plate's outline",
                ));
            };
            if reach <= tol {
                return Err(self.not_lowered(
                    id,
                    "flat patterns of a flange that develops shorter than its setback",
                ));
            }
            let s = lp[k].clone();
            let d = [s.end[0] - s.start[0], s.end[1] - s.start[1]];
            let param = |p: [f64; 2]| (p[0] - s.start[0]) * d[0] + (p[1] - s.start[1]) * d[1];
            let (p, q) = if param(e.start) <= param(e.end) {
                (e.start, e.end)
            } else {
                (e.end, e.start)
            };
            let near = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) <= tol;
            let c = &s.curve.0;
            let mut pieces = Vec::new();
            if !near(s.start, p) {
                pieces.push(line_seg(c, s.start, p));
            }
            // Each end of the strip: square, or a miter flange's mitre through its stations.
            let along = {
                let l = (e.end[0] - e.start[0]).hypot(e.end[1] - e.start[1]);
                [(e.end[0] - e.start[0]) / l, (e.end[1] - e.start[1]) / l]
            };
            let side = |from: [f64; 2], offsets: Option<&Vec<f64>>| -> Vec<[f64; 2]> {
                match (&e.mitre, offsets) {
                    (Some(m), Some(off)) => m
                        .stations
                        .iter()
                        .zip(off)
                        .map(|(u, z)| plus(plus(from, along, *z), e.out, e.shift + *u))
                        .collect(),
                    _ => vec![plus(from, e.out, reach)],
                }
            };
            let forward = param(e.start) <= param(e.end);
            let (ps, qs) = (
                e.mitre
                    .as_ref()
                    .map(|m| if forward { &m.start } else { &m.end }),
                e.mitre
                    .as_ref()
                    .map(|m| if forward { &m.end } else { &m.start }),
            );
            // p, out along the start end to the tip, back along the end end to q.
            let (out_side, mut back) = (side(p, ps), side(q, qs));
            let tip = out_side.len() - 1;
            back.reverse();
            let mut run = vec![(p, String::new())];
            for (k, v) in out_side.into_iter().chain(back).enumerate() {
                let name = match k {
                    0 => format!("{}.p", e.key),
                    k if k == tip + 1 => format!("{}.tip", e.key),
                    k => format!("{}.m{k}", e.key),
                };
                run.push((v, name));
            }
            run.push((q, format!("{}.q", e.key)));
            for w in run.windows(2) {
                if !near(w[0].0, w[1].0) {
                    pieces.push(line_seg(&w[1].1, w[0].0, w[1].0));
                }
            }
            if !near(q, s.end) {
                pieces.push(line_seg(&format!("{c}.{}", e.key), q, s.end));
            }
            lp.splice(k..=k, pieces);
        }
        // Corner reliefs and closed corners, in feature order.
        for edit in self.plate_corner_edits(maker) {
            let lp = out.loops.first_mut().expect("an outer loop");
            if !super::corners::apply_edit(lp, &edit, tol) {
                return Err(self.not_lowered(
                    id,
                    "flat patterns of a corner feature whose corner a later feature moved",
                ));
            }
        }
        if let Some((a, b)) = out.loops.first().and_then(|lp| first_crossing(lp, tol)) {
            return Err(self.error(
                id,
                None,
                DiagCode::RegenInvalid,
                format!(
                    "the flat blank overlaps itself where its sides {a} and {b} cross (walls that meet at a concave corner need the same sheet when flat), so it can't be cut from one sheet; rip one of the walls there or make it a separate part"
                ),
                Vec::new(),
            ));
        }
        Ok(out)
    }

    /// The bend lines of `flanges` on the developed plate.
    pub(super) fn flange_bend_lines(&self, flanges: &[FeatureId]) -> Vec<FlatBendLine> {
        self.flange_edges(flanges)
            .into_iter()
            .map(|e| e.line)
            .collect()
    }
}

/// The first two straight sides of a closed outline that cross each other (each through the
/// other's interior), by curve name; `None` when the outline is simple.
pub(super) fn first_crossing(lp: &[ProfileSegment], tol: f64) -> Option<(String, String)> {
    let lines: Vec<&ProfileSegment> = lp
        .iter()
        .filter(|s| matches!(s.kind, SegmentKind::Line))
        .collect();
    let orient = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let strict = |x: f64, y: f64, e: f64| (x > e && y < -e) || (x < -e && y > e);
    for i in 0..lines.len() {
        for j in i + 1..lines.len() {
            let (p, q) = (lines[i], lines[j]);
            let lp_ = (p.end[0] - p.start[0]).hypot(p.end[1] - p.start[1]);
            let lq = (q.end[0] - q.start[0]).hypot(q.end[1] - q.start[1]);
            let (d1, d2) = (
                orient(p.start, p.end, q.start),
                orient(p.start, p.end, q.end),
            );
            let (d3, d4) = (
                orient(q.start, q.end, p.start),
                orient(q.start, q.end, p.end),
            );
            if strict(d1, d2, tol * lp_.max(1e-9)) && strict(d3, d4, tol * lq.max(1e-9)) {
                return Some((p.curve.0.clone(), q.curve.0.clone()));
            }
        }
    }
    None
}
