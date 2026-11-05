//! Corner reliefs and closed corners where two edge flanges of a flat plate meet.
//!
//! Two flanges meet at a corner when their bends' plate-side tangent lines cross at a point
//! `P` that ends both flanges' edges. On the developed plate, with `a` and `b` the two
//! flanges' outward directions, the plate is the quadrant behind `P` (`α ≤ 0`, `β ≤ 0` in
//! `P + α a + β b`), `a`'s bend zone is `0 ≤ α ≤ BAₐ` with `β ≤ 0`, and `b`'s is
//! `0 ≤ β ≤ BA_b` with `α ≤ 0`. The quadrant ahead of `P` is open: the bends are ripped along
//! their tangent lines up to `P`.
//!
//! A corner relief removes the last `size` of each bend zone next to the corner, and on the
//! plate a `size` square at `P` (rectangular) or a quarter disc of radius `size` about `P`
//! (round). In the part, each bend zone's piece is the whole bend over that stretch of its
//! axis, so it is cut by the bend's annular sector swept along the axis; the plate's piece is
//! cut by a prism through the plate.
//!
//! A closed corner extends both flanges' flats along their bend lines towards each other
//! (both flanges stand at 90°), as [`CornerOverlap`] describes. Each extension is a rectangle
//! of the flat's length on the flat pattern and a prism on the flat in the part.
//!
//! Both record [`OutlineEdit`]s on the plate, which the developed outline applies after the
//! flanges' strips.

use super::super::families::{apply_rotation, inverse, line_seg};
use super::*;
use crate::model::sheet_metal::{CornerOverlap, CornerReliefKind};
use crate::regen::records::{OutlineEdit, PlateFlangeEdge};

/// Two flanges of a plate meeting at a corner, in the developed plate's frame.
struct Corner {
    a: PlateFlangeEdge,
    b: PlateFlangeEdge,
    /// Where the bends' plate-side tangent lines cross.
    at: [f64; 2],
    /// The rigid motion of the flat both flanges grew from (none: the plate's unmoved part).
    motion: Option<Transform>,
}

/// What corner features read about a plate.
struct PlateView {
    maker: FeatureId,
    body: usize,
    t: f64,
    frame: Frame,
    corners: Vec<Corner>,
}

fn p2(p: [f64; 2], a: [f64; 2], s: f64, b: [f64; 2], u: f64) -> [f64; 2] {
    [p[0] + s * a[0] + u * b[0], p[1] + s * a[1] + u * b[1]]
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn arc(curve: &str, center: [f64; 2], from: [f64; 2], to: [f64; 2]) -> ProfileSegment {
    let (u, v) = (
        [from[0] - center[0], from[1] - center[1]],
        [to[0] - center[0], to[1] - center[1]],
    );
    ProfileSegment {
        curve: crate::ids::SketchEntityId::new(curve),
        kind: SegmentKind::Arc {
            center,
            clockwise: u[0] * v[1] - u[1] * v[0] < 0.0,
        },
        start: from,
        end: to,
    }
}

/// A polyline path through `pts`, its segments named `<key>.<k>`.
fn path(key: &str, pts: &[[f64; 2]]) -> Vec<ProfileSegment> {
    pts.windows(2)
        .enumerate()
        .map(|(k, w)| line_seg(&format!("{key}.{k}"), w[0], w[1]))
        .collect()
}

fn reversed(path: &[ProfileSegment]) -> Vec<ProfileSegment> {
    path.iter()
        .rev()
        .map(|s| ProfileSegment {
            curve: s.curve.clone(),
            kind: match s.kind.clone() {
                SegmentKind::Arc { center, clockwise } => SegmentKind::Arc {
                    center,
                    clockwise: !clockwise,
                },
                k => k,
            },
            start: s.end,
            end: s.start,
        })
        .collect()
}

fn seg_len(s: &ProfileSegment) -> f64 {
    match &s.kind {
        SegmentKind::Arc { center, .. } => {
            let r = dist2(s.start, *center);
            let (u, v) = (
                [s.start[0] - center[0], s.start[1] - center[1]],
                [s.end[0] - center[0], s.end[1] - center[1]],
            );
            r * (u[0] * v[1] - u[1] * v[0]).atan2(dot2(u, v)).abs()
        }
        _ => dist2(s.start, s.end),
    }
}

/// Makes `p` the start of a segment of `lp` (splitting the line holding it); its index.
fn vertex_at(lp: &mut Vec<ProfileSegment>, p: [f64; 2], tol: f64) -> Option<usize> {
    if let Some(k) = lp.iter().position(|s| dist2(s.start, p) <= tol) {
        return Some(k);
    }
    let k = lp.iter().position(|s| {
        if !matches!(s.kind, SegmentKind::Line) {
            return false;
        }
        let d = [s.end[0] - s.start[0], s.end[1] - s.start[1]];
        let len = d[0].hypot(d[1]);
        let w = [p[0] - s.start[0], p[1] - s.start[1]];
        let along = dot2(w, d) / len;
        let across = (w[0] * d[1] - w[1] * d[0]).abs() / len;
        across <= tol && along > tol && along < len - tol
    })?;
    let s = lp[k].clone();
    let first = line_seg(&s.curve.0, s.start, p);
    let second = line_seg(&format!("{}.r", s.curve.0), p, s.end);
    lp.splice(k..=k, [first, second]);
    Some(k + 1)
}

/// Applies `edit` to the outline loop `lp`: the shorter way round from `from` to `to` is
/// replaced by the edit's path. False when either end is not on the outline.
pub(super) fn apply_edit(lp: &mut Vec<ProfileSegment>, edit: &OutlineEdit, tol: f64) -> bool {
    if vertex_at(lp, edit.from, tol).is_none() || vertex_at(lp, edit.to, tol).is_none() {
        return false;
    }
    let (Some(a), Some(b)) = (
        lp.iter().position(|s| dist2(s.start, edit.from) <= tol),
        lp.iter().position(|s| dist2(s.start, edit.to) <= tol),
    ) else {
        return false;
    };
    let n = lp.len();
    let forward: f64 = (0..(b + n - a) % n)
        .map(|k| seg_len(&lp[(a + k) % n]))
        .sum();
    let total: f64 = lp.iter().map(seg_len).sum();
    let (start, count, new) = if forward <= total - forward {
        (a, (b + n - a) % n, edit.path.clone())
    } else {
        (b, (a + n - b) % n, reversed(&edit.path))
    };
    lp.rotate_left(start);
    lp.splice(0..count, new);
    true
}

impl<'a, 'b> Run<'a, 'b> {
    /// The outline edits corner features recorded on plate `maker`, in feature order.
    pub(super) fn plate_corner_edits(&self, maker: &FeatureId) -> Vec<OutlineEdit> {
        self.doc
            .features
            .iter()
            .filter_map(|f| match self.st.records.get(&f.id) {
                Some(FeatureRecord::PlateCorner { base, edits }) if base == maker => {
                    Some(edits.clone())
                }
                _ => None,
            })
            .flatten()
            .collect()
    }

    /// The plate under `set`'s entities and the corners where its flanges meet.
    fn plate_view(
        &mut self,
        i: usize,
        id: &FeatureId,
        param: &str,
        set: &BTreeSet<EntityHandle>,
    ) -> Result<PlateView, Halt> {
        let operands = Self::bodies_of(set);
        let [body] = *operands.as_slice() else {
            return Err(self.error(
                id,
                Some(param),
                DiagCode::RefKindMismatch,
                "corner features act on one sheet".into(),
                Vec::new(),
            ));
        };
        let Some((maker, rules, section, _)) = self.sheet_of(body) else {
            return Err(self.error(
                id,
                Some(param),
                DiagCode::RefKindMismatch,
                "the picks are not on a sheet metal body".into(),
                Vec::new(),
            ));
        };
        if section.is_some() {
            return Err(self.not_lowered(
                id,
                "corners of flanges on a bent base flange (flange a flat plate)",
            ));
        }
        let profile_ref = match self.doc.feature(&maker).map(|f| &f.kind) {
            Some(FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { profile, .. })) => {
                profile.clone()
            }
            _ => return Err(self.not_lowered(id, "corners of this sheet")),
        };
        let frame = self.profile(i, id, param, &profile_ref)?.frame;
        let (bends, flanges, _) = self.plate_bends_and_others(body, &maker);
        let regions = self.plate_regions(&bends, &flanges);
        let edges: Vec<PlateFlangeEdge> = flanges
            .iter()
            .filter_map(|f| match self.st.records.get(f) {
                Some(FeatureRecord::PlateFlange { edges, .. }) => Some(edges.clone()),
                _ => None,
            })
            .flatten()
            .collect();
        let tol = 1e-7;
        let mut corners = Vec::new();
        for (x, e) in edges.iter().enumerate() {
            for f in &edges[x + 1..] {
                if e.parent != f.parent || dot2(e.out, f.out).abs() > 1e-9 {
                    continue;
                }
                // (p − e.start)·e.out = e.shift and the same for f, solved for p.
                let (ce, cf) = (
                    dot2(e.start, e.out) + e.shift,
                    dot2(f.start, f.out) + f.shift,
                );
                let det = e.out[0] * f.out[1] - e.out[1] * f.out[0];
                let at = [
                    (ce * f.out[1] - cf * e.out[1]) / det,
                    (e.out[0] * cf - f.out[0] * ce) / det,
                ];
                // Each edge ends on the other's tangent line, its span behind it.
                let ends = |g: &PlateFlangeEdge, other: [f64; 2]| {
                    let (s, t) = (
                        dot2([g.start[0] - at[0], g.start[1] - at[1]], other),
                        dot2([g.end[0] - at[0], g.end[1] - at[1]], other),
                    );
                    (s.abs() <= tol && t < -tol) || (t.abs() <= tol && s < -tol)
                };
                if !(ends(e, f.out) && ends(f, e.out)) {
                    continue;
                }
                let key = e.parent.clone().unwrap_or_else(|| "plate".into());
                let motion = regions.iter().find(|r| r.key == key).and_then(|r| r.motion);
                corners.push(Corner {
                    a: e.clone(),
                    b: f.clone(),
                    at,
                    motion,
                });
            }
        }
        Ok(PlateView {
            maker,
            body,
            t: rules.thickness,
            frame,
            corners,
        })
    }

    /// A developed point of the corner's parent flat, in the part.
    fn corner_world(v: &PlateView, c: &Corner, p: [f64; 2]) -> [f64; 3] {
        let w = v.frame.to_world(p);
        c.motion.as_ref().map_or(w, |m| m.apply(w))
    }

    /// A point of an entity: a vertex, an edge's midpoint or a face's centroid.
    fn entity_point(&self, h: &EntityHandle) -> [f64; 3] {
        let topo = &self.st.bodies[h.body as usize].topology;
        match h.kind {
            TopoKind::Vertex => topo.vertices[h.index as usize].point,
            TopoKind::Edge => {
                let g = &topo.edges[h.index as usize].geom;
                scale(add(g.start, g.end), 0.5)
            }
            _ => topo.faces[h.index as usize].geom.centroid,
        }
    }

    /// The bend of flange `e` swept along its axis over `[z0, z1]` (its frame's z), with a
    /// margin across the sheet: what removes the bend over that stretch.
    fn bend_sweep(e: &PlateFlangeEdge, t: f64, z0: f64, z1: f64, key: &str) -> LabeledTool {
        let (r, theta) = (e.line.inner_radius, e.line.angle);
        let c = [0.0, r];
        let a0 = -std::f64::consts::FRAC_PI_2;
        let a1 = a0 + theta;
        let at = |rad: f64, a: f64| [c[0] + rad * a.cos(), c[1] + rad * a.sin()];
        let (ri, ro) = (r * 0.5, r + 1.5 * t);
        let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{key}.{s}"));
        let mut segs = vec![ProfileSegment {
            curve: sid("out"),
            kind: SegmentKind::Arc {
                center: c,
                clockwise: false,
            },
            start: at(ro, a0),
            end: at(ro, a1),
        }];
        if ri > 1e-12 {
            segs.push(line_seg(&sid("e1").0, at(ro, a1), at(ri, a1)));
            segs.push(ProfileSegment {
                curve: sid("in"),
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: true,
                },
                start: at(ri, a1),
                end: at(ri, a0),
            });
            segs.push(line_seg(&sid("e0").0, at(ri, a0), at(ro, a0)));
        } else {
            segs.push(line_seg(&sid("e1").0, at(ro, a1), c));
            segs.push(line_seg(&sid("e0").0, c, at(ro, a0)));
        }
        let profile = CanonicalProfile {
            frame: e.frame,
            loops: vec![segs.clone()],
            merge_collinear: false,
        };
        let mut tool = LabeledTool::new(ToolShape::prism(profile, z0, z1));
        for (k, s) in segs.iter().enumerate() {
            tool.sides.push((
                s.curve.clone(),
                FaceLabel::key(Role::Relief, key).with_ordinal(k as u32 + 2),
            ));
        }
        tool.start = Some(FaceLabel::key(Role::Relief, key).with_ordinal(0));
        tool.end = Some(FaceLabel::key(Role::Relief, key).with_ordinal(1));
        tool
    }

    /// The z range, in flange `e`'s frame, of the corner's developed points `P + s d` for `s`
    /// in `[s0, s1]`.
    fn axis_range(
        v: &PlateView,
        c: &Corner,
        e: &PlateFlangeEdge,
        d: [f64; 2],
        s0: f64,
        s1: f64,
    ) -> (f64, f64) {
        let z = |s: f64| {
            let p = Self::corner_world(v, c, p2(c.at, d, s, d, 0.0));
            dot(sub(p, e.frame.origin), e.frame.z)
        };
        let (a, b) = (z(s0), z(s1));
        (a.min(b), a.max(b))
    }

    /// The corners `refs` pick on one plate (each the corner nearest its entity), in order.
    fn picked_corners(
        &mut self,
        i: usize,
        id: &FeatureId,
        param: &str,
        refs: &[Reference],
        kind: Option<TopoKind>,
    ) -> Result<(PlateView, Vec<usize>), Halt> {
        let set = self.list(i, id, param, refs, kind)?;
        let view = self.plate_view(i, id, param, &set)?;
        let mut picked = Vec::new();
        for h in &set {
            if h.kind == TopoKind::Face {
                // A face of a flange: each corner of that flange.
                let makers: Vec<FeatureId> = self
                    .st
                    .entity_names(h)
                    .iter()
                    .map(|n| n.maker().clone())
                    .collect();
                let of = |e: &PlateFlangeEdge| {
                    makers
                        .iter()
                        .any(|m| e.key.rsplit_once('.').map(|k| k.0) == Some(m.0.as_str()))
                };
                // A face of the plate itself: every corner of the plate.
                let whole = makers.contains(&view.maker);
                let hits: Vec<usize> = (0..view.corners.len())
                    .filter(|&k| whole || of(&view.corners[k].a) || of(&view.corners[k].b))
                    .collect();
                if !hits.is_empty() {
                    for k in hits {
                        if !picked.contains(&k) {
                            picked.push(k);
                        }
                    }
                    continue;
                }
                return Err(self.error(
                    id,
                    Some(param),
                    DiagCode::RegenInvalid,
                    "the picked face is on no flange that meets another at a corner".into(),
                    Vec::new(),
                ));
            }
            let p = self.entity_point(h);
            let near = view
                .corners
                .iter()
                .enumerate()
                .map(|(k, c)| {
                    let w = Self::corner_world(&view, c, c.at);
                    (k, dot(sub(p, w), sub(p, w)).sqrt())
                })
                .min_by(|x, y| x.1.total_cmp(&y.1));
            let reach = |c: &Corner| {
                c.a.allowance.max(c.b.allowance)
                    + c.a.line.inner_radius.max(c.b.line.inner_radius)
                    + 3.0 * view.t
            };
            match near {
                Some((k, d)) if d <= reach(&view.corners[k]) => {
                    if !picked.contains(&k) {
                        picked.push(k);
                    }
                }
                _ => {
                    return Err(self.error(
                        id,
                        Some(param),
                        DiagCode::RegenInvalid,
                        "no two flanges' bends meet at the picked corner".into(),
                        Vec::new(),
                    ))
                }
            }
        }
        Ok((view, picked))
    }

    pub(in crate::regen::engine) fn corner_relief(
        &mut self,
        i: usize,
        id: &FeatureId,
        corners: &[Reference],
        kind: &CornerReliefKind,
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let size = match kind {
            CornerReliefKind::Rectangular { size } | CornerReliefKind::Round { size } => {
                Some(self.value(id, "relief.size", size, Dimension::LENGTH)?)
            }
            CornerReliefKind::Tear => None,
        };
        let (view, picked) = self.picked_corners(i, id, "corners", corners, None)?;
        let Some(d) = size else {
            // A tear: the bends already part along their tangent lines up to the corner.
            return Ok(Compiled::Record(FeatureRecord::PlateCorner {
                base: view.maker.clone(),
                edits: Vec::new(),
            }));
        };
        let t = view.t;
        let mut tools = Vec::new();
        let mut edits = Vec::new();
        for k in picked {
            let c = &view.corners[k];
            let (ua, ub) = (c.a.out, c.b.out);
            let span = |e: &PlateFlangeEdge| dist2(e.start, e.end);
            if d <= 0.0 || d >= span(&c.a) || d >= span(&c.b) {
                return Err(self.error(
                    id,
                    Some("relief.size"),
                    DiagCode::RegenInvalid,
                    "a corner relief's size is positive and shorter than both flanges".into(),
                    Vec::new(),
                ));
            }
            let key = format!("{}.{}+{}", id.0, c.a.key, c.b.key);
            let (ba, bb) = (c.a.allowance, c.b.allowance);
            let at = |s: f64, u: f64| p2(c.at, ua, s, ub, u);
            // The flat pattern: from a's bend zone across to b's, round the plate's piece.
            let round = matches!(kind, CornerReliefKind::Round { .. });
            let mut segs = path(
                &key,
                &[
                    at(ba, 0.0),
                    at(ba, -d),
                    at(if round { 0.0 } else { -d }, -d),
                ],
            );
            if round {
                segs.push(arc(&format!("{key}.arc"), c.at, at(0.0, -d), at(-d, 0.0)));
                segs.extend(path(&format!("{key}.b"), &[at(-d, 0.0), at(-d, bb)]));
            } else {
                segs.extend(path(&format!("{key}.b"), &[at(-d, -d), at(-d, bb)]));
            }
            segs.push(line_seg(&format!("{key}.end"), at(-d, bb), at(0.0, bb)));
            edits.push(OutlineEdit {
                from: at(ba, 0.0),
                to: at(0.0, bb),
                path: segs,
            });
            // The part: each bend over its last `d` (and a thickness past the corner), and
            // the plate's piece through the plate.
            let (z0, z1) = Self::axis_range(&view, c, &c.a, ub, -d, t);
            tools.push(Self::bend_sweep(&c.a, t, z0, z1, &format!("{key}.a")));
            let (z0, z1) = Self::axis_range(&view, c, &c.b, ua, -d, t);
            tools.push(Self::bend_sweep(&c.b, t, z0, z1, &format!("{key}.b")));
            let mut plate = if round {
                vec![
                    line_seg(&format!("{key}.p0"), c.at, at(0.0, -d)),
                    arc(&format!("{key}.p1"), c.at, at(0.0, -d), at(-d, 0.0)),
                    line_seg(&format!("{key}.p2"), at(-d, 0.0), c.at),
                ]
            } else {
                path(
                    &format!("{key}.p"),
                    &[c.at, at(0.0, -d), at(-d, -d), at(-d, 0.0), c.at],
                )
            };
            // Counter-clockwise in the profile frame.
            let area: f64 = plate
                .iter()
                .map(|s| s.start[0] * s.end[1] - s.end[0] * s.start[1])
                .sum();
            if area < 0.0 {
                plate = reversed(&plate);
            }
            let mut pf = view.frame;
            if let Some(m) = &c.motion {
                pf = Frame {
                    origin: m.apply(pf.origin),
                    x: apply_rotation(m, pf.x),
                    y: apply_rotation(m, pf.y),
                    z: apply_rotation(m, pf.z),
                };
            }
            let profile = CanonicalProfile {
                frame: pf,
                loops: vec![plate.clone()],
                merge_collinear: false,
            };
            let mut tool = LabeledTool::new(ToolShape::prism(profile, -t, 2.0 * t));
            let pkey = format!("{key}.plate");
            for (k, s) in plate.iter().enumerate() {
                tool.sides.push((
                    s.curve.clone(),
                    FaceLabel::key(Role::Relief, pkey.clone()).with_ordinal(k as u32 + 2),
                ));
            }
            tool.start = Some(FaceLabel::key(Role::Relief, pkey.clone()).with_ordinal(0));
            tool.end = Some(FaceLabel::key(Role::Relief, pkey).with_ordinal(1));
            tools.push(tool);
        }
        self.st.records.insert(
            id.clone(),
            FeatureRecord::PlateCorner {
                base: view.maker.clone(),
                edits,
            },
        );
        Ok(op(
            labeled(tools, Vec::new(), Combine::Cut, false),
            vec![view.body],
            naming.clone(),
        ))
    }

    pub(in crate::regen::engine) fn closed_corner(
        &mut self,
        i: usize,
        id: &FeatureId,
        faces: &[Reference],
        gap: f64,
        overlap: CornerOverlap,
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let set = self.list(i, id, "faces", faces, Some(TopoKind::Face))?;
        let view = self.plate_view(i, id, "faces", &set)?;
        let t = view.t;
        if gap < 0.0 {
            return Err(self.error(
                id,
                Some("gap"),
                DiagCode::RegenInvalid,
                "a closed corner's gap is not negative".into(),
                Vec::new(),
            ));
        }
        // The flange flat each picked face lies on, by its developed position.
        let local = |p: [f64; 3]| {
            let q = sub(p, view.frame.origin);
            [
                dot(q, view.frame.x),
                dot(q, view.frame.y),
                dot(q, view.frame.z),
            ]
        };
        let mut extend: Vec<String> = Vec::new();
        for h in &set {
            // A face of the plate itself: each corner's first flange (in feature order).
            if self
                .st
                .entity_names(h)
                .iter()
                .any(|n| n.maker() == &view.maker)
            {
                for c in &view.corners {
                    if !extend.contains(&c.a.key) && !extend.contains(&c.b.key) {
                        extend.push(c.a.key.clone());
                    }
                }
                continue;
            }
            let p = self.entity_point(h);
            let found = view.corners.iter().flat_map(|c| [&c.a, &c.b]).find(|e| {
                let q = local(inverse(&e.motion).apply(p));
                let along = [e.end[0] - e.start[0], e.end[1] - e.start[1]];
                let w = [q[0] - e.start[0], q[1] - e.start[1]];
                let s = dot2(w, along) / dot2(along, along);
                let u = dot2(w, e.out);
                let tol = 1e-7;
                let flat = e.shift + e.allowance;
                (-tol..=t + tol).contains(&q[2])
                    && (-tol..=1.0 + tol).contains(&s)
                    && (flat - tol..=flat + e.flat + tol).contains(&u)
            });
            match found {
                Some(e) => {
                    if !extend.contains(&e.key) {
                        extend.push(e.key.clone());
                    }
                }
                None => return Err(self.error(
                    id,
                    Some("faces"),
                    DiagCode::RegenInvalid,
                    "a closed corner extends the flat of a flange that meets another at a corner"
                        .into(),
                    Vec::new(),
                )),
            }
        }
        let mut tools = Vec::new();
        let mut edits = Vec::new();
        let mut done: Vec<(String, String)> = Vec::new();
        for c in &view.corners {
            // `a` is the picked flange.
            let (a, b) = if extend.contains(&c.a.key) {
                (&c.a, &c.b)
            } else if extend.contains(&c.b.key) {
                (&c.b, &c.a)
            } else {
                continue;
            };
            let pair = (
                a.key.clone().min(b.key.clone()),
                a.key.clone().max(b.key.clone()),
            );
            if done.contains(&pair) {
                continue;
            }
            done.push(pair);
            let square =
                |e: &PlateFlangeEdge| (e.line.angle - std::f64::consts::FRAC_PI_2).abs() < 1e-9;
            if !(square(a) && square(b)) {
                return Err(self.not_lowered(id, "closed corners of flanges not at 90°"));
            }
            let (ra, rb) = (a.line.inner_radius, b.line.inner_radius);
            let (ea, eb) = match overlap {
                CornerOverlap::Overlap => (rb + t, ra - gap),
                CornerOverlap::Underlap => (rb - gap, ra + t),
                CornerOverlap::Butt => (rb - gap, ra - gap),
            };
            let (ua, ub) = (a.out, b.out);
            if eb > a.allowance + 1e-9 && ea > b.allowance + 1e-9 {
                return Err(self.not_lowered(
                    id,
                    "closed corners whose extended flats overlap on the flat pattern (a bend radius this small leaves no room)",
                ));
            }
            // Each flat extends along its bend line: `a`'s along `b`'s direction by `ea`,
            // `b`'s along `a`'s by `eb`.
            for (e, dir, ext, k) in [(a, ub, ea, "a"), (b, ua, eb, "b")] {
                if ext <= 1e-12 || e.flat <= 1e-12 {
                    continue;
                }
                let key = format!("{}.{}+{}.{k}", id.0, c.a.key, c.b.key);
                let other = if k == "a" { ua } else { ub };
                let ba = e.allowance;
                let pt = |s: f64, u: f64| p2(c.at, other, s, dir, u);
                edits.push(OutlineEdit {
                    from: pt(ba + e.flat, 0.0),
                    to: pt(ba, 0.0),
                    path: path(
                        &key,
                        &[
                            pt(ba + e.flat, 0.0),
                            pt(ba + e.flat, ext),
                            pt(ba, ext),
                            pt(ba, 0.0),
                        ],
                    ),
                });
                let r = e.line.inner_radius;
                let corners = vec![
                    (format!("{key}.in"), [r, r]),
                    (format!("{key}.tip"), [r, r + e.flat]),
                    (format!("{key}.out"), [r + t, r + e.flat]),
                    (format!("{key}.root"), [r + t, r]),
                ];
                let (z0, z1) = Self::axis_range(&view, c, e, dir, 0.0, ext);
                let mut tool =
                    LabeledTool::new(ToolShape::prism(polygon(e.frame, &corners), z0, z1));
                for (o, (s, _)) in corners.iter().enumerate() {
                    tool.sides.push((
                        crate::ids::SketchEntityId::new(s.clone()),
                        FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(o as u32 + 2),
                    ));
                }
                tool.start = Some(FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(0));
                tool.end = Some(FaceLabel::key(Role::SheetEdge, key).with_ordinal(1));
                tools.push(tool);
            }
        }
        if done.is_empty() {
            return Err(self.error(
                id,
                Some("faces"),
                DiagCode::RegenInvalid,
                "the picked flanges meet no other flange at a corner".into(),
                Vec::new(),
            ));
        }
        let record = FeatureRecord::PlateCorner {
            base: view.maker.clone(),
            edits,
        };
        if tools.is_empty() {
            return Ok(Compiled::Record(record));
        }
        self.st.records.insert(id.clone(), record);
        Ok(op(
            labeled(tools, Vec::new(), Combine::Add, false),
            vec![view.body],
            naming.clone(),
        ))
    }
}
