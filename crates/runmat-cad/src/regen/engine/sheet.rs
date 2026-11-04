//! Sheet metal features beyond the base flange's own lowering: the miter flange, and the
//! flat pattern and unfold of a plate with sketched bends and jogs.
//!
//! # Miter flange
//!
//! The profile is an open chain of lines drawn on a plane normal to one of the chain's edges,
//! starting on that edge. Its coordinates are read in the edge's section plane: `x` out of the
//! sheet across the edge (the thickness face's outward normal) and `y` along the outward normal
//! of the sheet face the edges lie on. The sheet itself is the stretch `x ≤ 0`, `−t ≤ y ≤ 0`.
//!
//! The section is the sheet's own mold line (`x ≤ 0`, `y = 0`) followed by the profile, with
//! the material below the mold line (on the sheet's side), so a bend of the sheet's radius
//! joins the sheet to the profile's first segment and a bend joins each later pair. The
//! section then moves out by the first bend's setback so that bend starts at the sheet edge
//! (nothing of the sheet is trimmed, as an edge flange placed `bend_outside`).
//!
//! The same section runs along every edge of the chain. At a corner shared by two edges each
//! side is clipped by the plane through the corner vertex that bisects the two flanges (it
//! holds the sheet normal and the sum of the two edges' outward directions); a positive `gap`
//! moves each clip plane back by `gap / 2`. Open chain ends stay square.
//!
//! Faces are named by the edge they grow from (the edge between its sheet face and thickness
//! face) and an ordinal. The flat of profile curve `j` takes `16 j` plus 0 for the face on the
//! profile's side (the inner face of a flange turned up), 1 for the other face, 1 for its free
//! end, 2 and 3 for its square ends and 4 and 5 for the ends that meet bends. The bend before
//! curve `j` takes `16 j + 8` plus the bend's own numbers (0 inner and 1 outer face, 2 to 5
//! its ends). A mitre cut adds 6 (at the chain's earlier corner) or 7 (at the later one).

use super::families::{bend_tool, frame_of, labeled, op, plane_normal, polygon};
use super::*;
use crate::model::sheet_metal::{
    section, FlatBendLine, FlatLayout, MaterialSide, Section, SheetMetalFeature, StripRegion,
};
use crate::refs::history::{FaceGeom, KernelTag};
use crate::refs::name::Role;
use crate::refs::naming::Restore;
use crate::regen::records::{FeatureRecord, SheetRules};

/// One edge of a miter flange's chain, oriented along the chain.
struct ChainEdge {
    index: u32,
    start: [f64; 3],
    end: [f64; 3],
    /// Outward normal of the thickness face along the edge (the section's x).
    out: [f64; 3],
}

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn miter_flange(
        &mut self,
        i: usize,
        id: &FeatureId,
        profile: &ProfileRef,
        edges: &[Reference],
        gap: &ValueExpr,
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let invalid = |me: &mut Self, param: &str, message: String| {
            me.error(id, Some(param), DiagCode::RegenInvalid, message, Vec::new())
        };
        let mismatch = |me: &mut Self, param: &str, message: &str| {
            me.error(
                id,
                Some(param),
                DiagCode::RefKindMismatch,
                message.into(),
                Vec::new(),
            )
        };
        let gap = self.value(id, "gap", gap, Dimension::LENGTH)?;
        if gap < 0.0 {
            return Err(invalid(
                self,
                "gap",
                "a miter gap cannot be negative".into(),
            ));
        }
        let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
        let operands = Self::bodies_of(&set);
        if operands.len() != 1 {
            return Err(mismatch(
                self,
                "edges",
                "miter flange edges must be on one sheet",
            ));
        }
        let body = operands[0];
        let Some((_, rules, _, _)) = self.sheet_of(body) else {
            return Err(mismatch(
                self,
                "edges",
                "the edges are not on a sheet metal body",
            ));
        };
        let (t, r) = (rules.thickness, rules.bend_radius);
        if r <= 0.0 {
            return Err(invalid(
                self,
                "edges",
                "a miter flange needs a sheet with a positive bend radius".into(),
            ));
        }
        // Each edge: straight, between a sheet face and a thickness face.
        let mut sheet_normal: Option<[f64; 3]> = None;
        let mut loose: Vec<ChainEdge> = Vec::new();
        for h in &set {
            let topo = &self.st.bodies[body].topology;
            let e = topo.edges[h.index as usize].clone();
            let chord = sub(e.geom.end, e.geom.start);
            let len = dot(chord, chord).sqrt();
            if len < 1e-12 || (e.geom.length - len).abs() > 1e-9 * len.max(1e-3) {
                return Err(mismatch(
                    self,
                    "edges",
                    "a miter flange runs along straight sheet edges",
                ));
            }
            let (ga, gb) = (
                topo.faces[e.faces[0] as usize].geom,
                topo.faces[e.faces[1] as usize].geom,
            );
            let (Some(na), Some(nb)) = (plane_normal(&ga), plane_normal(&gb)) else {
                return Err(mismatch(
                    self,
                    "edges",
                    "a miter flange edge lies between two flat faces",
                ));
            };
            let dir = scale(chord, 1.0 / len);
            let width = |g: &FaceGeom| {
                let ext = sub(g.bbox_max, g.bbox_min);
                let across = sub(ext, scale(dir, dot(ext, dir)));
                dot(across, across).sqrt()
            };
            let (n_sheet, out) = if width(&ga) >= width(&gb) {
                (na, nb)
            } else {
                (nb, na)
            };
            match sheet_normal {
                None => sheet_normal = Some(n_sheet),
                Some(n) if dot(n, n_sheet) > 1.0 - 1e-9 => {}
                Some(_) => {
                    return Err(mismatch(
                        self,
                        "edges",
                        "a miter flange's edges lie on one face of the sheet",
                    ))
                }
            }
            loose.push(ChainEdge {
                index: h.index,
                start: e.geom.start,
                end: e.geom.end,
                out,
            });
        }
        let Some(n_sheet) = sheet_normal else {
            return Err(mismatch(self, "edges", "a miter flange needs edges"));
        };
        let Some(chain) = order_chain(loose, 1e-7) else {
            return Err(invalid(
                self,
                "edges",
                "a miter flange's edges must join end to end in one chain".into(),
            ));
        };
        let closed = chain.len() > 2 && near(chain[0].start, chain[chain.len() - 1].end, 1e-7);
        // The profile, read in the section plane of the edge it starts on.
        let (sframe, segs) = self.chain(i, id, "profile", profile)?;
        if segs.iter().any(|s| !matches!(s.kind, SegmentKind::Line)) {
            return Err(self.error(
                id,
                Some("profile"),
                DiagCode::SketchOpenProfile,
                "a miter flange profile is drawn with lines; bends come from its corners".into(),
                Vec::new(),
            ));
        }
        let p0 = sframe.to_world(segs[0].start);
        let anchor = chain.iter().position(|e| {
            let d = normalize(sub(e.end, e.start));
            let along = dot(sub(p0, e.start), d);
            let off = sub(sub(p0, e.start), scale(d, along));
            let len = dot(sub(e.end, e.start), d);
            dot(off, off).sqrt() < 1e-7
                && along > -1e-7
                && along < len + 1e-7
                && dot(sframe.z, d).abs() > 1.0 - 1e-9
        });
        let Some(anchor) = anchor else {
            return Err(invalid(
                self,
                "profile",
                "the profile must start on one of the edges, drawn on a plane normal to it".into(),
            ));
        };
        let out0 = chain[anchor].out;
        let local = |p: [f64; 2]| {
            let w = sub(sframe.to_world(p), p0);
            [dot(w, out0), dot(w, n_sheet)]
        };
        // The section: the sheet's mold line, then the profile; material on the sheet's side.
        let span: f64 = segs
            .iter()
            .map(|s| (s.end[0] - s.start[0]).hypot(s.end[1] - s.start[1]))
            .sum();
        let lead = 1.0 + span + 1e3 * (r + t);
        let mut pts = vec![[-lead, 0.0], [0.0, 0.0]];
        pts.extend(segs.iter().map(|s| local(s.end)));
        let plate_curve = crate::ids::SketchEntityId::new(format!("{}.sheet", id.0));
        let mut curves = vec![plate_curve];
        curves.extend(segs.iter().map(|s| s.curve.clone()));
        let mut sec = section(&pts, &curves, MaterialSide::Right, t, r)
            .map_err(|m| invalid(self, "profile", m))?;
        if sec.bends.len() + 1 != sec.flats.len() {
            return Err(invalid(
                self,
                "profile",
                "the profile must turn at the sheet edge and at each of its corners".into(),
            ));
        }
        // Start the first bend at the sheet edge.
        let shift = -sec.flats[0].end[0];
        for f in &mut sec.flats {
            f.start[0] += shift;
            f.end[0] += shift;
        }
        for b in &mut sec.bends {
            b.center[0] += shift;
        }
        // How far the section reaches out of the sheet (for the mitred extensions).
        let reach = sec
            .flats
            .iter()
            .skip(1)
            .flat_map(|f| {
                [f.start, f.end]
                    .into_iter()
                    .flat_map(move |p| [f.near, f.far].map(|o| (p[0] + o * f.normal[0]).abs()))
            })
            .chain(
                sec.bends
                    .iter()
                    .map(|b| b.center[0].abs() + b.inner_radius + b.thickness),
            )
            .fold(0.0, f64::max);
        // The clip plane at each corner, as the side before it keeps it.
        let corners = chain.len() - if closed { 0 } else { 1 };
        let mut planes: Vec<([f64; 3], [f64; 3], f64)> = Vec::new();
        for k in 0..corners {
            let (a, b) = (&chain[k], &chain[(k + 1) % chain.len()]);
            let (da, db) = (
                normalize(sub(a.end, a.start)),
                normalize(sub(b.end, b.start)),
            );
            let turn = dot(da, db).clamp(-1.0, 1.0).acos();
            if turn > 170f64.to_radians() {
                return Err(invalid(
                    self,
                    "edges",
                    "a miter flange's edges turn by at most 170° at a corner".into(),
                ));
            }
            let bisector = add(a.out, b.out);
            let mut normal = normalize(cross(bisector, n_sheet));
            if dot(normal, da) < 0.0 {
                normal = scale(normal, -1.0);
            }
            // The distance along the edge the farthest section point needs past the corner.
            let ext = reach * (turn / 2.0).tan() + t;
            planes.push((a.end, normal, ext));
        }
        let mut tools = Vec::new();
        let n = chain.len();
        for (k, e) in chain.iter().enumerate() {
            let before = if k > 0 || closed {
                Some(&planes[(k + n - 1) % n])
            } else {
                None
            };
            let after = if k + 1 < n || closed {
                Some(&planes[k])
            } else {
                None
            };
            let fr = frame_of(e.start, e.out, n_sheet);
            let (s_a, s_b) = (0.0, dot(sub(e.end, e.start), fr.z));
            let sign = if s_b >= s_a { 1.0 } else { -1.0 };
            let za = s_a - sign * before.map_or(0.0, |p| p.2);
            let zb = s_b + sign * after.map_or(0.0, |p| p.2);
            let (z0, z1) = (za.min(zb), za.max(zb));
            let edge_tag = KernelTag::InputEdge {
                edge: OperandEntity {
                    operand: 0,
                    entity: e.index,
                },
            };
            let lab = |role: Role, o: u32| FaceLabel {
                role,
                tag: edge_tag.clone(),
                ordinal: Some(o),
            };
            let clips = |base: u32| {
                let mut v = Vec::new();
                if let Some((o, nrm, _)) = before {
                    // The side after the corner keeps the far half: the plane's normal turned.
                    let m = scale(*nrm, -1.0);
                    v.push(ClipPlane {
                        origin: sub(*o, scale(m, gap / 2.0)),
                        normal: m,
                        label: lab(Role::SheetEdge, base + 6),
                    });
                }
                if let Some((o, nrm, _)) = after {
                    v.push(ClipPlane {
                        origin: sub(*o, scale(*nrm, gap / 2.0)),
                        normal: *nrm,
                        label: lab(Role::SheetEdge, base + 7),
                    });
                }
                v
            };
            let last = sec.flats.len() - 1;
            for (m, f) in sec.flats.iter().enumerate().skip(1) {
                let j = (m - 1) as u32;
                let base = 16 * j;
                let off = |p: [f64; 2], o: f64| [p[0] + o * f.normal[0], p[1] + o * f.normal[1]];
                let corners = vec![
                    ("near".to_string(), off(f.start, f.near)),
                    ("b".to_string(), off(f.end, f.near)),
                    ("far".to_string(), off(f.end, f.far)),
                    ("a".to_string(), off(f.start, f.far)),
                ];
                let mut tool = LabeledTool::new(ToolShape::prism(polygon(fr, &corners), z0, z1));
                let sid = crate::ids::SketchEntityId::new;
                // Material lies right of the profile, so `far` (offset 0) is the face on the
                // profile's line.
                tool.sides
                    .push((sid("near"), lab(Role::SheetFace, base + 1)));
                tool.sides.push((sid("far"), lab(Role::SheetFace, base)));
                tool.sides.push((sid("a"), lab(Role::SheetEdge, base + 4)));
                tool.sides.push((
                    sid("b"),
                    lab(Role::SheetEdge, base + if m == last { 1 } else { 5 }),
                ));
                tool.start = Some(lab(Role::SheetEdge, base + 2));
                tool.end = Some(lab(Role::SheetEdge, base + 3));
                tool.clip = clips(base);
                tools.push(tool);
            }
            for (m, b) in sec.bends.iter().enumerate() {
                let base = 16 * m as u32 + 8;
                let mut tool = bend_tool(fr, b, 1.0, None);
                if let ToolShape::Prism { start, end, .. } = &mut tool.shape {
                    *start = z0;
                    *end = z1;
                }
                for (_, l) in tool.sides.iter_mut() {
                    *l = lab(l.role, base + l.ordinal.unwrap_or(0));
                }
                tool.start = Some(lab(Role::SheetEdge, base + 2));
                tool.end = Some(lab(Role::SheetEdge, base + 3));
                tool.clip = clips(base);
                tools.push(tool);
            }
        }
        Ok(op(
            labeled(tools, Vec::new(), Combine::Add, false),
            vec![body],
            naming.clone(),
        ))
    }
}

// ------------------------------------------------------------- plates with bends
//
// A sketched bend or jog keeps the plate's developed outline (its line is the bend centre
// line and the strip it consumes is its allowance), so the plate's flat pattern is the base
// flange's own profile with each bend's centre line on it. Each bend records its lines and
// the half of the plate it moved; a later bend must keep its own moved half clear of every
// earlier one, since it rebuilds its part of the plate from the base flange's outline.

impl<'a, 'b> Run<'a, 'b> {
    /// The bend features that made faces on `body`, besides the base flange `maker`, with
    /// their plate-bend records; `Err(feature)` names the first feature of another kind.
    fn plate_bends(
        &self,
        body: usize,
        maker: &FeatureId,
    ) -> Result<Vec<(FeatureId, Vec<FlatBendLine>)>, FeatureId> {
        let is_bend = |f: &FeatureId| {
            matches!(
                self.st.records.get(f),
                Some(FeatureRecord::PlateBend { base, .. }) if base == maker
            )
        };
        // Each face must answer to the plate or one of its bends (a fold's faces answer to
        // the names it restored as well as its own).
        let mut bends: Vec<&FeatureId> = Vec::new();
        for names in &self.st.bodies[body].names.faces {
            let mut accounted = false;
            for n in names {
                if n.maker() == maker {
                    accounted = true;
                } else if is_bend(n.maker()) {
                    accounted = true;
                    if !bends.contains(&n.maker()) {
                        bends.push(n.maker());
                    }
                }
            }
            if !accounted {
                if let Some(n) = names.iter().next() {
                    return Err(n.maker().clone());
                }
            }
        }
        let mut out = Vec::new();
        for f in &self.doc.features {
            if let Some(FeatureRecord::PlateBend { lines, .. }) = self.st.records.get(&f.id) {
                if bends.contains(&&f.id) {
                    out.push((f.id.clone(), lines.clone()));
                }
            }
        }
        Ok(out)
    }

    /// The sketched bends and jogs of plate `maker` made before feature `i`, in order.
    pub(super) fn bends_of_plate(&self, i: usize, maker: &FeatureId) -> Vec<FeatureId> {
        self.doc.features[..i]
            .iter()
            .filter(|f| {
                matches!(
                    self.st.records.get(&f.id),
                    Some(FeatureRecord::PlateBend { base, .. }) if base == maker
                ) && self.st.replays.contains_key(&f.id)
            })
            .map(|f| f.id.clone())
            .collect()
    }

    /// Adds to `flat` every bend that bends a flat a bend in `flat` moved: a bend of a moved
    /// flat cannot stay bent once that flat lies on the plate again.
    pub(super) fn close_flat(&self, flat: &mut Vec<FeatureId>) {
        loop {
            let more: Vec<FeatureId> = self
                .st
                .records
                .iter()
                .filter_map(|(f, r)| match r {
                    FeatureRecord::PlateBend {
                        parent: Some(p), ..
                    } if flat.contains(p) && !flat.contains(f) => Some(f.clone()),
                    _ => None,
                })
                .collect();
            if more.is_empty() {
                return;
            }
            flat.extend(more);
        }
    }

    /// `bends` with the bends whose moved flats they bend, recursively: folding a bend of a
    /// moved flat folds that flat's bend too.
    pub(super) fn with_parents(&self, bends: Vec<FeatureId>) -> Vec<FeatureId> {
        let mut out = bends;
        let mut k = 0;
        while k < out.len() {
            if let Some(FeatureRecord::PlateBend {
                parent: Some(p), ..
            }) = self.st.records.get(&out[k])
            {
                if !out.contains(p) {
                    out.push(p.clone());
                }
            }
            k += 1;
        }
        out
    }

    /// The bends of plate `maker` that the unfolds and folds before feature `i` left flat.
    pub(super) fn flattened_bends(&self, i: usize, maker: &FeatureId) -> Vec<FeatureId> {
        let all = self.bends_of_plate(i, maker);
        let mut flat: Vec<FeatureId> = Vec::new();
        let ours = |name: &EntityName| {
            let mut m = vec![name.maker().clone()];
            if let EntityName::Generated(g) = name {
                if let Some(step) = &g.step {
                    m.push(FeatureId::new(step));
                }
            }
            m.iter().any(|f| {
                f == maker
                    || matches!(
                        self.st.records.get(f),
                        Some(FeatureRecord::PlateBend { base, .. }) if base == maker
                    )
            })
        };
        // Whether the latest unfold was of this plate: a fold folds that one.
        let mut last_ours = false;
        for f in &self.doc.features[..i] {
            // Only unfolds and folds that ran change the state.
            let ran = self.st.records.contains_key(&f.id)
                || self
                    .st
                    .bodies
                    .iter()
                    .any(|b| b.names.faces.iter().flatten().any(|n| n.maker() == &f.id));
            if !ran {
                continue;
            }
            match &f.kind {
                FeatureKind::SheetMetal(SheetMetalFeature::Unfold { bends, fixed_face }) => {
                    last_ours = matches!(fixed_face, Reference::Name { name, .. } if ours(name));
                    if !last_ours {
                        continue;
                    }
                    let set = if bends.is_empty() {
                        all.clone()
                    } else {
                        Self::bend_ids_by_name(&self.st.records, maker, bends)
                    };
                    for b in set {
                        if !flat.contains(&b) {
                            flat.push(b);
                        }
                    }
                    self.close_flat(&mut flat);
                }
                FeatureKind::SheetMetal(SheetMetalFeature::Fold { bends }) => {
                    if !last_ours {
                        continue;
                    }
                    if bends.is_empty() {
                        flat.clear();
                    } else {
                        let set = self.with_parents(Self::bend_ids_by_name(
                            &self.st.records,
                            maker,
                            bends,
                        ));
                        flat.retain(|b| !set.contains(b));
                    }
                }
                _ => {}
            }
        }
        flat
    }

    /// The sketched bends and jogs of `maker` that `bends` name: a face a bend made (or a
    /// step of an unfold or fold made for that bend), by the name written in the reference.
    fn bend_ids_by_name(
        records: &BTreeMap<FeatureId, FeatureRecord>,
        maker: &FeatureId,
        bends: &[Reference],
    ) -> Vec<FeatureId> {
        let mut out = Vec::new();
        for r in bends {
            let Reference::Name { name, .. } = r else {
                continue;
            };
            let mut candidates = vec![name.maker().clone()];
            if let EntityName::Generated(g) = name {
                if let Some(step) = &g.step {
                    candidates.push(FeatureId::new(step));
                }
            }
            for c in candidates {
                if matches!(records.get(&c), Some(FeatureRecord::PlateBend { base, .. }) if base == maker)
                    && !out.contains(&c)
                {
                    out.push(c);
                }
            }
        }
        out
    }

    /// The bends `bends` name, each a sketched bend or jog of plate `maker`.
    pub(super) fn plate_bend_ids(
        &mut self,
        id: &FeatureId,
        maker: &FeatureId,
        bends: &[Reference],
    ) -> Result<Vec<FeatureId>, Halt> {
        let out = Self::bend_ids_by_name(&self.st.records, maker, bends);
        if out.len() != bends.len() {
            return Err(self.error(
                id,
                Some("bends"),
                DiagCode::RefKindMismatch,
                "each entry of `bends` names a face of one of the plate's sketched bends or jogs"
                    .into(),
                Vec::new(),
            ));
        }
        Ok(out)
    }

    /// Steps replaying the bends of plate `maker` that are not in `flat`, in order, on `body`.
    /// Each step is keyed by its bend and restores that bend's names.
    pub(super) fn replay_bends(
        &self,
        maker: &FeatureId,
        flat: &[FeatureId],
        body: usize,
        naming: &NamingContext,
    ) -> Vec<Step> {
        self.bends_of_plate(self.doc.features.len(), maker)
            .into_iter()
            .filter(|b| !flat.contains(b))
            .filter_map(|b| {
                Some(Step {
                    op: self.st.replays.get(&b)?.clone(),
                    operands: vec![body],
                    naming: naming.at_step(Some(b.0.clone()), vec![Restore::all(&b)]),
                })
            })
            .collect()
    }

    /// Unfolds the named bends of a plate and keeps the rest: the plate rebuilt flat in place,
    /// then the bends that stay bent replayed. A strip's partial unfold is not lowered.
    pub(super) fn unfold_some(
        &mut self,
        i: usize,
        id: &FeatureId,
        fixed: &EntityHandle,
        bends: &[Reference],
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let body = fixed.body as usize;
        if let Some((maker, _, Some(sec), _)) = self.sheet_of(body) {
            let fixed_key = self.st.entity_names(fixed).iter().find_map(|n| match n {
                EntityName::Generated(g) if g.feature == maker && g.role == Role::SheetFace => {
                    match &g.tag {
                        crate::refs::name::SourceTag::Key { key } => Some(key.clone()),
                        _ => None,
                    }
                }
                _ => None,
            });
            let Some(fixed_key) = fixed_key else {
                return Err(self.error(
                    id,
                    Some("fixed_face"),
                    DiagCode::RefKindMismatch,
                    "the fixed face is not a flat of the sheet".into(),
                    Vec::new(),
                ));
            };
            let Some(keys) = Self::strip_bend_keys(&sec, bends) else {
                return Err(self.error(
                    id,
                    Some("bends"),
                    DiagCode::RefKindMismatch,
                    "each entry of `bends` names a bend face of the sheet".into(),
                    Vec::new(),
                ));
            };
            return self.strip_some(i, id, body, &maker, Some(fixed_key), &keys, &[], naming);
        }
        let Some((maker, rules, None, _)) = self.sheet_of(body).or_else(|| {
            // A folded plate's body may answer to the fold instead of the base flange.
            let base =
                self.st.bodies[body]
                    .names
                    .faces
                    .iter()
                    .flatten()
                    .find_map(|n| match self.st.records.get(n.maker()) {
                        Some(FeatureRecord::Sheet { .. }) => Some(n.maker().clone()),
                        _ => None,
                    })?;
            match self.st.records.get(&base) {
                Some(FeatureRecord::Sheet {
                    rules,
                    section,
                    depth,
                }) => Some((base, rules.clone(), section.clone(), *depth)),
                _ => None,
            }
        }) else {
            return Err(self.not_lowered(
                id,
                "unfolding some bends of a sheet made by a bent base flange (unfold all of them: leave `bends` empty)",
            ));
        };
        let chosen = self.plate_bend_ids(id, &maker, bends)?;
        let mut flat = self.flattened_bends(i, &maker);
        for b in chosen {
            if !flat.contains(&b) {
                flat.push(b);
            }
        }
        self.close_flat(&mut flat);
        // The fixed face stays put: it lies on the plate's unmoved part or on a flat a bend
        // that stays bent moved.
        let names = self.st.entity_names(fixed);
        let stays = names.iter().any(|n| {
            n.maker() == &maker
                || matches!(
                    self.st.records.get(n.maker()),
                    Some(FeatureRecord::PlateBend { base, .. }) if base == &maker
                ) && !flat.contains(n.maker())
        });
        if !stays {
            return Err(self.not_lowered(
                id,
                "partial unfolds fixed on a face that moves (fix the unfold on the plate or on a flat of a bend that stays bent)",
            ));
        }
        if self.plate_bends(body, &maker).is_err() {
            return Err(self.not_lowered(
                id,
                "unfolding a plate that features besides its bends changed",
            ));
        }
        let Compiled::Op { op: develop, .. } =
            self.plate_flat(i, id, fixed, &maker, &rules, naming, false)?
        else {
            return Err(self.not_lowered(id, "unfolding this plate"));
        };
        // The record lists only the bends this unfold left flat.
        if let Some(FeatureRecord::FlatPattern { layout, .. }) = self.st.records.get_mut(id) {
            layout.bend_lines.retain(|l| flat.contains(&l.feature));
        }
        let Compiled::Op {
            op: develop,
            operands,
            ..
        } = self.in_place(
            Compiled::Op {
                op: develop,
                operands: Vec::new(),
                naming: naming.clone(),
                import: false,
                instances: None,
                tools: None,
            },
            body,
        )
        else {
            return Err(self.not_lowered(id, "unfolding this plate"));
        };
        let mut steps = vec![Step {
            op: develop,
            operands,
            naming: naming.clone(),
        }];
        steps.extend(self.replay_bends(&maker, &flat, body, naming));
        Ok(Compiled::Steps(steps))
    }

    /// Records a sketched bend or jog of the plate `maker` on `body`: refuses one whose moved
    /// half of the plate overlaps a half an earlier bend moved.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_plate_bend(
        &mut self,
        id: &FeatureId,
        body: usize,
        maker: &FeatureId,
        plate: &CanonicalProfile,
        lines: Vec<FlatBendLine>,
        moved: HalfPlane,
        motion: Transform,
        flat: CanonicalProfile,
        parent: Option<FeatureId>,
    ) -> Result<(), Halt> {
        // A bend of a moved flat moves part of what its parent (and the parent's parents)
        // moved: only the other bends must keep clear.
        let mut ancestors: Vec<FeatureId> = Vec::new();
        let mut up = parent.clone();
        while let Some(p) = up {
            up = match self.st.records.get(&p) {
                Some(FeatureRecord::PlateBend { parent, .. }) => parent.clone(),
                _ => None,
            };
            ancestors.push(p);
        }
        let mut earlier: Vec<(FeatureId, HalfPlane)> = Vec::new();
        for n in self.st.bodies[body].names.faces.iter().flatten() {
            if let Some(FeatureRecord::PlateBend { base, moved, .. }) =
                self.st.records.get(n.maker())
            {
                if base == maker
                    && !ancestors.contains(n.maker())
                    && !earlier.iter().any(|(f, _)| f == n.maker())
                {
                    earlier.push((n.maker().clone(), *moved));
                }
            }
        }
        let outline: Vec<[f64; 2]> = plate
            .loops
            .first()
            .map(|l| l.iter().map(|s| s.start).collect())
            .unwrap_or_default();
        for (f, other) in earlier {
            if half_planes_overlap_in(&outline, moved, other) {
                return Err(self.not_lowered(
                    id,
                    &format!(
                        "a bend of plate material `{f}` already moved (each sketched bend or jog moves a part of the plate no other bend moved)"
                    ),
                ));
            }
        }
        self.st.records.insert(
            id.clone(),
            FeatureRecord::PlateBend {
                base: maker.clone(),
                lines,
                moved,
                motion,
                flat,
                parent,
            },
        );
        Ok(())
    }

    /// The flat pattern of a plate with sketched bends and jogs: the base flange's profile,
    /// in place, with the bends' centre lines.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plate_flat(
        &mut self,
        i: usize,
        id: &FeatureId,
        fixed: &EntityHandle,
        maker: &FeatureId,
        rules: &SheetRules,
        naming: &NamingContext,
        follow_fixed: bool,
    ) -> Result<Compiled, Halt> {
        let body = fixed.body as usize;
        let names = self.st.entity_names(fixed);
        // Fixed on the plate's unmoved part, the flat lies in place; fixed on a flat a bend
        // moved, the developed plate moves with that bend so the fixed face stays put.
        let motion =
            if !follow_fixed || names.iter().any(|n| n.maker() == maker) {
                None
            } else {
                let moved_by = names
                    .iter()
                    .find_map(|n| match self.st.records.get(n.maker()) {
                        Some(FeatureRecord::PlateBend { base, motion, .. }) if base == maker => {
                            Some(*motion)
                        }
                        _ => None,
                    });
                match moved_by {
                    Some(m) => Some(m),
                    None => return Err(self.not_lowered(
                        id,
                        "flat patterns fixed on a bend face (fix the flat on a flat of the plate)",
                    )),
                }
            };
        let bends = match self.plate_bends(body, maker) {
            Ok(b) => b,
            Err(other) => {
                return Err(self.not_lowered(
                    id,
                    &format!(
                        "flat patterns of a plate that `{other}` changed (only sketched bends and jogs develop back to the plate)"
                    ),
                ))
            }
        };
        let profile_ref = match self.doc.feature(maker).map(|f| &f.kind) {
            Some(FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { profile, .. })) => {
                profile.clone()
            }
            _ => return Err(self.not_lowered(id, "flat patterns of this sheet")),
        };
        let plate = self.profile(i, id, "fixed_face", &profile_ref)?;
        let t = rules.thickness;
        let mut tool = LabeledTool::new(ToolShape::prism(plate.clone(), 0.0, t));
        for seg in plate.loops.iter().flatten() {
            tool.sides.push((
                seg.curve.clone(),
                FaceLabel::key(Role::SheetEdge, seg.curve.0.clone()),
            ));
        }
        tool.start = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(0));
        tool.end = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(1));
        let layout = FlatLayout {
            frame: plate.frame,
            depth: t,
            loops: plate.loops.clone(),
            bend_lines: bends.into_iter().flat_map(|(_, l)| l).collect(),
        };
        self.st.records.insert(
            id.clone(),
            FeatureRecord::FlatPattern {
                layout,
                development: None,
            },
        );
        let flat = op(
            labeled(vec![tool], Vec::new(), Combine::New, false),
            Vec::new(),
            naming.clone(),
        );
        let Some(motion) = motion else {
            return Ok(flat);
        };
        // Two steps: the developed plate, then the move that keeps the fixed flat in place.
        let Compiled::Op { op: develop, .. } = flat else {
            return Ok(flat);
        };
        if let Some(FeatureRecord::FlatPattern { layout, .. }) = self.st.records.get_mut(id) {
            layout.frame = Frame {
                origin: motion.apply(layout.frame.origin),
                x: rotate_by(&motion, layout.frame.x),
                y: rotate_by(&motion, layout.frame.y),
                z: rotate_by(&motion, layout.frame.z),
            };
        }
        let new_body = self.st.bodies.len();
        Ok(Compiled::Steps(vec![
            Step {
                op: develop,
                operands: Vec::new(),
                naming: naming.clone(),
            },
            Step {
                op: KernelOp::Transform { transform: motion },
                operands: vec![new_body],
                naming: naming.at_step(Some("place".into()), Vec::new()),
            },
        ]))
    }
}

/// A direction under a rigid motion's rotation.
fn rotate_by(t: &Transform, v: [f64; 3]) -> [f64; 3] {
    let r = &t.rotation;
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

/// A half-plane `(p − origin)·normal ≥ 0` of a plate's frame: `(origin, normal)`.
type HalfPlane = ([f64; 2], [f64; 2]);

/// Whether the half-planes `(p − o)·n ≥ 0` of `a` and `b` share an area inside the polygon
/// `outline` (its vertices; arcs count by their ends).
fn half_planes_overlap_in(outline: &[[f64; 2]], a: HalfPlane, b: HalfPlane) -> bool {
    let tol = 1e-9;
    let side =
        |h: ([f64; 2], [f64; 2]), p: [f64; 2]| (p[0] - h.0[0]) * h.1[0] + (p[1] - h.0[1]) * h.1[1];
    // The region's corners: outline vertices, where each line crosses the outline, and where
    // the two lines cross.
    let mut pts: Vec<[f64; 2]> = outline.to_vec();
    let n = outline.len();
    for h in [a, b] {
        for k in 0..n {
            let (p, q) = (outline[k], outline[(k + 1) % n]);
            let (sp, sq) = (side(h, p), side(h, q));
            if (sp < 0.0) != (sq < 0.0) && (sp - sq).abs() > 0.0 {
                let f = sp / (sp - sq);
                pts.push([p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])]);
            }
        }
    }
    let den = a.1[0] * b.1[1] - a.1[1] * b.1[0];
    if den.abs() > 1e-12 {
        // Lines n·p = n·o: solve the 2 × 2 system.
        let (ca, cb) = (
            a.1[0] * a.0[0] + a.1[1] * a.0[1],
            b.1[0] * b.0[0] + b.1[1] * b.0[1],
        );
        let x = [
            (ca * b.1[1] - cb * a.1[1]) / den,
            (a.1[0] * cb - b.1[0] * ca) / den,
        ];
        if inside(outline, x) {
            pts.push(x);
        }
    }
    let both: Vec<[f64; 2]> = pts
        .into_iter()
        .filter(|p| side(a, *p) >= -tol && side(b, *p) >= -tol)
        .collect();
    // A shared area has three corners that are not in a line.
    let scale = outline
        .iter()
        .map(|p| p[0].abs().max(p[1].abs()))
        .fold(1e-3, f64::max);
    let min_area = 1e-6 * scale * scale;
    for i in 0..both.len() {
        for j in i + 1..both.len() {
            for k in j + 1..both.len() {
                let (p, q, r) = (both[i], both[j], both[k]);
                let area =
                    0.5 * ((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])).abs();
                if area > min_area {
                    return true;
                }
            }
        }
    }
    false
}

/// Point in polygon (even-odd).
fn inside(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
    let mut c = false;
    let n = poly.len();
    for k in 0..n {
        let (a, b) = (poly[k], poly[(k + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        {
            c = !c;
        }
    }
    c
}

fn near(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    let d = sub(a, b);
    dot(d, d).sqrt() <= tol
}

/// Orders edges into one chain, end to end: an open chain starts at its free end; a closed
/// one at the first edge given. `None` when they do not form a single chain.
fn order_chain(mut loose: Vec<ChainEdge>, tol: f64) -> Option<Vec<ChainEdge>> {
    if loose.is_empty() {
        return None;
    }
    let touches = |p: [f64; 3], e: &ChainEdge| near(p, e.start, tol) || near(p, e.end, tol);
    // A free end: an endpoint no other edge shares.
    let mut first = 0;
    let mut flip = false;
    'find: for (k, e) in loose.iter().enumerate() {
        for (end, p) in [(false, e.start), (true, e.end)] {
            if !loose
                .iter()
                .enumerate()
                .any(|(j, o)| j != k && touches(p, o))
            {
                first = k;
                flip = end;
                break 'find;
            }
        }
    }
    let mut e = loose.remove(first);
    if flip {
        std::mem::swap(&mut e.start, &mut e.end);
    }
    let mut chain = vec![e];
    while !loose.is_empty() {
        let tip = chain.last().expect("non-empty").end;
        let found: Vec<usize> = (0..loose.len())
            .filter(|j| touches(tip, &loose[*j]))
            .collect();
        if found.len() != 1 {
            return None;
        }
        let mut e = loose.remove(found[0]);
        if !near(e.start, tip, tol) {
            std::mem::swap(&mut e.start, &mut e.end);
        }
        chain.push(e);
    }
    Some(chain)
}

/// A rigid motion of the section plane: `p ↦ R(angle) p + shift`.
#[derive(Clone, Copy, Debug)]
struct Motion2 {
    angle: f64,
    shift: [f64; 2],
}

impl Motion2 {
    const IDENTITY: Motion2 = Motion2 {
        angle: 0.0,
        shift: [0.0, 0.0],
    };
    fn turn(&self, v: [f64; 2]) -> [f64; 2] {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        [c * v[0] - s * v[1], s * v[0] + c * v[1]]
    }
    fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        let q = self.turn(p);
        [q[0] + self.shift[0], q[1] + self.shift[1]]
    }
    /// The motion that turns direction `d` to `d2` and carries `p` to `p2`.
    fn carrying(p: [f64; 2], d: [f64; 2], p2: [f64; 2], d2: [f64; 2]) -> Motion2 {
        let angle = d2[1].atan2(d2[0]) - d[1].atan2(d[0]);
        let m = Motion2 {
            angle,
            shift: [0.0, 0.0],
        };
        let q = m.turn(p);
        Motion2 {
            angle,
            shift: [p2[0] - q[0], p2[1] - q[1]],
        }
    }
}

impl<'a, 'b> Run<'a, 'b> {
    /// The keys of the bends of the bent sheet `maker` that `bends` name: faces any bend
    /// region carries (`BendFace` tagged by the bend key), made by the base flange or by an
    /// unfold or fold of it.
    pub(super) fn strip_bend_keys(sec: &Section, bends: &[Reference]) -> Option<Vec<String>> {
        let mut out = Vec::new();
        for r in bends {
            let Reference::Name {
                name: EntityName::Generated(g),
                ..
            } = r
            else {
                return None;
            };
            let crate::refs::name::SourceTag::Key { key } = &g.tag else {
                return None;
            };
            if g.role != Role::BendFace || !sec.bends.iter().any(|b| &b.key == key) {
                return None;
            }
            if !out.contains(key) {
                out.push(key.clone());
            }
        }
        Some(out)
    }

    /// The bend keys of the bent sheet `maker` the unfolds and folds before feature `i` left
    /// flat, and the flat the latest unfold fixed.
    pub(super) fn strip_flattened(
        &self,
        i: usize,
        maker: &FeatureId,
        sec: &Section,
    ) -> (Vec<String>, Option<String>) {
        let all: Vec<String> = sec.bends.iter().map(|b| b.key.clone()).collect();
        let mut flat: Vec<String> = Vec::new();
        let mut fixed = None;
        let mut last_ours = false;
        for f in &self.doc.features[..i] {
            let ran = self.st.records.contains_key(&f.id)
                || self
                    .st
                    .bodies
                    .iter()
                    .any(|b| b.names.faces.iter().flatten().any(|n| n.maker() == &f.id));
            if !ran {
                continue;
            }
            match &f.kind {
                FeatureKind::SheetMetal(SheetMetalFeature::Unfold { fixed_face, bends }) => {
                    let key = match fixed_face {
                        Reference::Name {
                            name: EntityName::Generated(g),
                            ..
                        } if g.role == Role::SheetFace => match &g.tag {
                            crate::refs::name::SourceTag::Key { key }
                                if sec.flats.iter().any(|x| &x.curve.0 == key) =>
                            {
                                Some(key.clone())
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    let on_sheet =
                        matches!(fixed_face, Reference::Name { name, .. } if name.maker() == maker);
                    last_ours = on_sheet && key.is_some();
                    if !last_ours {
                        continue;
                    }
                    fixed = key;
                    let set = if bends.is_empty() {
                        all.clone()
                    } else {
                        Self::strip_bend_keys(sec, bends).unwrap_or_default()
                    };
                    for k in set {
                        if !flat.contains(&k) {
                            flat.push(k);
                        }
                    }
                }
                FeatureKind::SheetMetal(SheetMetalFeature::Fold { bends }) if last_ours => {
                    if bends.is_empty() {
                        flat.clear();
                    } else {
                        let set = Self::strip_bend_keys(sec, bends).unwrap_or_default();
                        flat.retain(|k| !set.contains(k));
                    }
                }
                _ => {}
            }
        }
        (flat, fixed)
    }

    /// A partial unfold or fold of a bent sheet, rebuilt in place: the bends keyed in `add`
    /// are laid flat and those in `remove` bent again, on top of what the unfolds and folds
    /// before feature `i` left. The flat `fixed` stays put (a fold keeps the flat the latest
    /// unfold fixed).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn strip_some(
        &mut self,
        i: usize,
        id: &FeatureId,
        body: usize,
        maker: &FeatureId,
        fixed: Option<String>,
        add: &[String],
        remove: &[String],
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let Some(FeatureRecord::Sheet {
            rules,
            section: Some(sec),
            depth,
        }) = self.st.records.get(maker).cloned()
        else {
            return Err(self.not_lowered(id, "partial unfolds and folds of this sheet"));
        };
        let rebuilders = self.sheet_rebuilders(i, maker);
        self.only_made_by(id, body, &rebuilders, "a partial unfold or fold")?;
        let (mut flat, last_fixed) = self.strip_flattened(i, maker, &sec);
        let Some(fixed) = fixed.or(last_fixed) else {
            return Err(self.error(
                id,
                Some("fixed_face"),
                DiagCode::RefKindMismatch,
                "the fixed face is not a flat of the sheet".into(),
                Vec::new(),
            ));
        };
        for k in add {
            if !flat.contains(k) {
                flat.push(k.clone());
            }
        }
        flat.retain(|k| !remove.contains(k));
        let (op, naming) =
            self.strip_state(id, maker, &rules, &sec, depth, &fixed, &flat, naming)?;
        Ok(self.in_place(
            Compiled::Op {
                op,
                operands: Vec::new(),
                naming,
                import: false,
                instances: None,
                tools: None,
            },
            body,
        ))
    }

    /// The base flange and the unfolds and folds before feature `i`: what may have made the
    /// faces of a sheet a partial unfold or fold rebuilds.
    pub(super) fn sheet_rebuilders(&self, i: usize, maker: &FeatureId) -> Vec<FeatureId> {
        let mut out = vec![maker.clone()];
        for f in &self.doc.features[..i] {
            if matches!(
                f.kind,
                FeatureKind::SheetMetal(SheetMetalFeature::Unfold { .. })
                    | FeatureKind::SheetMetal(SheetMetalFeature::Fold { .. })
            ) {
                out.push(f.id.clone());
            }
        }
        out
    }

    /// A bent sheet with the bends keyed in `flat` laid out flat and the others bent: the flat
    /// `fixed` stays where the base flange put it, and each flattened bend becomes a straight
    /// strip of its allowance. Faces of the flats and of the bends that stay bent answer to the
    /// base flange's names again; the flattened strips are this feature's own.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn strip_state(
        &mut self,
        id: &FeatureId,
        maker: &FeatureId,
        rules: &SheetRules,
        sec: &Section,
        depth: f64,
        fixed: &str,
        flat: &[String],
        naming: &NamingContext,
    ) -> Result<(KernelOp, NamingContext), Halt> {
        let dev = crate::model::sheet_metal::develop(sec, &rules.rule)
            .map_err(|m| self.error(id, None, DiagCode::RegenInvalid, m, Vec::new()))?;
        let Some((_, frame)) = self.st.sketches.get(&self.sketch_of(maker)).cloned() else {
            return Err(self.error(
                id,
                None,
                DiagCode::RegenInvalid,
                "the sheet's sketch is gone".into(),
                Vec::new(),
            ));
        };
        let Some(k0) = sec.flats.iter().position(|f| f.curve.0 == fixed) else {
            return Err(self.not_lowered(id, "partial unfolds of this sheet"));
        };
        let allowance = |key: &str| {
            dev.regions.iter().find_map(|r| match r {
                StripRegion::Bend { key: k, length, .. } if k == key => Some(*length),
                _ => None,
            })
        };
        let dir = |f: &crate::model::sheet_metal::SectionFlat| {
            let l = f.length().max(1e-300);
            [(f.end[0] - f.start[0]) / l, (f.end[1] - f.start[1]) / l]
        };
        // Each flat's motion from its bent place, outward from the fixed flat.
        let n = sec.flats.len();
        let mut motion = vec![Motion2::IDENTITY; n];
        for k in k0 + 1..n {
            let b = &sec.bends[k - 1];
            motion[k] = if flat.contains(&b.key) {
                let prev = &sec.flats[k - 1];
                let d = motion[k - 1].turn(dir(prev));
                let e = motion[k - 1].apply(prev.end);
                let ba = allowance(&b.key).unwrap_or(0.0);
                Motion2::carrying(
                    sec.flats[k].start,
                    dir(&sec.flats[k]),
                    [e[0] + ba * d[0], e[1] + ba * d[1]],
                    d,
                )
            } else {
                motion[k - 1]
            };
        }
        for k in (0..k0).rev() {
            let b = &sec.bends[k];
            motion[k] = if flat.contains(&b.key) {
                let next = &sec.flats[k + 1];
                let d = motion[k + 1].turn(dir(next));
                let s = motion[k + 1].apply(next.start);
                let ba = allowance(&b.key).unwrap_or(0.0);
                Motion2::carrying(
                    sec.flats[k].end,
                    dir(&sec.flats[k]),
                    [s[0] - ba * d[0], s[1] - ba * d[1]],
                    d,
                )
            } else {
                motion[k + 1]
            };
        }
        let mut tools = Vec::new();
        for (k, f) in sec.flats.iter().enumerate() {
            let m = motion[k];
            let off =
                |p: [f64; 2], o: f64| m.apply([p[0] + o * f.normal[0], p[1] + o * f.normal[1]]);
            let c = f.curve.0.clone();
            let corners = vec![
                (format!("{}.near", c), off(f.start, f.near)),
                (format!("{}.b", c), off(f.end, f.near)),
                (format!("{}.far", c), off(f.end, f.far)),
                (format!("{}.a", c), off(f.start, f.far)),
            ];
            let mut tool = LabeledTool::new(ToolShape::prism(polygon(frame, &corners), 0.0, depth));
            let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{}.{}", c, s));
            tool.sides.push((
                sid("near"),
                FaceLabel::key(Role::SheetFace, c.clone()).with_ordinal(0),
            ));
            tool.sides.push((
                sid("far"),
                FaceLabel::key(Role::SheetFace, c.clone()).with_ordinal(1),
            ));
            tool.sides.push((
                sid("a"),
                FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(if k == 0 { 0 } else { 4 }),
            ));
            tool.sides.push((
                sid("b"),
                FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(if k + 1 == n {
                    1
                } else {
                    5
                }),
            ));
            tool.start = Some(FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(2));
            tool.end = Some(FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(3));
            tools.push(tool);
        }
        let mut kept: BTreeSet<String> = sec.flats.iter().map(|f| f.curve.0.clone()).collect();
        for (j, b) in sec.bends.iter().enumerate() {
            // A bend joins flats j and j + 1; it moves with the flat nearer the fixed one.
            let m = if j < k0 { motion[j + 1] } else { motion[j] };
            if flat.contains(&b.key) {
                let prev = &sec.flats[j];
                let (d, e) = if j < k0 {
                    let next = &sec.flats[j + 1];
                    let d = motion[j + 1].turn(dir(next));
                    let s = motion[j + 1].apply(next.start);
                    let ba = allowance(&b.key).unwrap_or(0.0);
                    (d, [s[0] - ba * d[0], s[1] - ba * d[1]])
                } else {
                    (motion[j].turn(dir(prev)), motion[j].apply(prev.end))
                };
                let nn = [-d[1], d[0]];
                let side = motion[j].turn(prev.normal);
                let sgn = if nn[0] * side[0] + nn[1] * side[1] >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let ba = allowance(&b.key).unwrap_or(0.0);
                let at = |x: f64, o: f64| {
                    [
                        e[0] + x * d[0] + sgn * o * nn[0],
                        e[1] + x * d[1] + sgn * o * nn[1],
                    ]
                };
                let key = b.key.clone();
                let corners = vec![
                    (format!("{}.near", key), at(0.0, prev.near)),
                    (format!("{}.x1", key), at(ba, prev.near)),
                    (format!("{}.far", key), at(ba, prev.far)),
                    (format!("{}.x0", key), at(0.0, prev.far)),
                ];
                let mut tool =
                    LabeledTool::new(ToolShape::prism(polygon(frame, &corners), 0.0, depth));
                let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{}.{}", key, s));
                for (s, label) in [
                    (
                        "near",
                        FaceLabel::key(Role::BendFace, key.clone()).with_ordinal(0),
                    ),
                    (
                        "far",
                        FaceLabel::key(Role::BendFace, key.clone()).with_ordinal(1),
                    ),
                    (
                        "x0",
                        FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(0),
                    ),
                    (
                        "x1",
                        FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(1),
                    ),
                ] {
                    tool.sides.push((sid(s), label));
                }
                tool.start = Some(FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(2));
                tool.end = Some(FaceLabel::key(Role::SheetEdge, key).with_ordinal(3));
                tools.push(tool);
            } else {
                let mut moved = b.clone();
                moved.center = m.apply(b.center);
                moved.from += m.angle;
                moved.to += m.angle;
                tools.push(super::families::bend_tool(frame, &moved, depth, None));
                kept.insert(b.key.clone());
            }
        }
        let restore = vec![Restore {
            feature: maker.clone(),
            keys: Some(kept),
        }];
        Ok((
            labeled(tools, Vec::new(), Combine::New, false),
            naming.at_step(None, restore),
        ))
    }
}
