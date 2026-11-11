//! Tabs and slots: rectangular tabs grown out of one sheet's edge, and matching slots cut
//! through the sheet the tabs reach into.
//!
//! The tab edge lies between a sheet face and the thickness face at the sheet's end. The tabs
//! stand on that end face, as thick as the tab sheet, `count` of them spaced evenly along the
//! edge (each centred in its share of the edge), each `length` long. The slot face is a face of
//! the other sheet, parallel to the end face and beyond it; the tabs reach through that sheet
//! to its far face. Each slot is its tab's footprint grown by `clearance` on every side, cut
//! through the slot sheet.
//!
//! Faces are named by the feature's key: `tab<k>` for tab `k`'s sides (ordinals 2 to 5 round
//! its outline, 1 its tip) and `slot<k>` for slot `k`'s walls.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::regen::engine) fn tab_and_slot(
        &mut self,
        i: usize,
        id: &FeatureId,
        edge: &Reference,
        slot_face: &Reference,
        count: f64,
        length: f64,
        clearance: f64,
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let mismatch = |me: &mut Self, param: &str, message: &str| {
            me.error(
                id,
                Some(param),
                DiagCode::RefKindMismatch,
                message.into(),
                Vec::new(),
            )
        };
        let invalid = |me: &mut Self, param: &str, message: String| {
            me.error(id, Some(param), DiagCode::RegenInvalid, message, Vec::new())
        };
        if count < 1.0 || count.fract() != 0.0 {
            return Err(invalid(
                self,
                "count",
                "the tab count is a whole number of at least 1".into(),
            ));
        }
        if length <= 0.0 {
            return Err(invalid(
                self,
                "length",
                "a tab's length must be positive".into(),
            ));
        }
        if clearance < 0.0 {
            return Err(invalid(
                self,
                "clearance",
                "a slot's clearance cannot be negative".into(),
            ));
        }
        let set: Vec<EntityHandle> = self
            .list(
                i,
                id,
                "edge",
                std::slice::from_ref(edge),
                Some(TopoKind::Edge),
            )?
            .into_iter()
            .collect();
        let [h] = set.as_slice() else {
            return Err(mismatch(self, "edge", "pick one edge for the tabs"));
        };
        let tab_body = h.body as usize;
        let Some((_, tab_rules, _, _)) = self.sheet_of(tab_body) else {
            return Err(mismatch(
                self,
                "edge",
                "the tab edge is not on a sheet metal body",
            ));
        };
        let topo = &self.st.bodies[tab_body].topology;
        let e = topo.edges[h.index as usize].clone();
        let chord = sub(e.geom.end, e.geom.start);
        let len = dot(chord, chord).sqrt();
        if len < 1e-12 || (e.geom.length - len).abs() > 1e-9 * len.max(1e-3) {
            return Err(mismatch(self, "edge", "tabs grow from a straight edge"));
        }
        let dir = scale(chord, 1.0 / len);
        let (ga, gb) = (
            topo.faces[e.faces[0] as usize].geom,
            topo.faces[e.faces[1] as usize].geom,
        );
        let (Some(na), Some(nb)) = (plane_normal(&ga), plane_normal(&gb)) else {
            return Err(mismatch(
                self,
                "edge",
                "the tab edge lies between two flat faces",
            ));
        };
        // The end face is the narrower of the two across the edge.
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
        let t1 = tab_rules.thickness;
        let s = self.face_handle(i, id, "slot_face", slot_face)?;
        let slot_body = s.body as usize;
        if slot_body == tab_body {
            return Err(mismatch(
                self,
                "slot_face",
                "the slot face is on another sheet than the tab edge",
            ));
        }
        let Some((_, slot_rules, _, _)) = self.sheet_of(slot_body) else {
            return Err(mismatch(
                self,
                "slot_face",
                "the slot face is not on a sheet metal body",
            ));
        };
        let sg = self.st.bodies[slot_body].topology.faces[s.index as usize].geom;
        let Some(ns) = plane_normal(&sg) else {
            return Err(mismatch(self, "slot_face", "the slot face is flat"));
        };
        if dot(ns, out).abs() < 1.0 - 1e-6 {
            return Err(mismatch(
                self,
                "slot_face",
                "the slot face is parallel to the end face the tabs grow from",
            ));
        }
        // Where the slot sheet lies along the tabs' direction, from the end face.
        let t2 = slot_rules.thickness;
        let at = dot(sub(sg.centroid, e.geom.start), out);
        let (near, far) = if dot(ns, out) < 0.0 {
            (at, at + t2)
        } else {
            (at - t2, at)
        };
        if near < -1e-7 {
            return Err(mismatch(
                self,
                "slot_face",
                "the slot sheet lies beyond the end face the tabs grow from",
            ));
        }
        let n = count as usize;
        if count * length > len + 1e-9 {
            return Err(invalid(
                self,
                "length",
                format!(
                    "{n} tabs of {:.3} mm don't fit on the {:.3} mm edge",
                    length * 1e3,
                    len * 1e3
                ),
            ));
        }
        // The tabs' frame: x along the edge, y into the sheet across the end face, z out.
        let frame = frame_of(e.geom.start, dir, scale(n_sheet, -1.0));
        let sign = if dot(frame.z, out) >= 0.0 { 1.0 } else { -1.0 };
        let span = |lo: f64, hi: f64| if sign > 0.0 { (lo, hi) } else { (-hi, -lo) };
        let eps = 1e-6;
        let mut tabs = Vec::new();
        let mut slots = Vec::new();
        for k in 0..n {
            let c = len * (k as f64 + 0.5) / count;
            let (x0, x1) = (c - length / 2.0, c + length / 2.0);
            let key = format!("{}.tab{k}", id.0);
            let corners = vec![
                (format!("{key}.0"), [x0, 0.0]),
                (format!("{key}.1"), [x1, 0.0]),
                (format!("{key}.2"), [x1, t1]),
                (format!("{key}.3"), [x0, t1]),
            ];
            let (z0, z1) = span(-eps, far);
            let mut tab = LabeledTool::new(ToolShape::prism(polygon(frame, &corners), z0, z1));
            for (m, (curve, _)) in corners.iter().enumerate() {
                tab.sides.push((
                    crate::ids::SketchEntityId::new(curve),
                    FaceLabel::key(Role::Side, key.clone()).with_ordinal(m as u32 + 2),
                ));
            }
            let tip = FaceLabel::key(Role::Side, key.clone()).with_ordinal(1);
            if sign > 0.0 {
                tab.end = Some(tip);
            } else {
                tab.start = Some(tip);
            }
            tabs.push(tab);
            let key = format!("{}.slot{k}", id.0);
            let g = clearance;
            let corners = vec![
                (format!("{key}.0"), [x0 - g, -g]),
                (format!("{key}.1"), [x1 + g, -g]),
                (format!("{key}.2"), [x1 + g, t1 + g]),
                (format!("{key}.3"), [x0 - g, t1 + g]),
            ];
            let (z0, z1) = span(near - eps, far + eps);
            let mut slot = LabeledTool::new(ToolShape::prism(polygon(frame, &corners), z0, z1));
            for (m, (curve, _)) in corners.iter().enumerate() {
                slot.sides.push((
                    crate::ids::SketchEntityId::new(curve),
                    FaceLabel::key(Role::Relief, key.clone()).with_ordinal(m as u32),
                ));
            }
            slots.push(slot);
        }
        Ok(Compiled::Steps(vec![
            Step {
                op: labeled(tabs, Vec::new(), Combine::Add, false),
                operands: vec![tab_body],
                naming: naming.at_step(Some("tabs".into()), Vec::new()),
            },
            Step {
                op: labeled(slots, Vec::new(), Combine::Cut, false),
                operands: vec![slot_body],
                naming: naming.at_step(Some("slots".into()), Vec::new()),
            },
        ]))
    }
}
