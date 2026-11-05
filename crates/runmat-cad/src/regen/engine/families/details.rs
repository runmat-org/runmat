//! Detail features: cosmetic threads, emboss and deboss, and ribs.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn thread(
        &mut self,
        i: usize,
        id: &FeatureId,
        t: &Thread,
    ) -> Result<Compiled, Halt> {
        let h = self.face_handle(i, id, "face", &t.face)?;
        if !matches!(
            self.st.bodies[h.body as usize].topology.faces[h.index as usize]
                .geom
                .surface,
            SurfaceKind::Cylinder { .. }
        ) {
            return Err(self.error(
                id,
                Some("face"),
                DiagCode::RefKindMismatch,
                "a thread goes on a cylindrical face".into(),
                Vec::new(),
            ));
        }
        if t.modelled {
            return Err(self.not_lowered(id, "modelled (helical) threads"));
        }
        let pitch = self.value(id, "spec.pitch", &t.spec.pitch, Dimension::LENGTH)?;
        let depth = self.value(id, "spec.depth", &t.spec.depth, Dimension::LENGTH)?;
        let length = match &t.length {
            Some(l) => Some(self.value(id, "length", l, Dimension::LENGTH)?),
            None => None,
        };
        Ok(Compiled::Record(FeatureRecord::CosmeticThread {
            face: self.st.entity_names(&h),
            spec_designation: t.spec.designation.clone(),
            pitch,
            depth,
            length,
        }))
    }

    pub(super) fn emboss(
        &mut self,
        i: usize,
        id: &FeatureId,
        e: &Emboss,
    ) -> Result<Compiled, Halt> {
        let h = self.face_handle(i, id, "face", &e.face)?;
        if self.face_frame(&h).is_none() {
            return Err(self.not_lowered(id, "wrapping onto curved faces"));
        }
        let p = self.profile(i, id, "profile", &e.profile)?;
        let n = self.face_normal(&h).expect("planar");
        let height = self.value(id, "height", &e.height, Dimension::LENGTH)?;
        let g = self.st.bodies[h.body as usize].topology.faces[h.index as usize].geom;
        // From the face plane, along the face normal (raise) or into the body (sink).
        let start = dot(sub(g.centroid, p.frame.origin), p.frame.z) * dot(n, p.frame.z).signum();
        let s = dot(n, p.frame.z).signum();
        let mut tool = LabeledTool::new(ToolShape::prism(p, s * start, s * (start + height)));
        tool.end = Some(FaceLabel::key(Role::EmbossTop, "top"));
        let combine = if height >= 0.0 {
            Combine::Add
        } else {
            Combine::Cut
        };
        Ok(op(
            labeled(vec![tool], Vec::new(), combine, false),
            vec![h.body as usize],
            NamingContext::simple(id, vec![e.profile.sketch.clone()]),
        ))
    }

    pub(super) fn rib(&mut self, i: usize, id: &FeatureId, r: &Rib) -> Result<Compiled, Halt> {
        let (frame, segs) = self.chain(i, id, "profile", &r.profile)?;
        let t = self.value(id, "thickness", &r.thickness, Dimension::LENGTH)?;
        let depth = self.value(id, "depth", &r.depth, Dimension::LENGTH)?;
        let mut tools = Vec::new();
        for s in &segs {
            if !matches!(s.kind, SegmentKind::Line) {
                return Err(self.not_lowered(id, "ribs along arcs"));
            }
            let d = [s.end[0] - s.start[0], s.end[1] - s.start[1]];
            let l = d[0].hypot(d[1]);
            let (u, n) = ([d[0] / l, d[1] / l], [-d[1] / l, d[0] / l]);
            // Each line thickened symmetrically and extended by half the thickness at both
            // ends so neighbouring segments overlap at corners.
            let e = t / 2.0;
            let p = |a: [f64; 2], k: f64, o: f64| {
                [a[0] + k * u[0] + o * n[0], a[1] + k * u[1] + o * n[1]]
            };
            let c = s.curve.0.clone();
            let corners = vec![
                (format!("{}.r", c), p(s.start, -e, -e)),
                (format!("{}.end", c), p(s.end, e, -e)),
                (format!("{}.l", c), p(s.end, e, e)),
                (format!("{}.start", c), p(s.start, -e, e)),
            ];
            let mut tool = LabeledTool::new(ToolShape::prism(
                polygon(frame, &corners),
                0.0,
                if r.reverse { -depth } else { depth },
            ));
            let sid = |x: &str| crate::ids::SketchEntityId::new(format!("{}.{}", c, x));
            tool.sides.push((
                sid("r"),
                FaceLabel::key(Role::Rib, c.clone()).with_ordinal(0),
            ));
            tool.sides.push((
                sid("l"),
                FaceLabel::key(Role::Rib, c.clone()).with_ordinal(1),
            ));
            tool.sides.push((
                sid("start"),
                FaceLabel::key(Role::Rib, c.clone()).with_ordinal(2),
            ));
            tool.sides.push((
                sid("end"),
                FaceLabel::key(Role::Rib, c.clone()).with_ordinal(3),
            ));
            tool.start = Some(FaceLabel::key(Role::Rib, c.clone()).with_ordinal(4));
            tool.end = Some(FaceLabel::key(Role::Rib, c.clone()).with_ordinal(5));
            tools.push(tool);
        }
        let operands = self.implicit_bodies(i, id)?;
        Ok(op(
            labeled(tools, Vec::new(), Combine::Add, false),
            operands,
            NamingContext::simple(id, Vec::new()),
        ))
    }
}
