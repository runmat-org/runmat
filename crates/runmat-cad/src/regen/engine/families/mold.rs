//! Mold features (shrink, parting line) and the draft analysis they record.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn mold(
        &mut self,
        i: usize,
        id: &FeatureId,
        m: &MoldFeature,
    ) -> Result<Compiled, Halt> {
        let simple = NamingContext::simple(id, Vec::new());
        match m {
            MoldFeature::Shrink { bodies, x, y, z } => {
                let set = self.list(i, id, "bodies", bodies, Some(TopoKind::Body))?;
                let k = [
                    1.0 + self.value(id, "x", x, Dimension::DIMENSIONLESS)?,
                    1.0 + self.value(id, "y", y, Dimension::DIMENSIONLESS)?,
                    1.0 + self.value(id, "z", z, Dimension::DIMENSIONLESS)?,
                ];
                Ok(op(
                    KernelOp::Scale {
                        center: [0.0; 3],
                        factors: k,
                    },
                    Self::bodies_of(&set),
                    simple,
                ))
            }
            MoldFeature::PartingLine {
                body,
                pull,
                min_draft,
                ..
            } => {
                let b = self.single(i, id, "body", body, Some(TopoKind::Body))?;
                let bi = b.entities().iter().next().expect("one").body as usize;
                let (_, d) = self.axis(i, id, "pull", pull)?;
                let min = self.value(id, "min_draft", min_draft, Dimension::ANGLE)?;
                Ok(Compiled::Record(self.draft_analysis(bi, normalize(d), min)))
            }
            MoldFeature::Undercuts { body, pull } => {
                let b = self.single(i, id, "body", body, Some(TopoKind::Body))?;
                let bi = b.entities().iter().next().expect("one").body as usize;
                let (_, d) = self.axis(i, id, "pull", pull)?;
                Ok(Compiled::Record(self.draft_analysis(bi, normalize(d), 0.0)))
            }
            MoldFeature::PartingSurface { plane, extent } => {
                let fr = self.plane(i, id, "plane", plane)?;
                let e = self.value(id, "extent", extent, Dimension::LENGTH)?;
                let corners = vec![
                    ("s0".to_string(), [-e, -e]),
                    ("s1".to_string(), [e, -e]),
                    ("s2".to_string(), [e, e]),
                    ("s3".to_string(), [-e, e]),
                ];
                Ok(op(
                    KernelOp::Surface(SurfaceOp::Planar {
                        patches: vec![SurfacePatch {
                            profile: polygon(fr, &corners),
                            label: FaceLabel::key(Role::PartingFace, "parting"),
                        }],
                    }),
                    Vec::new(),
                    NamingContext::simple(id, vec![id.clone()]),
                ))
            }
            MoldFeature::ShutOff { .. } => Err(self.not_lowered(id, "shut-off surfaces")),
            MoldFeature::CoreCavity { .. } => Err(self.not_lowered(id, "core/cavity split")),
        }
    }

    /// Draft of every planar face against a pull direction; cylinders parallel to the pull
    /// have zero draft. Faces facing away from the pull and covered by other material along
    /// it are undercuts.
    pub(super) fn draft_analysis(&self, body: usize, pull: [f64; 3], min: f64) -> FeatureRecord {
        let b = &self.st.bodies[body];
        let mut positive = Vec::new();
        let mut negative = Vec::new();
        let mut insufficient = Vec::new();
        let mut sign = vec![0i8; b.topology.faces.len()];
        for (fi, f) in b.topology.faces.iter().enumerate() {
            let draft = match f.geom.surface {
                SurfaceKind::Plane { normal } => {
                    dot(normalize(normal), pull).clamp(-1.0, 1.0).asin()
                }
                SurfaceKind::Cylinder { axis_direction, .. }
                    if dot(normalize(axis_direction), pull).abs() > 1.0 - 1e-9 =>
                {
                    0.0
                }
                _ => continue,
            };
            let names = b.names.faces[fi].iter().cloned().collect::<Vec<_>>();
            let rec = FaceDraft { face: names, draft };
            if draft.abs() < min.max(1e-12) && draft.abs() < std::f64::consts::FRAC_PI_2 - 1e-9 {
                insufficient.push(rec.clone());
            }
            if draft > 1e-12 {
                sign[fi] = 1;
                positive.push(rec);
            } else if draft < -1e-12 {
                sign[fi] = -1;
                negative.push(rec);
            }
        }
        // Undercut: a face drafted away from the pull whose centroid has material of the
        // body above it along the pull (a face of the body facing it lies further along).
        let mut undercuts = Vec::new();
        for (fi, f) in b.topology.faces.iter().enumerate() {
            if sign[fi] >= 0 {
                continue;
            }
            let c = f.geom.centroid;
            let blocked = b.topology.faces.iter().enumerate().any(|(gj, g)| {
                gj != fi
                    && sign[gj] < 0
                    && dot(sub(g.geom.centroid, c), pull) < -1e-9
                    && (0..3).all(|ax| {
                        pull[ax].abs() > 0.5
                            || (c[ax] >= g.geom.bbox_min[ax] - 1e-9
                                && c[ax] <= g.geom.bbox_max[ax] + 1e-9)
                    })
            });
            if blocked {
                undercuts.push(b.names.faces[fi].iter().cloned().collect());
            }
        }
        let mut parting_edges = Vec::new();
        for (ei, e) in b.topology.edges.iter().enumerate() {
            let (sa, sb) = (sign[e.faces[0] as usize], sign[e.faces[1] as usize]);
            let side = |s: i8| s != 0;
            if side(sa) && side(sb) && sa != sb {
                if let Some(n) = crate::refs::naming::edge_name(&b.names, &b.topology, ei as u32) {
                    parting_edges.push(n);
                }
            }
        }
        FeatureRecord::Mold {
            positive,
            negative,
            insufficient,
            undercuts,
            parting_edges,
        }
    }

    // ----------------------------------------------------------------------- details
}
