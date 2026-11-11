//! A feature's inputs: values and counts, and the frames, planes, axes, points, profiles and
//! paths its references resolve to.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn value(
        &mut self,
        f: &FeatureId,
        param: &str,
        v: &ValueExpr,
        dim: Dimension,
    ) -> Result<f64, Halt> {
        let mut ins = BTreeSet::new();
        let mut ls = BTreeSet::new();
        v.collect_names(&mut ins, &mut ls);
        let failed: Option<String> = ins
            .iter()
            .map(|n| format!("input.{}", n))
            .chain(ls.iter().map(|n| format!("let.{}", n)))
            .find(|k| self.values.failed.contains_key(k));
        if let Some(k) = failed {
            let root = self.failure_diag.get(&k).copied().unwrap_or(0);
            let d = self.diags.push(
                DiagCode::RegenCascade,
                Severity::Error,
                Site::feature(f, Some(param)),
                format!("`{}` reads `{}`, which failed to evaluate", f, k),
                Cause::Cascade { root },
                Vec::new(),
            );
            return Err(Halt::Block(f.clone(), d));
        }
        self.ev.eval_as(v, self.values, dim).map_err(|e| {
            let (code, message, repairs) =
                value_error_diagnostic(&Site::feature(f, Some(param)), &e);
            self.error(f, Some(param), code, message, repairs)
        })
    }

    pub(super) fn count(&mut self, f: &FeatureId, param: &str, v: &ValueExpr) -> Result<u32, Halt> {
        let n = self.value(f, param, v, Dimension::DIMENSIONLESS)?;
        if n < 1.0 || n.fract() != 0.0 || n > 10_000.0 {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::ValueFailed,
                format!("count must be a whole number from 1, got {}", n),
                Vec::new(),
            ));
        }
        Ok(n as u32)
    }

    // ------------------------------------------------------------- geometry from references

    /// The planar faces a hole starts on: one face, or several coplanar faces (a boss
    /// pattern's end caps, `feature.bosses.face.end`), with the bodies they belong to and the
    /// plane's frame. Faces off the first one's plane are an error.
    pub(super) fn hole_faces(
        &mut self,
        i: usize,
        id: &FeatureId,
        r: &Reference,
    ) -> Result<(Vec<usize>, Frame), Halt> {
        let faces = self.list(i, id, "face", std::slice::from_ref(r), Some(TopoKind::Face))?;
        let mut frame: Option<Frame> = None;
        let mut bodies = Vec::new();
        for h in &faces {
            let Some(f) = self.face_frame(h) else {
                return Err(self.error(
                    id,
                    Some("face"),
                    DiagCode::RefKindMismatch,
                    "hole face is not planar".into(),
                    Vec::new(),
                ));
            };
            if let Some(first) = &frame {
                let off = sub(f.origin, first.origin);
                if dot(f.z, first.z) < 1.0 - 1e-9 || dot(off, off).sqrt() > 1e-9 {
                    return Err(self.error(
                        id,
                        Some("face"),
                        DiagCode::RefKindMismatch,
                        "a hole's faces must lie in one plane".into(),
                        Vec::new(),
                    ));
                }
            } else {
                frame = Some(f);
            }
            if !bodies.contains(&(h.body as usize)) {
                bodies.push(h.body as usize);
            }
        }
        let frame = frame.ok_or_else(|| {
            self.error(
                id,
                Some("face"),
                DiagCode::RefKindMismatch,
                "a hole needs a face to start on".into(),
                Vec::new(),
            )
        })?;
        Ok((bodies, frame))
    }

    pub(super) fn face_frame(&self, h: &EntityHandle) -> Option<Frame> {
        let g = &self.st.bodies[h.body as usize].topology.faces[h.index as usize].geom;
        let SurfaceKind::Plane { normal } = g.surface else {
            return None;
        };
        let z = normalize(normal);
        let origin = scale(z, dot(z, g.centroid));
        let seed = if dot(z, [1.0, 0.0, 0.0]).abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let x = normalize(sub(seed, scale(z, dot(seed, z))));
        let y = cross(z, x);
        Some(Frame { origin, x, y, z })
    }

    pub(super) fn datum_frame(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        d: &DatumRef,
    ) -> Result<Frame, Halt> {
        match d {
            DatumRef::OriginPlane { plane } => Ok(plane.frame()),
            DatumRef::OriginFrame => Ok(Frame::WORLD),
            DatumRef::Feature { feature } => {
                self.gate_feature(i, f, feature, param)?;
                match self.st.datums.get(feature) {
                    Some(DatumValue::Plane { frame }) | Some(DatumValue::Frame { frame }) => {
                        Ok(*frame)
                    }
                    _ => Err(self.error(
                        f,
                        Some(param),
                        DiagCode::RefKindMismatch,
                        format!("`{}` is not a plane", feature),
                        Vec::new(),
                    )),
                }
            }
            other => Err(self.error(
                f,
                Some(param),
                DiagCode::RefKindMismatch,
                format!("{} is not a plane", other),
                Vec::new(),
            )),
        }
    }

    pub(super) fn plane(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        p: &PlaneRef,
    ) -> Result<Frame, Halt> {
        match p {
            PlaneRef::Datum { datum } => self.datum_frame(i, f, param, datum),
            PlaneRef::Face { face } => {
                let r = self.single(i, f, param, face, Some(TopoKind::Face))?;
                let h = *r.entities().iter().next().ok_or_else(|| {
                    self.error(
                        f,
                        Some(param),
                        DiagCode::RefEmpty,
                        "no face".into(),
                        Vec::new(),
                    )
                })?;
                self.face_frame(&h).ok_or_else(|| {
                    self.error(
                        f,
                        Some(param),
                        DiagCode::RefKindMismatch,
                        format!("`{}` is not planar", face),
                        Vec::new(),
                    )
                })
            }
        }
    }

    pub(super) fn axis(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        a: &AxisRef,
    ) -> Result<([f64; 3], [f64; 3]), Halt> {
        match a {
            AxisRef::Datum { datum } => match datum {
                DatumRef::OriginAxis { axis } => Ok((
                    [0.0; 3],
                    match axis {
                        OriginAxis::X => [1.0, 0.0, 0.0],
                        OriginAxis::Y => [0.0, 1.0, 0.0],
                        OriginAxis::Z => [0.0, 0.0, 1.0],
                    },
                )),
                DatumRef::Feature { feature } => {
                    self.gate_feature(i, f, feature, param)?;
                    match self.st.datums.get(feature) {
                        Some(DatumValue::Axis { point, direction }) => Ok((*point, *direction)),
                        Some(DatumValue::Frame { frame }) => Ok((frame.origin, frame.z)),
                        _ => Err(self.error(
                            f,
                            Some(param),
                            DiagCode::RefKindMismatch,
                            format!("`{}` is not an axis", feature),
                            Vec::new(),
                        )),
                    }
                }
                other => Err(self.error(
                    f,
                    Some(param),
                    DiagCode::RefKindMismatch,
                    format!("{} is not an axis", other),
                    Vec::new(),
                )),
            },
            AxisRef::Edge { edge } => {
                let r = self.single(i, f, param, edge, Some(TopoKind::Edge))?;
                let h = *r.entities().iter().next().expect("exactly one");
                let g = self.st.bodies[h.body as usize].topology.edges[h.index as usize].geom;
                match g.curve {
                    EdgeCurve::Line => Ok((g.start, normalize(sub(g.end, g.start)))),
                    EdgeCurve::Circle { center, normal, .. } => Ok((center, normalize(normal))),
                    EdgeCurve::Other => Err(self.error(
                        f,
                        Some(param),
                        DiagCode::RefKindMismatch,
                        format!(
                            "`{}` is neither straight nor circular, so it gives no axis; pick a \
                             straight edge, a circular edge, a cylindrical face or a datum axis",
                            edge
                        ),
                        Vec::new(),
                    )),
                }
            }
            AxisRef::Face { face } => {
                let r = self.single(i, f, param, face, Some(TopoKind::Face))?;
                let h = *r.entities().iter().next().expect("exactly one");
                self.face_axis(&h).ok_or_else(|| {
                    self.error(
                        f,
                        Some(param),
                        DiagCode::RefKindMismatch,
                        format!(
                            "`{}` is neither cylindrical nor bounded by a circle, so it gives no \
                             axis; pick a cylindrical or conical face, a circular edge or a datum axis",
                            face
                        ),
                        Vec::new(),
                    )
                })
            }
            AxisRef::SketchLine { sketch, entity } => {
                self.gate_feature(i, f, sketch, param)?;
                let line = self.st.sketches.get(sketch).and_then(|(solved, frame)| {
                    let at = |role| solved.point(entity, role).map(|p| frame.to_world(p));
                    match (
                        at(PointRole::Start),
                        at(PointRole::End),
                        at(PointRole::Center),
                    ) {
                        (Some(a), Some(b), None) if a != b => Some((a, normalize(sub(b, a)))),
                        _ => None,
                    }
                });
                line.ok_or_else(|| {
                    self.error(
                        f,
                        Some(param),
                        DiagCode::RefKindMismatch,
                        format!(
                            "`sketch.{}.{}` is not a line of that sketch, so it gives no axis; \
                             pick a sketch line, such as a construction centerline",
                            sketch, entity
                        ),
                        Vec::new(),
                    )
                })
            }
        }
    }

    /// A face's axis: a cylinder's own, else the axis of a circle on its boundary (a cone, a
    /// revolved or a turned face). `None` for a face with neither.
    pub(super) fn face_axis(&self, h: &EntityHandle) -> Option<([f64; 3], [f64; 3])> {
        let topo = &self.st.bodies[h.body as usize].topology;
        if let SurfaceKind::Cylinder {
            axis_point,
            axis_direction,
            ..
        } = topo.faces[h.index as usize].geom.surface
        {
            return Some((axis_point, normalize(axis_direction)));
        }
        topo.edges
            .iter()
            .filter(|e| e.faces.contains(&h.index))
            .find_map(|e| match e.geom.curve {
                EdgeCurve::Circle { center, normal, .. } => Some((center, normalize(normal))),
                _ => None,
            })
    }

    pub(super) fn point(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        p: &PointRef,
    ) -> Result<[f64; 3], Halt> {
        match p {
            PointRef::Datum { datum } => match datum {
                DatumRef::OriginPoint | DatumRef::OriginFrame => Ok([0.0; 3]),
                DatumRef::Feature { feature } => {
                    self.gate_feature(i, f, feature, param)?;
                    match self.st.datums.get(feature) {
                        Some(DatumValue::Point { at }) => Ok(*at),
                        Some(DatumValue::Frame { frame }) | Some(DatumValue::Plane { frame }) => {
                            Ok(frame.origin)
                        }
                        Some(DatumValue::Axis { point, .. }) => Ok(*point),
                        None => Err(self.error(
                            f,
                            Some(param),
                            DiagCode::RefKindMismatch,
                            format!("`{}` is not a point", feature),
                            Vec::new(),
                        )),
                    }
                }
                other => Err(self.error(
                    f,
                    Some(param),
                    DiagCode::RefKindMismatch,
                    format!("{} is not a point", other),
                    Vec::new(),
                )),
            },
            PointRef::Coords { x, y, z } => Ok([
                self.value(f, param, x, Dimension::LENGTH)?,
                self.value(f, param, y, Dimension::LENGTH)?,
                self.value(f, param, z, Dimension::LENGTH)?,
            ]),
            PointRef::Vertex { vertex } => {
                let r = self.single(i, f, param, vertex, Some(TopoKind::Vertex))?;
                let h = *r.entities().iter().next().expect("exactly one");
                Ok(self.st.bodies[h.body as usize].topology.vertices[h.index as usize].point)
            }
            PointRef::Sketch {
                sketch,
                entity,
                role,
            } => {
                self.gate_feature(i, f, sketch, param)?;
                let Some((solved, frame)) = self.st.sketches.get(sketch) else {
                    return Err(self.error(
                        f,
                        Some(param),
                        DiagCode::RefKindMismatch,
                        format!("`{}` is not a sketch", sketch),
                        Vec::new(),
                    ));
                };
                match solved.point(entity, *role) {
                    Some(p) => Ok(frame.to_world(p)),
                    None => Err(self.error(
                        f,
                        Some(param),
                        DiagCode::RefNotGenerated,
                        format!("sketch `{}` has no point `{}.{:?}`", sketch, entity, role),
                        Vec::new(),
                    )),
                }
            }
        }
    }

    pub(super) fn profile(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        p: &ProfileRef,
    ) -> Result<CanonicalProfile, Halt> {
        self.gate_feature(i, f, &p.sketch, param)?;
        let Some((solved, frame)) = self.st.sketches.get(&p.sketch).cloned() else {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::RefKindMismatch,
                format!("`{}` is not a sketch", p.sketch),
                Vec::new(),
            ));
        };
        let Some(sketch_feature) = self.doc.feature(&p.sketch) else {
            unreachable!("gated")
        };
        let FeatureKind::Sketch(sketch) = &sketch_feature.kind else {
            unreachable!("sketch state")
        };
        match build_loops(sketch, &solved, &p.loops) {
            Ok(loops) => Ok(CanonicalProfile {
                frame,
                loops,
                merge_collinear: p.merge_collinear,
            }),
            Err(msg) => {
                Err(self.error(f, Some(param), DiagCode::SketchOpenProfile, msg, Vec::new()))
            }
        }
    }

    /// A sweep path: the sketch's curves chained end to end (open or closed), as a polyline in
    /// part coordinates plus the path curve each segment samples.
    pub(super) fn path(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        p: &ProfileRef,
    ) -> Result<(Vec<[f64; 3]>, Vec<u32>), Halt> {
        self.gate_feature(i, f, &p.sketch, param)?;
        let Some((solved, frame)) = self.st.sketches.get(&p.sketch).cloned() else {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::RefKindMismatch,
                format!("`{}` is not a sketch", p.sketch),
                Vec::new(),
            ));
        };
        let Some(FeatureKind::Sketch(sketch)) = self.doc.feature(&p.sketch).map(|x| &x.kind) else {
            unreachable!("gated")
        };
        match build_path(sketch, &solved) {
            Ok(segs) => {
                let mut pts = Vec::new();
                let mut curves = Vec::new();
                for (ci, seg) in segs.iter().enumerate() {
                    let samples = segment_samples(seg);
                    if pts.is_empty() {
                        pts.push(frame.to_world(samples[0]));
                    }
                    for q in &samples[1..] {
                        pts.push(frame.to_world(*q));
                        curves.push(ci as u32);
                    }
                }
                Ok((pts, curves))
            }
            Err(msg) => {
                Err(self.error(f, Some(param), DiagCode::SketchOpenProfile, msg, Vec::new()))
            }
        }
    }
}
