//! Reference curves (helix, spiral, projected, composite, split line) and mate connectors.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    /// A point and unit tangent at parameter `t ∈ [0, 1]` along a curve or edge.
    pub(super) fn curve_at(
        &mut self,
        i: usize,
        id: &FeatureId,
        c: &CurveRef,
        t: f64,
    ) -> Result<([f64; 3], [f64; 3]), Halt> {
        let pts = self.curve_points(i, id, c)?;
        let total: f64 = pts
            .windows(2)
            .map(|w| dot(sub(w[1], w[0]), sub(w[1], w[0])).sqrt())
            .sum();
        let mut want = t.clamp(0.0, 1.0) * total;
        for w in pts.windows(2) {
            let d = sub(w[1], w[0]);
            let l = dot(d, d).sqrt();
            if want <= l || std::ptr::eq(w, pts.windows(2).last().expect("two points")) {
                let s = if l > 0.0 { (want / l).min(1.0) } else { 0.0 };
                return Ok((add(w[0], scale(d, s)), normalize(d)));
            }
            want -= l;
        }
        Err(self.error(
            id,
            Some("curve"),
            DiagCode::RegenInvalid,
            "the curve is empty".into(),
            Vec::new(),
        ))
    }

    pub(super) fn connector(
        &mut self,
        i: usize,
        id: &FeatureId,
        m: &MateConnectorDef,
    ) -> Result<DatumValue, Halt> {
        let mut frame = match &m.placement {
            ConnectorPlacement::Frame { frame } => {
                let tmp = FeatureKind::Frame(frame.clone());
                let FeatureKind::Frame(fd) = &tmp else {
                    unreachable!()
                };
                match fd {
                    FrameDef::OnFace { face } => {
                        self.plane(i, id, "face", &PlaneRef::Face { face: face.clone() })?
                    }
                    FrameDef::At { origin, z, x } => {
                        let o = self.point(i, id, "origin", origin)?;
                        let (_, zd) = self.axis(i, id, "z", z)?;
                        let xd = match x {
                            Some(x) => self.axis(i, id, "x", x)?.1,
                            None if dot(zd, [1.0, 0.0, 0.0]).abs() < 0.9 => [1.0, 0.0, 0.0],
                            None => [0.0, 1.0, 0.0],
                        };
                        let xd = normalize(sub(xd, scale(zd, dot(xd, zd))));
                        Frame {
                            origin: o,
                            x: xd,
                            y: cross(zd, xd),
                            z: zd,
                        }
                    }
                }
            }
            ConnectorPlacement::Center { of } => {
                let h = self.face_handle(i, id, "of", of)?;
                let g = self.st.bodies[h.body as usize].topology.faces[h.index as usize].geom;
                let (o, z) = match g.surface {
                    SurfaceKind::Cylinder {
                        axis_point,
                        axis_direction,
                        ..
                    } => {
                        let ad = normalize(axis_direction);
                        (
                            add(axis_point, scale(ad, dot(sub(g.centroid, axis_point), ad))),
                            ad,
                        )
                    }
                    SurfaceKind::Plane { normal } => (g.centroid, normalize(normal)),
                    SurfaceKind::Other => (g.centroid, [0.0, 0.0, 1.0]),
                };
                let seed = if dot(z, [1.0, 0.0, 0.0]).abs() < 0.9 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let x = normalize(sub(seed, scale(z, dot(seed, z))));
                Frame {
                    origin: o,
                    x,
                    y: cross(z, x),
                    z,
                }
            }
            ConnectorPlacement::EdgeMidpoint { edge } => {
                let r = self.single(i, id, "edge", edge, Some(TopoKind::Edge))?;
                let h = *r.entities().iter().next().expect("one");
                let g = self.st.bodies[h.body as usize].topology.edges[h.index as usize].geom;
                let z = normalize(sub(g.end, g.start));
                let seed = if dot(z, [1.0, 0.0, 0.0]).abs() < 0.9 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let x = normalize(sub(seed, scale(z, dot(seed, z))));
                Frame {
                    origin: scale(add(g.start, g.end), 0.5),
                    x,
                    y: cross(z, x),
                    z,
                }
            }
            ConnectorPlacement::Vertex { vertex } => {
                let at = self.point(
                    i,
                    id,
                    "vertex",
                    &PointRef::Vertex {
                        vertex: vertex.clone(),
                    },
                )?;
                Frame {
                    origin: at,
                    ..Frame::WORLD
                }
            }
        };
        if let Some(o) = &m.offset {
            let d = [
                self.value(id, "offset", &o[0], Dimension::LENGTH)?,
                self.value(id, "offset", &o[1], Dimension::LENGTH)?,
                self.value(id, "offset", &o[2], Dimension::LENGTH)?,
            ];
            frame.origin = add(
                frame.origin,
                add(
                    add(scale(frame.x, d[0]), scale(frame.y, d[1])),
                    scale(frame.z, d[2]),
                ),
            );
        }
        if let Some(r) = &m.rotation {
            let a = self.value(id, "rotation", r, Dimension::ANGLE)?;
            frame.x = rotate(frame.x, frame.z, a);
            frame.y = rotate(frame.y, frame.z, a);
        }
        if m.flip {
            frame.z = scale(frame.z, -1.0);
            frame.y = scale(frame.y, -1.0);
        }
        Ok(DatumValue::Frame { frame })
    }

    // ---------------------------------------------------------------------- curves

    pub(super) fn curve(
        &mut self,
        i: usize,
        id: &FeatureId,
        c: &CurveDef,
    ) -> Result<Compiled, Halt> {
        let value = match c {
            CurveDef::Helix {
                axis,
                start,
                pitch,
                turns,
                left_hand,
                taper,
            } => {
                let (ap, ad) = self.axis(i, id, "axis", axis)?;
                let s = self.point(i, id, "start", start)?;
                let pitch = self.value(id, "pitch", pitch, Dimension::LENGTH)?;
                let turns = self.value(id, "turns", turns, Dimension::DIMENSIONLESS)?;
                let taper = match taper {
                    Some(t) => self.value(id, "taper", t, Dimension::LENGTH)?,
                    None => 0.0,
                };
                let ad = normalize(ad);
                let rel = sub(s, ap);
                let foot = add(ap, scale(ad, dot(rel, ad)));
                let radial = sub(s, foot);
                let r0 = dot(radial, radial).sqrt();
                let u = normalize(radial);
                let v = cross(ad, u);
                let sign = if *left_hand { -1.0 } else { 1.0 };
                let n = ((turns * 48.0).ceil() as usize).max(2);
                let points = (0..=n)
                    .map(|k| {
                        let t = turns * k as f64 / n as f64;
                        let a = sign * t * std::f64::consts::TAU;
                        let r = r0 + taper * t;
                        add(
                            add(foot, scale(ad, pitch * t)),
                            add(scale(u, r * a.cos()), scale(v, r * a.sin())),
                        )
                    })
                    .collect();
                CurveValue {
                    points,
                    closed: false,
                }
            }
            CurveDef::Spiral {
                plane,
                start_radius,
                pitch,
                turns,
            } => {
                let fr = self.plane(i, id, "plane", plane)?;
                let r0 = self.value(id, "start_radius", start_radius, Dimension::LENGTH)?;
                let pitch = self.value(id, "pitch", pitch, Dimension::LENGTH)?;
                let turns = self.value(id, "turns", turns, Dimension::DIMENSIONLESS)?;
                let n = ((turns * 48.0).ceil() as usize).max(2);
                let points = (0..=n)
                    .map(|k| {
                        let t = turns * k as f64 / n as f64;
                        let a = t * std::f64::consts::TAU;
                        let r = r0 + pitch * t;
                        fr.to_world([r * a.cos(), r * a.sin()])
                    })
                    .collect();
                CurveValue {
                    points,
                    closed: false,
                }
            }
            CurveDef::ThroughPoints { points, closed } => {
                let mut pts = Vec::new();
                for (k, p) in points.iter().enumerate() {
                    pts.push(self.point(i, id, &format!("points[{}]", k), p)?);
                }
                CurveValue {
                    points: pts,
                    closed: *closed,
                }
            }
            CurveDef::Composite { edges } => {
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let mut segs: Vec<([f64; 3], [f64; 3])> = set
                    .iter()
                    .map(|h| {
                        let g =
                            self.st.bodies[h.body as usize].topology.edges[h.index as usize].geom;
                        (g.start, g.end)
                    })
                    .collect();
                let close = |a: [f64; 3], b: [f64; 3]| dot(sub(a, b), sub(a, b)).sqrt() < 1e-9;
                // Grow the chain from either end, so the edges may come in any order and
                // any direction (a split edge's pieces come back in the kernel's order).
                let mut pts: std::collections::VecDeque<[f64; 3]> = Default::default();
                if let Some(first) = segs.first().copied() {
                    segs.remove(0);
                    pts.push_back(first.0);
                    pts.push_back(first.1);
                    while !segs.is_empty() {
                        let (head, tail) = (pts[0], pts[pts.len() - 1]);
                        if let Some(k) = segs.iter().position(|s| close(s.0, tail) || close(s.1, tail)) {
                            let s = segs.remove(k);
                            pts.push_back(if close(s.0, tail) { s.1 } else { s.0 });
                        } else if let Some(k) = segs.iter().position(|s| close(s.0, head) || close(s.1, head)) {
                            let s = segs.remove(k);
                            pts.push_front(if close(s.0, head) { s.1 } else { s.0 });
                        } else {
                            return Err(self.error(
                                id,
                                Some("edges"),
                                DiagCode::RegenInvalid,
                                "the edges do not join end to end".into(),
                                Vec::new(),
                            ));
                        }
                    }
                }
                let mut pts: Vec<[f64; 3]> = pts.into();
                let closed = pts.len() > 2 && close(pts[0], *pts.last().expect("points"));
                if closed {
                    pts.pop();
                }
                CurveValue {
                    points: pts,
                    closed,
                }
            }
            CurveDef::Intersection { a, b } => {
                let h = self.face_handle(i, id, "a", a)?;
                let fr = self.plane(i, id, "b", b)?;
                let g = self.st.bodies[h.body as usize].topology.faces[h.index as usize].geom;
                let SurfaceKind::Plane { normal } = g.surface else {
                    return Err(self.not_lowered(id, "intersections of curved faces"));
                };
                let n = normalize(normal);
                let d = cross(n, fr.z);
                if dot(d, d).sqrt() < 1e-12 {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RegenInvalid,
                        "the face is parallel to the plane".into(),
                        Vec::new(),
                    ));
                }
                // The line where the planes meet, clipped to the face's box.
                let d = normalize(d);
                let (ha, hb) = (dot(n, g.centroid), dot(fr.z, fr.origin));
                let c = dot(n, fr.z);
                let den = 1.0 - c * c;
                let p0 = add(
                    scale(n, (ha - hb * c) / den),
                    scale(fr.z, (hb - ha * c) / den),
                );
                let mut lo = f64::NEG_INFINITY;
                let mut hi = f64::INFINITY;
                for ax in 0..3 {
                    if d[ax].abs() > 1e-12 {
                        let (t0, t1) = (
                            (g.bbox_min[ax] - p0[ax]) / d[ax],
                            (g.bbox_max[ax] - p0[ax]) / d[ax],
                        );
                        lo = lo.max(t0.min(t1));
                        hi = hi.min(t0.max(t1));
                    }
                }
                if lo > hi {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RegenInvalid,
                        "the plane misses the face".into(),
                        Vec::new(),
                    ));
                }
                CurveValue {
                    points: vec![add(p0, scale(d, lo)), add(p0, scale(d, hi))],
                    closed: false,
                }
            }
            CurveDef::Equation {
                parameter,
                x,
                y,
                z,
                t0,
                t1,
                samples,
            } => {
                let a = self.value(id, "t0", t0, Dimension::DIMENSIONLESS)?;
                let b = self.value(id, "t1", t1, Dimension::DIMENSIONLESS)?;
                let n = (*samples).max(2) as usize;
                let mut points = Vec::with_capacity(n + 1);
                for k in 0..=n {
                    let t = a + (b - a) * k as f64 / n as f64;
                    let sub_t =
                        |e: &ValueExpr| e.substitute_input(parameter, &ValueExpr::number(t));
                    let (ex, ey, ez) = (sub_t(x), sub_t(y), sub_t(z));
                    points.push([
                        self.value(id, "x", &ex, Dimension::LENGTH)?,
                        self.value(id, "y", &ey, Dimension::LENGTH)?,
                        self.value(id, "z", &ez, Dimension::LENGTH)?,
                    ]);
                }
                CurveValue {
                    points,
                    closed: false,
                }
            }
            CurveDef::Projected { .. } => return Err(self.not_lowered(id, "projected curves")),
            CurveDef::SplitLine { .. } => {
                return Err(self.not_lowered(id, "split lines (face splitting)"))
            }
        };
        Ok(Compiled::Record(FeatureRecord::Curve(value)))
    }

    // ------------------------------------------------------------------ direct edits
}
