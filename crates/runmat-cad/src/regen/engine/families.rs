//! Lowering of the feature families (direct edits, surfaces, sheet metal, weldments, mold,
//! details, reference curves, mate connectors and the added datum definitions) to canonical
//! kernel operations and typed records.
//!
//! Each family lowers to the kernel seam ([`KernelOp::Labeled`], [`KernelOp::MoveFaces`],
//! [`KernelOp::Surface`], ...) with labels from the definition, or to a [`FeatureRecord`] when
//! it makes no geometry. A case the seam has no operation for yet fails the feature with
//! `cad.kernel.not_supported` naming the case: never a quietly different body.

use super::*;
use crate::model::curves::{ConnectorPlacement, CurveDef, CurveValue, MateConnectorDef};
use crate::model::detail::{Emboss, Rib, Thread};
use crate::model::direct::{DirectEdit, DirectMotion, LiveRelation};
use crate::model::mold::MoldFeature;
use crate::model::sheet_metal::{
    section, BendRule, BendRuleValue, HemKind, LengthReference, SheetMetalFeature, SheetMetalRules,
    StripRegion,
};
use crate::model::surface::SurfaceFeature;
use crate::model::weldment::{cut_list, member_cuts, CornerTreatment, WeldmentFeature};
use crate::refs::history::{FaceGeom, KernelTag};
use crate::refs::name::Role;
use crate::regen::records::{FaceDraft, FeatureRecord, SheetRules};

pub(super) fn line_seg(curve: &str, a: [f64; 2], b: [f64; 2]) -> ProfileSegment {
    ProfileSegment {
        curve: crate::ids::SketchEntityId::new(curve),
        kind: SegmentKind::Line,
        start: a,
        end: b,
    }
}

pub(super) fn polygon(frame: Frame, pts: &[(String, [f64; 2])]) -> CanonicalProfile {
    let n = pts.len();
    CanonicalProfile {
        frame,
        loops: vec![(0..n)
            .map(|k| line_seg(&pts[k].0, pts[k].1, pts[(k + 1) % n].1))
            .collect()],
        merge_collinear: false,
    }
}

pub(super) fn frame_of(origin: [f64; 3], x: [f64; 3], y: [f64; 3]) -> Frame {
    let x = normalize(x);
    let y = normalize(sub(y, scale(x, dot(y, x))));
    Frame {
        origin,
        x,
        y,
        z: cross(x, y),
    }
}

pub(super) fn op(op: KernelOp, operands: Vec<usize>, naming: NamingContext) -> Compiled {
    Compiled::Op {
        op,
        operands,
        naming,
        import: false,
        instances: None,
        tools: None,
    }
}

pub(super) fn labeled(
    tools: Vec<LabeledTool>,
    carve: Vec<LabeledTool>,
    combine: Combine,
    separate: bool,
) -> KernelOp {
    KernelOp::Labeled {
        tools,
        carve,
        combine,
        separate,
    }
}

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn compile_family(
        &mut self,
        i: usize,
        f: &Feature,
    ) -> Result<Option<Compiled>, Halt> {
        let id = &f.id;
        Ok(Some(match &f.kind {
            FeatureKind::Plane(
                PlaneDef::ThreePoint { .. }
                | PlaneDef::Mid { .. }
                | PlaneDef::NormalToCurve { .. }
                | PlaneDef::TangentToFace { .. },
            )
            | FeatureKind::Axis(AxisDef::TwoPlanes { .. } | AxisDef::Normal { .. })
            | FeatureKind::Point(
                PointDef::CenterOf { .. }
                | PointDef::OnCurve { .. }
                | PointDef::Intersection { .. },
            ) => Compiled::Datum(self.datum_extra(i, id, &f.kind)?),
            FeatureKind::MateConnector(m) => Compiled::Datum(self.connector(i, id, m)?),
            FeatureKind::Curve(c) => self.curve(i, id, c)?,
            FeatureKind::Direct(d) => self.direct(i, id, d)?,
            FeatureKind::Surface(s) => self.surface_feature(i, id, s)?,
            FeatureKind::SheetMetal(s) => self.sheet_metal(i, id, s)?,
            FeatureKind::Weldment(w) => self.weldment(i, id, w)?,
            FeatureKind::Mold(m) => self.mold(i, id, m)?,
            FeatureKind::Thread(t) => self.thread(i, id, t)?,
            FeatureKind::Emboss(e) => self.emboss(i, id, e)?,
            FeatureKind::Rib(r) => self.rib(i, id, r)?,
            FeatureKind::Library(_) => {
                return Err(self.error(
                    id,
                    None,
                    DiagCode::RegenInvalid,
                    "library features are expanded before regeneration; this one was not \
                     (its definition is missing)"
                        .into(),
                    Vec::new(),
                ))
            }
            _ => return Ok(None),
        }))
    }

    pub(super) fn not_lowered(&mut self, id: &FeatureId, what: &str) -> Halt {
        self.error(
            id,
            None,
            DiagCode::KernelNotSupported,
            format!("not supported yet: {}", what),
            Vec::new(),
        )
    }

    pub(super) fn face_handle(
        &mut self,
        i: usize,
        id: &FeatureId,
        param: &str,
        r: &Reference,
    ) -> Result<EntityHandle, Halt> {
        let res = self.single(i, id, param, r, Some(TopoKind::Face))?;
        Ok(*res.entities().iter().next().expect("one"))
    }

    pub(super) fn face_normal(&self, h: &EntityHandle) -> Option<[f64; 3]> {
        match self.st.bodies[h.body as usize].topology.faces[h.index as usize]
            .geom
            .surface
        {
            SurfaceKind::Plane { normal } => Some(normalize(normal)),
            _ => None,
        }
    }

    pub(super) fn chain(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        p: &ProfileRef,
    ) -> Result<(Frame, Vec<ProfileSegment>), Halt> {
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
        build_path(sketch, &solved)
            .map(|segs| (frame, segs))
            .map_err(|m| self.error(f, Some(param), DiagCode::SketchOpenProfile, m, Vec::new()))
    }

    // --------------------------------------------------------------------- datums

    fn datum_extra(
        &mut self,
        i: usize,
        id: &FeatureId,
        kind: &FeatureKind,
    ) -> Result<DatumValue, Halt> {
        Ok(match kind {
            FeatureKind::Plane(PlaneDef::ThreePoint { a, b, c }) => {
                let (pa, pb, pc) = (
                    self.point(i, id, "a", a)?,
                    self.point(i, id, "b", b)?,
                    self.point(i, id, "c", c)?,
                );
                let n = cross(sub(pb, pa), sub(pc, pa));
                if dot(n, n).sqrt() < 1e-12 {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RegenInvalid,
                        "the three points are collinear".into(),
                        Vec::new(),
                    ));
                }
                DatumValue::Plane {
                    frame: frame_of(pa, sub(pb, pa), cross(normalize(n), sub(pb, pa))),
                }
            }
            FeatureKind::Plane(PlaneDef::Mid { a, b }) => {
                let (fa, fb) = (self.plane(i, id, "a", a)?, self.plane(i, id, "b", b)?);
                if dot(fa.z, fb.z).abs() < 1.0 - 1e-9 {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RegenInvalid,
                        "a mid-plane needs parallel planes".into(),
                        Vec::new(),
                    ));
                }
                let d = dot(sub(fb.origin, fa.origin), fa.z);
                DatumValue::Plane {
                    frame: fa.offset(d / 2.0),
                }
            }
            FeatureKind::Plane(PlaneDef::NormalToCurve { curve, parameter }) => {
                let t = self.value(id, "parameter", parameter, Dimension::DIMENSIONLESS)?;
                let (p, tangent) = self.curve_at(i, id, curve, t)?;
                let seed = if dot(tangent, [0.0, 0.0, 1.0]).abs() < 0.9 {
                    [0.0, 0.0, 1.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let x = normalize(cross(seed, tangent));
                DatumValue::Plane {
                    frame: Frame {
                        origin: p,
                        x,
                        y: cross(tangent, x),
                        z: tangent,
                    },
                }
            }
            FeatureKind::Plane(PlaneDef::TangentToFace { face, at }) => {
                let h = self.face_handle(i, id, "face", face)?;
                let p = self.point(i, id, "at", at)?;
                let SurfaceKind::Cylinder {
                    axis_point,
                    axis_direction,
                    radius,
                } = self.st.bodies[h.body as usize].topology.faces[h.index as usize]
                    .geom
                    .surface
                else {
                    return Err(self.error(
                        id,
                        Some("face"),
                        DiagCode::RefKindMismatch,
                        "a tangent plane needs a cylindrical face".into(),
                        Vec::new(),
                    ));
                };
                let ad = normalize(axis_direction);
                let rel = sub(p, axis_point);
                let radial = normalize(sub(rel, scale(ad, dot(rel, ad))));
                let foot = add(
                    add(axis_point, scale(ad, dot(rel, ad))),
                    scale(radial, radius),
                );
                DatumValue::Plane {
                    frame: Frame {
                        origin: foot,
                        x: ad,
                        y: cross(radial, ad),
                        z: radial,
                    },
                }
            }
            FeatureKind::Axis(AxisDef::TwoPlanes { a, b }) => {
                let (fa, fb) = (self.plane(i, id, "a", a)?, self.plane(i, id, "b", b)?);
                let d = cross(fa.z, fb.z);
                let l = dot(d, d).sqrt();
                if l < 1e-12 {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RegenInvalid,
                        "parallel planes do not meet".into(),
                        Vec::new(),
                    ));
                }
                let d = scale(d, 1.0 / l);
                // A point on both: solve along the in-plane directions.
                let (ha, hb) = (dot(fa.z, fa.origin), dot(fb.z, fb.origin));
                let c = dot(fa.z, fb.z);
                let den = 1.0 - c * c;
                let ka = (ha - hb * c) / den;
                let kb = (hb - ha * c) / den;
                DatumValue::Axis {
                    point: add(scale(fa.z, ka), scale(fb.z, kb)),
                    direction: d,
                }
            }
            FeatureKind::Axis(AxisDef::Normal { plane, through }) => {
                let fr = self.plane(i, id, "plane", plane)?;
                DatumValue::Axis {
                    point: self.point(i, id, "through", through)?,
                    direction: fr.z,
                }
            }
            FeatureKind::Point(PointDef::CenterOf { of }) => {
                let h = self.face_handle(i, id, "of", of)?;
                let g = self.st.bodies[h.body as usize].topology.faces[h.index as usize].geom;
                let at = match g.surface {
                    SurfaceKind::Cylinder {
                        axis_point,
                        axis_direction,
                        ..
                    } => {
                        let ad = normalize(axis_direction);
                        add(axis_point, scale(ad, dot(sub(g.centroid, axis_point), ad)))
                    }
                    _ => g.centroid,
                };
                DatumValue::Point { at }
            }
            FeatureKind::Point(PointDef::OnCurve { curve, parameter }) => {
                let t = self.value(id, "parameter", parameter, Dimension::DIMENSIONLESS)?;
                DatumValue::Point {
                    at: self.curve_at(i, id, curve, t)?.0,
                }
            }
            FeatureKind::Point(PointDef::Intersection { axis, plane }) => {
                let (p, d) = self.axis(i, id, "axis", axis)?;
                let fr = self.plane(i, id, "plane", plane)?;
                let den = dot(d, fr.z);
                if den.abs() < 1e-12 {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RegenInvalid,
                        "the axis is parallel to the plane".into(),
                        Vec::new(),
                    ));
                }
                let t = dot(sub(fr.origin, p), fr.z) / den;
                DatumValue::Point {
                    at: add(p, scale(d, t)),
                }
            }
            _ => unreachable!("datum_extra"),
        })
    }

    /// A point and unit tangent at parameter `t ∈ [0, 1]` along a curve or edge.
    fn curve_at(
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

    fn connector(
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

    fn curve(&mut self, i: usize, id: &FeatureId, c: &CurveDef) -> Result<Compiled, Halt> {
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
                let mut pts = Vec::new();
                if let Some(first) = segs.first().copied() {
                    segs.remove(0);
                    pts.push(first.0);
                    pts.push(first.1);
                    while !segs.is_empty() {
                        let last = *pts.last().expect("points");
                        let Some(k) = segs
                            .iter()
                            .position(|s| close(s.0, last) || close(s.1, last))
                        else {
                            return Err(self.error(
                                id,
                                Some("edges"),
                                DiagCode::RegenInvalid,
                                "the edges do not join end to end".into(),
                                Vec::new(),
                            ));
                        };
                        let s = segs.remove(k);
                        pts.push(if close(s.0, last) { s.1 } else { s.0 });
                    }
                }
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

    fn motion(&mut self, i: usize, id: &FeatureId, m: &DirectMotion) -> Result<FaceMotion, Halt> {
        Ok(match m {
            DirectMotion::Offset { distance } => FaceMotion::Offset {
                distance: self.value(id, "motion.distance", distance, Dimension::LENGTH)?,
            },
            DirectMotion::Translate { dx, dy, dz } => FaceMotion::Translate {
                vector: [
                    self.value(id, "motion.dx", dx, Dimension::LENGTH)?,
                    self.value(id, "motion.dy", dy, Dimension::LENGTH)?,
                    self.value(id, "motion.dz", dz, Dimension::LENGTH)?,
                ],
            },
            DirectMotion::Rotate { axis, angle } => {
                let (p, d) = self.axis(i, id, "motion.axis", axis)?;
                FaceMotion::Rotate {
                    axis_point: p,
                    axis_direction: d,
                    angle: self.value(id, "motion.angle", angle, Dimension::ANGLE)?,
                }
            }
        })
    }

    fn direct(&mut self, i: usize, id: &FeatureId, d: &DirectEdit) -> Result<Compiled, Halt> {
        let simple = NamingContext::simple(id, Vec::new());
        match d {
            DirectEdit::MoveFaces {
                faces,
                motion,
                keep,
            } => {
                let set = self.list(i, id, "faces", faces, Some(TopoKind::Face))?;
                let operands = Self::bodies_of(&set);
                if operands.len() != 1 {
                    return Err(self.error(
                        id,
                        Some("faces"),
                        DiagCode::RefKindMismatch,
                        "moved faces must be on one body".into(),
                        Vec::new(),
                    ));
                }
                let motion = self.motion(i, id, motion)?;
                self.check_relations(i, id, keep, &set, &motion)?;
                let faces = Self::operand_entities(&set, &operands);
                Ok(op(KernelOp::MoveFaces { faces, motion }, operands, simple))
            }
            DirectEdit::ReplaceFaces { faces, with } => {
                let set = self.list(i, id, "faces", faces, Some(TopoKind::Face))?;
                let target = Self::bodies_of(&set);
                let w = self.face_handle(i, id, "with", with)?;
                if target.len() != 1 || target[0] == w.body as usize {
                    return Err(self.error(
                        id,
                        Some("with"),
                        DiagCode::RefKindMismatch,
                        "replace faces of one body by a face of a surface body".into(),
                        Vec::new(),
                    ));
                }
                let operands = vec![target[0], w.body as usize];
                let faces = Self::operand_entities(&set, &operands);
                let with = OperandEntity {
                    operand: 1,
                    entity: w.index,
                };
                Ok(op(KernelOp::ReplaceFaces { faces, with }, operands, simple))
            }
            DirectEdit::DeleteFaces { faces, heal } => {
                let set = self.list(i, id, "faces", faces, Some(TopoKind::Face))?;
                let operands = Self::bodies_of(&set);
                if operands.len() != 1 {
                    return Err(self.error(
                        id,
                        Some("faces"),
                        DiagCode::RefKindMismatch,
                        "deleted faces must be on one body".into(),
                        Vec::new(),
                    ));
                }
                let faces = Self::operand_entities(&set, &operands);
                Ok(op(
                    KernelOp::DeleteFaces { faces, heal: *heal },
                    operands,
                    simple,
                ))
            }
            DirectEdit::MoveBodies {
                bodies,
                motion,
                copy,
            } => {
                let set = self.list(i, id, "bodies", bodies, Some(TopoKind::Body))?;
                let operands = Self::bodies_of(&set);
                let transform = match self.motion(i, id, motion)? {
                    FaceMotion::Offset { .. } => {
                        return Err(self.error(
                            id,
                            Some("motion"),
                            DiagCode::RegenInvalid,
                            "a body moves by a translation or a rotation".into(),
                            Vec::new(),
                        ))
                    }
                    FaceMotion::Translate { vector } => Transform::translation(vector),
                    FaceMotion::Rotate {
                        axis_point,
                        axis_direction,
                        angle,
                    } => Transform::rotation_about(axis_point, axis_direction, angle),
                };
                if *copy {
                    let key = InstanceKey::Row { key: "copy".into() };
                    Ok(Compiled::Op {
                        op: KernelOp::PatternBodies {
                            instances: vec![Instance {
                                key: key.clone(),
                                transform,
                            }],
                        },
                        operands,
                        naming: simple,
                        import: false,
                        instances: Some([key].into_iter().collect()),
                        tools: None,
                    })
                } else {
                    Ok(op(KernelOp::Transform { transform }, operands, simple))
                }
            }
        }
    }

    /// A live relation fails the edit when the motion cannot keep it.
    fn check_relations(
        &mut self,
        i: usize,
        id: &FeatureId,
        keep: &[LiveRelation],
        moved: &BTreeSet<EntityHandle>,
        motion: &FaceMotion,
    ) -> Result<(), Halt> {
        for (k, rel) in keep.iter().enumerate() {
            let param = format!("keep[{}]", k);
            let resolve = |s: &mut Self, r: &Reference| -> Result<(bool, Option<[f64; 3]>), Halt> {
                let set = s.list(i, id, &param, std::slice::from_ref(r), Some(TopoKind::Face))?;
                let any_moved = set.iter().any(|h| moved.contains(h));
                let n = set.iter().next().and_then(|h| s.face_normal(h));
                Ok((any_moved, n))
            };
            let broken = match rel {
                LiveRelation::Fixed { face } => resolve(self, face)?.0,
                LiveRelation::Parallel { a, b }
                | LiveRelation::Perpendicular { a, b }
                | LiveRelation::Coplanar { a, b } => {
                    let (ma, na) = resolve(self, a)?;
                    let (mb, _) = resolve(self, b)?;
                    let one_moves = ma != mb;
                    match motion {
                        FaceMotion::Rotate {
                            axis_direction,
                            angle,
                            ..
                        } => {
                            one_moves
                                && angle.abs() > 1e-12
                                && na
                                    .map(|n| {
                                        dot(normalize(n), normalize(*axis_direction)).abs()
                                            < 1.0 - 1e-9
                                    })
                                    .unwrap_or(true)
                        }
                        FaceMotion::Offset { distance } => {
                            matches!(rel, LiveRelation::Coplanar { .. })
                                && one_moves
                                && distance.abs() > 1e-12
                        }
                        FaceMotion::Translate { vector } => {
                            matches!(rel, LiveRelation::Coplanar { .. })
                                && one_moves
                                && na.map(|n| dot(n, *vector).abs() > 1e-12).unwrap_or(true)
                        }
                    }
                }
                LiveRelation::Concentric { a, b } | LiveRelation::Tangent { a, b } => {
                    let (ma, _) = resolve(self, a)?;
                    let (mb, _) = resolve(self, b)?;
                    ma != mb
                }
                LiveRelation::Symmetric { a, b, .. } => {
                    let (ma, _) = resolve(self, a)?;
                    let (mb, _) = resolve(self, b)?;
                    ma != mb
                }
            };
            if broken {
                return Err(self.error(
                    id,
                    Some(&param),
                    DiagCode::DirectRelationBroken,
                    format!("the edit would break the live relation {}", param),
                    Vec::new(),
                ));
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------------- surfaces

    fn surface_feature(
        &mut self,
        i: usize,
        id: &FeatureId,
        s: &SurfaceFeature,
    ) -> Result<Compiled, Halt> {
        let simple = |p: Vec<FeatureId>| NamingContext::simple(id, p);
        let sheet_bodies =
            |run: &mut Self, param: &str, refs: &[Reference]| -> Result<Vec<usize>, Halt> {
                let set = run.list(i, id, param, refs, Some(TopoKind::Body))?;
                Ok(Self::bodies_of(&set))
            };
        Ok(match s {
            SurfaceFeature::Extrude {
                profile,
                start,
                end,
            } => {
                let (frame, segs) = self.chain(i, id, "profile", profile)?;
                let (a, b) = (
                    self.value(id, "start", start, Dimension::LENGTH)?,
                    self.value(id, "end", end, Dimension::LENGTH)?,
                );
                let shape = ToolShape::prism(
                    CanonicalProfile {
                        frame,
                        loops: vec![segs],
                        merge_collinear: profile.merge_collinear,
                    },
                    a,
                    b,
                );
                op(
                    KernelOp::Surface(SurfaceOp::Swept {
                        tool: LabeledTool::new(shape),
                    }),
                    Vec::new(),
                    simple(vec![profile.sketch.clone()]),
                )
            }
            SurfaceFeature::Revolve {
                profile,
                axis,
                angle,
            } => {
                let (frame, segs) = self.chain(i, id, "profile", profile)?;
                let (p, d) = self.axis(i, id, "axis", axis)?;
                let a = self.value(id, "angle", angle, Dimension::ANGLE)?;
                let shape = ToolShape::Revolve {
                    profile: CanonicalProfile {
                        frame,
                        loops: vec![segs],
                        merge_collinear: false,
                    },
                    axis_point: p,
                    axis_direction: d,
                    angle: a,
                };
                op(
                    KernelOp::Surface(SurfaceOp::Swept {
                        tool: LabeledTool::new(shape),
                    }),
                    Vec::new(),
                    simple(vec![profile.sketch.clone()]),
                )
            }
            SurfaceFeature::Planar { profile } => {
                let p = self.profile(i, id, "profile", profile)?;
                op(
                    KernelOp::Surface(SurfaceOp::Planar {
                        patches: vec![SurfacePatch {
                            profile: p,
                            label: FaceLabel {
                                role: Role::SurfacePatch,
                                tag: KernelTag::None,
                                ordinal: None,
                            },
                        }],
                    }),
                    Vec::new(),
                    simple(vec![profile.sketch.clone()]),
                )
            }
            SurfaceFeature::Offset { faces, distance } => {
                let set = self.list(i, id, "faces", faces, Some(TopoKind::Face))?;
                let operands = Self::bodies_of(&set);
                let faces = Self::operand_entities(&set, &operands);
                let distance = self.value(id, "distance", distance, Dimension::LENGTH)?;
                op(
                    KernelOp::Surface(SurfaceOp::Offset { faces, distance }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Extend { edges, distance } => {
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let operands = Self::bodies_of(&set);
                let edges = Self::operand_entities(&set, &operands);
                let distance = self.value(id, "distance", distance, Dimension::LENGTH)?;
                op(
                    KernelOp::Surface(SurfaceOp::Extend { edges, distance }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Trim { bodies, tool, keep } => {
                let operands = sheet_bodies(self, "bodies", bodies)?;
                let frame = self.plane(i, id, "tool", tool)?;
                op(
                    KernelOp::Surface(SurfaceOp::Trim { frame, keep: *keep }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Knit { bodies, solid } => {
                let operands = sheet_bodies(self, "bodies", bodies)?;
                op(
                    KernelOp::Surface(SurfaceOp::Knit { solid: *solid }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Thicken {
                bodies,
                thickness,
                both_sides,
            } => {
                let operands = sheet_bodies(self, "bodies", bodies)?;
                let distance = self.value(id, "thickness", thickness, Dimension::LENGTH)?;
                op(
                    KernelOp::Surface(SurfaceOp::Thicken {
                        distance,
                        both: *both_sides,
                    }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::MidSurface { body, pairs } => {
                let b = self.single(i, id, "body", body, Some(TopoKind::Body))?;
                let body = b.entities().iter().next().expect("one").body as usize;
                let mut out = Vec::new();
                for (k, (a, c)) in pairs.iter().enumerate() {
                    let ha = self.face_handle(i, id, &format!("pairs[{}].0", k), a)?;
                    let hc = self.face_handle(i, id, &format!("pairs[{}].1", k), c)?;
                    if ha.body as usize != body || hc.body as usize != body {
                        return Err(self.error(
                            id,
                            Some("pairs"),
                            DiagCode::RefKindMismatch,
                            "mid-surface faces must be on the body".into(),
                            Vec::new(),
                        ));
                    }
                    out.push((
                        OperandEntity {
                            operand: 0,
                            entity: ha.index,
                        },
                        OperandEntity {
                            operand: 0,
                            entity: hc.index,
                        },
                    ));
                }
                op(
                    KernelOp::Surface(SurfaceOp::MidSurface { pairs: out }),
                    vec![body],
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Sweep { profile, path } => {
                let (frame, segs) = self.chain(i, id, "profile", profile)?;
                let (path_points, path_curves) = self.path(i, id, "path", path)?;
                let shape = ToolShape::Sweep {
                    profile: CanonicalProfile {
                        frame,
                        loops: vec![segs],
                        merge_collinear: false,
                    },
                    path: path_points,
                    path_curves,
                };
                op(
                    KernelOp::Surface(SurfaceOp::Swept {
                        tool: LabeledTool::new(shape),
                    }),
                    Vec::new(),
                    simple(vec![profile.sketch.clone(), path.sketch.clone()]),
                )
            }
            SurfaceFeature::Loft { profiles, guides } => {
                if !guides.is_empty() {
                    return Err(self.not_lowered(id, "lofts along guide curves"));
                }
                let mut sections = Vec::new();
                for (k, p) in profiles.iter().enumerate() {
                    let (frame, segs) = self.chain(i, id, &format!("profiles[{}]", k), p)?;
                    sections.push(CanonicalProfile {
                        frame,
                        loops: vec![segs],
                        merge_collinear: false,
                    });
                }
                op(
                    KernelOp::Surface(SurfaceOp::Swept {
                        tool: LabeledTool::new(ToolShape::Loft { profiles: sections }),
                    }),
                    Vec::new(),
                    simple(profiles.iter().map(|p| p.sketch.clone()).collect()),
                )
            }
            SurfaceFeature::Boundary { .. } => {
                return Err(self.not_lowered(id, "boundary surfaces"))
            }
            SurfaceFeature::Fill { boundary } => {
                let refs: Vec<Reference> = boundary.iter().map(|b| b.edge.clone()).collect();
                let mut edges = Vec::new();
                let mut operands: Vec<usize> = Vec::new();
                for (k, (r, b)) in refs.iter().zip(boundary).enumerate() {
                    let h =
                        self.single(i, id, &format!("boundary[{}]", k), r, Some(TopoKind::Edge))?;
                    let e = *h.entities().iter().next().expect("one");
                    let body = e.body as usize;
                    let operand = match operands.iter().position(|x| *x == body) {
                        Some(p) => p,
                        None => {
                            operands.push(body);
                            operands.len() - 1
                        }
                    };
                    edges.push((
                        OperandEntity {
                            operand: operand as u32,
                            entity: e.index,
                        },
                        b.continuity,
                    ));
                }
                op(
                    KernelOp::Surface(SurfaceOp::Fill { edges }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Ruled {
                edges,
                distance,
                angle,
            } => {
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let operands = Self::bodies_of(&set);
                let edges = Self::operand_entities(&set, &operands);
                let distance = self.value(id, "distance", distance, Dimension::LENGTH)?;
                let angle = self.value(id, "angle", angle, Dimension::ANGLE)?;
                op(
                    KernelOp::Surface(SurfaceOp::Ruled {
                        edges,
                        distance,
                        angle,
                    }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Untrim { .. } => return Err(self.not_lowered(id, "untrim")),
            SurfaceFeature::DeleteHole { edges } => {
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let operands = Self::bodies_of(&set);
                let edges = Self::operand_entities(&set, &operands);
                op(
                    KernelOp::Surface(SurfaceOp::DeleteHole { edges }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::CutWithSurface {
                body,
                surface,
                keep,
            } => {
                let b = self.single(i, id, "body", body, Some(TopoKind::Body))?;
                let s = self.single(i, id, "surface", surface, Some(TopoKind::Body))?;
                let solid = b.entities().iter().next().expect("one").body as usize;
                let sheet = s.entities().iter().next().expect("one").body as usize;
                if solid == sheet {
                    return Err(self.error(
                        id,
                        Some("surface"),
                        DiagCode::RefKindMismatch,
                        "a body cannot be cut with itself".into(),
                        Vec::new(),
                    ));
                }
                op(
                    KernelOp::Surface(SurfaceOp::CutWithSurface { keep: *keep }),
                    vec![solid, sheet],
                    simple(Vec::new()),
                )
            }
        })
    }

    // ------------------------------------------------------------------- sheet metal

    fn sheet_rules(&mut self, id: &FeatureId, r: &SheetMetalRules) -> Result<SheetRules, Halt> {
        let (mut t, mut radius) = (
            self.value(id, "rules.thickness", &r.thickness, Dimension::LENGTH)?,
            self.value(id, "rules.bend_radius", &r.bend_radius, Dimension::LENGTH)?,
        );
        let mut rule = match &r.rule {
            BendRule::KFactor { k } => BendRuleValue::KFactor {
                k: self.value(id, "rules.k", k, Dimension::DIMENSIONLESS)?,
            },
            BendRule::Table { table } => BendRuleValue::Table {
                table: table.clone(),
            },
            BendRule::Deduction { value } => BendRuleValue::Deduction {
                value: self.value(id, "rules.deduction", value, Dimension::LENGTH)?,
            },
            BendRule::Allowance { value } => BendRuleValue::Allowance {
                value: self.value(id, "rules.allowance", value, Dimension::LENGTH)?,
            },
        };
        if let Some((table, gauge)) = &r.gauge {
            let Some(row) = table.rows.iter().find(|g| &g.gauge == gauge) else {
                return Err(self.error(
                    id,
                    Some("rules.gauge"),
                    DiagCode::RegenInvalid,
                    format!("gauge table `{}` has no gauge `{}`", table.name, gauge),
                    Vec::new(),
                ));
            };
            t = row.thickness;
            radius = row.bend_radius;
            if matches!(r.rule, BendRule::KFactor { .. }) {
                rule = BendRuleValue::KFactor { k: row.k_factor };
            }
        }
        if t <= 0.0 || radius < 0.0 {
            return Err(self.error(
                id,
                Some("rules"),
                DiagCode::RegenInvalid,
                "sheet thickness must be positive and the bend radius non-negative".into(),
                Vec::new(),
            ));
        }
        Ok(SheetRules {
            thickness: t,
            bend_radius: radius,
            rule,
        })
    }

    /// The sheet record of the base flange that made a body.
    pub(super) fn sheet_of(
        &self,
        body: usize,
    ) -> Option<(
        FeatureId,
        SheetRules,
        Option<crate::model::sheet_metal::Section>,
        f64,
    )> {
        for n in &self.st.bodies[body].names.body {
            if let Some(FeatureRecord::Sheet {
                rules,
                section,
                depth,
            }) = self.st.records.get(n.maker())
            {
                return Some((n.maker().clone(), rules.clone(), section.clone(), *depth));
            }
        }
        None
    }

    fn sheet_metal(
        &mut self,
        i: usize,
        id: &FeatureId,
        s: &SheetMetalFeature,
    ) -> Result<Compiled, Halt> {
        let simple = NamingContext::simple(id, Vec::new());
        match s {
            SheetMetalFeature::BaseFlange {
                profile,
                depth,
                side,
                rules,
            } => {
                let rules = self.sheet_rules(id, rules)?;
                let (t, r) = (rules.thickness, rules.bend_radius);
                let body = None;
                let Some(depth) = depth else {
                    // A closed profile: a flat plate.
                    let p = self.profile(i, id, "profile", profile)?;
                    let mut tool = LabeledTool::new(ToolShape::prism(p.clone(), 0.0, t));
                    for seg in p.loops.iter().flatten() {
                        tool.sides.push((
                            seg.curve.clone(),
                            FaceLabel::key(Role::SheetEdge, seg.curve.0.clone()),
                        ));
                    }
                    tool.start = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(0));
                    tool.end = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(1));
                    tool.body = body;
                    self.st.records.insert(
                        id.clone(),
                        FeatureRecord::Sheet {
                            rules,
                            section: None,
                            depth: 0.0,
                        },
                    );
                    return Ok(op(
                        labeled(vec![tool], Vec::new(), Combine::New, false),
                        Vec::new(),
                        simple,
                    ));
                };
                let depth = self.value(id, "depth", depth, Dimension::LENGTH)?;
                let (frame, segs) = self.chain(i, id, "profile", profile)?;
                if segs.iter().any(|s| !matches!(s.kind, SegmentKind::Line)) {
                    return Err(self.error(
                        id,
                        Some("profile"),
                        DiagCode::SketchOpenProfile,
                        "a sheet metal chain is drawn with lines; bends come from its corners"
                            .into(),
                        Vec::new(),
                    ));
                }
                if r <= 0.0 {
                    return Err(self.error(
                        id,
                        Some("rules.bend_radius"),
                        DiagCode::RegenInvalid,
                        "bent sheets need a positive bend radius".into(),
                        Vec::new(),
                    ));
                }
                let mut pts: Vec<[f64; 2]> = segs.iter().map(|s| s.start).collect();
                pts.push(segs.last().expect("segments").end);
                let curves: Vec<_> = segs.iter().map(|s| s.curve.clone()).collect();
                let sec = section(&pts, &curves, *side, t, r).map_err(|m| {
                    self.error(id, Some("profile"), DiagCode::RegenInvalid, m, Vec::new())
                })?;
                let mut tools = Vec::new();
                let nflats = sec.flats.len();
                for (k, f) in sec.flats.iter().enumerate() {
                    let off =
                        |p: [f64; 2], o: f64| [p[0] + o * f.normal[0], p[1] + o * f.normal[1]];
                    let c = f.curve.0.clone();
                    let corners = vec![
                        (format!("{}.near", c), off(f.start, f.near)),
                        (format!("{}.b", c), off(f.end, f.near)),
                        (format!("{}.far", c), off(f.end, f.far)),
                        (format!("{}.a", c), off(f.start, f.far)),
                    ];
                    let mut tool =
                        LabeledTool::new(ToolShape::prism(polygon(frame, &corners), 0.0, depth));
                    let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{}.{}", c, s));
                    tool.sides.push((
                        sid("near"),
                        FaceLabel::key(Role::SheetFace, c.clone()).with_ordinal(0),
                    ));
                    tool.sides.push((
                        sid("far"),
                        FaceLabel::key(Role::SheetFace, c.clone()).with_ordinal(1),
                    ));
                    // Chain ends are sheet edges; ends meeting a bend lie inside the body.
                    tool.sides.push((
                        sid("a"),
                        FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(if k == 0 {
                            0
                        } else {
                            4
                        }),
                    ));
                    tool.sides.push((
                        sid("b"),
                        FaceLabel::key(Role::SheetEdge, c.clone())
                            .with_ordinal(if k + 1 == nflats { 1 } else { 5 }),
                    ));
                    tool.start = Some(FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(2));
                    tool.end = Some(FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(3));
                    tool.body = body.clone();
                    tools.push(tool);
                }
                for b in &sec.bends {
                    tools.push(bend_tool(frame, b, depth, body.clone()));
                }
                self.st.records.insert(
                    id.clone(),
                    FeatureRecord::Sheet {
                        rules,
                        section: Some(sec),
                        depth,
                    },
                );
                Ok(op(
                    labeled(tools, Vec::new(), Combine::New, false),
                    Vec::new(),
                    simple,
                ))
            }
            SheetMetalFeature::EdgeFlange {
                edges,
                length,
                angle,
                length_reference,
                position,
                radius,
                ..
            } => {
                let l = self.value(id, "length", length, Dimension::LENGTH)?;
                let theta = self.value(id, "angle", angle, Dimension::ANGLE)?;
                let rr = match radius {
                    Some(r) => Some(self.value(id, "radius", r, Dimension::LENGTH)?),
                    None => None,
                };
                self.flange(
                    i,
                    id,
                    edges,
                    l,
                    theta,
                    *length_reference,
                    *position,
                    rr,
                    &simple,
                )
            }
            SheetMetalFeature::Hem {
                edges,
                kind,
                length,
            } => {
                let l = self.value(id, "length", length, Dimension::LENGTH)?;
                let gap = match kind {
                    HemKind::Closed => None,
                    HemKind::Open { gap } => {
                        Some(self.value(id, "kind.gap", gap, Dimension::LENGTH)?)
                    }
                    HemKind::Teardrop { radius, angle } | HemKind::Rolled { radius, angle } => {
                        // A roll of `angle` at inner radius `radius`, then the straight leg
                        // of `length` (a teardrop's leg runs back towards the sheet).
                        let r = self.value(id, "kind.radius", radius, Dimension::LENGTH)?;
                        let theta = self.value(id, "kind.angle", angle, Dimension::ANGLE)?;
                        if r <= 0.0 {
                            return Err(self.error(
                                id,
                                Some("kind.radius"),
                                DiagCode::RegenInvalid,
                                "a rolled or teardrop hem needs a positive radius".into(),
                                Vec::new(),
                            ));
                        }
                        if matches!(kind, HemKind::Teardrop { .. })
                            && theta <= std::f64::consts::PI + 1e-12
                        {
                            return Err(self.error(
                                id,
                                Some("kind.angle"),
                                DiagCode::RegenInvalid,
                                "a teardrop hem turns more than 180°; use an open hem".into(),
                                Vec::new(),
                            ));
                        }
                        return self.flange_with(
                            i,
                            id,
                            edges,
                            l,
                            theta,
                            LengthReference::Tangent,
                            crate::model::sheet_metal::FlangePosition::BendOutside,
                            Some(r),
                            &simple,
                            true,
                        );
                    }
                };
                // A closed hem folds flat on the sheet: an inner radius of a hundredth of the
                // thickness (the fold line); an open hem leaves `gap` between the faces.
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let body = set.iter().next().map(|h| h.body as usize).unwrap_or(0);
                let t = self.sheet_of(body).map(|s| s.1.thickness).unwrap_or(0.0);
                let r = gap.map(|g| g / 2.0).unwrap_or(t * 0.01);
                self.flange(
                    i,
                    id,
                    edges,
                    l,
                    std::f64::consts::PI,
                    LengthReference::Tangent,
                    crate::model::sheet_metal::FlangePosition::BendOutside,
                    Some(r),
                    &simple,
                )
            }
            SheetMetalFeature::BreakCorner { edges, kind } => {
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let operands = Self::bodies_of(&set);
                let es = Self::operand_entities(&set, &operands);
                Ok(op(
                    match kind {
                        crate::model::sheet_metal::BreakKind::Fillet { radius } => {
                            KernelOp::Fillet {
                                edges: es,
                                radius: self.value(id, "kind.radius", radius, Dimension::LENGTH)?,
                                variable: Vec::new(),
                                profile: BlendProfile::Circular,
                                setbacks: Vec::new(),
                            }
                        }
                        crate::model::sheet_metal::BreakKind::Chamfer { distance } => {
                            KernelOp::Chamfer {
                                edges: es,
                                size: ChamferSize::Equal {
                                    distance: self.value(
                                        id,
                                        "kind.distance",
                                        distance,
                                        Dimension::LENGTH,
                                    )?,
                                },
                            }
                        }
                    },
                    operands,
                    simple,
                ))
            }
            SheetMetalFeature::FlatPattern { fixed_face, .. } => {
                let h = self.face_handle(i, id, "fixed_face", fixed_face)?;
                if let Some((maker, rules, None, _)) = self.sheet_of(h.body as usize) {
                    return self.plate_flat(i, id, &h, &maker, &rules, &simple);
                }
                let Some((maker, rules, Some(sec), depth)) = self.sheet_of(h.body as usize) else {
                    return Err(self.error(
                        id,
                        Some("fixed_face"),
                        DiagCode::RefKindMismatch,
                        "the fixed face is not on a bent sheet".into(),
                        Vec::new(),
                    ));
                };
                let names = self.st.entity_names(&h);
                let fixed = names.iter().find_map(|n| match n {
                    EntityName::Generated(g) if g.feature == maker && g.role == Role::SheetFace => {
                        match &g.tag {
                            crate::refs::name::SourceTag::Key { key } => Some(key.clone()),
                            _ => None,
                        }
                    }
                    _ => None,
                });
                let Some(fixed) = fixed else {
                    return Err(self.error(
                        id,
                        Some("fixed_face"),
                        DiagCode::RefKindMismatch,
                        "the fixed face is not a flat of the sheet".into(),
                        Vec::new(),
                    ));
                };
                let dev = crate::model::sheet_metal::develop(&sec, &rules.rule)
                    .map_err(|m| self.error(id, None, DiagCode::RegenInvalid, m, Vec::new()))?;
                let Some(((_, frame), flat)) = self
                    .st
                    .sketches
                    .get(&self.sketch_of(&maker))
                    .cloned()
                    .zip(sec.flats.iter().find(|f| f.curve.0 == fixed).cloned())
                else {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RegenInvalid,
                        "the sheet's sketch is gone".into(),
                        Vec::new(),
                    ));
                };
                let xf = dev
                    .regions
                    .iter()
                    .find_map(|r| match r {
                        StripRegion::Flat { curve, start, .. } if curve.0 == fixed => Some(*start),
                        _ => None,
                    })
                    .unwrap_or(0.0);
                let p0 = frame.to_world([
                    flat.start[0] + flat.near * flat.normal[0],
                    flat.start[1] + flat.near * flat.normal[1],
                ]);
                let p1 = frame.to_world([
                    flat.end[0] + flat.near * flat.normal[0],
                    flat.end[1] + flat.near * flat.normal[1],
                ]);
                let xdir = normalize(sub(p1, p0));
                let n3 = normalize(add(
                    scale(frame.x, flat.normal[0]),
                    scale(frame.y, flat.normal[1]),
                ));
                let fr = frame_of(add(p0, scale(xdir, -xf)), xdir, frame.z);
                let t = rules.thickness;
                let thick = if dot(fr.z, n3) >= 0.0 { t } else { -t };
                let mut tools = Vec::new();
                for r in &dev.regions {
                    let (key, start, len, role) = match r {
                        StripRegion::Flat {
                            curve,
                            start,
                            length,
                        } => (curve.0.clone(), *start, *length, Role::SheetFace),
                        StripRegion::Bend { key, start, length } => {
                            (key.clone(), *start, *length, Role::BendFace)
                        }
                    };
                    let corners = vec![
                        (format!("{}.y0", key), [start, 0.0]),
                        (format!("{}.x1", key), [start + len, 0.0]),
                        (format!("{}.y1", key), [start + len, depth]),
                        (format!("{}.x0", key), [start, depth]),
                    ];
                    let mut tool =
                        LabeledTool::new(ToolShape::prism(polygon(fr, &corners), 0.0, thick));
                    let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{}.{}", key, s));
                    for (s, o) in [("y0", 2), ("y1", 3), ("x0", 0), ("x1", 1)] {
                        tool.sides.push((
                            sid(s),
                            FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(o),
                        ));
                    }
                    tool.start = Some(FaceLabel::key(role, key.clone()).with_ordinal(0));
                    tool.end = Some(FaceLabel::key(role, key.clone()).with_ordinal(1));
                    tools.push(tool);
                }
                // The layout: the strip in `fr`, its bend lines across it.
                let up_side = dot(fr.z, n3) >= 0.0;
                let corner = |x: f64, y: f64| [x, y];
                let outline = [
                    ("y0", corner(0.0, 0.0), corner(dev.length, 0.0)),
                    ("x1", corner(dev.length, 0.0), corner(dev.length, depth)),
                    ("y1", corner(dev.length, depth), corner(0.0, depth)),
                    ("x0", corner(0.0, depth), corner(0.0, 0.0)),
                ];
                let layout = crate::model::sheet_metal::FlatLayout {
                    frame: fr,
                    depth: thick,
                    loops: vec![outline
                        .iter()
                        .map(|(c, a, b)| line_seg(c, *a, *b))
                        .collect()],
                    bend_lines: dev
                        .bend_lines
                        .iter()
                        .map(|b| crate::model::sheet_metal::FlatBendLine {
                            key: b.key.clone(),
                            feature: maker.clone(),
                            start: [b.position, 0.0],
                            end: [b.position, depth],
                            angle: b.angle,
                            inner_radius: b.inner_radius,
                            allowance: b.allowance,
                            deduction: b.deduction,
                            direction: if (b.direction
                                == crate::model::sheet_metal::BendDirection::Up)
                                == up_side
                            {
                                crate::model::sheet_metal::BendDirection::Up
                            } else {
                                crate::model::sheet_metal::BendDirection::Down
                            },
                        })
                        .collect(),
                };
                self.st.records.insert(
                    id.clone(),
                    FeatureRecord::FlatPattern {
                        layout,
                        development: Some(dev),
                    },
                );
                Ok(op(
                    labeled(tools, Vec::new(), Combine::New, false),
                    Vec::new(),
                    simple,
                ))
            }
            SheetMetalFeature::MiterFlange {
                profile,
                edges,
                gap,
            } => self.miter_flange(i, id, profile, edges, gap, &simple),
            SheetMetalFeature::Jog {
                face,
                line,
                offset,
                angle,
            } => {
                let theta = self.value(id, "angle", angle, Dimension::ANGLE)?;
                let rise = self.value(id, "offset", offset, Dimension::LENGTH)?;
                self.bend_chain(i, id, face, line, theta, None, Some(rise), None, &simple)
            }
            SheetMetalFeature::SketchedBend {
                face,
                line,
                angle,
                radius,
                fixed,
            } => {
                let theta = self.value(id, "angle", angle, Dimension::ANGLE)?;
                let r = match radius {
                    Some(r) => Some(self.value(id, "radius", r, Dimension::LENGTH)?),
                    None => None,
                };
                self.bend_chain(i, id, face, line, theta, r, None, Some(*fixed), &simple)
            }
            SheetMetalFeature::LoftedBend { .. } => Err(self.not_lowered(id, "lofted bends")),
            SheetMetalFeature::ConvertToSheetMetal { .. } => {
                Err(self.not_lowered(id, "convert to sheet metal"))
            }
            SheetMetalFeature::ClosedCorner { .. } => Err(self.not_lowered(id, "closed corners")),
            SheetMetalFeature::CornerRelief { .. } => Err(self.not_lowered(id, "corner reliefs")),
            SheetMetalFeature::TabAndSlot { .. } => Err(self.not_lowered(id, "tabs and slots")),
            SheetMetalFeature::FormTool { .. } => Err(self.not_lowered(id, "form tools")),
            SheetMetalFeature::Unfold { fixed_face, bends } => {
                if !bends.is_empty() {
                    return Err(self.not_lowered(
                        id,
                        "unfolding some of a sheet's bends (unfold all of them: leave `bends` empty)",
                    ));
                }
                let h = self.face_handle(i, id, "fixed_face", fixed_face)?;
                let body = h.body as usize;
                let Some((maker, _, sec, _)) = self.sheet_of(body) else {
                    return Err(self.error(
                        id,
                        Some("fixed_face"),
                        DiagCode::RefKindMismatch,
                        "the fixed face is not on a sheet metal body".into(),
                        Vec::new(),
                    ));
                };
                // A plate's flat pattern checks that only its bends changed it.
                if sec.is_some() {
                    self.only_made_by(id, body, &[maker], "unfold")?;
                }
                // The flat pattern's strip, in place of the bent sheet.
                let flat = SheetMetalFeature::FlatPattern {
                    fixed_face: fixed_face.clone(),
                    grain: None,
                };
                let compiled = self.sheet_metal(i, id, &flat)?;
                Ok(self.in_place(compiled, body))
            }
            SheetMetalFeature::Fold { bends } => {
                if !bends.is_empty() {
                    return Err(self.not_lowered(
                        id,
                        "folding some of a sheet's bends (fold all of them: leave `bends` empty)",
                    ));
                }
                // The latest unfold before this feature, and the sheet it flattened.
                let unfold = self.doc.features[..i].iter().rev().find(|f| {
                    matches!(
                        f.kind,
                        FeatureKind::SheetMetal(SheetMetalFeature::Unfold { .. })
                    ) && self.st.records.contains_key(&f.id)
                });
                let Some(unfold) = unfold.map(|f| f.id.clone()) else {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RefKindMismatch,
                        "there is no unfolded sheet to fold".into(),
                        Vec::new(),
                    ));
                };
                let body = self
                    .st
                    .bodies
                    .iter()
                    .position(|b| b.names.faces.iter().flatten().any(|n| n.maker() == &unfold));
                let Some(body) = body else {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RefKindMismatch,
                        format!("the sheet `{unfold}` unfolded is gone"),
                        Vec::new(),
                    ));
                };
                // The sheet: the body's own record, else the base flange the unfold's fixed
                // face was named on (a rebuilt body can lose its body name).
                let from_face = match self.doc.feature(&unfold).map(|f| &f.kind) {
                    Some(FeatureKind::SheetMetal(SheetMetalFeature::Unfold {
                        fixed_face: Reference::Name { name, .. },
                        ..
                    })) => Some(name.maker().clone()),
                    _ => None,
                };
                let maker = self.sheet_of(body).map(|s| s.0).or(from_face).filter(|m| {
                    matches!(
                        self.st.records.get(m),
                        Some(FeatureRecord::Sheet {
                            section: Some(_),
                            ..
                        })
                    )
                });
                let Some(maker) = maker else {
                    return Err(self.not_lowered(
                        id,
                        "folding a plate's sketched bends and jogs back (suppress the unfold to see the bent plate)",
                    ));
                };
                // Faces the rebuild kept may still carry the base flange's names.
                self.only_made_by(id, body, &[unfold, maker.clone()], "fold")?;
                let Some(base) = self.doc.feature(&maker).cloned() else {
                    return Err(self.not_lowered(id, "folding this sheet"));
                };
                let FeatureKind::SheetMetal(base_kind) = &base.kind else {
                    return Err(self.not_lowered(id, "folding this sheet"));
                };
                // The base flange's own tools and names: references to the bent sheet's
                // faces resolve again after the fold.
                let compiled = self.sheet_metal(i, &maker, base_kind)?;
                Ok(self.in_place(compiled, body))
            }
        }
    }

    /// Refuses a whole-sheet rebuild (unfold, fold) of a body that features besides `makers`
    /// changed: the rebuild comes from the base flange's section and would drop them.
    fn only_made_by(
        &mut self,
        id: &FeatureId,
        body: usize,
        makers: &[FeatureId],
        what: &str,
    ) -> Result<(), Halt> {
        let changed = self.st.bodies[body]
            .names
            .faces
            .iter()
            .any(|names| !names.iter().any(|n| makers.contains(n.maker())));
        if changed {
            return Err(self.not_lowered(
                id,
                &format!(
                    "{what} of a sheet that features after its base flange changed (flanges, hems, cuts)"
                ),
            ));
        }
        Ok(())
    }

    /// A new-body sheet op turned into a rebuild of `body` in place: the body is carved away
    /// whole and the op's tools take its place.
    fn in_place(&self, compiled: Compiled, body: usize) -> Compiled {
        let Compiled::Op {
            op: KernelOp::Labeled { tools, .. },
            naming,
            ..
        } = compiled
        else {
            return compiled;
        };
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for f in &self.st.bodies[body].topology.faces {
            for k in 0..3 {
                lo[k] = lo[k].min(f.geom.bbox_min[k]);
                hi[k] = hi[k].max(f.geom.bbox_max[k]);
            }
        }
        let margin = 1e-3 + 0.01 * (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
        let fr = frame_of([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let corners = vec![
            ("c0".to_string(), [lo[0] - margin, lo[1] - margin]),
            ("c1".to_string(), [hi[0] + margin, lo[1] - margin]),
            ("c2".to_string(), [hi[0] + margin, hi[1] + margin]),
            ("c3".to_string(), [lo[0] - margin, hi[1] + margin]),
        ];
        let carve = LabeledTool::new(ToolShape::prism(
            polygon(fr, &corners),
            lo[2] - margin,
            hi[2] + margin,
        ));
        let tools = tools
            .into_iter()
            .map(|mut t| {
                t.body = None;
                t
            })
            .collect();
        op(
            labeled(tools, vec![carve], Combine::Add, false),
            vec![body],
            naming,
        )
    }

    fn sketch_of(&self, base: &FeatureId) -> FeatureId {
        match self.doc.feature(base).map(|f| &f.kind) {
            Some(FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { profile, .. })) => {
                profile.sketch.clone()
            }
            _ => base.clone(),
        }
    }

    /// Flanges (and hems) on sheet edges: a bend and a flat grown from each edge's thickness
    /// face, labelled by the edge they grow from.
    #[allow(clippy::too_many_arguments)]
    fn flange(
        &mut self,
        i: usize,
        id: &FeatureId,
        edges: &[Reference],
        length: f64,
        theta: f64,
        reference: LengthReference,
        position: crate::model::sheet_metal::FlangePosition,
        radius: Option<f64>,
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        self.flange_with(
            i, id, edges, length, theta, reference, position, radius, naming, false,
        )
    }

    /// [`Self::flange`]; `roll` lets the bend turn past 180° (rolled and teardrop hems), up
    /// to where its outer face would meet the sheet.
    #[allow(clippy::too_many_arguments)]
    fn flange_with(
        &mut self,
        i: usize,
        id: &FeatureId,
        edges: &[Reference],
        length: f64,
        theta: f64,
        reference: LengthReference,
        position: crate::model::sheet_metal::FlangePosition,
        radius: Option<f64>,
        naming: &NamingContext,
        roll: bool,
    ) -> Result<Compiled, Halt> {
        use crate::model::sheet_metal::FlangePosition as P;
        let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
        let operands = Self::bodies_of(&set);
        if operands.len() != 1 {
            return Err(self.error(
                id,
                Some("edges"),
                DiagCode::RefKindMismatch,
                "flange edges must be on one sheet".into(),
                Vec::new(),
            ));
        }
        let body = operands[0];
        let Some((_, rules, _, _)) = self.sheet_of(body) else {
            return Err(self.error(
                id,
                Some("edges"),
                DiagCode::RefKindMismatch,
                "the edges are not on a sheet metal body".into(),
                Vec::new(),
            ));
        };
        let t = rules.thickness;
        let r = radius.unwrap_or(rules.bend_radius);
        // A roll's outer face meets the sheet's outer face where r + (r + t) sin a = 0.
        let max = if roll {
            1.5 * std::f64::consts::PI + (r / (r + t)).asin()
        } else {
            std::f64::consts::PI
        };
        if theta <= 0.0 || theta > max + 1e-12 {
            return Err(self.error(
                id,
                Some(if roll { "kind.angle" } else { "angle" }),
                DiagCode::RegenInvalid,
                if roll {
                    format!(
                        "this roll turns at most {:.1}° before it meets the sheet",
                        max.to_degrees()
                    )
                } else {
                    "a flange angle lies in (0°, 180°]".into()
                },
                Vec::new(),
            ));
        }
        let mut tools = Vec::new();
        let mut carve = Vec::new();
        for h in &set {
            let topo = &self.st.bodies[body].topology;
            let e = topo.edges[h.index as usize].clone();
            let (fa, fb) = (e.faces[0], e.faces[1]);
            let (ga, gb) = (topo.faces[fa as usize].geom, topo.faces[fb as usize].geom);
            let (Some(na), Some(nb)) = (plane_normal(&ga), plane_normal(&gb)) else {
                return Err(self.error(
                    id,
                    Some("edges"),
                    DiagCode::RefKindMismatch,
                    "a flange edge lies between two flat faces".into(),
                    Vec::new(),
                ));
            };
            // The thickness face is the narrower one (by its extent across the edge).
            let dir = normalize(sub(e.geom.end, e.geom.start));
            let width = |g: &FaceGeom| {
                let ext = sub(g.bbox_max, g.bbox_min);
                let across = sub(ext, scale(dir, dot(ext, dir)));
                dot(across, across).sqrt()
            };
            let (n_sheet, n_thick) = if width(&ga) >= width(&gb) {
                (na, nb)
            } else {
                (nb, na)
            };
            let ossb = |rad: f64| (theta / 2.0).tan() * rad;
            let shift = match position {
                P::BendOutside => 0.0,
                P::MaterialInside => -ossb(r + t),
                P::MaterialOutside => -ossb(r),
            };
            let flat = match reference {
                LengthReference::Tangent => length,
                LengthReference::OuterVirtualSharp => length - ossb(r + t),
                LengthReference::InnerVirtualSharp => length - ossb(r),
            };
            if roll && flat > 0.0 {
                // The straight leg must stay off the sheet (y ≥ 0 over the sheet, x ≤ 0).
                let d = [theta.cos(), theta.sin()];
                let c = [0.0, r];
                for rad in [r, r + t] {
                    let a1 = -std::f64::consts::FRAC_PI_2 + theta;
                    let tip = [
                        c[0] + rad * a1.cos() + flat * d[0],
                        c[1] + rad * a1.sin() + flat * d[1],
                    ];
                    if tip[0] <= 1e-9 && tip[1] < -1e-9 {
                        return Err(self.error(
                            id,
                            Some("length"),
                            DiagCode::RegenInvalid,
                            "the hem's straight leg runs into the sheet; shorten it".into(),
                            Vec::new(),
                        ));
                    }
                }
            }
            if flat <= 0.0 && theta < std::f64::consts::PI - 1e-9 {
                return Err(self.error(
                    id,
                    Some("length"),
                    DiagCode::RegenInvalid,
                    "the flange is shorter than its bend".into(),
                    Vec::new(),
                ));
            }
            let origin = add(e.geom.start, scale(n_thick, shift));
            let fr = frame_of(origin, n_thick, n_sheet);
            let s0 = dot(sub(e.geom.start, fr.origin), fr.z);
            let s1 = dot(sub(e.geom.end, fr.origin), fr.z);
            let (z0, z1) = (s0.min(s1), s0.max(s1));
            let edge_tag = KernelTag::InputEdge {
                edge: OperandEntity {
                    operand: 0,
                    entity: h.index,
                },
            };
            let lab = |role: Role, o: u32| FaceLabel {
                role,
                tag: edge_tag.clone(),
                ordinal: Some(o),
            };
            // The bend: centre (0, r), from −90° by theta.
            let c = [0.0, r];
            let a0 = -std::f64::consts::FRAC_PI_2;
            let a1 = a0 + theta;
            let at = |rad: f64, a: f64| [c[0] + rad * a.cos(), c[1] + rad * a.sin()];
            let sid = |s: &str| crate::ids::SketchEntityId::new(s);
            let bend_profile = CanonicalProfile {
                frame: fr,
                loops: vec![vec![
                    ProfileSegment {
                        curve: sid("in"),
                        kind: SegmentKind::Arc {
                            center: c,
                            clockwise: false,
                        },
                        start: at(r, a0),
                        end: at(r, a1),
                    },
                    line_seg("e1", at(r, a1), at(r + t, a1)),
                    ProfileSegment {
                        curve: sid("out"),
                        kind: SegmentKind::Arc {
                            center: c,
                            clockwise: true,
                        },
                        start: at(r + t, a1),
                        end: at(r + t, a0),
                    },
                    line_seg("e0", at(r + t, a0), at(r, a0)),
                ]],
                merge_collinear: false,
            };
            let mut bend = LabeledTool::new(ToolShape::prism(bend_profile, z0, z1));
            bend.sides.push((sid("in"), lab(Role::BendFace, 0)));
            bend.sides.push((sid("out"), lab(Role::BendFace, 1)));
            bend.sides.push((sid("e1"), lab(Role::SheetEdge, 6)));
            bend.sides.push((sid("e0"), lab(Role::SheetEdge, 7)));
            bend.start = Some(lab(Role::SheetEdge, 2));
            bend.end = Some(lab(Role::SheetEdge, 3));
            tools.push(bend);
            if flat > 1e-12 {
                let d = [theta.cos(), theta.sin()];
                let (ci, co) = (at(r, a1), at(r + t, a1));
                let fwd = |p: [f64; 2]| [p[0] + flat * d[0], p[1] + flat * d[1]];
                let corners = vec![
                    ("in".to_string(), ci),
                    ("tip".to_string(), fwd(ci)),
                    ("out".to_string(), fwd(co)),
                    ("root".to_string(), co),
                ];
                let mut ft = LabeledTool::new(ToolShape::prism(polygon(fr, &corners), z0, z1));
                ft.sides.push((sid("in"), lab(Role::SheetFace, 0)));
                ft.sides.push((sid("out"), lab(Role::SheetFace, 1)));
                ft.sides.push((sid("tip"), lab(Role::SheetEdge, 1)));
                ft.sides.push((sid("root"), lab(Role::SheetEdge, 8)));
                ft.start = Some(lab(Role::SheetEdge, 4));
                ft.end = Some(lab(Role::SheetEdge, 5));
                tools.push(ft);
            }
            if shift < 0.0 {
                // Trim the sheet back to where the bend now starts.
                let corners = vec![
                    ("c0".to_string(), [0.0, -t - 1e-6]),
                    ("c1".to_string(), [-shift + 1e-9, -t - 1e-6]),
                    ("c2".to_string(), [-shift + 1e-9, 1e-6]),
                    ("c3".to_string(), [0.0, 1e-6]),
                ];
                let mut ct = LabeledTool::new(ToolShape::prism(polygon(fr, &corners), z0, z1));
                ct.sides.push((sid("c0"), lab(Role::Relief, 0)));
                carve.push(ct);
            }
        }
        Ok(op(
            labeled(tools, carve, Combine::Add, false),
            operands,
            naming.clone(),
        ))
    }

    /// A sketched bend (one bend) or a jog (a bend, a flat and the opposite bend) of a flat
    /// plate along a sketch line.
    ///
    /// The line is the first bend's centre line in the flat, so the plate's developed length
    /// is kept: the strip each bend consumes is its bend allowance, centred on the line for
    /// the first bend and following it for the rest. The side `fixed` (by the line's left
    /// normal; the jog keeps the negative side) stays; the rest turns by `angle` towards the
    /// face's outward normal (away from it when negative). A jog's offset is how far the
    /// moved flat's faces move along the sheet normal; its middle flat is what is left of
    /// the offset after the two bends.
    ///
    /// The plate is the base flange's closed profile, unchanged by later features; its
    /// edges cross the bend strip square to the line, and holes stay clear of the strip.
    #[allow(clippy::too_many_arguments)]
    fn bend_chain(
        &mut self,
        i: usize,
        id: &FeatureId,
        face: &Reference,
        line: &(FeatureId, crate::ids::SketchEntityId),
        theta: f64,
        radius: Option<f64>,
        jog: Option<f64>,
        fixed: Option<crate::refs::name::PlaneSide>,
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        use crate::refs::name::PlaneSide;
        let h = self.face_handle(i, id, "face", face)?;
        let body = h.body as usize;
        let invalid = |me: &mut Self, param: &str, message: String| {
            me.error(id, Some(param), DiagCode::RegenInvalid, message, Vec::new())
        };
        let Some((maker, rules, None, _)) = self.sheet_of(body) else {
            return Err(self.not_lowered(
                id,
                "bends of an already bent sheet (bend a flat plate, or add the bend to the base flange's chain)",
            ));
        };
        let Some(n_face) = self.face_normal(&h) else {
            return Err(invalid(self, "face", "the bend face is not flat".into()));
        };
        let t = rules.thickness;
        let r = radius.unwrap_or(rules.bend_radius);
        if r <= 0.0 {
            return Err(invalid(
                self,
                "radius",
                "a bend needs a positive radius".into(),
            ));
        }
        let a = theta.abs();
        if a <= 1e-9 || a > std::f64::consts::PI + 1e-12 {
            return Err(invalid(
                self,
                "angle",
                "a bend angle lies in (0°, 180°]".into(),
            ));
        }
        let ba = crate::model::sheet_metal::bend_allowance(&rules.rule, t, r, a)
            .map_err(|m| invalid(self, "angle", m))?;
        // The middle flat of a jog: the rise left after two bends.
        let middle = match jog {
            None => None,
            Some(rise) => {
                if a >= std::f64::consts::PI - 1e-9 {
                    return Err(invalid(
                        self,
                        "angle",
                        "a jog's angle lies in (0°, 180°)".into(),
                    ));
                }
                let flat = (rise.abs() - (2.0 * r + t) * (1.0 - a.cos())) / a.sin();
                if flat < -1e-12 {
                    return Err(invalid(
                        self,
                        "offset",
                        format!(
                            "the offset is less than the two bends rise ({:.3} mm at this angle and radius)",
                            (2.0 * r + t) * (1.0 - a.cos()) * 1e3
                        ),
                    ));
                }
                Some(flat.max(0.0))
            }
        };
        // The plate: the base flange's profile, between z = 0 and z = t of its frame.
        let profile_ref = match self.doc.feature(&maker).map(|f| &f.kind) {
            Some(FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { profile, .. })) => {
                profile.clone()
            }
            _ => return Err(self.not_lowered(id, "bends of this sheet")),
        };
        let plate = self.profile(i, id, "face", &profile_ref)?;
        let pf = plate.frame;
        let n = pf.z;
        // The bend turns towards b; the face whose outward normal is b is the inner face.
        let toward = if theta > 0.0 {
            n_face
        } else {
            scale(n_face, -1.0)
        };
        let b = if dot(toward, n) >= 0.0 {
            n
        } else {
            scale(n, -1.0)
        };
        let inner_height = if dot(b, n) > 0.0 { t } else { 0.0 };
        // The line, in world coordinates on the inner face.
        self.gate_feature(i, id, &line.0, "line")?;
        let Some((solved, sframe)) = self.st.sketches.get(&line.0).cloned() else {
            return Err(self.error(
                id,
                Some("line"),
                DiagCode::RefKindMismatch,
                format!("`{}` is not a sketch", line.0),
                Vec::new(),
            ));
        };
        use crate::model::sketch::PointRole;
        let (Some(p0), Some(p1)) = (
            solved
                .points
                .get(&(line.1.clone(), PointRole::Start))
                .copied(),
            solved
                .points
                .get(&(line.1.clone(), PointRole::End))
                .copied(),
        ) else {
            return Err(self.error(
                id,
                Some("line"),
                DiagCode::RefKindMismatch,
                format!("`{}` has no line `{}`", line.0, line.1),
                Vec::new(),
            ));
        };
        let (w0, w1) = (sframe.to_world(p0), sframe.to_world(p1));
        let on_inner = |p: [f64; 3]| {
            let h = dot(sub(p, pf.origin), n);
            add(p, scale(n, inner_height - h))
        };
        let l0 = on_inner(w0);
        let u = sub(on_inner(w1), l0);
        if dot(u, u).sqrt() < 1e-9 {
            return Err(invalid(
                self,
                "line",
                "the bend line does not lie across the plate".into(),
            ));
        }
        let u = normalize(u);
        // Left normal of the line in the sketch, in world coordinates; `fixed` keeps a side.
        let d2 = [p1[0] - p0[0], p1[1] - p0[1]];
        let left = normalize(add(scale(sframe.x, -d2[1]), scale(sframe.y, d2[0])));
        let left = normalize(sub(left, scale(n, dot(left, n))));
        let m = match fixed.unwrap_or(PlaneSide::Negative) {
            PlaneSide::Positive => scale(left, -1.0),
            PlaneSide::Negative => left,
        };
        let sw = |p: [f64; 3]| {
            let q = sub(on_inner(p), l0);
            [dot(q, m), dot(q, u)]
        };
        // The strip the bends consume, in developed length along m.
        let s0 = -ba / 2.0;
        let s1 = s0 + ba + middle.map_or(0.0, |f| f + ba);
        let tol = 1e-9;
        let to_sw = |seg: &ProfileSegment| (sw(pf.to_world(seg.start)), sw(pf.to_world(seg.end)));
        let reach = |seg: &ProfileSegment| -> (f64, f64) {
            let (a, b) = to_sw(seg);
            match seg.kind {
                SegmentKind::Line => (a[0].min(b[0]), a[0].max(b[0])),
                SegmentKind::Arc { center, .. } => {
                    let c = sw(pf.to_world(center));
                    let rad = ((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2)).sqrt();
                    (c[0] - rad, c[0] + rad)
                }
                SegmentKind::Circle { center, radius } => {
                    let c = sw(pf.to_world(center));
                    (c[0] - radius, c[0] + radius)
                }
                // Ellipses and splines: anywhere, so they count as crossing the strip.
                _ => (f64::NEG_INFINITY, f64::INFINITY),
            }
        };
        let Some(outer) = plate.loops.first() else {
            return Err(invalid(self, "face", "the plate has no outline".into()));
        };
        // Faces other features made (earlier bends, flanges, cuts) must lie wholly on the
        // fixed side: the moved part is rebuilt from the plate's outline.
        let foreign = {
            let live = &self.st.bodies[body];
            live.names.faces.iter().enumerate().any(|(index, names)| {
                if names.iter().any(|n| n.maker() == &maker) {
                    return false;
                }
                let g = &live.topology.faces[index].geom;
                (0..8).any(|k| {
                    let c = [
                        if k & 1 == 0 {
                            g.bbox_min[0]
                        } else {
                            g.bbox_max[0]
                        },
                        if k & 2 == 0 {
                            g.bbox_min[1]
                        } else {
                            g.bbox_max[1]
                        },
                        if k & 4 == 0 {
                            g.bbox_min[2]
                        } else {
                            g.bbox_max[2]
                        },
                    ];
                    dot(sub(c, l0), m) > s0 - tol
                })
            })
        };
        if foreign {
            return Err(self.not_lowered(
                id,
                "bends whose moving side holds faces of other features (bend the plate before adding them there)",
            ));
        }
        let big = 2.0
            * outer
                .iter()
                .map(|g| {
                    let (a2, b2) = to_sw(g);
                    a2[0]
                        .abs()
                        .max(a2[1].abs())
                        .max(b2[0].abs())
                        .max(b2[1].abs())
                })
                .fold(0.0, f64::max)
            + 10.0 * t;
        // Square crossings: every outline segment over the strip is a line of constant w.
        let mut chord: Vec<f64> = Vec::new();
        for seg in outer {
            let (lo, hi) = reach(seg);
            if hi <= s0 + tol || lo >= s1 - tol {
                continue;
            }
            let (a2, b2) = to_sw(seg);
            if !matches!(seg.kind, SegmentKind::Line) || (a2[1] - b2[1]).abs() > tol {
                return Err(self.not_lowered(
                    id,
                    "bends whose strip the plate's edges cross at an angle or along a curve (edges must cross the bend zone square to the line)",
                ));
            }
            if !chord.iter().any(|w| (w - a2[1]).abs() <= tol) {
                chord.push(a2[1]);
            }
        }
        if chord.len() != 2 {
            return Err(invalid(
                self,
                "line",
                "the bend line must cross the plate once, from edge to edge".into(),
            ));
        }
        let (wa, wb) = (chord[0].min(chord[1]), chord[0].max(chord[1]));
        for inner in plate.loops.iter().skip(1) {
            let lo = inner
                .iter()
                .map(|g| reach(g).0)
                .fold(f64::INFINITY, f64::min);
            let hi = inner
                .iter()
                .map(|g| reach(g).1)
                .fold(f64::NEG_INFINITY, f64::max);
            if !(hi < s0 - tol || lo > s1 + tol) {
                return Err(self.not_lowered(id, "bends through a hole or cut-out of the plate"));
            }
        }
        // The moved flat: the outline beyond s1 (shifted to start at 0) and the holes there.
        let mut moved: Vec<ProfileSegment> = Vec::new();
        let n_out = outer.len();
        let start_at = (0..n_out).find(|k| {
            let (a2, b2) = to_sw(&outer[*k]);
            a2[0] < s1 - tol && b2[0] >= s1 - tol
        });
        let Some(start_at) = start_at else {
            return Err(invalid(
                self,
                "line",
                "nothing of the plate lies past the bend".into(),
            ));
        };
        let cut = |a2: [f64; 2], b2: [f64; 2]| {
            let f = (s1 - a2[0]) / (b2[0] - a2[0]);
            [s1, a2[1] + f * (b2[1] - a2[1])]
        };
        let local = |p: [f64; 2]| [p[0] - s1, p[1]];
        let mut entry = None;
        let mut exit = None;
        for step in 0..n_out {
            let seg = &outer[(start_at + step) % n_out];
            let (a2, b2) = to_sw(seg);
            let (ina, inb) = (a2[0] >= s1 - tol, b2[0] >= s1 - tol);
            if step == 0 {
                let e = cut(a2, b2);
                entry = Some(e);
                moved.push(line_seg(&seg.curve.0, local(e), local(b2)));
                continue;
            }
            match (ina, inb) {
                (true, true) => {
                    let mut g = seg.clone();
                    g.start = local(a2);
                    g.end = local(b2);
                    if let SegmentKind::Arc { center, clockwise } = seg.kind {
                        // Developed coordinates keep the sketch's handedness only when (m, u)
                        // is right-handed with the plate normal.
                        let c = local(sw(pf.to_world(center)));
                        let flip = dot(cross(m, u), n) < 0.0;
                        g.kind = SegmentKind::Arc {
                            center: c,
                            clockwise: clockwise ^ flip,
                        };
                    }
                    moved.push(g);
                }
                (true, false) => {
                    if !matches!(seg.kind, SegmentKind::Line) {
                        return Err(self.not_lowered(id, "bends whose strip a curved edge crosses"));
                    }
                    let e = cut(a2, b2);
                    exit = Some(e);
                    moved.push(line_seg(&seg.curve.0, local(a2), local(e)));
                    break;
                }
                _ => {
                    return Err(invalid(
                        self,
                        "line",
                        "the bend line must cross the plate once, from edge to edge".into(),
                    ))
                }
            }
        }
        let (Some(entry), Some(exit)) = (entry, exit) else {
            return Err(invalid(
                self,
                "line",
                "the plate does not cross the bend line".into(),
            ));
        };
        let close_id = format!("{}.root", line.1 .0);
        moved.push(line_seg(&close_id, local(exit), local(entry)));
        let mut loops = vec![moved];
        for inner in plate.loops.iter().skip(1) {
            let lo = inner
                .iter()
                .map(|g| reach(g).0)
                .fold(f64::INFINITY, f64::min);
            if lo > s1 + tol {
                let flip = dot(cross(m, u), n) < 0.0;
                loops.push(
                    inner
                        .iter()
                        .map(|seg| {
                            let (a2, b2) = to_sw(seg);
                            let mut g = seg.clone();
                            g.start = local(a2);
                            g.end = local(b2);
                            if let SegmentKind::Arc { center, clockwise } = seg.kind {
                                g.kind = SegmentKind::Arc {
                                    center: local(sw(pf.to_world(center))),
                                    clockwise: clockwise ^ flip,
                                };
                            }
                            g
                        })
                        .collect(),
                );
            }
        }
        // Walk the section: a cursor on the inner (b-side) face, x along the developed
        // direction, y the face normal; z is the line direction for every local frame.
        let key = line.1 .0.clone();
        let mut o = add(l0, scale(m, s0));
        let (mut x, mut y) = (m, b);
        let mut tools = Vec::new();
        let span = (wa, wb);
        let mut turns = vec![(true, key.clone())];
        if middle.is_some() {
            turns.push((false, format!("{key}.2")));
        }
        for (k, (up, bend_key)) in turns.iter().enumerate() {
            let fr = Frame {
                origin: o,
                x,
                y,
                z: cross(x, y),
            };
            let (wz0, wz1) = {
                let along = dot(u, fr.z);
                let (p, q) = (span.0 * along, span.1 * along);
                let base = dot(sub(l0, o), fr.z);
                (base + p.min(q), base + p.max(q))
            };
            tools.push(sheet_bend_tool(fr, *up, r, t, a, bend_key, (wz0, wz1)));
            // The cursor after the bend.
            let (c, s) = (a.cos(), a.sin());
            let (dx, dy, nx, ny) = if *up {
                (r * s, r - r * c, [c, s], [-s, c])
            } else {
                ((r + t) * s, -t - r + (r + t) * c, [c, -s], [s, c])
            };
            o = add(o, add(scale(x, dx), scale(y, dy)));
            let (x2, y2) = (
                add(scale(x, nx[0]), scale(y, nx[1])),
                add(scale(x, ny[0]), scale(y, ny[1])),
            );
            x = normalize(x2);
            y = normalize(y2);
            if k == 0 {
                if let Some(flat) = middle {
                    if flat > 1e-12 {
                        let fr = frame_of(o, x, u);
                        let sign = if dot(fr.z, y) >= 0.0 { 1.0 } else { -1.0 };
                        let corners = vec![
                            ("y0".to_string(), [0.0, wa]),
                            ("x1".to_string(), [flat, wa]),
                            ("y1".to_string(), [flat, wb]),
                            ("x0".to_string(), [0.0, wb]),
                        ];
                        let mut tool = LabeledTool::new(ToolShape::prism(
                            polygon(fr, &corners),
                            0.0,
                            -t * sign,
                        ));
                        let mid = format!("{key}.mid");
                        for (side, o) in [("y0", 2), ("y1", 3), ("x0", 4), ("x1", 5)] {
                            tool.sides.push((
                                crate::ids::SketchEntityId::new(side),
                                FaceLabel::key(Role::SheetEdge, mid.clone()).with_ordinal(o),
                            ));
                        }
                        tool.start =
                            Some(FaceLabel::key(Role::SheetFace, mid.clone()).with_ordinal(0));
                        tool.end = Some(FaceLabel::key(Role::SheetFace, mid).with_ordinal(1));
                        tools.push(tool);
                        o = add(o, scale(x, flat));
                    }
                }
            }
        }
        // The moved flat, from the last cursor.
        let fr = frame_of(o, x, u);
        let sign = if dot(fr.z, y) >= 0.0 { 1.0 } else { -1.0 };
        let profile = CanonicalProfile {
            frame: fr,
            loops,
            merge_collinear: false,
        };
        let mut flat = LabeledTool::new(ToolShape::prism(profile.clone(), 0.0, -t * sign));
        for lp in &profile.loops {
            for seg in lp {
                let label = if seg.curve.0 == close_id {
                    FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(9)
                } else {
                    FaceLabel::key(Role::SheetEdge, seg.curve.0.clone()).with_ordinal(0)
                };
                flat.sides.push((seg.curve.clone(), label));
            }
        }
        let moved_key = format!("{key}.flat");
        flat.start = Some(FaceLabel::key(Role::SheetFace, moved_key.clone()).with_ordinal(0));
        flat.end = Some(FaceLabel::key(Role::SheetFace, moved_key).with_ordinal(1));
        tools.push(flat);
        // The bend lines on the developed plate, for its flat pattern.
        let to2d = |p: [f64; 3]| {
            let q = sub(p, pf.origin);
            [dot(q, pf.x), dot(q, pf.y)]
        };
        let bd = crate::model::sheet_metal::bend_deduction(&rules.rule, t, r, a)
            .map_err(|m| invalid(self, "angle", m))?;
        let first_up = dot(b, n) > 0.0;
        let mut lines = Vec::new();
        for (k, at) in std::iter::once(0.0)
            .chain(middle.map(|f| ba + f))
            .enumerate()
        {
            let up = first_up == (k == 0);
            lines.push(crate::model::sheet_metal::FlatBendLine {
                key: if k == 0 {
                    key.clone()
                } else {
                    format!("{key}.2")
                },
                feature: id.clone(),
                start: to2d(add(l0, add(scale(m, at), scale(u, wa)))),
                end: to2d(add(l0, add(scale(m, at), scale(u, wb)))),
                angle: a,
                inner_radius: r,
                allowance: ba,
                deduction: bd,
                direction: if up {
                    crate::model::sheet_metal::BendDirection::Up
                } else {
                    crate::model::sheet_metal::BendDirection::Down
                },
            });
        }
        let moved = (to2d(add(l0, scale(m, s0))), [dot(m, pf.x), dot(m, pf.y)]);
        self.record_plate_bend(id, body, &maker, &plate, lines, moved)?;
        // Carve everything of the plate past the fixed flat.
        let cf = frame_of(add(l0, scale(m, s0)), m, u);
        let csign = if dot(cf.z, b) >= 0.0 { 1.0 } else { -1.0 };
        let corners = vec![
            ("c0".to_string(), [0.0, -big]),
            ("c1".to_string(), [big, -big]),
            ("c2".to_string(), [big, big]),
            ("c3".to_string(), [0.0, big]),
        ];
        let mut carve = LabeledTool::new(ToolShape::prism(
            polygon(cf, &corners),
            1e-6 * csign,
            -(t + 1e-6) * csign,
        ));
        carve.sides.push((
            crate::ids::SketchEntityId::new("c0"),
            FaceLabel::key(Role::Relief, key).with_ordinal(0),
        ));
        Ok(op(
            labeled(tools, vec![carve], Combine::Add, false),
            vec![body],
            naming.clone(),
        ))
    }

    // --------------------------------------------------------------------- weldments

    fn weldment(
        &mut self,
        i: usize,
        id: &FeatureId,
        w: &WeldmentFeature,
    ) -> Result<Compiled, Halt> {
        let simple = NamingContext::simple(id, Vec::new());
        match w {
            WeldmentFeature::StructuralMember { groups } => {
                let mut tools = Vec::new();
                let mut cuts_all = Vec::new();
                let mut rows_in = Vec::new();
                for (gi, g) in groups.iter().enumerate() {
                    self.gate_feature(i, id, &g.sketch, &format!("groups[{}].sketch", gi))?;
                    let Some((solved, frame)) = self.st.sketches.get(&g.sketch).cloned() else {
                        return Err(self.error(
                            id,
                            Some("groups"),
                            DiagCode::RefKindMismatch,
                            format!("`{}` is not a sketch", g.sketch),
                            Vec::new(),
                        ));
                    };
                    let rot = self.value(id, "groups.rotation", &g.rotation, Dimension::ANGLE)?;
                    let mut segs = Vec::new();
                    for s in &g.segments {
                        let (Some(a), Some(b)) = (
                            solved.point(s, PointRole::Start),
                            solved.point(s, PointRole::End),
                        ) else {
                            return Err(self.error(
                                id,
                                Some("groups.segments"),
                                DiagCode::RefKindMismatch,
                                format!("`{}` is not a sketch line", s),
                                Vec::new(),
                            ));
                        };
                        segs.push((s.clone(), frame.to_world(a), frame.to_world(b)));
                    }
                    let outline = g.profile.outline(24);
                    let depth = outline[0]
                        .iter()
                        .map(|(_, p)| p[1])
                        .fold(f64::NEG_INFINITY, f64::max)
                        - outline[0]
                            .iter()
                            .map(|(_, p)| p[1])
                            .fold(f64::INFINITY, f64::min);
                    let cuts = member_cuts(&segs, g.corner, depth);
                    for (s, a, b) in &segs {
                        let z = normalize(sub(*b, *a));
                        let seed = if dot(z, frame.z).abs() < 0.9 {
                            frame.z
                        } else {
                            frame.x
                        };
                        let y0 = normalize(sub(seed, scale(z, dot(seed, z))));
                        let y = rotate(y0, z, rot);
                        let x = cross(y, z);
                        let mf = Frame {
                            origin: *a,
                            x,
                            y,
                            z,
                        };
                        let l = dot(sub(*b, *a), z);
                        let loops: Vec<Vec<ProfileSegment>> = outline
                            .iter()
                            .map(|lp| {
                                let n = lp.len();
                                (0..n)
                                    .map(|k| line_seg(&lp[k].0 .0, lp[k].1, lp[(k + 1) % n].1))
                                    .collect()
                            })
                            .collect();
                        // Ends: extended past the sketch point for a mitre and clipped on the
                        // bisector plane; butt joints trim or extend square.
                        let mut ext = [0.0f64; 2];
                        let mut clip = Vec::new();
                        for (end, p) in [(0usize, *a), (1usize, *b)] {
                            let other = segs.iter().find(|(o, c, e)| {
                                o != s && (dist3(*c, p) < 1e-9 || dist3(*e, p) < 1e-9)
                            });
                            let Some((_, c, e)) = other else { continue };
                            let od = if dist3(*c, p) < 1e-9 {
                                normalize(sub(*e, *c))
                            } else {
                                normalize(sub(*c, *e))
                            };
                            let mine = if end == 0 { z } else { scale(z, -1.0) };
                            match g.corner {
                                CornerTreatment::Miter => {
                                    ext[end] = depth;
                                    // The bisector plane through the joint; keep this member's side.
                                    let n = normalize(sub(od, mine));
                                    clip.push(ClipPlane {
                                        origin: p,
                                        normal: n,
                                        label: FaceLabel::key(Role::MemberEnd, s.0.clone())
                                            .with_ordinal(end as u32),
                                    });
                                }
                                CornerTreatment::Butt1 | CornerTreatment::Butt2 => {
                                    let mine_first =
                                        groups[gi].segments.iter().position(|x| x == s)
                                            < groups[gi].segments.iter().position(|x| {
                                                x == other.map(|o| &o.0).expect("other")
                                            });
                                    let through =
                                        (g.corner == CornerTreatment::Butt1) == mine_first;
                                    ext[end] = if through { depth / 2.0 } else { -depth / 2.0 };
                                }
                                CornerTreatment::None => {}
                            }
                        }
                        let mut tool = LabeledTool::new(ToolShape::prism(
                            CanonicalProfile {
                                frame: mf,
                                loops,
                                merge_collinear: false,
                            },
                            -ext[0],
                            l + ext[1],
                        ));
                        for lp in &outline {
                            for (cid, _) in lp {
                                if !tool.sides.iter().any(|(c, _)| c == cid) {
                                    tool.sides.push((
                                        cid.clone(),
                                        FaceLabel::key(Role::MemberSide, format!("{}/{}", s, cid)),
                                    ));
                                }
                            }
                        }
                        tool.start =
                            Some(FaceLabel::key(Role::MemberEnd, s.0.clone()).with_ordinal(0));
                        tool.end =
                            Some(FaceLabel::key(Role::MemberEnd, s.0.clone()).with_ordinal(1));
                        tool.body = Some(FaceLabel::key(Role::Body, s.0.clone()));
                        tool.clip = clip;
                        tools.push(tool);
                    }
                    rows_in.push((
                        format!("{:?}", g.profile),
                        g.material.clone(),
                        None,
                        cuts.clone(),
                    ));
                    cuts_all.extend(cuts);
                }
                let list = cut_list(&rows_in, 1e-6);
                self.st.records.insert(
                    id.clone(),
                    FeatureRecord::Weldment {
                        cuts: cuts_all,
                        cut_list: list,
                    },
                );
                Ok(op(
                    labeled(tools, Vec::new(), Combine::New, true),
                    Vec::new(),
                    simple,
                ))
            }
            WeldmentFeature::EndCap {
                end_face,
                thickness,
                inset,
            } => {
                let h = self.face_handle(i, id, "end_face", end_face)?;
                let Some(fr) = self.face_frame(&h) else {
                    return Err(self.error(
                        id,
                        Some("end_face"),
                        DiagCode::RefKindMismatch,
                        "an end cap sits on a flat end face".into(),
                        Vec::new(),
                    ));
                };
                let t = self.value(id, "thickness", thickness, Dimension::LENGTH)?;
                let inset = match inset {
                    Some(v) => self.value(id, "inset", v, Dimension::LENGTH)?,
                    None => 0.0,
                };
                let g = self.st.bodies[h.body as usize].topology.faces[h.index as usize].geom;
                let corners = face_box(&fr, &g, inset);
                let mut tool = LabeledTool::new(ToolShape::prism(polygon(fr, &corners), 0.0, t));
                for (c, _) in &corners {
                    tool.sides.push((
                        crate::ids::SketchEntityId::new(c.clone()),
                        FaceLabel::key(Role::CapPlate, c.clone()),
                    ));
                }
                tool.end = Some(FaceLabel::key(Role::CapPlate, "outer"));
                tool.start = Some(FaceLabel::key(Role::CapPlate, "inner"));
                tool.body = Some(FaceLabel::key(Role::Body, "cap"));
                Ok(op(
                    labeled(vec![tool], Vec::new(), Combine::New, false),
                    Vec::new(),
                    simple,
                ))
            }
            WeldmentFeature::Gusset {
                faces,
                leg,
                thickness,
            } => {
                let ha = self.face_handle(i, id, "faces[0]", &faces[0])?;
                let hb = self.face_handle(i, id, "faces[1]", &faces[1])?;
                let (Some(na), Some(nb)) = (self.face_normal(&ha), self.face_normal(&hb)) else {
                    return Err(self.error(
                        id,
                        Some("faces"),
                        DiagCode::RefKindMismatch,
                        "a gusset joins two flat faces".into(),
                        Vec::new(),
                    ));
                };
                let leg = self.value(id, "leg", leg, Dimension::LENGTH)?;
                let t = self.value(id, "thickness", thickness, Dimension::LENGTH)?;
                let ga = self.st.bodies[ha.body as usize].topology.faces[ha.index as usize].geom;
                let gb = self.st.bodies[hb.body as usize].topology.faces[hb.index as usize].geom;
                // The corner line: where the two face planes meet; the gusset sits at its
                // middle, in the plane spanned by the two normals.
                let axis = normalize(cross(na, nb));
                let (ha_, hb_) = (dot(na, ga.centroid), dot(nb, gb.centroid));
                let c = dot(na, nb);
                let den = 1.0 - c * c;
                if den.abs() < 1e-12 {
                    return Err(self.error(
                        id,
                        Some("faces"),
                        DiagCode::RegenInvalid,
                        "gusset faces must meet at an angle".into(),
                        Vec::new(),
                    ));
                }
                let p0 = add(
                    scale(na, (ha_ - hb_ * c) / den),
                    scale(nb, (hb_ - ha_ * c) / den),
                );
                let mid = scale(add(ga.centroid, gb.centroid), 0.5);
                let origin = add(p0, scale(axis, dot(sub(mid, p0), axis)));
                let fr = frame_of(origin, nb, na);
                let corners = vec![
                    ("leg_a".to_string(), [0.0, 0.0]),
                    ("hyp".to_string(), [leg, 0.0]),
                    ("leg_b".to_string(), [0.0, leg]),
                ];
                let mut tool =
                    LabeledTool::new(ToolShape::prism(polygon(fr, &corners), -t / 2.0, t / 2.0));
                for (s, _) in &corners {
                    tool.sides.push((
                        crate::ids::SketchEntityId::new(s.clone()),
                        FaceLabel::key(Role::Gusset, s.clone()),
                    ));
                }
                tool.start = Some(FaceLabel::key(Role::Gusset, "side").with_ordinal(0));
                tool.end = Some(FaceLabel::key(Role::Gusset, "side").with_ordinal(1));
                tool.body = Some(FaceLabel::key(Role::Body, "gusset"));
                Ok(op(
                    labeled(vec![tool], Vec::new(), Combine::New, false),
                    Vec::new(),
                    simple,
                ))
            }
            WeldmentFeature::WeldBead { edges, kind } => {
                let leg = match kind {
                    crate::model::weldment::WeldBeadKind::Fillet { leg } => {
                        self.value(id, "kind.leg", leg, Dimension::LENGTH)?
                    }
                    crate::model::weldment::WeldBeadKind::Groove { .. } => {
                        return Err(self.not_lowered(id, "groove welds"))
                    }
                };
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let mut tools = Vec::new();
                for h in &set {
                    let topo = &self.st.bodies[h.body as usize].topology;
                    let e = topo.edges[h.index as usize].clone();
                    let (Some(na), Some(nb)) = (
                        plane_normal(&topo.faces[e.faces[0] as usize].geom),
                        plane_normal(&topo.faces[e.faces[1] as usize].geom),
                    ) else {
                        return Err(self.error(
                            id,
                            Some("edges"),
                            DiagCode::RefKindMismatch,
                            "a fillet weld runs between flat faces".into(),
                            Vec::new(),
                        ));
                    };
                    let fr = frame_of(e.geom.start, na, nb);
                    let s1 = dot(sub(e.geom.end, fr.origin), fr.z);
                    let corners = vec![
                        ("a".to_string(), [0.0, 0.0]),
                        ("face".to_string(), [leg, 0.0]),
                        ("b".to_string(), [0.0, leg]),
                    ];
                    let mut tool = LabeledTool::new(ToolShape::prism(
                        polygon(fr, &corners),
                        s1.min(0.0),
                        s1.max(0.0),
                    ));
                    let tag = KernelTag::InputEdge {
                        edge: OperandEntity {
                            operand: 0,
                            entity: h.index,
                        },
                    };
                    tool.sides.push((
                        crate::ids::SketchEntityId::new("face"),
                        FaceLabel {
                            role: Role::WeldBead,
                            tag: tag.clone(),
                            ordinal: None,
                        },
                    ));
                    tool.start = Some(FaceLabel {
                        role: Role::WeldBead,
                        tag: tag.clone(),
                        ordinal: Some(0),
                    });
                    tool.end = Some(FaceLabel {
                        role: Role::WeldBead,
                        tag,
                        ordinal: Some(1),
                    });
                    tools.push(tool);
                }
                let operands = Self::bodies_of(&set);
                Ok(op(
                    labeled(tools, Vec::new(), Combine::Add, false),
                    operands,
                    simple,
                ))
            }
            WeldmentFeature::TrimExtend { .. } => {
                Err(self.not_lowered(id, "trim/extend of members"))
            }
        }
    }

    // -------------------------------------------------------------------------- mold

    fn mold(&mut self, i: usize, id: &FeatureId, m: &MoldFeature) -> Result<Compiled, Halt> {
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
    fn draft_analysis(&self, body: usize, pull: [f64; 3], min: f64) -> FeatureRecord {
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

    fn thread(&mut self, i: usize, id: &FeatureId, t: &Thread) -> Result<Compiled, Halt> {
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

    fn emboss(&mut self, i: usize, id: &FeatureId, e: &Emboss) -> Result<Compiled, Halt> {
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

    fn rib(&mut self, i: usize, id: &FeatureId, r: &Rib) -> Result<Compiled, Halt> {
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

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}

pub(super) fn plane_normal(g: &FaceGeom) -> Option<[f64; 3]> {
    match g.surface {
        SurfaceKind::Plane { normal } => Some(normalize(normal)),
        _ => None,
    }
}

/// The rectangle a face's box makes in a frame on it, inset.
fn face_box(fr: &Frame, g: &FaceGeom, inset: f64) -> Vec<(String, [f64; 2])> {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for k in 0..8 {
        let c = [
            if k & 1 == 0 {
                g.bbox_min[0]
            } else {
                g.bbox_max[0]
            },
            if k & 2 == 0 {
                g.bbox_min[1]
            } else {
                g.bbox_max[1]
            },
            if k & 4 == 0 {
                g.bbox_min[2]
            } else {
                g.bbox_max[2]
            },
        ];
        let d = sub(c, fr.origin);
        let q = [dot(d, fr.x), dot(d, fr.y)];
        for a in 0..2 {
            lo[a] = lo[a].min(q[a]);
            hi[a] = hi[a].max(q[a]);
        }
    }
    vec![
        ("bottom".to_string(), [lo[0] + inset, lo[1] + inset]),
        ("right".to_string(), [hi[0] - inset, lo[1] + inset]),
        ("top".to_string(), [hi[0] - inset, hi[1] - inset]),
        ("left".to_string(), [lo[0] + inset, hi[1] - inset]),
    ]
}

/// The annular sector of a bend swept along the sketch normal, labelled by the bend key.
/// A bend of a sheet section in `fr` (origin on the inner face where the bend starts, x
/// along the sheet, y the inner face's normal for an upward bend; z along the bend line):
/// `up` turns towards y about a centre `r` above the face, otherwise away from it about a
/// centre `r` below the outer face. Prism over `w` along z.
pub(super) fn sheet_bend_tool(
    fr: Frame,
    up: bool,
    r: f64,
    t: f64,
    angle: f64,
    key: &str,
    w: (f64, f64),
) -> LabeledTool {
    use std::f64::consts::FRAC_PI_2;
    let (c, from, to) = if up {
        ([0.0, r], -FRAC_PI_2, -FRAC_PI_2 + angle)
    } else {
        ([0.0, -t - r], FRAC_PI_2 - angle, FRAC_PI_2)
    };
    let (ri, ro) = (r, r + t);
    let at = |rad: f64, a: f64| [c[0] + rad * a.cos(), c[1] + rad * a.sin()];
    let sid = |s: &str| crate::ids::SketchEntityId::new(s);
    let profile = CanonicalProfile {
        frame: fr,
        loops: vec![vec![
            ProfileSegment {
                curve: sid("in"),
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: false,
                },
                start: at(ri, from),
                end: at(ri, to),
            },
            line_seg("e1", at(ri, to), at(ro, to)),
            ProfileSegment {
                curve: sid("out"),
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: true,
                },
                start: at(ro, to),
                end: at(ro, from),
            },
            line_seg("e0", at(ro, from), at(ri, from)),
        ]],
        merge_collinear: false,
    };
    let mut tool = LabeledTool::new(ToolShape::prism(profile, w.0, w.1));
    // Inner and outer as the bend sees them: an upward bend's inner face is on y's side.
    let (inner, outer) = if up { ("in", "out") } else { ("out", "in") };
    tool.sides.push((
        sid(inner),
        FaceLabel::key(Role::BendFace, key).with_ordinal(0),
    ));
    tool.sides.push((
        sid(outer),
        FaceLabel::key(Role::BendFace, key).with_ordinal(1),
    ));
    tool.sides.push((
        sid("e0"),
        FaceLabel::key(Role::SheetEdge, key).with_ordinal(4),
    ));
    tool.sides.push((
        sid("e1"),
        FaceLabel::key(Role::SheetEdge, key).with_ordinal(5),
    ));
    tool.start = Some(FaceLabel::key(Role::SheetEdge, key).with_ordinal(2));
    tool.end = Some(FaceLabel::key(Role::SheetEdge, key).with_ordinal(3));
    tool
}

pub(super) fn bend_tool(
    frame: Frame,
    b: &crate::model::sheet_metal::SectionBend,
    depth: f64,
    body: Option<FaceLabel>,
) -> LabeledTool {
    let (ri, ro) = (b.inner_radius, b.inner_radius + b.thickness);
    let c = b.center;
    let at = |r: f64, a: f64| [c[0] + r * a.cos(), c[1] + r * a.sin()];
    let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{}.{}", b.key, s));
    let profile = CanonicalProfile {
        frame,
        loops: vec![vec![
            ProfileSegment {
                curve: sid("in"),
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: false,
                },
                start: at(ri, b.from),
                end: at(ri, b.to),
            },
            line_seg(&format!("{}.e1", b.key), at(ri, b.to), at(ro, b.to)),
            ProfileSegment {
                curve: sid("out"),
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: true,
                },
                start: at(ro, b.to),
                end: at(ro, b.from),
            },
            line_seg(&format!("{}.e0", b.key), at(ro, b.from), at(ri, b.from)),
        ]],
        merge_collinear: false,
    };
    let mut tool = LabeledTool::new(ToolShape::prism(profile, 0.0, depth));
    tool.sides.push((
        sid("in"),
        FaceLabel::key(Role::BendFace, b.key.clone()).with_ordinal(0),
    ));
    tool.sides.push((
        sid("out"),
        FaceLabel::key(Role::BendFace, b.key.clone()).with_ordinal(1),
    ));
    tool.sides.push((
        sid("e0"),
        FaceLabel::key(Role::SheetEdge, b.key.clone()).with_ordinal(4),
    ));
    tool.sides.push((
        sid("e1"),
        FaceLabel::key(Role::SheetEdge, b.key.clone()).with_ordinal(5),
    ));
    tool.start = Some(FaceLabel::key(Role::SheetEdge, b.key.clone()).with_ordinal(2));
    tool.end = Some(FaceLabel::key(Role::SheetEdge, b.key.clone()).with_ordinal(3));
    tool.body = body;
    tool
}
