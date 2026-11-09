//! The added datum definitions (frames, planes, axes and points from geometry) as typed records.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn datum_extra(
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
}
