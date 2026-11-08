//! Form tools stamped into a flat of a plate: louvers, dimples and beads.
//!
//! Each form is built in the frame of the face it is stamped on (see
//! [`FormToolShape`]): the sheet lies at `−t ≤ z ≤ 0`, and the form rises out of the face.
//! The lowering first carves the material the form displaces out of the sheet, then adds the
//! formed shell, both in one labelled operation:
//!
//! - louver: the lance's rectangle comes out of the sheet; a hood of thickness `t` hinged at
//!   `y = 0` rises to `height` at `y = width`. The hood's section is a parallelogram of base
//!   `width` and vertical thickness `t`, so the part keeps its volume;
//! - dimple and bead: the area inside the walls comes out of the sheet; the walls (thickness
//!   `t`) rise `height` from the face and a lid of thickness `t` closes them. The part gains
//!   the wall's footprint times `height`.
//!
//! The flat pattern keeps the plate as cut before the forms: a form's material stays where it
//! was on the developed sheet.

use super::*;
use crate::model::sheet_metal::FormToolShape;

fn circle(curve: &str, center: [f64; 2], radius: f64) -> ProfileSegment {
    ProfileSegment {
        curve: crate::ids::SketchEntityId::new(curve),
        kind: SegmentKind::Circle { center, radius },
        start: [center[0] + radius, center[1]],
        end: [center[0] + radius, center[1]],
    }
}

fn rect(key: &str, x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<ProfileSegment> {
    let p = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    (0..4)
        .map(|k| super::super::families::line_seg(&format!("{key}.{k}"), p[k], p[(k + 1) % 4]))
        .collect()
}

/// A prism of `loops` in `frame` over `z0..z1`, its faces labelled `key` by segment.
fn tool(frame: Frame, loops: Vec<Vec<ProfileSegment>>, z0: f64, z1: f64, key: &str) -> LabeledTool {
    let segs: Vec<ProfileSegment> = loops.iter().flatten().cloned().collect();
    let profile = CanonicalProfile {
        frame,
        loops,
        merge_collinear: false,
    };
    let mut t = LabeledTool::new(ToolShape::prism(profile, z0, z1));
    for (k, s) in segs.iter().enumerate() {
        t.sides.push((
            s.curve.clone(),
            FaceLabel::key(Role::SheetFace, key).with_ordinal(k as u32 + 2),
        ));
    }
    t.start = Some(FaceLabel::key(Role::SheetFace, key).with_ordinal(0));
    t.end = Some(FaceLabel::key(Role::SheetFace, key).with_ordinal(1));
    t
}

impl<'a, 'b> Run<'a, 'b> {
    /// Refuses a form tool dimension at or below its least value (for the sheet's thickness).
    fn form_limits(&mut self, id: &FeatureId, dims: &[(&str, f64, f64)]) -> Result<(), Halt> {
        for (field, v, min) in dims {
            if v <= min {
                return Err(self.error(
                    id,
                    Some(&format!("tool.{field}")),
                    DiagCode::RegenInvalid,
                    format!(
                        "a form tool's {field} must exceed {:.3} mm for this sheet",
                        min * 1e3
                    ),
                    Vec::new(),
                ));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::regen::engine) fn form_tool(
        &mut self,
        i: usize,
        id: &FeatureId,
        shape: &FormToolShape,
        face: &Reference,
        at: &crate::model::datum::PointRef,
        rotation: &crate::value::ValueExpr,
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let h = self.face_handle(i, id, "face", face)?;
        let body = h.body as usize;
        let Some((maker, rules, section, _)) = self.sheet_of(body) else {
            return Err(self.error(
                id,
                Some("face"),
                DiagCode::RefKindMismatch,
                "the face is not on a sheet metal body".into(),
                Vec::new(),
            ));
        };
        if section.is_some() {
            return Err(self.not_lowered(id, "form tools on a bent base flange (stamp a plate)"));
        }
        let geom = self.st.bodies[body].topology.faces[h.index as usize].geom;
        let Some(n) = plane_normal(&geom) else {
            return Err(self.error(
                id,
                Some("face"),
                DiagCode::RefKindMismatch,
                "a form tool is stamped on a flat face".into(),
                Vec::new(),
            ));
        };
        let p = self.point(i, id, "at", at)?;
        let p = sub(p, scale(n, dot(sub(p, geom.centroid), n)));
        let turn = self.value(id, "rotation", rotation, Dimension::ANGLE)?;
        // The face frame's x: the plate's x carried with the flat the face lies on.
        let profile_ref = match self.doc.feature(&maker).map(|f| &f.kind) {
            Some(FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { profile, .. })) => {
                profile.clone()
            }
            _ => return Err(self.not_lowered(id, "form tools on this sheet")),
        };
        let pf = self.profile(i, id, "face", &profile_ref)?.frame;
        let t = rules.thickness;
        let (bends, flanges, _) = self.plate_bends_and_others(body, &maker);
        let regions = self.plate_regions(&bends, &flanges);
        let local = |q: [f64; 3]| {
            let q = sub(q, pf.origin);
            [dot(q, pf.x), dot(q, pf.y), dot(q, pf.z)]
        };
        let region = regions.iter().find(|r| {
            let back = r.motion.as_ref().map(super::super::families::inverse);
            let q = local(back.as_ref().map_or(p, |m| m.apply(p)));
            q[2] >= -1e-7 && q[2] <= t + 1e-7 && r.holds([q[0], q[1]], 1e-7)
        });
        let Some(region) = region else {
            return Err(self.not_lowered(
                id,
                "form tools on a bend zone or a jog's middle flat (stamp a flat)",
            ));
        };
        let xb = region
            .motion
            .as_ref()
            .map_or(pf.x, |m| super::super::families::apply_rotation(m, pf.x));
        let xb = normalize(sub(xb, scale(n, dot(xb, n))));
        let (s, c) = turn.sin_cos();
        let x = add(scale(xb, c), scale(cross(n, xb), s));
        let fr = Frame {
            origin: p,
            x,
            y: cross(n, x),
            z: n,
        };
        let e = 1e-6;
        let key = |s: &str| format!("{}.{s}", id.0);
        let (tools, carve) = match shape {
            FormToolShape::Louver {
                length,
                width,
                height,
            } => {
                let l = self.value(id, "tool.length", length, Dimension::LENGTH)?;
                let w = self.value(id, "tool.width", width, Dimension::LENGTH)?;
                let hh = self.value(id, "tool.height", height, Dimension::LENGTH)?;
                self.form_limits(
                    id,
                    &[("length", l, 0.0), ("width", w, 0.0), ("height", hh, t)],
                )?;
                let lance = tool(
                    fr,
                    vec![rect(&key("lance"), -l / 2.0, 0.0, l / 2.0, w)],
                    -t - e,
                    e,
                    &key("lance"),
                );
                // The hood's section in (y, z), swept along x.
                let hood_frame = Frame {
                    origin: add(p, scale(x, -l / 2.0)),
                    x: fr.y,
                    y: fr.z,
                    z: fr.x,
                };
                let pts = [[0.0, -t], [w, hh - t], [w, hh], [0.0, 0.0]];
                let hood: Vec<ProfileSegment> = (0..4)
                    .map(|k| {
                        super::super::families::line_seg(
                            &format!("{}.{k}", key("hood")),
                            pts[k],
                            pts[(k + 1) % 4],
                        )
                    })
                    .collect();
                (
                    vec![tool(hood_frame, vec![hood], 0.0, l, &key("hood"))],
                    vec![lance],
                )
            }
            FormToolShape::Dimple { diameter, height } => {
                let d = self.value(id, "tool.diameter", diameter, Dimension::LENGTH)?;
                let hh = self.value(id, "tool.height", height, Dimension::LENGTH)?;
                self.form_limits(id, &[("diameter", d, 2.0 * t), ("height", hh, t)])?;
                let r = d / 2.0;
                let ri = r - t;
                let o = [0.0, 0.0];
                (
                    vec![
                        tool(
                            fr,
                            vec![
                                vec![circle(&key("wall.o"), o, r)],
                                vec![circle(&key("wall.i"), o, ri)],
                            ],
                            0.0,
                            hh,
                            &key("wall"),
                        ),
                        tool(
                            fr,
                            vec![vec![circle(&key("lid"), o, ri)]],
                            hh - t,
                            hh,
                            &key("lid"),
                        ),
                    ],
                    vec![tool(
                        fr,
                        vec![vec![circle(&key("hole"), o, ri)]],
                        -t - e,
                        e,
                        &key("hole"),
                    )],
                )
            }
            FormToolShape::Bead {
                length,
                width,
                height,
            } => {
                let l = self.value(id, "tool.length", length, Dimension::LENGTH)?;
                let w = self.value(id, "tool.width", width, Dimension::LENGTH)?;
                let hh = self.value(id, "tool.height", height, Dimension::LENGTH)?;
                self.form_limits(
                    id,
                    &[
                        ("length", l, 2.0 * t),
                        ("width", w, 2.0 * t),
                        ("height", hh, t),
                    ],
                )?;
                let (a, b) = (l / 2.0, w / 2.0);
                let (ai, bi) = (a - t, b - t);
                (
                    vec![
                        tool(
                            fr,
                            vec![
                                rect(&key("wall.o"), -a, -b, a, b),
                                rect(&key("wall.i"), -ai, -bi, ai, bi),
                            ],
                            0.0,
                            hh,
                            &key("wall"),
                        ),
                        tool(
                            fr,
                            vec![rect(&key("lid"), -ai, -bi, ai, bi)],
                            hh - t,
                            hh,
                            &key("lid"),
                        ),
                    ],
                    vec![tool(
                        fr,
                        vec![rect(&key("hole"), -ai, -bi, ai, bi)],
                        -t - e,
                        e,
                        &key("hole"),
                    )],
                )
            }
        };
        self.st.records.insert(
            id.clone(),
            FeatureRecord::PlateForm {
                base: maker.clone(),
            },
        );
        let made = labeled(tools, carve, Combine::Add, false);
        self.st.replays.insert(id.clone(), made.clone());
        Ok(super::super::families::op(made, vec![body], naming.clone()))
    }
}
