//! Sheet-metal edge, miter and hem flanges: the flange tool built on a plate edge.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    /// Flanges (and hems) on sheet edges: a bend and a flat grown from each edge's thickness
    /// face, labelled by the edge they grow from.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn flange(
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
    pub(super) fn flange_with(
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
        let Some((maker, rules, section, _)) = self.sheet_of(body) else {
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
        // On a flat plate, what each flange develops to on the plate's flat pattern.
        let mut developed = Vec::new();
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
            developed.push(crate::regen::engine::sheet::FlangeEdgeIn {
                start: e.geom.start,
                end: e.geom.end,
                frame: fr,
                shift,
                theta,
                radius: r,
                flat,
            });
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
        let made = labeled(tools, carve, Combine::Add, false);
        if section.is_none() {
            self.record_plate_flange(i, id, body, &maker, &rules, developed)?;
            // A fold of the plate replays the flange after the plate's bends before it.
            self.st.replays.insert(id.clone(), made.clone());
        }
        Ok(op(made, operands, naming.clone()))
    }
}
