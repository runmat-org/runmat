//! Weldments: structural members along sketch segments, their end cuts and the cut list.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn weldment(
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
                    let profile = g.profile.try_map(|name, size| {
                        self.value(id, &format!("groups.profile.{name}"), size, Dimension::LENGTH)
                    })?;
                    let outline = profile.outline(24);
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
                    // Mass per metre from the material's density (kg/m³) and the section (m²).
                    let per_metre = g
                        .material
                        .as_deref()
                        .and_then(crate::model::materials::material_density)
                        .map(|density| density * profile.area());
                    rows_in.push((
                        profile.description(),
                        g.material.clone(),
                        per_metre,
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
            WeldmentFeature::TrimExtend { members, to } => {
                // The member's end face nearest the target moves onto the target's plane: a
                // member that runs past it is trimmed, one that stops short is extended.
                let set = self.list(i, id, "members", members, None)?;
                let bodies = Self::bodies_of(&set);
                let w = self.face_handle(i, id, "to", to)?;
                if bodies.len() != 1 || bodies[0] == w.body as usize {
                    return Err(self.error(
                        id,
                        Some("members"),
                        DiagCode::RefKindMismatch,
                        "trim or extend one member to a face of another body".into(),
                        Vec::new(),
                    ));
                }
                let target = self.st.bodies[w.body as usize].topology.faces[w.index as usize].geom;
                let Some(n) = plane_normal(&target) else {
                    return Err(self.error(
                        id,
                        Some("to"),
                        DiagCode::RefKindMismatch,
                        "a member trims or extends to a flat face".into(),
                        Vec::new(),
                    ));
                };
                let member = bodies[0];
                // An end face crosses the target's normal (a side face runs along it); of the
                // ends, the one nearest the target's plane.
                let end = self.st.bodies[member]
                    .topology
                    .faces
                    .iter()
                    .enumerate()
                    .filter_map(|(k, f)| {
                        let m = plane_normal(&f.geom)?;
                        (dot(m, n).abs() > 0.3).then(|| (k, dot(n, sub(f.geom.centroid, target.centroid)).abs()))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(k, _)| k as u32);
                let Some(end) = end else {
                    return Err(self.error(
                        id,
                        Some("members"),
                        DiagCode::RegenInvalid,
                        "the member has no end face that meets the target's plane".into(),
                        Vec::new(),
                    ));
                };
                let operands = vec![member, w.body as usize];
                Ok(op(
                    KernelOp::ReplaceFaces {
                        faces: vec![OperandEntity { operand: 0, entity: end }],
                        with: OperandEntity { operand: 1, entity: w.index },
                    },
                    operands,
                    simple,
                ))
            }
        }
    }

    // -------------------------------------------------------------------------- mold
}
